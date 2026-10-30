"""Multi-vendor model abstraction layer.

Provides a unified interface to call LLMs from different vendors
(AWS Bedrock, OpenAI) through LangChain.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from connections import create_bedrock_llm, get_bedrock_client, load_config

logger = logging.getLogger(__name__)

PRICE_PER_1K = {
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 0.003, "output": 0.015},
    "us.meta.llama3-3-70b-instruct-v1:0":           {"input": 0.00072, "output": 0.00072},
    "meta.llama3-70b-instruct-v1:0":                 {"input": 0.00265, "output": 0.0035},
    "mistral.mistral-large-2402-v1:0":               {"input": 0.004, "output": 0.012},
    "us.anthropic.claude-opus-4-20250514-v1:0":      {"input": 0.015, "output": 0.075},
    "us.anthropic.claude-haiku-3-5-20241022-v1:0":   {"input": 0.0008, "output": 0.004},
    "us.amazon.nova-pro-v1:0":                       {"input": 0.0008, "output": 0.0032},
    "cohere.command-r-plus-v1:0":                    {"input": 0.003, "output": 0.015},
    "gpt-4o":                                        {"input": 0.0025, "output": 0.01},
}
# Unknown models are priced pessimistically so the ceiling errs on the safe side.
_DEFAULT_PRICE = {"input": 0.015, "output": 0.075}

# Hard spend ceiling, written into the code. config.yaml may lower it but never raise it.
HARD_CEILING_USD = 25.00


class SpendCeilingExceeded(RuntimeError):
    pass


class SpendCeiling:
    """Tracks cumulative spend and enforces a hard dollar ceiling."""

    def __init__(self, max_usd: float = 5.00, warn_pct: float = 80):
        self.max_usd = min(max_usd, HARD_CEILING_USD)
        self.warn_pct = warn_pct
        self.total_usd = 0.0
        self._warned = False
        # Usage is counted for every call, free (local) models included, to measure cost-to-serve.
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def snapshot(self) -> Dict[str, float]:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "usd": self.total_usd}

    def record(self, model_id: str, input_tokens: int, output_tokens: int, free: bool = False) -> float:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if free:
            return 0.0
        prices = PRICE_PER_1K.get(model_id, _DEFAULT_PRICE)
        cost = (input_tokens / 1000) * prices["input"] + (output_tokens / 1000) * prices["output"]
        self.total_usd += cost

        if not self._warned and self.total_usd >= self.max_usd * (self.warn_pct / 100):
            logger.warning(f"Spend warning: ${self.total_usd:.4f} of ${self.max_usd:.2f} ceiling ({self.warn_pct}% threshold)")
            self._warned = True

        if self.total_usd > self.max_usd:
            raise SpendCeilingExceeded(
                f"Spend ceiling exceeded: ${self.total_usd:.4f} > ${self.max_usd:.2f} limit. "
                f"Raise spend_ceiling.max_total_usd in config.yaml (capped at HARD_CEILING_USD "
                f"= ${HARD_CEILING_USD:.2f} in src/models.py)."
            )
        return cost


_global_spend: Optional[SpendCeiling] = None


def get_spend_tracker(config: Optional[dict] = None) -> SpendCeiling:
    global _global_spend
    if _global_spend is None:
        if config is None:
            config = load_config()
        ceiling_cfg = config.get("spend_ceiling", {})
        _global_spend = SpendCeiling(
            max_usd=ceiling_cfg.get("max_total_usd", 5.00),
            warn_pct=ceiling_cfg.get("warn_at_pct", 80),
        )
    return _global_spend


@dataclass
class ModelResponse:
    model_label: str
    vendor: str
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    truncated: bool = False


@dataclass
class DebateModel:
    model_id: str
    vendor: str
    label: str
    llm: BaseChatModel = field(repr=False, default=None)

    def invoke(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))

        start = time.time()
        response = self.llm.invoke(messages)
        latency = (time.time() - start) * 1000

        usage = response.usage_metadata or {}
        input_tok = usage.get("input_tokens", 0)
        output_tok = usage.get("output_tokens", 0)

        tracker = get_spend_tracker()
        cost = tracker.record(self.model_id, input_tok, output_tok, free=_is_free(self.llm))

        meta = getattr(response, "response_metadata", None) or {}
        stop = meta.get("done_reason") or meta.get("stopReason") or meta.get("finish_reason")
        truncated = stop in ("length", "max_tokens")
        if truncated:
            logger.warning(f"{self.label}: output hit the max_tokens limit and was cut off")
        # Ollama silently drops the start of a prompt that overflows num_ctx — make that visible.
        num_ctx = getattr(self.llm, "num_ctx", None)
        if num_ctx and input_tok >= num_ctx - 64:
            logger.warning(f"{self.label}: prompt filled the {num_ctx}-token context window; "
                           f"its beginning may have been dropped — raise num_ctx in config.yaml")

        return ModelResponse(
            model_label=self.label,
            vendor=self.vendor,
            content=_message_text(response.content),
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=latency,
            cost_usd=cost,
            truncated=truncated,
        )


def _message_text(content) -> str:
    """Flatten LangChain message content (str or list of content blocks) to plain text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


class MockChatModel:
    """Offline stand-in for an LLM, used by --dry-run to exercise the full pipeline at zero cost.

    Returns well-formed outputs for each prompt type (answers, scores, verdicts) with
    deterministic per-model variation, so parsing, aggregation and reporting all run for real.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._calls = 0

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self._calls += 1
        prompt = "\n".join(_message_text(m.content) for m in messages)
        seed = (sum(map(ord, self.model_id)) + self._calls) % 7

        if "COMPLETENESS: <score>/10" in prompt:
            s = [4 + (seed + 2 * i) % 6 for i in range(4)]
            text = (f"COMPLETENESS: {s[0]}/10\nACCURACY: {s[1]}/10\nREASONING: {s[2]}/10\n"
                    f"NUANCE: {s[3]}/10\nJUSTIFICATION: [mock] deterministic dry-run score.")
        elif "WINNER: A or B or TIE" in prompt:
            text = f"WINNER: {'AB'[seed % 2]}\nREASON: [mock] deterministic dry-run preference."
        elif "VERDICT: ACCEPT or REVERT" in prompt:
            text = "BASE_SCORE: 7/10\nMERGED_SCORE: 8/10\nVERDICT: ACCEPT\nREASON: [mock] merge adds coverage."
        elif "Suggest improvements" in prompt:
            text = f"[mock {self.model_id}] Add a worked example and a counter-argument."
        else:
            text = (f"[mock {self.model_id}] Answer #{self._calls}. Key considerations, tradeoffs "
                    f"and a recommendation would appear here. " * (2 + seed))
        return AIMessage(content=text, usage_metadata={
            "input_tokens": len(prompt) // 4, "output_tokens": len(text) // 4,
            "total_tokens": (len(prompt) + len(text)) // 4,
        })


def _init_openai_model(model_id: str, temperature: float, max_tokens: int) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY env var required for OpenAI models")
    return ChatOpenAI(model=model_id, temperature=temperature, max_tokens=max_tokens, api_key=api_key)


def _init_ollama_model(mcfg: Dict, temperature: float, max_tokens: int) -> BaseChatModel:
    """Local open-weight model served by Ollama: no account, no key, no spend."""
    from langchain_ollama import ChatOllama
    kwargs = {
        "model": mcfg["id"],
        "temperature": temperature,
        "num_predict": max_tokens,
        # Debate prompts carry all three previous answers; Ollama's default context would truncate them.
        "num_ctx": mcfg.get("num_ctx", 16384),
        "base_url": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    }
    if "reasoning" in mcfg:
        kwargs["reasoning"] = mcfg["reasoning"]
    return ChatOllama(**kwargs)


def _is_free(llm) -> bool:
    return isinstance(llm, MockChatModel) or type(llm).__name__ == "ChatOllama"


def is_dry_run() -> bool:
    return os.getenv("DEBATE_DRY_RUN", "").lower() in ("1", "true", "yes")


def build_llm(mcfg: Dict, temperature: float, max_tokens: int, _clients: Dict) -> BaseChatModel:
    """Create the LLM for one model config entry. `_clients` caches the Bedrock client."""
    vendor = mcfg["vendor"]
    if is_dry_run() or vendor == "mock":
        return MockChatModel(mcfg["id"])
    if vendor == "bedrock":
        if "bedrock" not in _clients:
            _clients["bedrock"] = get_bedrock_client()
        bedrock_client, boto_cfg = _clients["bedrock"]
        return create_bedrock_llm(mcfg["id"], bedrock_client, boto_cfg, temperature, max_tokens)
    if vendor == "openai":
        return _init_openai_model(mcfg["id"], temperature, max_tokens)
    if vendor == "ollama":
        return _init_ollama_model(mcfg, temperature, max_tokens)
    raise ValueError(f"Unknown vendor: {vendor}")


def initialize_models(config: Optional[dict] = None) -> List[DebateModel]:
    if config is None:
        config = load_config()

    debate_cfg = config.get("debate", {})
    temperature = debate_cfg.get("temperature", 0.7)
    max_tokens = debate_cfg.get("max_tokens", 4096)

    model_configs = config.get("models", [])
    if len(model_configs) < 3:
        raise ValueError(f"Need at least 3 models, got {len(model_configs)}")

    model_vendors = {m.get("model_vendor", m["vendor"]) for m in model_configs}
    if len(model_vendors) < 2:
        raise ValueError(f"Need at least 2 model vendors, got {model_vendors}")

    clients: Dict = {}
    models: List[DebateModel] = []

    for mcfg in model_configs:
        llm = build_llm(mcfg, temperature, max_tokens, clients)
        model_vendor = mcfg.get("model_vendor", mcfg["vendor"])
        models.append(DebateModel(model_id=mcfg["id"], vendor=model_vendor, label=mcfg["label"], llm=llm))

    logger.info(f"Initialized {len(models)} models: {[m.label for m in models]}")
    return models
