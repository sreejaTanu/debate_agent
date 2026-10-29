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
}
_DEFAULT_PRICE = {"input": 0.005, "output": 0.015}


class SpendCeiling:
    """Tracks cumulative spend and enforces a hard dollar ceiling."""

    def __init__(self, max_usd: float = 5.00, warn_pct: float = 80):
        self.max_usd = max_usd
        self.warn_pct = warn_pct
        self.total_usd = 0.0
        self._warned = False

    def record(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        prices = PRICE_PER_1K.get(model_id, _DEFAULT_PRICE)
        cost = (input_tokens / 1000) * prices["input"] + (output_tokens / 1000) * prices["output"]
        self.total_usd += cost

        if not self._warned and self.total_usd >= self.max_usd * (self.warn_pct / 100):
            logger.warning(f"Spend warning: ${self.total_usd:.4f} of ${self.max_usd:.2f} ceiling ({self.warn_pct}% threshold)")
            self._warned = True

        if self.total_usd > self.max_usd:
            raise RuntimeError(
                f"Spend ceiling exceeded: ${self.total_usd:.4f} > ${self.max_usd:.2f} limit. "
                f"Increase spend_ceiling.max_total_usd in config.yaml to continue."
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
        cost = tracker.record(self.model_id, input_tok, output_tok)

        return ModelResponse(
            model_label=self.label,
            vendor=self.vendor,
            content=response.content,
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=latency,
            cost_usd=cost,
        )


def _init_openai_model(model_id: str, temperature: float, max_tokens: int) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY env var required for OpenAI models")
    return ChatOpenAI(model=model_id, temperature=temperature, max_tokens=max_tokens, api_key=api_key)


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

    bedrock_client = None
    boto_cfg = None
    models: List[DebateModel] = []

    for mcfg in model_configs:
        vendor = mcfg["vendor"]
        model_id = mcfg["id"]
        label = mcfg["label"]

        if vendor == "bedrock":
            if bedrock_client is None:
                bedrock_client, boto_cfg = get_bedrock_client()
            llm = create_bedrock_llm(model_id, bedrock_client, boto_cfg, temperature, max_tokens)
        elif vendor == "openai":
            llm = _init_openai_model(model_id, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown vendor: {vendor}")

        model_vendor = mcfg.get("model_vendor", vendor)
        models.append(DebateModel(model_id=model_id, vendor=model_vendor, label=label, llm=llm))

    logger.info(f"Initialized {len(models)} models: {[m.label for m in models]}")
    return models
