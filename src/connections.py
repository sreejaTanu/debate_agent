"""Connection factory for multi-model debate agent.

Handles AWS Bedrock session creation and model initialization.
"""

import os
from pathlib import Path

import boto3
import yaml
from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel

_CONFIG_CACHE = None


def _resolve_config_path() -> Path:
    env_path = os.getenv("CONFIG_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path(__file__).resolve().parents[1] / "config" / "config.yaml")

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "config.yaml not found. Checked: " + ", ".join(str(p) for p in candidates)
    )


def load_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    config_path = _resolve_config_path()
    with open(config_path, "r") as f:
        _CONFIG_CACHE = yaml.safe_load(f) or {}
    return _CONFIG_CACHE


def _build_session(region_name: str) -> boto3.Session:
    """Build a boto3 session from environment variables only (no ~/.aws profiles)."""
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")

    if access_key and secret_key:
        return boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
            region_name=region_name,
        )

    # Bedrock API key: boto3 reads AWS_BEARER_TOKEN_BEDROCK from the environment itself.
    if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        return boto3.Session(region_name=region_name)

    raise EnvironmentError(
        "No AWS credentials in environment. Set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY "
        "(optionally AWS_SESSION_TOKEN), or AWS_BEARER_TOKEN_BEDROCK. "
        "To test without credentials, run with --dry-run."
    )


def get_bedrock_client(region_name: str = "us-east-1"):
    config = load_config()
    aws_cfg = config.get("aws", {})
    region = os.getenv("AWS_REGION") or aws_cfg.get("region_name", region_name)

    session = _build_session(region)
    boto_cfg = BotoConfig(
        read_timeout=300,
        connect_timeout=300,
        retries={"max_attempts": 10, "mode": "standard"},
    )
    return session.client("bedrock-runtime", region_name=region, config=boto_cfg), boto_cfg


def create_bedrock_llm(
    model_id: str,
    bedrock_client,
    boto_cfg: BotoConfig,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> BaseChatModel:
    llm_kwargs = {
        "model": model_id,
        "client": bedrock_client,
        "max_tokens": max_tokens,
        "config": boto_cfg,
    }
    if "opus-4-8" not in (model_id or ""):
        llm_kwargs["temperature"] = temperature

    return ChatBedrockConverse(**llm_kwargs)
