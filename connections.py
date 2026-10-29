"""Connection factory for multi-model debate agent.

Handles AWS Bedrock session creation and model initialization.
"""

import os
from pathlib import Path
from typing import Optional

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


def _build_session(region_name: str, profile_name: Optional[str] = None) -> boto3.Session:
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

    profile_name = profile_name or os.getenv("AWS_PROFILE") or os.getenv("AWS_DEFAULT_PROFILE")
    if profile_name:
        return boto3.Session(profile_name=profile_name, region_name=region_name)

    return boto3.Session(region_name=region_name)


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
