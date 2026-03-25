"""
Configuration loader for wakanda-voice pipeline.
Reads YAML config and provides typed access to all settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelServerConfig:
    url: str
    model: str
    streaming: bool = False
    language: Optional[str] = None
    sample_rate: int = 16000
    options: dict = field(default_factory=dict)


@dataclass
class LLMConfig:
    url: str
    model: str
    provider: str = "openai_compat"
    api_key: Optional[str] = None
    options: dict = field(default_factory=dict)


@dataclass
class VADConfig:
    threshold: float = 0.5
    min_silence_ms: int = 700
    min_speech_ms: int = 250


@dataclass
class S2STConfig:
    source_lang: str = "yor_Latn"
    target_lang: str = "eng_Latn"


@dataclass
class ToolConfig:
    enabled: bool = False
    provider: str = ""
    api_key: Optional[str] = None


@dataclass
class LoggingConfig:
    dir: str = "./logs"
    format: str = "jsonl"


@dataclass
class PipelineConfig:
    # LiveKit
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "secret"

    # Pipeline mode
    mode: str = "s2st"  # "s2st" | "multilingual_voice_bot"
    default_lang: str = "yor"

    # Model servers
    stt: ModelServerConfig = field(default_factory=lambda: ModelServerConfig(
        url="ws://localhost:8001",
        model="facebook/mms-1b-all",
    ))
    mt: ModelServerConfig = field(default_factory=lambda: ModelServerConfig(
        url="ws://localhost:8002",
        model="facebook/nllb-200-3.3B",
    ))
    tts: ModelServerConfig = field(default_factory=lambda: ModelServerConfig(
        url="ws://localhost:8003",
        model="facebook/mms-tts-yor",
        sample_rate=16000,
    ))
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(
        url="http://localhost:8000/v1",
        model="almanach/AfriqueLLM-Qwen2.5-3B",
    ))

    # VAD
    vad: VADConfig = field(default_factory=VADConfig)

    # S2ST specific
    s2st: S2STConfig = field(default_factory=S2STConfig)

    # Tools
    tools: dict[str, ToolConfig] = field(default_factory=dict)

    # Logging
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _resolve_env(value):
    """Resolve ${ENV_VAR} references in string values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_key = value[2:-1]
        return os.environ.get(env_key, "")
    return value


def _resolve_dict(d: dict) -> dict:
    """Recursively resolve environment variables in a dict."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _resolve_dict(v)
        elif isinstance(v, str):
            result[k] = _resolve_env(v)
        else:
            result[k] = v
    return result


def load_config(path: str | Path) -> PipelineConfig:
    """Load pipeline configuration from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    raw = _resolve_dict(raw)

    # Build config from raw dict
    config = PipelineConfig()

    # LiveKit
    if "livekit" in raw:
        lk = raw["livekit"]
        config.livekit_url = lk.get("url", config.livekit_url)
        config.livekit_api_key = lk.get("api_key", config.livekit_api_key)
        config.livekit_api_secret = lk.get("api_secret", config.livekit_api_secret)

    config.mode = raw.get("mode", config.mode)
    config.default_lang = raw.get("default_lang", config.default_lang)

    # Model servers
    if "stt" in raw:
        config.stt = ModelServerConfig(**{k: v for k, v in raw["stt"].items()})
    if "mt" in raw:
        config.mt = ModelServerConfig(**{k: v for k, v in raw["mt"].items()})
    if "tts" in raw:
        config.tts = ModelServerConfig(**{k: v for k, v in raw["tts"].items()})
    if "llm" in raw:
        config.llm = LLMConfig(**{k: v for k, v in raw["llm"].items()})

    # VAD
    if "vad" in raw:
        config.vad = VADConfig(**raw["vad"])

    # S2ST
    if "s2st" in raw:
        config.s2st = S2STConfig(**raw["s2st"])

    # Tools
    if "tools" in raw:
        for name, tool_raw in raw["tools"].items():
            config.tools[name] = ToolConfig(**tool_raw)

    # Logging
    if "logging" in raw:
        config.logging = LoggingConfig(**raw["logging"])

    return config
