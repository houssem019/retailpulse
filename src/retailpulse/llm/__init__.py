from __future__ import annotations

from pathlib import Path

from retailpulse.config import Settings, settings
from retailpulse.llm.base import LLMClient, LLMResponse, Usage

__all__ = ["LLMClient", "LLMResponse", "Usage", "get_llm"]


def get_llm(cfg: Settings = settings, cassette: Path | None = None) -> LLMClient:
    """Build the configured LLM. `fake` replays recorded responses from `cassette` (CI default)."""
    if cfg.llm_provider == "anthropic":
        from retailpulse.llm.anthropic_client import AnthropicLLM

        return AnthropicLLM(cfg.llm_model, effort=cfg.llm_effort)
    if cfg.llm_provider == "ollama":
        from retailpulse.llm.ollama_client import OllamaLLM

        return OllamaLLM(cfg.ollama_model, cfg.ollama_host)
    from retailpulse.llm.replay import ReplayLLM

    return ReplayLLM(cassette or cfg.cassette_dir / "default.json")
