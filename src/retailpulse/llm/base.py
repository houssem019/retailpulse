"""Provider-neutral LLM interface used by the agent.

Messages and tool calls use the Anthropic Messages format as the canonical shape (content blocks:
text / tool_use / tool_result / thinking). Other providers translate to and from it, so the agent
graph never contains provider-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

# USD per million tokens: (input, output, cache read). Keep in sync with provider pricing pages.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def cost_usd(self, model: str) -> float:
        inp, out, cached = PRICING.get(model, (0.0, 0.0, 0.0))  # local models are free
        return (
            self.input_tokens * inp
            + self.cache_write_tokens * inp * 1.25
            + self.cache_read_tokens * cached
            + self.output_tokens * out
        ) / 1e6


@dataclass
class LLMResponse:
    content: list[dict]  # Anthropic-format content blocks, kept verbatim (thinking blocks included)
    stop_reason: str  # end_turn | tool_use | max_tokens | refusal
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")

    @property
    def tool_calls(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...
