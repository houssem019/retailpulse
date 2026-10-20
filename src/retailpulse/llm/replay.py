"""Deterministic LLM doubles for tests and CI: no API key, no network, no cost.

* ScriptedLLM - returns a fixed list of responses in order (unit tests).
* ReplayLLM   - a "cassette": in record mode it wraps a real LLM and saves every response keyed by a
  hash of the request (system prompt + tools + conversation, minus tool result payloads); in replay
  mode it serves them back. Any change to the prompt, tool definitions or the agent's own decisions
  changes the hash, so CI fails loudly ("cassette miss") instead of silently testing stale
  behaviour. Re-record with a real model then.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from retailpulse.llm.base import LLMClient, LLMResponse, Usage


def _block(text: str | None = None, tool: tuple[str, dict] | None = None, idx: int = 0) -> dict:
    if tool:
        return {"type": "tool_use", "id": f"toolu_scripted_{idx}", "name": tool[0], "input": tool[1]}
    return {"type": "text", "text": text or ""}


class ScriptedLLM:
    """Each script step is either a final answer string or a (tool_name, tool_input) tuple / list of them."""

    def __init__(self, steps: list, model: str = "scripted"):
        self.model = model
        self.steps = list(steps)
        self.calls: list[dict] = []

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        step = self.steps.pop(0)
        if isinstance(step, str):
            return LLMResponse([_block(step)], "end_turn", self.model, Usage(100, 20))
        calls = step if isinstance(step, list) else [step]
        blocks = [_block(tool=c, idx=len(self.calls) * 10 + i) for i, c in enumerate(calls)]
        return LLMResponse(blocks, "tool_use", self.model, Usage(100, 20))


class CassetteMiss(RuntimeError):
    pass


def _without_tool_output(messages: list[dict]) -> list[dict]:
    """Drop tool *results* from the key. They are data, not decisions: a forecast that differs in the 6th
    decimal between Windows and Linux (thread scheduling in LightGBM) must not invalidate a recording.
    Changed data is still caught, because answers are graded against gold values computed at eval time."""
    out = []
    for m in messages:
        if isinstance(m["content"], list):
            m = {
                **m,
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": b["tool_use_id"],
                        "is_error": b.get("is_error", False),
                    }
                    if b.get("type") == "tool_result"
                    else b
                    for b in m["content"]
                ],
            }
        out.append(m)
    return out


def request_key(system: str, messages: list[dict], tools: list[dict]) -> str:
    payload = json.dumps(
        {"system": system, "messages": _without_tool_output(messages), "tools": tools},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class ReplayLLM:
    def __init__(self, cassette: Path, inner: LLMClient | None = None):
        """inner=None -> replay only; inner given -> record (calls inner on misses and saves)."""
        self.cassette = cassette
        self.inner = inner
        self.model = inner.model if inner else "replay"
        self.entries: dict = json.loads(cassette.read_text(encoding="utf-8")) if cassette.exists() else {}

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        key = request_key(system, messages, tools)
        if key in self.entries:
            e = self.entries[key]
            return LLMResponse(
                e["content"], e["stop_reason"], e["model"], Usage(**e["usage"]), e["latency_ms"]
            )
        if self.inner is None:
            raise CassetteMiss(
                f"No recorded response for request {key} in {self.cassette.name}. The prompt, tools or data "
                "changed: re-record with a real model (see README: evals)."
            )
        resp = self.inner.complete(system, messages, tools)
        self.entries[key] = {
            "content": resp.content,
            "stop_reason": resp.stop_reason,
            "model": resp.model,
            "usage": vars(resp.usage),
            "latency_ms": resp.latency_ms,
        }
        self.cassette.parent.mkdir(parents=True, exist_ok=True)
        self.cassette.write_text(json.dumps(self.entries, indent=1, default=str), encoding="utf-8")
        return resp
