"""Local open-weight models via Ollama (free, runs on CPU). Translates Anthropic-format messages."""

from __future__ import annotations

import json
import time
import uuid

import httpx

from retailpulse.llm.base import LLMResponse, Usage


def to_ollama_messages(system: str, messages: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    tool_names: dict[str, str] = {}
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue
        if msg["role"] == "assistant":
            text = "".join(b.get("text", "") for b in content if b["type"] == "text")
            calls = []
            for b in content:
                if b["type"] == "tool_use":
                    tool_names[b["id"]] = b["name"]
                    calls.append({"function": {"name": b["name"], "arguments": b["input"]}})
            out.append({"role": "assistant", "content": text, **({"tool_calls": calls} if calls else {})})
        else:
            for b in content:
                if b["type"] == "tool_result":
                    body = b["content"] if isinstance(b["content"], str) else json.dumps(b["content"])
                    out.append(
                        {"role": "tool", "content": body, "tool_name": tool_names.get(b["tool_use_id"], "")}
                    )
                elif b["type"] == "text":
                    out.append({"role": "user", "content": b["text"]})
    return out


def to_ollama_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
        }
        for t in tools
    ]


def from_ollama_message(message: dict) -> list[dict]:
    blocks: list[dict] = []
    if message.get("content"):
        blocks.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        fn = call["function"]
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args)
        blocks.append(
            {"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:20]}", "name": fn["name"], "input": args}
        )
    return blocks


class OllamaLLM:
    def __init__(self, model: str, host: str, timeout_s: float = 600.0):
        self.model = model
        self.http = httpx.Client(base_url=host, timeout=timeout_s)

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        start = time.perf_counter()
        r = self.http.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0},
                "messages": to_ollama_messages(system, messages),
                "tools": to_ollama_tools(tools),
            },
        )
        r.raise_for_status()
        body = r.json()
        content = from_ollama_message(body["message"])
        return LLMResponse(
            content=content,
            stop_reason="tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn",
            model=self.model,
            usage=Usage(
                input_tokens=body.get("prompt_eval_count", 0), output_tokens=body.get("eval_count", 0)
            ),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
