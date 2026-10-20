"""Tracing and run logging for the agent.

Every question produces:
  * one structured JSON log line (latency, tokens, cost, tools, stop reason) - always on, zero setup;
  * a Langfuse trace when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set:
        agent "retailpulse.ask"
          ├─ generation "llm"  (model, token usage, cost, latency, tool calls requested)
          ├─ tool "run_sql"    (input, output, sql, error flag, approval)
          └─ ...
Spans are created explicitly from the root (not via implicit OpenTelemetry context), so nesting stays
correct even when LangGraph runs nodes on worker threads or a run pauses for approval and resumes
in a later HTTP request.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from retailpulse.llm.base import LLMResponse

log = logging.getLogger("retailpulse.agent")


class Span:
    def end(self, **data: Any) -> None:  # noqa: B027 - intentionally a no-op default
        pass


class Run:
    """One question. The base class only logs; LangfuseRun also exports a trace."""

    def __init__(self, question: str, thread_id: str):
        self.question = question
        self.thread_id = thread_id

    def start_llm(self, step: int, last_message: dict) -> Span:
        return Span()

    def start_tool(self, name: str, args: dict) -> Span:
        return Span()

    def finish(self, result) -> None:
        log.info(
            json.dumps(
                {
                    "event": "agent_run",
                    "thread_id": self.thread_id,
                    "question": self.question[:200],
                    "stop": result.stop,
                    "latency_ms": result.latency_ms,
                    "llm_calls": len(result.llm_calls),
                    "tools": result.tools_used,
                    "tool_errors": sum(t["is_error"] for t in result.tool_calls),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cost_usd": result.cost_usd,
                }
            )
        )


class Observer:
    def start_run(self, question: str, thread_id: str) -> Run:
        return Run(question, thread_id)


# ---- Langfuse ------------------------------------------------------------------------------------------
def _content_summary(content: list[dict]) -> list[dict]:
    """What the model produced, minus thinking blocks (empty/opaque, and large)."""
    out = []
    for b in content:
        if b.get("type") == "text":
            out.append({"type": "text", "text": b["text"]})
        elif b.get("type") == "tool_use":
            out.append({"type": "tool_use", "name": b["name"], "input": b["input"]})
    return out


class _LangfuseSpan(Span):
    def __init__(self, obs, kind: str):
        self.obs = obs
        self.kind = kind

    def end(self, **data: Any) -> None:
        if self.kind == "llm":
            resp: LLMResponse = data["response"]
            u = resp.usage
            self.obs.update(
                model=resp.model,
                output=_content_summary(resp.content),
                usage_details={
                    "input": u.input_tokens,
                    "output": u.output_tokens,
                    "cache_read_input_tokens": u.cache_read_tokens,
                    "cache_creation_input_tokens": u.cache_write_tokens,
                },
                cost_details={"total": u.cost_usd(resp.model)},
                metadata={"stop_reason": resp.stop_reason, "latency_ms": resp.latency_ms},
            )
        else:
            event = data["event"]
            self.obs.update(
                output=event.get("output"),
                level="ERROR" if event["is_error"] else "DEFAULT",
                metadata={
                    k: event.get(k)
                    for k in ("sql", "row_count", "elapsed_ms", "approval", "estimated_rows_scanned")
                    if event.get(k) is not None
                },
            )
        self.obs.end()


class LangfuseRun(Run):
    def __init__(self, client, question: str, thread_id: str):
        super().__init__(question, thread_id)
        self.client = client
        self.root = client.start_observation(
            name="retailpulse.ask",
            as_type="agent",
            input={"question": question},
            metadata={"thread_id": thread_id},
        )

    def start_llm(self, step: int, last_message: dict) -> Span:
        obs = self.root.start_observation(
            name="llm",
            as_type="generation",
            input={"step": step, "last_message": last_message},
        )
        return _LangfuseSpan(obs, "llm")

    def start_tool(self, name: str, args: dict) -> Span:
        return _LangfuseSpan(self.root.start_observation(name=name, as_type="tool", input=args), "tool")

    def finish(self, result) -> None:
        super().finish(result)
        self.root.update(
            output={"answer": result.answer},
            metadata={
                "thread_id": self.thread_id,
                "stop": result.stop,
                "cost_usd": result.cost_usd,
                "tools": result.tools_used,
            },
            level="WARNING" if result.stop != "end_turn" else "DEFAULT",
        )
        self.root.end()
        self.client.flush()


class LangfuseObserver(Observer):
    def __init__(self, client=None):
        from langfuse import Langfuse

        self.client = client or Langfuse()

    def start_run(self, question: str, thread_id: str) -> Run:
        return LangfuseRun(self.client, question, thread_id)


def get_observer() -> Observer:
    """Langfuse if configured (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST), else log-only."""
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        return LangfuseObserver()
    return Observer()
