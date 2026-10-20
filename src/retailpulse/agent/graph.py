"""The RetailPulse agent as a LangGraph state machine.

    START -> model --(tool_use)--> tools --> model --> ... --(final answer / refusal / step limit)--> END
                                     |
                                     +-- interrupt() for human approval of expensive SQL, resumed with
                                         Command(resume=True|False)

LangGraph gives us explicit state, a checkpointer (so a run can pause for approval and resume, even in a
later HTTP request, and a conversation can continue on the same thread), and a hard recursion limit. The
model call goes through our provider-neutral `llm` module, so the graph has no provider-specific code.

Public API:
  ask(question, approve=callback)     -> AgentResult              (CLI, evals: approvals via callback)
  start(question) / resume(thread, ok) -> AgentResult | PendingApproval   (HTTP: approval is a round trip)
  stream(question) / stream_resume(...) -> iterator of progress events     (SSE)
"""

from __future__ import annotations

import operator
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from retailpulse.agent.prompts import build_system_prompt
from retailpulse.llm.base import LLMClient
from retailpulse.observability import Observer, Run
from retailpulse.tools.registry import TOOL_SPECS, Toolbox, ToolOutcome
from retailpulse.tools.schema import describe_schema

DECLINED = (
    "Error: the user declined this query because it would scan too much data. Do not retry it; narrow it "
    "(filter by store, date range or category) or aggregate differently."
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    steps: int
    trace: Annotated[list[dict], operator.add]  # one event per LLM call and per tool call


@dataclass
class AgentResult:
    question: str
    answer: str
    tool_calls: list[dict]
    llm_calls: list[dict]
    latency_ms: float
    stop: str  # end_turn | refusal | step_limit | max_tokens
    thread_id: str = ""

    @property
    def tools_used(self) -> list[str]:
        return [t["name"] for t in self.tool_calls]

    @property
    def approvals(self) -> list[dict]:
        return [
            {"tool": t["name"], "sql": t.get("sql") or t["input"].get("query"), "approved": t["approval"]}
            for t in self.tool_calls
            if t.get("approval") is not None
        ]

    @property
    def input_tokens(self) -> int:
        return sum(
            c["input_tokens"] + c["cache_read_tokens"] + c["cache_write_tokens"] for c in self.llm_calls
        )

    @property
    def output_tokens(self) -> int:
        return sum(c["output_tokens"] for c in self.llm_calls)

    @property
    def cost_usd(self) -> float:
        return round(sum(c["cost_usd"] for c in self.llm_calls), 6)

    def summary(self) -> dict:
        return {
            "status": "completed",
            "thread_id": self.thread_id,
            "answer": self.answer,
            "stop": self.stop,
            "tools": [
                {
                    "name": t["name"],
                    "input": t["input"],
                    "is_error": t["is_error"],
                    "sql": t.get("sql"),
                    "elapsed_ms": t.get("elapsed_ms"),
                    "chart_path": t.get("chart_path"),
                }
                for t in self.tool_calls
            ],
            "approvals": self.approvals,
            "llm_calls": len(self.llm_calls),
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass
class PendingApproval:
    thread_id: str
    question: str
    request: dict  # {"type": "sql_approval", "tool", "sql", "estimated_rows_scanned"}

    def summary(self) -> dict:
        return {"status": "approval_required", "thread_id": self.thread_id, "approval": self.request}


@dataclass
class _Turn:
    question: str
    trace_start: int  # index into the thread's trace where this question's events begin
    started: float
    run: Run
    extra: dict = field(default_factory=dict)


class RetailAgent:
    def __init__(
        self, llm: LLMClient, toolbox: Toolbox, *, max_steps: int = 10, observer: Observer | None = None
    ):
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps
        self.observer = observer or Observer()
        self.system = build_system_prompt(describe_schema(toolbox.sandbox._con.cursor()))
        self.graph = self._build()
        self._turns: dict[str, _Turn] = {}  # in-flight questions (incl. paused for approval), per thread

    # ---- nodes -------------------------------------------------------------------------------------
    def _run(self, config: RunnableConfig) -> Run:
        return self._turns[config["configurable"]["thread_id"]].run

    def _model(self, state: AgentState, config: RunnableConfig) -> dict:
        step = state.get("steps", 0) + 1
        span = self._run(config).start_llm(step, state["messages"][-1])
        resp = self.llm.complete(self.system, state["messages"], TOOL_SPECS)
        span.end(response=resp)
        event = {
            "type": "llm",
            "model": resp.model,
            "stop_reason": resp.stop_reason,
            "latency_ms": resp.latency_ms,
            "cost_usd": resp.usage.cost_usd(resp.model),
            **vars(resp.usage),
        }
        return {"messages": [{"role": "assistant", "content": resp.content}], "steps": step, "trace": [event]}

    def _tools(self, state: AgentState, config: RunnableConfig) -> dict:
        calls = [b for b in state["messages"][-1]["content"] if b.get("type") == "tool_use"]
        # Ask for every approval first: on resume LangGraph re-runs this node from the top, so no tool
        # may execute before all interrupts are answered (otherwise it would run twice).
        decisions: dict[str, bool] = {}
        for call in calls:
            req = self.toolbox.approval_needed(call["name"], call["input"])
            if req:
                decisions[call["id"]] = bool(
                    interrupt(
                        {
                            "type": "sql_approval",
                            "tool": req.tool,
                            "sql": req.sql,
                            "estimated_rows_scanned": req.estimated_rows_scanned,
                        }
                    )
                )

        run = self._run(config)
        results, trace = [], []
        for call in calls:
            span = run.start_tool(call["name"], call["input"])
            approved = decisions.get(call["id"])
            if approved is False:
                outcome = ToolOutcome(DECLINED, is_error=True)
            else:
                outcome = self.toolbox.execute(call["name"], call["input"], approved=bool(approved))
            result = {"type": "tool_result", "tool_use_id": call["id"], "content": outcome.content}
            if outcome.is_error:
                result["is_error"] = True
            results.append(result)
            event = {
                "type": "tool",
                "name": call["name"],
                "input": call["input"],
                "is_error": outcome.is_error,
                "approval": approved,
                "output": outcome.content[:2000],
                **outcome.meta,
            }
            span.end(event=event)
            trace.append(event)
        return {"messages": [{"role": "user", "content": results}], "trace": trace}

    def _route(self, state: AgentState) -> str:
        wants_tools = any(b.get("type") == "tool_use" for b in state["messages"][-1]["content"])
        return "tools" if wants_tools and state["steps"] < self.max_steps else END

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("model", self._model)
        g.add_node("tools", self._tools)
        g.add_edge(START, "model")
        g.add_conditional_edges("model", self._route, {"tools": "tools", END: END})
        g.add_edge("tools", "model")
        return g.compile(checkpointer=InMemorySaver())

    # ---- run bookkeeping ---------------------------------------------------------------------------
    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 2 * self.max_steps + 5}

    def _begin(self, question: str, thread_id: str | None) -> tuple[str, dict]:
        thread_id = thread_id or uuid.uuid4().hex
        if thread_id in self._turns:
            raise ValueError(f"Thread {thread_id} has a question awaiting approval; resume it first.")
        config = self._config(thread_id)
        trace_start = len(self.graph.get_state(config).values.get("trace", []))
        self._turns[thread_id] = _Turn(
            question, trace_start, time.perf_counter(), self.observer.start_run(question, thread_id)
        )
        return thread_id, config

    def _settle(self, thread_id: str) -> AgentResult | PendingApproval:
        snapshot = self.graph.get_state(self._config(thread_id))
        turn = self._turns[thread_id]
        if snapshot.interrupts:
            return PendingApproval(thread_id, turn.question, snapshot.interrupts[0].value)
        del self._turns[thread_id]
        values = snapshot.values
        trace = values["trace"][turn.trace_start :]
        llm_calls = [e for e in trace if e["type"] == "llm"]
        text = "".join(
            b.get("text", "") for b in values["messages"][-1]["content"] if b.get("type") == "text"
        )
        stop = llm_calls[-1]["stop_reason"] if llm_calls else "end_turn"
        if stop == "tool_use":
            stop = "step_limit"
            text = text or "I could not finish this analysis within the step limit. Try a narrower question."
        elif stop == "refusal":
            text = text or "The model declined to answer this request."
        result = AgentResult(
            question=turn.question,
            answer=text.strip(),
            tool_calls=[e for e in trace if e["type"] == "tool"],
            llm_calls=llm_calls,
            latency_ms=round((time.perf_counter() - turn.started) * 1000, 1),
            stop=stop,
            thread_id=thread_id,
        )
        turn.run.finish(result)
        return result

    def _invoke(self, thread_id: str, payload: Any) -> AgentResult | PendingApproval:
        try:
            self.graph.invoke(payload, self._config(thread_id))
        except Exception:
            self._turns.pop(thread_id, None)
            raise
        return self._settle(thread_id)

    # ---- public API --------------------------------------------------------------------------------
    def start(self, question: str, *, thread_id: str | None = None) -> AgentResult | PendingApproval:
        """Run until the answer, or until a query needs human approval (then call resume())."""
        thread_id, _ = self._begin(question, thread_id)
        return self._invoke(thread_id, {"messages": [{"role": "user", "content": question}], "steps": 0})

    def resume(self, thread_id: str, approve: bool) -> AgentResult | PendingApproval:
        if thread_id not in self._turns:
            raise KeyError(f"No question awaiting approval on thread {thread_id}.")
        return self._invoke(thread_id, Command(resume=bool(approve)))

    def ask(
        self, question: str, *, approve: Callable[[dict], bool] | None = None, thread_id: str | None = None
    ) -> AgentResult:
        """Answer one question. `approve(request) -> bool` decides on expensive queries (default: deny).
        Reusing a thread_id continues the same conversation (follow-up questions)."""
        out = self.start(question, thread_id=thread_id)
        while isinstance(out, PendingApproval):
            out = self.resume(out.thread_id, bool(approve(out.request)) if approve else False)
        return out

    def stream(self, question: str, *, thread_id: str | None = None) -> Iterator[dict]:
        """Progress events as the agent works: tool_call, tool_result, then answer or approval_required."""
        thread_id, _ = self._begin(question, thread_id)
        yield from self._stream(thread_id, {"messages": [{"role": "user", "content": question}], "steps": 0})

    def stream_resume(self, thread_id: str, approve: bool) -> Iterator[dict]:
        if thread_id not in self._turns:
            raise KeyError(f"No question awaiting approval on thread {thread_id}.")
        yield from self._stream(thread_id, Command(resume=bool(approve)))

    def _stream(self, thread_id: str, payload: Any) -> Iterator[dict]:
        try:
            for update in self.graph.stream(payload, self._config(thread_id), stream_mode="updates"):
                for node, delta in update.items():
                    if node == "model":
                        for b in delta["messages"][0]["content"]:
                            if b.get("type") == "tool_use":
                                yield {"type": "tool_call", "name": b["name"], "input": b["input"]}
                    elif node == "tools":
                        for e in delta["trace"]:
                            yield {
                                "type": "tool_result",
                                "name": e["name"],
                                "is_error": e["is_error"],
                                "elapsed_ms": e.get("elapsed_ms"),
                                "output": e["output"][:500],
                            }
        except Exception:
            self._turns.pop(thread_id, None)
            raise
        out = self._settle(thread_id)
        if isinstance(out, PendingApproval):
            yield {"type": "approval_required", **out.summary()}
        else:
            yield {"type": "answer", **out.summary()}
