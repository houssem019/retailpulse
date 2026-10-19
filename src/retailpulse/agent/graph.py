"""The RetailPulse agent as a LangGraph state machine.

    START -> model --(tool_use)--> tools --> model --> ... --(final answer / refusal / step limit)--> END
                                     |
                                     +-- interrupt() for human approval of expensive SQL, resumed with
                                         Command(resume=True|False)

LangGraph gives us explicit state, a checkpointer (so a run can pause for approval and resume, and a
conversation can continue on the same thread), and a hard recursion limit. The model call itself goes
through our provider-neutral `llm` module, so the graph has no Anthropic- or Ollama-specific code.
"""

from __future__ import annotations

import operator
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from retailpulse.agent.prompts import build_system_prompt
from retailpulse.llm.base import LLMClient
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
    approvals: list[dict] = field(default_factory=list)

    @property
    def tools_used(self) -> list[str]:
        return [t["name"] for t in self.tool_calls]

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


class RetailAgent:
    def __init__(self, llm: LLMClient, toolbox: Toolbox, *, max_steps: int = 10):
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps
        self.system = build_system_prompt(describe_schema(toolbox.sandbox._con.cursor()))
        self.graph = self._build()

    # ---- nodes -------------------------------------------------------------------------------------
    def _model(self, state: AgentState) -> dict:
        resp = self.llm.complete(self.system, state["messages"], TOOL_SPECS)
        event = {
            "type": "llm",
            "model": resp.model,
            "stop_reason": resp.stop_reason,
            "latency_ms": resp.latency_ms,
            "cost_usd": resp.usage.cost_usd(resp.model),
            **vars(resp.usage),
        }
        return {
            "messages": [{"role": "assistant", "content": resp.content}],
            "steps": state.get("steps", 0) + 1,
            "trace": [event],
        }

    def _tools(self, state: AgentState) -> dict:
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

        results, trace = [], []
        for call in calls:
            approved = decisions.get(call["id"])
            if approved is False:
                outcome = ToolOutcome(DECLINED, is_error=True)
            else:
                outcome = self.toolbox.execute(call["name"], call["input"], approved=bool(approved))
            result = {"type": "tool_result", "tool_use_id": call["id"], "content": outcome.content}
            if outcome.is_error:
                result["is_error"] = True
            results.append(result)
            trace.append(
                {
                    "type": "tool",
                    "name": call["name"],
                    "input": call["input"],
                    "is_error": outcome.is_error,
                    "approval": approved,
                    "output": outcome.content[:2000],
                    **outcome.meta,
                }
            )
        return {"messages": [{"role": "user", "content": results}], "trace": trace}

    def _route(self, state: AgentState) -> str:
        last = state["messages"][-1]
        wants_tools = any(b.get("type") == "tool_use" for b in last["content"])
        if wants_tools and state["steps"] < self.max_steps:
            return "tools"
        return END

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("model", self._model)
        g.add_node("tools", self._tools)
        g.add_edge(START, "model")
        g.add_conditional_edges("model", self._route, {"tools": "tools", END: END})
        g.add_edge("tools", "model")
        return g.compile(checkpointer=InMemorySaver())

    # ---- public API --------------------------------------------------------------------------------
    def ask(
        self,
        question: str,
        *,
        approve: Callable[[dict], bool] | None = None,
        thread_id: str | None = None,
    ) -> AgentResult:
        """Answer one question. `approve(request) -> bool` decides on expensive queries (default: deny).
        Reusing a thread_id continues the same conversation (follow-up questions)."""
        config = {
            "configurable": {"thread_id": thread_id or uuid.uuid4().hex},
            "recursion_limit": 2 * self.max_steps + 5,
        }
        start = time.perf_counter()
        before = len(self.graph.get_state(config).values.get("trace", [])) if thread_id else 0
        state = self.graph.invoke({"messages": [{"role": "user", "content": question}], "steps": 0}, config)
        approvals = []
        while state.get("__interrupt__"):
            request = state["__interrupt__"][0].value
            decision = bool(approve(request)) if approve else False
            approvals.append(request | {"approved": decision})
            state = self.graph.invoke(Command(resume=decision), config)

        trace = state["trace"][before:]
        llm_calls = [e for e in trace if e["type"] == "llm"]
        last = state["messages"][-1]
        text = "".join(b.get("text", "") for b in last["content"] if b.get("type") == "text")
        stop = llm_calls[-1]["stop_reason"] if llm_calls else "end_turn"
        if stop == "tool_use":
            stop = "step_limit"
            text = text or "I could not finish this analysis within the step limit. Try a narrower question."
        elif stop == "refusal":
            text = text or "The model declined to answer this request."
        return AgentResult(
            question=question,
            answer=text.strip(),
            tool_calls=[e for e in trace if e["type"] == "tool"],
            llm_calls=llm_calls,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            stop=stop,
            approvals=approvals,
        )
