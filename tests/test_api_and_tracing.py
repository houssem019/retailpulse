import json
import logging

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from retailpulse.agent.graph import RetailAgent
from retailpulse.api import create_app
from retailpulse.llm.replay import ScriptedLLM
from retailpulse.observability import LangfuseObserver

REVENUE_SQL = "SELECT store_id, round(sum(revenue)) AS rev FROM sales GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
EXPENSIVE_SQL = "SELECT count(*) FROM sales a JOIN sales b USING (item_id, store_id, date)"


def _client(toolbox, steps) -> TestClient:
    return TestClient(create_app(RetailAgent(ScriptedLLM(steps), toolbox)))


def test_ask_returns_answer_and_usage(toolbox):
    with _client(toolbox, [("run_sql", {"query": REVENUE_SQL, "purpose": "p"}), "CA_1 leads."]) as c:
        body = c.post("/ask", json={"question": "Top store?"}).json()
    assert body["status"] == "completed" and body["answer"] == "CA_1 leads."
    assert body["tools"][0]["name"] == "run_sql" and body["tools"][0]["sql"].startswith("SELECT")
    assert body["llm_calls"] == 2 and body["thread_id"]


@pytest.mark.parametrize("approve", [True, False])
def test_approval_round_trip_over_http(toolbox, approve):
    steps = [("run_sql", {"query": EXPENSIVE_SQL, "purpose": "self join"}), "done"]
    with _client(toolbox, steps) as c:
        paused = c.post("/ask", json={"question": "big query"}).json()
        assert paused["status"] == "approval_required"
        assert paused["approval"]["estimated_rows_scanned"] > 0
        # the thread is busy until the approval is answered
        assert (
            c.post("/ask", json={"question": "another one", "thread_id": paused["thread_id"]}).status_code
            == 409
        )
        done = c.post("/approve", json={"thread_id": paused["thread_id"], "approve": approve}).json()
    assert done["status"] == "completed"
    assert done["approvals"][0]["approved"] is approve
    assert done["tools"][0]["is_error"] is (not approve)


def test_approve_unknown_thread_is_404(toolbox):
    with _client(toolbox, []) as c:
        assert c.post("/approve", json={"thread_id": "nope", "approve": True}).status_code == 404


def test_stream_emits_progress_then_answer(toolbox):
    steps = [("run_sql", {"query": REVENUE_SQL, "purpose": "p"}), "CA_1 leads."]
    with _client(toolbox, steps) as c, c.stream("POST", "/ask/stream", json={"question": "Top store?"}) as r:
        events = [
            line.removeprefix("event: ").strip() for line in r.iter_lines() if line.startswith("event:")
        ]
    assert events == ["tool_call", "tool_result", "answer"]


def test_follow_up_on_same_thread(toolbox):
    with _client(toolbox, ["First.", "Second."]) as c:
        first = c.post("/ask", json={"question": "one"}).json()
        second = c.post("/ask", json={"question": "two", "thread_id": first["thread_id"]}).json()
    assert second["answer"] == "Second."


def test_run_is_logged_as_json(toolbox, caplog):
    caplog.set_level(logging.INFO, logger="retailpulse.agent")
    RetailAgent(ScriptedLLM(["ok"]), toolbox).ask("q")
    record = json.loads(caplog.records[-1].getMessage())
    assert record["event"] == "agent_run" and record["llm_calls"] == 1 and "cost_usd" in record


def test_langfuse_trace_structure(toolbox):
    """Export spans to memory instead of a Langfuse server and check the trace tree."""
    from langfuse import Langfuse

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="http://localhost:1",
        span_exporter=exporter,
        flush_at=1,
    )
    agent = RetailAgent(
        ScriptedLLM([("run_sql", {"query": REVENUE_SQL, "purpose": "p"}), "done"]),
        toolbox,
        observer=LangfuseObserver(client),
    )
    agent.ask("Top store?")
    client.flush()
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert names.count("llm") == 2 and "run_sql" in names and "retailpulse.ask" in names
    root = next(s for s in spans if s.name == "retailpulse.ask")
    children = [s for s in spans if s.parent is not None and s.parent.span_id == root.context.span_id]
    assert {s.name for s in children} == {"llm", "run_sql"}
    assert all(s.context.trace_id == root.context.trace_id for s in spans)
