import pytest

from retailpulse.agent.graph import RetailAgent
from retailpulse.llm.ollama_client import from_ollama_message, to_ollama_messages
from retailpulse.llm.replay import CassetteMiss, ReplayLLM, ScriptedLLM

REVENUE_SQL = "SELECT store_id, round(sum(revenue)) AS rev FROM sales GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
EXPENSIVE_SQL = "SELECT count(*) FROM sales a JOIN sales b USING (item_id, store_id, date)"


def test_agent_tool_loop_and_answer(toolbox):
    llm = ScriptedLLM(
        [("run_sql", {"query": REVENUE_SQL, "purpose": "top store"}), "CA_1 had the most revenue."]
    )
    result = RetailAgent(llm, toolbox).ask("Which store had the highest revenue?")
    assert result.answer == "CA_1 had the most revenue."
    assert result.tools_used == ["run_sql"] and result.stop == "end_turn"
    # the model's second call saw the tool result
    tool_result = llm.calls[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and "rev" in tool_result["content"]


def test_system_prompt_contains_schema_and_is_stable(toolbox):
    a, b = RetailAgent(ScriptedLLM([]), toolbox), RetailAgent(ScriptedLLM([]), toolbox)
    assert a.system == b.system  # byte-stable -> prompt-cacheable
    assert "inventory_signals" in a.system and "Values of cat_id: FOODS, HOBBIES, HOUSEHOLD" in a.system


def test_agent_recovers_from_sql_error(toolbox):
    llm = ScriptedLLM(
        [
            ("run_sql", {"query": "SELECT revenu FROM sales", "purpose": "typo"}),
            ("run_sql", {"query": REVENUE_SQL, "purpose": "fixed"}),
            "Fixed it.",
        ]
    )
    result = RetailAgent(llm, toolbox).ask("q")
    assert [t["is_error"] for t in result.tool_calls] == [True, False]


@pytest.mark.parametrize("decision", [True, False])
def test_expensive_query_pauses_for_human_approval(toolbox, decision):
    seen = []
    llm = ScriptedLLM([("run_sql", {"query": EXPENSIVE_SQL, "purpose": "self join"}), "done"])
    result = RetailAgent(llm, toolbox).ask("q", approve=lambda req: seen.append(req) or decision)
    assert len(seen) == 1 and seen[0]["type"] == "sql_approval"
    assert result.approvals[0]["approved"] is decision
    assert result.tool_calls[0]["is_error"] is (not decision)


def test_parallel_tool_calls_in_one_turn(toolbox):
    llm = ScriptedLLM(
        [
            [
                ("run_sql", {"query": REVENUE_SQL, "purpose": "a"}),
                ("forecast_item", {"item_id": "FOODS_1_001", "store_id": "CA_1", "horizon": 7}),
            ],
            "Both done.",
        ]
    )
    result = RetailAgent(llm, toolbox).ask("q")
    assert result.tools_used == ["run_sql", "forecast_item"]
    assert len(llm.calls[1]["messages"][-1]["content"]) == 2  # both results returned in ONE user message


def test_step_limit(toolbox):
    llm = ScriptedLLM([("run_sql", {"query": REVENUE_SQL, "purpose": "loop"})] * 3)
    result = RetailAgent(llm, toolbox, max_steps=3).ask("q")
    assert result.stop == "step_limit" and len(result.llm_calls) == 3


def test_follow_up_question_keeps_context(toolbox):
    llm = ScriptedLLM(["First answer.", "Second answer."])
    agent = RetailAgent(llm, toolbox)
    agent.ask("first", thread_id="t1")
    agent.ask("and the second?", thread_id="t1")
    assert len(llm.calls[1]["messages"]) == 3  # user, assistant, user


def test_replay_cassette_records_then_replays(toolbox, tmp_path):
    cassette = tmp_path / "c.json"
    recorder = ReplayLLM(
        cassette, inner=ScriptedLLM([("run_sql", {"query": REVENUE_SQL, "purpose": "p"}), "Hi."])
    )
    first = RetailAgent(recorder, toolbox).ask("Which store?")
    replayed = RetailAgent(ReplayLLM(cassette), toolbox).ask("Which store?")
    assert replayed.answer == first.answer and replayed.tools_used == ["run_sql"]
    with pytest.raises(CassetteMiss):
        RetailAgent(ReplayLLM(cassette), toolbox).ask("A different question")


def test_ollama_message_translation():
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "run_sql",
                    "input": {"query": "SELECT 1", "purpose": "p"},
                },
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "[[1]]"}]},
    ]
    out = to_ollama_messages("sys", messages)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[2]["tool_calls"][0]["function"]["name"] == "run_sql" and out[3]["tool_name"] == "run_sql"
    blocks = from_ollama_message(
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "run_sql", "arguments": '{"query": "SELECT 1", "purpose": "p"}'}}
            ],
        }
    )
    assert blocks[0]["type"] == "tool_use" and blocks[0]["input"]["query"] == "SELECT 1"
