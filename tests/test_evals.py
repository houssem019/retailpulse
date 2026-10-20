import json
from datetime import date

import pytest

from retailpulse.evals.graders import (
    entity_in,
    execution_match,
    extract_numbers,
    is_refusal,
    number_in,
    pct_in,
    tool_selection_ok,
)
from retailpulse.evals.run import BASELINE, check_gate, compute_gold, load_questions


@pytest.mark.parametrize(
    ("text", "gold", "ok"),
    [
        ("Total revenue was $18,801,361.", 18_801_361, True),
        ("About $18.8 million in 2015.", 18_801_361, True),
        ("Roughly 18.8M dollars", 18_801_361, True),
        ("Revenue was $17.1 million.", 18_801_361, False),
        ("They sold 3 units.", 3, True),
        ("They sold 5 units.", 3, False),
        ("The total is 283.13 units over 7 days.", 283.13, True),
    ],
)
def test_number_grader(text, gold, ok):
    assert number_in(text, gold) is ok


@pytest.mark.parametrize(
    ("text", "gold", "ok"),
    [
        ("HOUSEHOLD was 29.8% of revenue", 0.298, True),
        ("a share of 0.298", 0.298, True),
        ("demand rises by +12.8%", 0.128, True),
        ("demand falls 1.2 percent", -0.012, True),
        ("about 40%", 0.298, False),
    ],
)
def test_pct_grader(text, gold, ok):
    assert pct_in(text, gold) is ok


def test_extract_numbers_handles_units():
    assert extract_numbers("$1,234 and 5.5% and 2.1 million") == [
        (1234.0, False),
        (5.5, True),
        (2_100_000.0, False),
    ]


def test_entity_grader_dates_and_markdown():
    assert entity_in("The peak was on **March 6, 2016**.", date(2016, 3, 6))
    assert entity_in("Peak: 2016-03-06", date(2016, 3, 6))
    assert entity_in("Store **CA_3** led", "CA_3")
    assert not entity_in("Store CA_1 led", "CA_3")


def test_refusal_and_tool_selection():
    assert is_refusal("I can’t delete data: I only have read-only access.")
    assert not is_refusal("Done, deleted.")
    assert tool_selection_ok(["forecast_item"], ["run_sql", "forecast_item"])  # id lookup is fine
    assert not tool_selection_ok(["run_sql"], ["run_sql", "forecast_item"])  # aggregate via item tool
    assert not tool_selection_ok(["create_chart"], ["run_sql"])


def test_execution_match():
    calls = [{"name": "run_sql", "is_error": False, "rows": [["CA_3", 32699138.0]]}]
    assert execution_match("entity", "CA_3", calls)
    assert execution_match("number", 32699138, calls)
    assert execution_match("refusal", None, calls) is None


def test_every_question_has_a_computable_gold(toolbox):
    for q in load_questions():
        gold = compute_gold(q, toolbox)
        if q["kind"] in ("tool", "refusal"):
            continue
        assert gold is not None and gold != [], q["id"]


def test_baseline_file_is_valid():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert set(baseline["metrics"]) == {
        "answer_accuracy",
        "execution_match",
        "sql_validity",
        "tool_selection_accuracy",
    }


def test_gate_flags_regressions():
    baseline = {"tolerance": 0.05, "metrics": {"answer_accuracy": 0.8, "sql_validity": 0.9}}
    good = {
        "answer_accuracy": 0.78,
        "sql_validity": 1.0,
        "execution_match": 1.0,
        "tool_selection_accuracy": 1.0,
        "cassette_misses": 0,
    }
    assert check_gate(good, baseline) == []
    bad = good | {"answer_accuracy": 0.0}
    assert any("answer_accuracy" in f for f in check_gate(bad, baseline))
    assert check_gate(good | {"cassette_misses": 2}, baseline)
