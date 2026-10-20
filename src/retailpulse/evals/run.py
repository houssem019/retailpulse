"""Run the RetailPulse benchmark and (optionally) gate on regressions.

    # CI: replay recorded model responses on the fixture warehouse (no key, no network, ~30 s)
    uv run python -m retailpulse.evals.run --provider fake --cassette evals/cassettes/qwen2.5-7b.json --gate

    # Record a new cassette with a real model (free local Ollama, or Claude with an API key)
    uv run python -m retailpulse.evals.run --provider ollama --cassette evals/cassettes/qwen2.5-7b.json --record

    # Live: full warehouse + running forecast API
    uv run python -m retailpulse.evals.run --env live --provider anthropic

Metrics:
  answer_accuracy          answer contains the gold value (computed at eval time from gold SQL / tools)
  execution_match          some successful agent query returned the gold value (right data, any phrasing)
  sql_validity             share of the agent's SQL calls that executed without error
  tool_selection_accuracy  expected tools used and no unexpected action tools
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

from retailpulse.agent.graph import RetailAgent
from retailpulse.config import PROJECT_ROOT, settings
from retailpulse.evals.env import fixture_toolbox, live_toolbox
from retailpulse.evals.graders import SQL_TOOLS, execution_match, grade_answer, tool_selection_ok
from retailpulse.llm import LLMClient, get_llm
from retailpulse.llm.replay import CassetteMiss, ReplayLLM
from retailpulse.tools.registry import Toolbox

EVAL_DIR = PROJECT_ROOT / "evals"
QUESTIONS = EVAL_DIR / "questions.yaml"
BASELINE = EVAL_DIR / "baseline.json"
METRICS = ["answer_accuracy", "execution_match", "sql_validity", "tool_selection_accuracy"]


@dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    kind: str
    gold: object
    answer: str
    correct: bool
    tool_selection_ok: bool
    execution_match: bool | None
    tools_used: list[str]
    sql_calls: int
    sql_errors: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    stop: str
    error: str | None = None
    sql: list[str] = field(default_factory=list)


def load_questions(path: Path = QUESTIONS, only: list[str] | None = None) -> list[dict]:
    qs = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [q for q in qs if not only or q["id"] in only]


def compute_gold(q: dict, toolbox: Toolbox):
    if "gold_sql" in q:
        res = toolbox.sandbox.run(q["gold_sql"], approved=True)
        if q["kind"] == "entities":
            # one row of several columns (item, store) or one column of several rows (top-3 list)
            return list(res.rows[0]) if len(res.columns) > 1 else [r[0] for r in res.rows]
        return res.rows[0][0] if res.rows else None
    if "gold_tool" in q:
        spec = q["gold_tool"]
        out = toolbox.execute(spec["name"], spec["args"])
        return out.meta.get(spec["field"]) if not out.is_error else None
    return None


def run_question(agent: RetailAgent, q: dict, toolbox: Toolbox) -> QuestionResult:
    gold = compute_gold(q, toolbox)
    try:
        r = agent.ask(q["question"], approve=lambda _: False)
    except CassetteMiss as e:
        return QuestionResult(
            q["id"],
            q["category"],
            q["question"],
            q["kind"],
            gold,
            "",
            False,
            False,
            None,
            [],
            0,
            0,
            0,
            0,
            0,
            0.0,
            0.0,
            "cassette_miss",
            error=str(e),
        )
    sql_calls = [t for t in r.tool_calls if t["name"] in SQL_TOOLS]
    return QuestionResult(
        id=q["id"],
        category=q["category"],
        question=q["question"],
        kind=q["kind"],
        gold=gold,
        answer=r.answer,
        correct=grade_answer(q["kind"], r.answer, gold, r.tool_calls, q["expected_tools"]),
        tool_selection_ok=tool_selection_ok(q["expected_tools"], r.tools_used),
        execution_match=execution_match(q["kind"], gold, r.tool_calls) if "gold_sql" in q else None,
        tools_used=r.tools_used,
        sql_calls=len(sql_calls),
        sql_errors=sum(t["is_error"] for t in sql_calls),
        llm_calls=len(r.llm_calls),
        input_tokens=r.input_tokens,
        output_tokens=r.output_tokens,
        cost_usd=r.cost_usd,
        latency_ms=r.latency_ms,
        stop=r.stop,
        sql=[t.get("sql") or t["input"].get("query", "") for t in sql_calls],
    )


def summarize(results: list[QuestionResult]) -> dict:
    n = len(results)
    em = [r.execution_match for r in results if r.execution_match is not None]
    sql_total = sum(r.sql_calls for r in results)
    by_cat: dict[str, list[bool]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r.correct)
    return {
        "questions": n,
        "answer_accuracy": round(sum(r.correct for r in results) / n, 3),
        "execution_match": round(sum(em) / len(em), 3) if em else None,
        "sql_validity": round(1 - sum(r.sql_errors for r in results) / sql_total, 3) if sql_total else None,
        "tool_selection_accuracy": round(sum(r.tool_selection_ok for r in results) / n, 3),
        "by_category": {c: round(sum(v) / len(v), 3) for c, v in sorted(by_cat.items())},
        "avg_llm_calls": round(sum(r.llm_calls for r in results) / n, 2),
        "avg_latency_s": round(sum(r.latency_ms for r in results) / n / 1000, 2),
        "total_input_tokens": sum(r.input_tokens for r in results),
        "total_output_tokens": sum(r.output_tokens for r in results),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 4),
        "cassette_misses": sum(r.stop == "cassette_miss" for r in results),
    }


def check_gate(summary: dict, baseline: dict) -> list[str]:
    """Regression gate: every metric must stay within `tolerance` of the committed baseline."""
    tol = baseline.get("tolerance", 0.0)
    failures = []
    if summary["cassette_misses"]:
        failures.append(
            f"{summary['cassette_misses']} cassette misses: prompt/tools changed, re-record the cassette"
        )
    for m in METRICS:
        floor = baseline["metrics"].get(m)
        if floor is not None and summary[m] is not None and summary[m] < floor - tol:
            failures.append(f"{m} {summary[m]:.3f} < baseline {floor:.3f} - tolerance {tol}")
    return failures


def to_markdown(summary: dict, results: list[QuestionResult], label: str) -> str:
    lines = [f"# RetailPulse eval: {label}", "", "| metric | value |", "|---|---|"]
    for k in METRICS + [
        "avg_llm_calls",
        "avg_latency_s",
        "total_input_tokens",
        "total_output_tokens",
        "total_cost_usd",
    ]:
        lines.append(f"| {k} | {summary[k]} |")
    lines += ["", "| category | accuracy |", "|---|---|"]
    lines += [f"| {c} | {v} |" for c, v in summary["by_category"].items()]
    lines += [
        "",
        "| id | correct | tools ok | exec match | tools used | gold | answer |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ans = r.answer.replace("\n", " ").replace("|", "/")[:140]
        lines.append(
            f"| {r.id} | {'✅' if r.correct else '❌'} | {'✅' if r.tool_selection_ok else '❌'} | "
            f"{'' if r.execution_match is None else ('✅' if r.execution_match else '❌')} | "
            f"{', '.join(r.tools_used)} | {r.gold} | {ans} |"
        )
    return "\n".join(lines) + "\n"


def _json_default(o):
    return o.isoformat() if isinstance(o, (date, datetime)) else str(o)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env", choices=["fixture", "live"], default="fixture")
    p.add_argument(
        "--provider",
        choices=["fake", "ollama", "anthropic"],
        default=None,
        help="Overrides LLM_PROVIDER. 'fake' replays --cassette.",
    )
    p.add_argument("--cassette", type=Path, help="Replay from (fake) or record into (--record) this file")
    p.add_argument(
        "--record", action="store_true", help="Call the real provider on misses and save responses"
    )
    p.add_argument("--only", nargs="*", help="Question ids to run")
    p.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    p.add_argument("--gate", action="store_true", help=f"Exit 1 if metrics regress vs {BASELINE.name}")
    args = p.parse_args()

    cfg = settings.model_copy(update={"llm_provider": args.provider} if args.provider else {})
    if cfg.llm_provider == "fake":
        if not args.cassette:
            p.error("--provider fake needs --cassette")
        llm: LLMClient = ReplayLLM(args.cassette)
    else:
        llm = get_llm(cfg)
        if args.record:
            if not args.cassette:
                p.error("--record needs --cassette")
            llm = ReplayLLM(args.cassette, inner=llm)
    label = f"{args.env}-{cfg.llm_provider}-{getattr(llm, 'model', 'model').replace(':', '-')}"

    questions = load_questions(only=args.only)
    env: AbstractContextManager[Toolbox] = fixture_toolbox() if args.env == "fixture" else live_toolbox()
    results: list[QuestionResult] = []
    start = time.perf_counter()
    with env as toolbox:
        agent = RetailAgent(llm, toolbox, max_steps=settings.agent_max_steps)
        for i, q in enumerate(questions, 1):
            r = run_question(agent, q, toolbox)
            results.append(r)
            print(
                f"[{i:>2}/{len(questions)}] {r.id} {'PASS' if r.correct else 'FAIL'} "
                f"tools={','.join(r.tools_used) or '-'} {r.latency_ms / 1000:.1f}s",
                flush=True,
            )

    summary = summarize(results)
    summary["label"], summary["wall_clock_s"] = label, round(time.perf_counter() - start, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{label}.json").write_text(
        json.dumps(
            {"summary": summary, "results": [asdict(r) for r in results]}, indent=1, default=_json_default
        ),
        encoding="utf-8",
    )
    (args.out / f"{label}.md").write_text(to_markdown(summary, results, label), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, indent=1))

    if args.gate:
        failures = check_gate(summary, json.loads(BASELINE.read_text(encoding="utf-8")))
        if failures:
            print("EVAL GATE FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
            return 1
        print("eval gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
