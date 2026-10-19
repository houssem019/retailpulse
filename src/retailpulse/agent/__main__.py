"""Ask RetailPulse a question from the terminal.

    uv run python -m retailpulse.agent "Which store had the highest revenue in 2015?"

Provider comes from LLM_PROVIDER (anthropic | ollama | fake). The forecast tools need the forecast API
running (uv run uvicorn retailpulse.forecasting.serve:app --port 8001).
"""

from __future__ import annotations

import argparse
import json

from retailpulse.agent.graph import RetailAgent
from retailpulse.config import settings
from retailpulse.llm import get_llm
from retailpulse.tools.registry import Toolbox


def _ask_human(request: dict) -> bool:
    print(f"\n[approval needed] {request['tool']} would scan ~{request['estimated_rows_scanned']:,} rows:")
    print(f"  {request['sql']}")
    return input("Run it? [y/N] ").strip().lower() == "y"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RetailPulse agent a question")
    parser.add_argument("question")
    parser.add_argument(
        "--auto-approve", action="store_true", help="Approve expensive queries without asking"
    )
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    args = parser.parse_args()

    agent = RetailAgent(get_llm(), Toolbox.from_settings(), max_steps=settings.agent_max_steps)
    result = agent.ask(args.question, approve=(lambda _: True) if args.auto_approve else _ask_human)
    if args.json:
        print(json.dumps(vars(result), indent=1, default=str))
        return
    print(f"\n{result.answer}\n")
    for t in result.tool_calls:
        status = "error" if t["is_error"] else "ok"
        detail = t.get("sql") or json.dumps(t["input"])
        print(f"  - {t['name']} [{status}, {t['elapsed_ms']} ms]: {detail[:160]}")
    print(
        f"\n  {len(result.llm_calls)} LLM calls · {result.input_tokens:,} in / {result.output_tokens:,} out tokens"
        f" · ${result.cost_usd:.4f} · {result.latency_ms / 1000:.1f}s · stop={result.stop}"
    )


if __name__ == "__main__":
    main()
