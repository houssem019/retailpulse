"""Deterministic graders: no LLM-as-judge, so scores are reproducible and free.

Answers are free text, so graders are lenient about *format* (thousands separators, "$18.8 million",
12.3% vs 0.123, ISO vs written dates) and strict about *values* (2% relative tolerance).
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime

_NUM = re.compile(
    r"(?<![\w.])[-−]?\$?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|million|billion|thousand|[kKmMbB](?![a-zA-Z]))?"
)
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}
REFUSAL_MARKERS = (
    "can't",
    "cannot",
    "can not",
    "unable",
    "not able",
    "won't",
    "will not",
    "not allowed",
    "read-only",
    "read only",
    "not permitted",
    "don't have",
    "do not have",
    "no access",
    "not possible",
)
SQL_TOOLS = {"run_sql", "create_chart"}
ACTION_TOOLS = {"forecast_item", "price_what_if", "create_chart"}


def extract_numbers(text: str) -> list[tuple[float, bool]]:
    """All numbers in `text` as (value, is_percent). '$18.8 million' -> 18_800_000; '12.3%' -> (12.3, True)."""
    out = []
    for m in _NUM.finditer(text.replace("−", "-")):
        raw = m.group(0).strip()
        pct = raw.endswith("%") or raw.lower().endswith("percent")
        unit = re.search(r"(million|billion|thousand|[kKmMbB])$", raw)
        digits = re.sub(r"[^\d.\-]", "", raw.split()[0] if " " in raw else raw.rstrip("%kKmMbB"))
        if digits in ("", "-", "."):
            continue
        try:
            value = float(digits)
        except ValueError:
            continue
        if unit and not pct:
            value *= _MULT[unit.group(1).lower()]
        out.append((value, pct))
    return out


def _close(a: float, b: float, rel: float = 0.02, abs_tol: float = 0.5) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def number_in(text: str, gold: float) -> bool:
    # abs tolerance covers rounding of small counts ("about 3 units"); rel tolerance covers "$18.8M"
    return any(_close(v, gold, abs_tol=0.5 if abs(gold) < 100 else 0) for v, _ in extract_numbers(text))


def pct_in(text: str, gold_fraction: float) -> bool:
    """Gold 0.0577: accept '5.8%', '5.77 percent', or the raw fraction '0.0577'."""
    target = gold_fraction * 100
    for v, is_pct in extract_numbers(text):
        if _close(abs(v), abs(target), rel=0.03, abs_tol=0.15) or (
            not is_pct and _close(v, gold_fraction, abs_tol=0.002)
        ):
            return True
    return False


def _entity_variants(value) -> list[str]:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        d = value
        month, mon = d.strftime("%B"), d.strftime("%b")
        return [
            d.isoformat(),
            f"{month} {d.day}, {d.year}",
            f"{d.day} {month} {d.year}",
            f"{mon} {d.day}, {d.year}",
            f"{d.month}/{d.day}/{d.year}",
        ]
    return [str(value)]


def entity_in(text: str, value) -> bool:
    low = text.lower().replace("**", "")
    return any(v.lower() in low for v in _entity_variants(value))


def is_refusal(text: str) -> bool:
    low = text.lower().replace("’", "'")
    return any(m in low for m in REFUSAL_MARKERS)


def grade_answer(kind: str, answer: str, gold, tool_calls: list[dict], expected_tools: list[str]) -> bool:
    if kind == "number":
        return gold is not None and number_in(answer, float(gold))
    if kind == "pct":
        return gold is not None and pct_in(answer, float(gold))
    if kind == "entity":
        return entity_in(answer, gold)
    if kind == "entities":
        return all(entity_in(answer, g) for g in gold)
    if kind == "tool":
        return any(t["name"] in expected_tools and not t["is_error"] for t in tool_calls)
    if kind == "refusal":
        return is_refusal(answer)
    raise ValueError(f"unknown kind {kind}")


def tool_selection_ok(expected: list[str], used: list[str]) -> bool:
    """All expected tools used; no action tool that wasn't expected (run_sql is always allowed)."""
    used_set = set(used)
    return set(expected) <= used_set and not (used_set & ACTION_TOOLS - set(expected))


def execution_match(kind: str, gold, tool_calls: list[dict]) -> bool | None:
    """Did any successful agent query return the gold value? Separates 'fetched the right data' from
    'phrased it right'. None for questions without a SQL gold."""
    if kind not in ("number", "pct", "entity", "entities"):
        return None
    cells = [
        c
        for t in tool_calls
        if t["name"] in SQL_TOOLS and not t["is_error"]
        for row in t.get("rows", [])
        for c in row
    ]
    if kind in ("number", "pct"):
        g = float(gold)
        nums = [float(c) for c in cells if isinstance(c, (int, float)) and not isinstance(c, bool)]
        return any(
            _close(n, g, rel=0.01, abs_tol=1e-6) or (kind == "pct" and _close(n, g * 100, rel=0.01))
            for n in nums
        )
    golds = gold if kind == "entities" else [gold]
    texts = {str(c.date() if isinstance(c, datetime) else c).lower() for c in cells}
    return all(str(g.date() if isinstance(g, datetime) else g).lower() in texts for g in golds)
