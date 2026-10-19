"""Chart tool: run a (sandboxed) query and render it with Plotly to a standalone HTML file."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import plotly.express as px

from retailpulse.tools.sql import QueryResult, SqlRejected


def render_chart(
    result: QueryResult, *, chart_type: str, x: str, y: str, color: str | None, title: str, out_dir: Path
) -> dict:
    df = pd.DataFrame(result.rows, columns=result.columns)
    missing = [c for c in (x, y, color) if c and c not in df.columns]
    if missing:
        raise SqlRejected(f"Columns {missing} not in query result columns {result.columns}.")
    plot = {"line": px.line, "bar": px.bar, "scatter": px.scatter}[chart_type]
    fig = plot(df, x=x, y=y, color=color, title=title)
    fig.update_layout(template="plotly_white")

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50] or "chart"
    digest = hashlib.sha1(result.sql.encode()).hexdigest()[:8]
    path = out_dir / f"{slug}-{digest}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return {"chart_path": str(path), "points": len(df), "truncated": result.truncated}
