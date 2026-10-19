"""Read-only SQL sandbox for LLM-generated queries.

Defence in depth: each layer would stop most bad queries on its own.
  1. Parse with sqlglot: exactly one statement, must be a SELECT/WITH/UNION query, may only read the
     warehouse tables, and may not call file/network table functions.
  2. Execute on a DuckDB connection opened read_only with enable_external_access=false, so even a
     query that slipped past the parser cannot write, ATTACH, COPY or read files.
  3. A LIMIT is injected and results are capped; a timer interrupts queries that run too long.
  4. EXPLAIN estimates rows scanned; above a threshold the query needs human approval (the agent
     graph pauses and asks) instead of silently burning compute.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {
    "sales",
    "items",
    "stores",
    "calendar",
    "prices",
    "forecasts",
    "forecast_accuracy",
    "inventory_signals",
}
BLOCKED_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "parquet_scan",
    "read_json",
    "read_json_auto",
    "read_ndjson",
    "read_text",
    "read_blob",
    "glob",
    "sniff_csv",
    "query",
    "query_table",
    "getenv",
    "current_setting",
    "duckdb_settings",
    "duckdb_secrets",
}


class SqlRejected(ValueError):
    """The query violates the sandbox policy (message is safe to show to the model)."""


@dataclass
class CheckedQuery:
    sql: str  # the rewritten SQL that will actually run (LIMIT injected)
    estimated_rows_scanned: int
    needs_approval: bool


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[list]
    row_count: int  # rows returned (<= max_rows)
    truncated: bool
    elapsed_ms: float
    estimated_rows_scanned: int = 0
    extra: dict = field(default_factory=dict)

    def to_model_payload(self, max_rows_shown: int = 50) -> str:
        """Compact JSON for the LLM; it never needs more than ~50 rows to answer a question."""
        shown = self.rows[:max_rows_shown]
        payload = {
            "columns": self.columns,
            "rows": shown,
            "row_count": self.row_count,
            "truncated": self.truncated or len(shown) < self.row_count,
        }
        return json.dumps(payload, default=str)


class SqlSandbox:
    def __init__(
        self,
        db_path: Path,
        *,
        max_rows: int = 200,
        timeout_s: float = 20.0,
        approval_rows: int = 60_000_000,
    ):
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self.approval_rows = approval_rows
        self._con = duckdb.connect(str(db_path), read_only=True, config={"enable_external_access": False})

    def close(self) -> None:
        self._con.close()

    # ---- layer 1: static validation ---------------------------------------------------------------
    def validate(self, sql: str) -> str:
        try:
            statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
        except sqlglot.errors.ParseError as e:
            raise SqlRejected(f"SQL syntax error: {e}") from e
        if len(statements) != 1:
            raise SqlRejected("Exactly one SQL statement is allowed.")
        stmt = statements[0]
        if not isinstance(stmt, exp.Query):
            raise SqlRejected("Only read-only SELECT queries are allowed.")
        for node in stmt.walk():
            if isinstance(node, (exp.DML, exp.DDL, exp.Command)):
                raise SqlRejected("Only read-only SELECT queries are allowed.")

        cte_names = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
        for table in stmt.find_all(exp.Table):
            name = table.name.lower()
            if not name and table.this is not None and not isinstance(table.this, exp.Identifier):
                raise SqlRejected("Table functions are not allowed; query the warehouse tables directly.")
            if name and name not in ALLOWED_TABLES and name not in cte_names:
                raise SqlRejected(
                    f"Unknown table '{table.name}'. Available: {', '.join(sorted(ALLOWED_TABLES))}."
                )
        for func in stmt.find_all(exp.Func):
            fname = (func.sql_name() if not isinstance(func, exp.Anonymous) else func.name).lower()
            if fname in BLOCKED_FUNCTIONS:
                raise SqlRejected(f"Function '{fname}' is not allowed.")

        if stmt.args.get("limit") is None:
            stmt = stmt.limit(self.max_rows + 1)
        return stmt.sql(dialect="duckdb")

    # ---- layer 4: cost estimate --------------------------------------------------------------------
    def estimate_rows_scanned(self, sql: str) -> int:
        plan = json.loads(self._con.cursor().execute(f"EXPLAIN (FORMAT JSON) {sql}").fetchall()[0][1])
        total = 0

        def walk(node: dict) -> None:
            nonlocal total
            info = node.get("extra_info", {})
            # Scans with no projected columns only read row ids (DuckDB's late-materialisation plan
            # for `SELECT * ... LIMIT n` adds one); they are cheap, so don't count them.
            if "SCAN" in node.get("name", "") and info.get("Projections", "") != "":
                est = info.get("Estimated Cardinality", "0")
                total += int(str(est).lstrip("~") or 0)
            for child in node.get("children", []):
                walk(child)

        for root in plan:
            walk(root)
        return total

    def check(self, sql: str) -> CheckedQuery:
        safe_sql = self.validate(sql)
        try:
            est = self.estimate_rows_scanned(safe_sql)
        except duckdb.Error as e:
            raise SqlRejected(f"SQL error: {e}") from e
        return CheckedQuery(safe_sql, est, needs_approval=est > self.approval_rows)

    # ---- layers 2+3: execution ---------------------------------------------------------------------
    def run(self, sql: str, *, approved: bool = False) -> QueryResult:
        checked = self.check(sql)
        if checked.needs_approval and not approved:
            raise SqlRejected(
                f"Query would scan ~{checked.estimated_rows_scanned:,} rows and needs human approval."
            )
        cur = self._con.cursor()
        timer = threading.Timer(self.timeout_s, cur.interrupt)
        start = time.perf_counter()
        timer.start()
        try:
            cur.execute(checked.sql)
            columns = [d[0] for d in cur.description]
            rows = [list(r) for r in cur.fetchmany(self.max_rows + 1)]
        except duckdb.InterruptException as e:
            raise SqlRejected(
                f"Query exceeded the {self.timeout_s:.0f}s time limit; add filters or aggregate."
            ) from e
        except duckdb.Error as e:
            raise SqlRejected(f"SQL error: {e}") from e
        finally:
            timer.cancel()
            cur.close()
        truncated = len(rows) > self.max_rows
        rows = rows[: self.max_rows]
        return QueryResult(
            sql=checked.sql,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            elapsed_ms=round((time.perf_counter() - start) * 1000, 1),
            estimated_rows_scanned=checked.estimated_rows_scanned,
        )
