"""Environments the agent can be evaluated in.

* fixture: the committed 105-series M5 sample -> DuckDB -> tiny LightGBM model -> in-process forecast API.
  Deterministic and fast (~15 s); used by tests and by the CI regression gate.
* live: the full warehouse and the running forecast API (settings), for real-scale evaluation runs.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import duckdb

from retailpulse.config import PROJECT_ROOT
from retailpulse.data.ingest import build_database
from retailpulse.forecasting.train import TrainConfig, train
from retailpulse.tools.forecast import ForecastClient
from retailpulse.tools.registry import Toolbox
from retailpulse.tools.sql import SqlSandbox

FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "m5_sample"
FIXTURE_STORES = ["CA_1", "TX_1", "WI_1"]


def train_fixture_model(fixture_db: Path, workdir: Path) -> tuple[TrainConfig, dict]:
    """Train a tiny demand model on a copy of the fixture warehouse, into a throwaway MLflow store."""
    db = workdir / "db.duckdb"
    shutil.copy(fixture_db, db)
    cfg = TrainConfig(
        db_path=db,
        stores=FIXTURE_STORES,
        train_start=date(2015, 1, 1),
        num_boost_round=30,
        early_stopping_rounds=10,
        mlflow_uri=f"sqlite:///{(workdir / 'mlflow.db').as_posix()}",
        mlflow_artifacts=workdir / "artifacts",
        model_name="fixture-demand",
    )
    return cfg, train(cfg)


def approval_threshold(db: Path, full_scans: float = 1.28) -> int:
    """Scale the 'needs approval' threshold to the warehouse size (production: 60M rows on a 47M-row
    sales table = 1.28 full scans), so fixture evals exercise the same policy as production."""
    with duckdb.connect(str(db), read_only=True) as con:
        return int(con.execute("SELECT count(*) FROM sales").fetchone()[0] * full_scans)


@contextmanager
def fixture_toolbox(workdir: Path | None = None) -> Iterator[Toolbox]:
    from fastapi.testclient import TestClient

    from retailpulse.forecasting.serve import create_app

    tmp = Path(workdir or tempfile.mkdtemp(prefix="retailpulse-eval-"))
    base = tmp / "fixture.duckdb"
    build_database(FIXTURE_DIR, base)
    cfg, _ = train_fixture_model(base, tmp)
    app = create_app(db_path=cfg.db_path, mlflow_uri=cfg.mlflow_uri, model_name=cfg.model_name)
    with TestClient(app) as http:
        sandbox_db = tmp / "sandbox.duckdb"  # separate file: DuckDB forbids mixed configs in one process
        shutil.copy(cfg.db_path, sandbox_db)
        sandbox = SqlSandbox(sandbox_db, approval_rows=approval_threshold(sandbox_db))
        try:
            yield Toolbox(sandbox, ForecastClient(http), tmp / "charts")
        finally:
            sandbox.close()


@contextmanager
def live_toolbox() -> Iterator[Toolbox]:
    toolbox = Toolbox.from_settings()
    try:
        yield toolbox
    finally:
        toolbox.sandbox.close()
