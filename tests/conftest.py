import shutil
from datetime import date
from pathlib import Path

import pytest

from retailpulse.data.ingest import build_database
from retailpulse.forecasting.train import TrainConfig, train

FIXTURE = Path(__file__).parent / "fixtures" / "m5_sample"


@pytest.fixture(scope="session")
def fixture_db(tmp_path_factory) -> Path:
    """A DuckDB warehouse built from the committed raw-format M5 sample via the real ingest path."""
    path = tmp_path_factory.mktemp("db") / "fixture.duckdb"
    build_database(FIXTURE, path)
    return path


@pytest.fixture(scope="session")
def trained(fixture_db, tmp_path_factory):
    work = tmp_path_factory.mktemp("train")
    db = work / "db.duckdb"
    shutil.copy(fixture_db, db)
    cfg = TrainConfig(
        db_path=db,
        stores=["CA_1", "TX_1", "WI_1"],
        train_start=date(2015, 1, 1),
        num_boost_round=30,
        early_stopping_rounds=10,
        mlflow_uri=f"sqlite:///{(work / 'mlflow.db').as_posix()}",
        mlflow_artifacts=work / "artifacts",
        model_name="test-demand",
    )
    return cfg, train(cfg)


@pytest.fixture(scope="session")
def toolbox(trained, tmp_path_factory):
    """Real Toolbox over the trained fixture warehouse; the forecast API runs in-process via TestClient."""
    from fastapi.testclient import TestClient

    from retailpulse.forecasting.serve import create_app
    from retailpulse.tools.forecast import ForecastClient
    from retailpulse.tools.registry import Toolbox
    from retailpulse.tools.sql import SqlSandbox

    cfg, _ = trained
    app = create_app(db_path=cfg.db_path, mlflow_uri=cfg.mlflow_uri, model_name=cfg.model_name)
    with TestClient(app) as http:
        # DuckDB refuses two in-process connections to one file with different configs (the API's plain
        # read-only one vs the sandbox's locked-down one). In production they are separate processes.
        sandbox_db = tmp_path_factory.mktemp("sandbox") / "db.duckdb"
        shutil.copy(cfg.db_path, sandbox_db)
        # threshold sits between one and two full scans of the fixture's sales table (~70k rows)
        sandbox = SqlSandbox(sandbox_db, approval_rows=100_000)
        yield Toolbox(sandbox, ForecastClient(http), tmp_path_factory.mktemp("charts"))
        sandbox.close()
