import shutil
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from retailpulse.forecasting.features import FEATURES, build_features, last_actual_date
from retailpulse.forecasting.metrics import wrmsse
from retailpulse.forecasting.serve import create_app
from retailpulse.forecasting.train import TrainConfig, train

ITEM, STORE = "FOODS_1_001", "CA_1"


@pytest.fixture(scope="module")
def feats(fixture_db):
    con = duckdb.connect(str(fixture_db), read_only=True)
    last = last_actual_date(con)
    df = build_features(con, date(2015, 6, 1), last + timedelta(days=28))
    return df, last, con


def test_lag_28_is_exactly_28_days_back(feats):
    df, _, _ = feats
    s = df[(df.item_id == ITEM) & (df.store_id == STORE)].set_index("date")
    d = date(2016, 1, 15)
    assert s.loc[d, "lag_28"] == s.loc[d - timedelta(days=28), "units"]


def test_no_leakage_into_horizon(feats):
    """Future rows (no actuals) must still have every lag/rolling feature: they only look >=28 days back."""
    df, last, _ = feats
    future = df[df["date"] > last]
    assert len(future) > 0 and future["units"].isna().all()
    for col in ["lag_28", "rmean_7", "rmean_28", "rmean_56"]:
        assert future[col].notna().all(), col


def test_features_identical_between_train_and_serve_queries(feats, fixture_db):
    """Training builds features for all series over years; serving for one series over 28 days.
    Both must produce the same numbers for the same row (no training/serving skew)."""
    df, last, con = feats
    d = last + timedelta(days=5)
    wide = df[(df.item_id == ITEM) & (df.store_id == STORE) & (df.date == d)]
    narrow = build_features(con, d, d, stores=[STORE], items=[ITEM])
    num = [c for c in FEATURES if c not in ("item_id", "store_id", "dept_id", "cat_id", "event_type_1")]
    np.testing.assert_allclose(wide[num].to_numpy(), narrow[num].to_numpy(), rtol=1e-6)


def test_wrmsse_perfect_and_weighted():
    actuals = pd.DataFrame(
        {
            "item_id": ["a"] * 2 + ["b"] * 2,
            "store_id": ["s"] * 4,
            "units": [1.0, 3.0, 10.0, 10.0],
            "perfect": [1.0, 3.0, 10.0, 10.0],
            "off": [1.0, 3.0, 12.0, 12.0],
        }
    )
    scales = pd.DataFrame(
        {"item_id": ["a", "b"], "store_id": ["s", "s"], "scale": [1.0, 4.0], "recent_revenue": [0.0, 100.0]}
    )
    assert wrmsse(actuals, "perfect", scales) == 0
    assert wrmsse(actuals, "off", scales) == pytest.approx(1.0)  # sqrt(4/4) on the only weighted series


@pytest.fixture(scope="module")
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


def test_training_registers_and_publishes(trained):
    cfg, m = trained
    assert m["promoted"] and m["model_version"] == "1"
    assert m["holdout_wrmsse"] > 0 and m["n_series"] == 105
    con = duckdb.connect(str(cfg.db_path), read_only=True)
    kinds = dict(con.execute("SELECT kind, count(*) FROM forecasts GROUP BY 1").fetchall())
    assert kinds["backtest"] > 0 and kinds["future"] > 0
    assert con.execute("SELECT count(*) FROM forecast_accuracy").fetchone()[0] == kinds["backtest"]


def test_serving_forecast_and_what_if(trained):
    cfg, _ = trained
    app = create_app(db_path=cfg.db_path, mlflow_uri=cfg.mlflow_uri, model_name=cfg.model_name)
    with TestClient(app) as client:
        assert client.get("/health").json()["model_version"] == "1"
        r = client.post("/forecast", json={"item_id": ITEM, "store_id": STORE, "horizon": 7})
        assert r.status_code == 200 and len(r.json()["points"]) == 7
        r = client.post("/what-if", json={"item_id": ITEM, "store_id": STORE, "price_change_pct": -0.2})
        body = r.json()
        assert r.status_code == 200 and len(body["points"]) == 28
        assert body["scenario_total"] >= 0
        assert client.post("/forecast", json={"item_id": "NOPE", "store_id": STORE}).status_code == 404


def test_price_cut_never_lowers_demand(trained):
    """Monotone constraints: a price cut can only keep or raise predicted demand, on every day."""
    cfg, _ = trained
    app = create_app(db_path=cfg.db_path, mlflow_uri=cfg.mlflow_uri, model_name=cfg.model_name)
    with TestClient(app) as client:
        for pct, sign in ((-0.3, 1), (0.3, -1)):
            body = client.post(
                "/what-if", json={"item_id": ITEM, "store_id": STORE, "price_change_pct": pct}
            ).json()
            for p in body["points"]:
                assert sign * (p["scenario"] - p["baseline"]) >= -1e-9
