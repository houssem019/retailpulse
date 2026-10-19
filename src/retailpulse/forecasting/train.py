"""Train, evaluate, register and publish the LightGBM demand model.

Pipeline:
  1. Build features (features.py) from train_start up to 28 days past the last actual sale.
  2. Split by time:  train | valid (28d, early stopping) | holdout (last 28d actuals) | future (28d).
  3. Model A = fit on train, early-stopped on valid  -> scored on holdout vs two baselines.
  4. Model B = refit on everything up to the last actual with A's best iteration count
     -> the model we register and use for future forecasts.
  5. Log params/metrics/model to MLflow; promote to the `champion` alias only if holdout WRMSSE
     beats the current champion.
  6. Write backtest + future forecasts to DuckDB (`forecasts` table, `forecast_accuracy` view)
     so the agent can answer forecast and accuracy questions with SQL.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import duckdb  # noqa: E402
import lightgbm as lgb  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from retailpulse.config import settings  # noqa: E402
from retailpulse.forecasting.features import FEATURES, HORIZON, TARGET, build_features, last_actual_date  # noqa: E402
from retailpulse.forecasting.metrics import point_metrics, series_scales_and_weights, wrmsse  # noqa: E402

EXPERIMENT = "retailpulse-demand"
CHAMPION = "champion"

PRICE_FEATURES = {"sell_price", "price_ratio", "price_change_wow"}

# Tweedie loss suits retail demand: many zeros plus a long right tail (it was the standard choice in M5).
LGB_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
    # Demand must not rise when price rises. Without this the model learned "low price -> low sales"
    # from clearance markdowns (confounding) and a 20% price cut predicted -54% demand.
    "monotone_constraints": [-1 if f in PRICE_FEATURES else 0 for f in FEATURES],
    "monotone_constraints_method": "advanced",
}


@dataclass
class TrainConfig:
    db_path: Path = settings.db_path
    stores: list[str] = field(default_factory=lambda: list(settings.forecast_stores))
    train_start: date = date.fromisoformat(settings.train_start)
    num_boost_round: int = 1500
    early_stopping_rounds: int = 50
    mlflow_uri: str = settings.mlflow_uri
    mlflow_artifacts: Path = settings.mlflow_artifacts
    model_name: str = settings.model_name


def _setup_mlflow(cfg: TrainConfig) -> None:
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        cfg.mlflow_artifacts.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(EXPERIMENT, artifact_location=cfg.mlflow_artifacts.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def _baselines(df: pd.DataFrame, holdout: pd.DataFrame, holdout_start: date) -> pd.DataFrame:
    """Two classic baselines a model must beat to be worth deploying."""
    hist = df[(df["date"] < holdout_start) & (df["date"] >= holdout_start - timedelta(days=28))]
    keys = ["item_id", "store_id"]
    ma28 = hist.groupby(keys, observed=True)["units"].mean().rename("ma28").reset_index()
    last_week = hist[hist["date"] >= holdout_start - timedelta(days=7)]
    snaive = last_week[keys + ["wday", "units"]].rename(columns={"units": "snaive7"})
    out = holdout.merge(ma28, on=keys, how="left").merge(snaive, on=keys + ["wday"], how="left")
    return out.fillna({"ma28": 0, "snaive7": 0})


def _champion_wrmsse(client: MlflowClient, model_name: str) -> float | None:
    try:
        mv = client.get_model_version_by_alias(model_name, CHAMPION)
    except Exception:  # no registered model / alias yet
        return None
    return client.get_run(mv.run_id).data.metrics.get("holdout_wrmsse")


def _write_forecasts(db_path: Path, stores: list[str], frames: dict[str, pd.DataFrame], version: str) -> int:
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            item_id VARCHAR, store_id VARCHAR, date DATE, forecast DOUBLE,
            kind VARCHAR,            -- 'backtest' (holdout, comparable to actuals) | 'future'
            model_version VARCHAR, created_at TIMESTAMP DEFAULT current_timestamp
        )""")
    con.execute("DELETE FROM forecasts WHERE store_id IN (SELECT unnest(?::VARCHAR[]))", [stores])
    for kind, frame in frames.items():
        fc = frame[["item_id", "store_id", "date", "pred"]].astype({"item_id": str, "store_id": str})
        con.register("fc", fc)
        con.execute(
            "INSERT INTO forecasts (item_id, store_id, date, forecast, kind, model_version) "
            "SELECT item_id, store_id, date, pred, ?, ? FROM fc",
            [kind, version],
        )
        con.unregister("fc")
    con.execute("""
        CREATE OR REPLACE VIEW forecast_accuracy AS
        SELECT f.item_id, f.store_id, f.date, f.forecast, s.units AS actual,
               f.forecast - s.units AS error, abs(f.forecast - s.units) AS abs_error, f.model_version
        FROM forecasts f JOIN sales s USING (item_id, store_id, date)
        WHERE f.kind = 'backtest'""")
    n = con.execute("SELECT count(*) FROM forecasts").fetchone()[0]
    con.close()
    return n


def train(cfg: TrainConfig) -> dict:
    t0 = time.perf_counter()
    con = duckdb.connect(str(cfg.db_path), read_only=True)
    last = last_actual_date(con)
    holdout_start = last - timedelta(days=HORIZON - 1)
    valid_start = holdout_start - timedelta(days=HORIZON)

    df = build_features(con, cfg.train_start, last + timedelta(days=HORIZON), stores=cfg.stores)
    scales = series_scales_and_weights(con, holdout_start, cfg.stores)
    con.close()
    t_features = time.perf_counter() - t0

    train_df = df[df["date"] < valid_start]
    valid_df = df[(df["date"] >= valid_start) & (df["date"] < holdout_start)]
    holdout = df[(df["date"] >= holdout_start) & (df["date"] <= last)].copy()
    future = df[df["date"] > last].copy()

    # --- Model A: honest evaluation -------------------------------------------------------------
    dtrain = lgb.Dataset(train_df[FEATURES], train_df[TARGET])
    dvalid = lgb.Dataset(valid_df[FEATURES], valid_df[TARGET], reference=dtrain)
    model_a = lgb.train(
        LGB_PARAMS,
        dtrain,
        cfg.num_boost_round,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False), lgb.log_evaluation(100)],
    )
    best_iter = model_a.best_iteration or cfg.num_boost_round
    holdout["pred"] = model_a.predict(holdout[FEATURES], num_iteration=best_iter).clip(min=0)
    scored = _baselines(df, holdout, holdout_start)

    metrics = {
        "holdout_wrmsse": wrmsse(scored, "pred", scales),
        "baseline_ma28_wrmsse": wrmsse(scored, "ma28", scales),
        "baseline_snaive7_wrmsse": wrmsse(scored, "snaive7", scales),
        **{f"holdout_{k}": v for k, v in point_metrics(scored["units"], scored["pred"]).items()},
        "best_iteration": best_iter,
        "n_series": int(df.groupby(["item_id", "store_id"], observed=True).ngroups),
        "n_train_rows": len(train_df),
    }

    # --- Model B: production refit on all actuals -----------------------------------------------
    full = df[df["date"] <= last]
    model_b = lgb.train(LGB_PARAMS, lgb.Dataset(full[FEATURES], full[TARGET]), best_iter)
    future["pred"] = model_b.predict(future[FEATURES]).clip(min=0)
    metrics["train_seconds"] = time.perf_counter() - t0
    metrics["feature_seconds"] = t_features

    # --- MLflow tracking + registry -------------------------------------------------------------
    _setup_mlflow(cfg)
    client = MlflowClient()
    with mlflow.start_run(run_name=f"lgbm-{'-'.join(cfg.stores)}") as run:
        mlflow.log_params(
            {
                **LGB_PARAMS,
                "stores": ",".join(cfg.stores),
                "train_start": str(cfg.train_start),
                "horizon": HORIZON,
                "features": ",".join(FEATURES),
                "last_actual": str(last),
            }
        )
        mlflow.log_metrics(metrics)
        importance = pd.DataFrame({"feature": FEATURES, "gain": model_b.feature_importance("gain")})
        mlflow.log_table(importance.sort_values("gain", ascending=False), "feature_importance.json")
        info = mlflow.lightgbm.log_model(
            model_b,
            name="model",
            registered_model_name=cfg.model_name,
            # No input_example: MLflow round-trips it through JSON, which drops the pandas `category`
            # dtypes LightGBM needs. The real input contract is features.build_features().
        )
        version = str(info.registered_model_version)

    previous = _champion_wrmsse(client, cfg.model_name)
    promoted = previous is None or metrics["holdout_wrmsse"] < previous
    if promoted:
        client.set_registered_model_alias(cfg.model_name, CHAMPION, version)
    metrics.update(
        run_id=run.info.run_id, model_version=version, promoted=promoted, previous_champion_wrmsse=previous
    )

    # Only publish forecasts from a promoted model, so the agent never serves a worse model's numbers.
    if promoted:
        metrics["forecast_rows"] = _write_forecasts(
            cfg.db_path, cfg.stores, {"backtest": holdout, "future": future}, version
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register the RetailPulse demand model")
    parser.add_argument("--stores", nargs="+", default=list(settings.forecast_stores))
    parser.add_argument(
        "--train-start", type=date.fromisoformat, default=date.fromisoformat(settings.train_start)
    )
    parser.add_argument("--rounds", type=int, default=1500)
    args = parser.parse_args()

    result = train(TrainConfig(stores=args.stores, train_start=args.train_start, num_boost_round=args.rounds))
    for k, v in result.items():
        print(f"  {k:<26} {v:.4f}" if isinstance(v, float) else f"  {k:<26} {v}")


if __name__ == "__main__":
    main()
