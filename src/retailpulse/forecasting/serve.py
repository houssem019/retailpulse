"""FastAPI service exposing the champion demand model: item-level forecasts and price what-ifs.

Run:  uv run uvicorn retailpulse.forecasting.serve:app --port 8001

Features are computed on request with the exact same SQL used in training (features.py), which is
how we avoid training/serving skew without a separate feature store.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import duckdb  # noqa: E402
import mlflow  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from retailpulse.config import settings  # noqa: E402
from retailpulse.forecasting.features import FEATURES, HORIZON, build_features, last_actual_date  # noqa: E402
from retailpulse.forecasting.train import CHAMPION  # noqa: E402


class ForecastRequest(BaseModel):
    item_id: str = Field(examples=["FOODS_3_090"])
    store_id: str = Field(examples=["CA_1"])
    horizon: int = Field(HORIZON, ge=1, le=HORIZON, description="Days ahead (direct model: max 28)")


class WhatIfRequest(ForecastRequest):
    price_change_pct: float = Field(ge=-0.9, le=2.0, description="-0.1 = 10% price cut", examples=[-0.1])


class ForecastPoint(BaseModel):
    date: date
    forecast: float


class ForecastResponse(BaseModel):
    item_id: str
    store_id: str
    model_version: str
    total: float
    points: list[ForecastPoint]


class WhatIfPoint(BaseModel):
    date: date
    baseline: float
    scenario: float


class WhatIfResponse(BaseModel):
    item_id: str
    store_id: str
    model_version: str
    price_change_pct: float
    baseline_total: float
    scenario_total: float
    delta_pct: float
    points: list[WhatIfPoint]


def create_app(
    db_path: Path | None = None, mlflow_uri: str | None = None, model_name: str | None = None
) -> FastAPI:
    db_path = db_path or settings.db_path
    mlflow_uri = mlflow_uri or settings.mlflow_uri
    model_name = model_name or settings.model_name
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        mlflow.set_tracking_uri(mlflow_uri)
        mv = MlflowClient().get_model_version_by_alias(model_name, CHAMPION)
        state["model"] = mlflow.lightgbm.load_model(f"models:/{model_name}@{CHAMPION}")
        state["version"] = str(mv.version)
        state["con"] = duckdb.connect(str(db_path), read_only=True)
        state["last"] = last_actual_date(state["con"])
        yield
        state["con"].close()

    app = FastAPI(title="RetailPulse Forecast API", version="0.1.0", lifespan=lifespan)

    def _features(req: ForecastRequest):
        start = state["last"] + timedelta(days=1)
        # DuckDB connections aren't safe to share across threads; a cursor per request is.
        df = build_features(
            state["con"].cursor(),
            start,
            start + timedelta(days=req.horizon - 1),
            stores=[req.store_id],
            items=[req.item_id],
        )
        if df.empty:
            raise HTTPException(
                404, f"No listed series {req.item_id} @ {req.store_id} in the forecast window"
            )
        return df

    def _predict(df):
        return state["model"].predict(df[FEATURES]).clip(min=0)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_version": state.get("version"),
            "last_actual_date": str(state.get("last")),
        }

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(req: ForecastRequest) -> ForecastResponse:
        df = _features(req)
        preds = _predict(df)
        return ForecastResponse(
            item_id=req.item_id,
            store_id=req.store_id,
            model_version=state["version"],
            total=round(float(preds.sum()), 2),
            points=[
                ForecastPoint(date=d, forecast=round(float(p), 3))
                for d, p in zip(df["date"], preds, strict=False)
            ],
        )

    @app.post("/what-if", response_model=WhatIfResponse)
    def what_if(req: WhatIfRequest) -> WhatIfResponse:
        """Re-score the horizon with a different price. Only price features move; demand history is fixed."""
        df = _features(req)
        base = _predict(df)
        scenario_df = df.copy()
        factor = 1 + req.price_change_pct
        scenario_df["sell_price"] *= factor
        scenario_df["price_ratio"] *= factor
        # the price "jumps" on day 1 of the scenario relative to last week's actual price
        first_week = scenario_df["date"] < scenario_df["date"].min() + timedelta(days=7)
        scenario_df.loc[first_week, "price_change_wow"] = (
            1 + scenario_df.loc[first_week, "price_change_wow"].fillna(0)
        ) * factor - 1
        scen = _predict(scenario_df)
        b, s = float(base.sum()), float(scen.sum())
        return WhatIfResponse(
            item_id=req.item_id,
            store_id=req.store_id,
            model_version=state["version"],
            price_change_pct=req.price_change_pct,
            baseline_total=round(b, 2),
            scenario_total=round(s, 2),
            delta_pct=round((s - b) / b, 4) if b else 0.0,
            points=[
                WhatIfPoint(date=d, baseline=round(float(x), 3), scenario=round(float(y), 3))
                for d, x, y in zip(df["date"], base, scen, strict=False)
            ],
        )

    return app


app = create_app()
