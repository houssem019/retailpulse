"""Agent tools that call the forecasting service over HTTP (the model is served, not imported).

Keeping the model behind an API means the agent and the model can be deployed, scaled and rolled
back independently; the agent only depends on the request/response contract.
"""

from __future__ import annotations

import httpx


class ForecastServiceError(RuntimeError):
    pass


class ForecastClient:
    def __init__(self, http: httpx.Client):
        self.http = http

    @classmethod
    def from_url(cls, base_url: str, timeout_s: float = 30.0) -> ForecastClient:
        return cls(httpx.Client(base_url=base_url, timeout=timeout_s))

    def _post(self, path: str, payload: dict) -> dict:
        try:
            r = self.http.post(path, json=payload)
        except httpx.HTTPError as e:
            raise ForecastServiceError(f"Forecast service unavailable ({e.__class__.__name__}).") from e
        if r.status_code == 404:
            raise ForecastServiceError(r.json().get("detail", "Series not found."))
        if r.status_code >= 400:
            raise ForecastServiceError(f"Forecast service error {r.status_code}: {r.text[:200]}")
        return r.json()

    def forecast(self, item_id: str, store_id: str, horizon: int = 28) -> dict:
        body = self._post("/forecast", {"item_id": item_id, "store_id": store_id, "horizon": horizon})
        return {k: body[k] for k in ("item_id", "store_id", "model_version", "total")} | {
            "daily": [[p["date"], p["forecast"]] for p in body["points"]]
        }

    def what_if(self, item_id: str, store_id: str, price_change_pct: float, horizon: int = 28) -> dict:
        body = self._post(
            "/what-if",
            {
                "item_id": item_id,
                "store_id": store_id,
                "horizon": horizon,
                "price_change_pct": price_change_pct,
            },
        )
        return {
            k: body[k]
            for k in (
                "item_id",
                "store_id",
                "model_version",
                "price_change_pct",
                "baseline_total",
                "scenario_total",
                "delta_pct",
            )
        }
