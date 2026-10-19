# RetailPulse — Agentic Retail Analytics Assistant

LLM agent that answers natural-language questions on sales, inventory and forecast accuracy over the M5 Walmart dataset,
using text-to-SQL (DuckDB), a LightGBM forecasting service and chart generation — exposed as an MCP server.

> 🚧 Work in progress. See `docs/BUILD_LOG.md`.

## Quickstart
```bash
uv sync --python 3.11
kaggle competitions download -c m5-forecasting-accuracy -p data/raw && unzip data/raw/*.zip -d data/raw
uv run python -m retailpulse.data.ingest            # DuckDB warehouse (~90 s, 46.9M sales rows)
uv run python -m retailpulse.forecasting.train      # LightGBM -> MLflow registry (~6 min, 3 stores)
uv run uvicorn retailpulse.forecasting.serve:app --port 8001   # /forecast, /what-if
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db       # inspect runs & registry
uv run python -m retailpulse.agent "Which state had the highest revenue in 2015?"   # needs LLM_PROVIDER (see .env.example)
uv run pytest                                        # runs on a committed 105-series fixture, no Kaggle / API key needed
```

## Demand forecasting results
28-day holdout (2016-04-25 to 2016-05-22), 3 stores (CA_1, TX_1, WI_1), 9,147 item-store series:

| Model | WRMSSE (item×store) ↓ |
|---|---|
| **LightGBM (Tweedie, monotone price constraints)** | **0.823** |
| 28-day moving average | 0.841 |
| Seasonal naive (same weekday last week) | 1.076 |

The price what-if is monotone by construction. At CA_1, a 20% price cut raises forecast demand by +7% to +13%, depending on the item.
