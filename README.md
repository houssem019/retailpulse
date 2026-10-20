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

## Use it from Claude Code (MCP)
The repo ships a `.mcp.json`, so opening Claude Code in this folder offers the `retailpulse` MCP server. From another folder, register it with:
```bash
claude mcp add retailpulse -- uv --directory /path/to/retailpulse run python -m retailpulse.mcp_server
```
Start the forecast API, which the forecast tools call (`uv run uvicorn retailpulse.forecasting.serve:app --port 8001`). Then try prompts like:
- *"Which 5 FOODS items lost the most units to likely stockouts, and in which stores?"*
- *"How accurate was the forecast for HOUSEHOLD in TX_1 over the backtest window? Give WAPE and bias."*
- *"If we cut the price of FOODS_3_090 in CA_1 by 15%, what happens to demand over the next 4 weeks?"*
- *"Chart monthly FOODS revenue by state for 2015."*

Tools: `get_schema`, `run_sql`, `forecast_item`, `price_what_if`, `create_chart`. They share the same guardrails as the built-in agent. Queries that scan very large amounts of data ask for your approval.

## Agent API, tracing, Docker
```bash
uv run uvicorn retailpulse.api:app --port 8000      # POST /ask, /approve, /ask/stream (SSE); GET /health
curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Top 3 departments by 2015 units?"}'
```
If a query needs approval, the response is `{"status": "approval_required", "thread_id": ..., "approval": {"sql": ..., "estimated_rows_scanned": ...}}`. Answer it with `POST /approve {"thread_id": ..., "approve": true}`.

**Tracing:** set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` to get a trace per question. Each trace has an agent span, then one generation per LLM call (tokens and cost), then one span per tool call. Without them, each question still produces a JSON log line with latency, tokens and cost.

**Docker:** build the warehouse and train the model on the host, then:
```bash
uv run python scripts/export_champion.py && docker compose up --build    # agent :8000, forecast :8001, MCP :8002/mcp
```

## Agent evaluation
25 benchmark questions: SQL analytics, inventory, forecast accuracy, item forecasts, price what-ifs, charts and guardrail attacks. Gold answers are computed at eval time.

Baseline, with a free local model (`qwen2.5:7b` via Ollama on CPU), on the fixture warehouse:

| metric | score |
|---|---|
| Answer accuracy | **68%** |
| Execution match (agent's SQL returned the gold value) | 61% |
| SQL validity (queries that ran without error) | 53% |
| Tool-selection accuracy | 96% |
| Avg LLM calls / latency per question | 2.6 / ~2 min (CPU) |
| Cost | $0 |

CI replays the recorded responses (33 s, no API key) and fails if any metric drops below `evals/baseline.json`. The failure analysis is in `docs/CHALLENGES.md` (#17).

```bash
uv run python -m retailpulse.evals.run --provider fake --cassette evals/cassettes/qwen2.5-7b.json --gate   # what CI runs
uv run python -m retailpulse.evals.run --provider ollama --cassette evals/cassettes/qwen2.5-7b.json --record   # re-record
```

## Demand forecasting results
28-day holdout (2016-04-25 to 2016-05-22), 3 stores (CA_1, TX_1, WI_1), 9,147 item-store series:

| Model | WRMSSE (item×store) ↓ |
|---|---|
| **LightGBM (Tweedie, monotone price constraints)** | **0.823** |
| 28-day moving average | 0.841 |
| Seasonal naive (same weekday last week) | 1.076 |

The price what-if is monotone by construction. At CA_1, a 20% price cut raises forecast demand by +7% to +13%, depending on the item.
