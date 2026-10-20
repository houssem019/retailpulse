# syntax=docker/dockerfile:1
# One image, three entrypoints (see docker-compose.yml): agent API, forecast API, MCP server.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    RETAILPULSE_DB_PATH=/data/retailpulse.duckdb \
    RETAILPULSE_MODEL_DIR=/models/champion \
    RETAILPULSE_CHARTS_DIR=/app/outputs/charts

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Dependencies first (cached layer), project code second
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

RUN useradd --create-home app && mkdir -p /app/outputs/charts && chown -R app /app/outputs
USER app

EXPOSE 8000 8001 8002
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\", \"8000\")}/health')" || exit 1
CMD ["uvicorn", "retailpulse.api:app", "--host", "0.0.0.0", "--port", "8000"]
