from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str) -> AliasChoices:
    """Accept both the shared un-prefixed name (e.g. LLM_PROVIDER) and RETAILPULSE_<name>."""
    return AliasChoices(name, f"RETAILPULSE_{name}")


class Settings(BaseSettings):
    """Runtime configuration, read from env vars / .env (prefix RETAILPULSE_ for project settings)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="RETAILPULSE_", extra="ignore", populate_by_name=True
    )

    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    db_path: Path = PROJECT_ROOT / "data" / "processed" / "retailpulse.duckdb"

    # Forecasting / MLflow
    mlflow_uri: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow_artifacts: Path = PROJECT_ROOT / "mlartifacts"
    model_dir: Path | None = None  # serve an exported champion folder instead of the registry (containers)
    model_name: str = "retailpulse-demand"
    forecast_stores: list[str] = ["CA_1", "TX_1", "WI_1"]  # one store per state keeps training ~minutes
    train_start: str = "2014-01-01"
    forecast_api_url: str = "http://localhost:8001"

    # LLM provider: anthropic (Claude API) | ollama (free, local) | fake (replay recorded responses)
    llm_provider: Literal["anthropic", "ollama", "fake"] = Field(
        "anthropic", validation_alias=_env("LLM_PROVIDER")
    )
    llm_model: str = Field("claude-opus-5-5", validation_alias=_env("LLM_MODEL"))
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        "medium", validation_alias=_env("LLM_EFFORT")
    )
    ollama_host: str = Field("http://localhost:11434", validation_alias=_env("OLLAMA_HOST"))
    ollama_model: str = Field("qwen2.5:7b", validation_alias=_env("OLLAMA_MODEL"))
    cassette_dir: Path = PROJECT_ROOT / "tests" / "cassettes"

    # Agent guardrails
    agent_max_steps: int = 10  # model calls per question
    sql_timeout_s: float = 20.0
    sql_max_rows: int = 200  # rows returned to the agent (the model sees at most 50)
    sql_approval_rows: int = 60_000_000  # estimated rows scanned above which a human must approve
    charts_dir: Path = PROJECT_ROOT / "outputs" / "charts"


settings = Settings()
