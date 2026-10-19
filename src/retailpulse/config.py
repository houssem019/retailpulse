from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration, read from env vars / .env (prefix RETAILPULSE_ for project settings)."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RETAILPULSE_", extra="ignore")

    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    db_path: Path = PROJECT_ROOT / "data" / "processed" / "retailpulse.duckdb"

    # Forecasting / MLflow
    mlflow_uri: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow_artifacts: Path = PROJECT_ROOT / "mlartifacts"
    model_name: str = "retailpulse-demand"
    forecast_stores: list[str] = ["CA_1", "TX_1", "WI_1"]  # one store per state keeps training ~minutes
    train_start: str = "2014-01-01"


settings = Settings()
