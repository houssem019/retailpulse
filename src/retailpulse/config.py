from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration, read from env vars / .env (prefix RETAILPULSE_ for project settings)."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RETAILPULSE_", extra="ignore")

    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    db_path: Path = PROJECT_ROOT / "data" / "processed" / "retailpulse.duckdb"


settings = Settings()
