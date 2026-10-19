from pathlib import Path

import pytest

from retailpulse.data.ingest import build_database

FIXTURE = Path(__file__).parent / "fixtures" / "m5_sample"


@pytest.fixture(scope="session")
def fixture_db(tmp_path_factory) -> Path:
    """A DuckDB warehouse built from the committed raw-format M5 sample via the real ingest path."""
    path = tmp_path_factory.mktemp("db") / "fixture.duckdb"
    build_database(FIXTURE, path)
    return path
