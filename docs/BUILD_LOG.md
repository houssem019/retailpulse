# Build Log

Chronological record of what was built, **why**, and which alternatives were rejected.
Read this before an interview to retell the story of the project.

| Step | What | Why / alternatives considered |
|------|------|-------------------------------|
| 1 | **Data warehouse.** Downloaded M5 (Kaggle `m5-forecasting-accuracy`). `retailpulse.data.ingest` reshapes the wide sales file (1,941 day columns) into a DuckDB star schema: `items`, `stores`, `calendar`, `prices`, `sales`. | **DuckDB** over Postgres/SQLite: it's embedded (no server), columnar and fast for analytics, and it reads CSV natively. A read-only connection is also a simple sandbox for the agent later. **The reshape runs in SQL (`UNPIVOT`), not pandas:** pandas `melt` on 59M rows needs several GB of RAM, while DuckDB streams it in 92s. **`revenue` is precomputed** in the fact table so the LLM doesn't need a three-way join for dollar questions. That means fewer chances for wrong SQL. |
| 2 | **Test fixture.** `scripts/make_fixture.py` cuts 105 series (3 stores × 5 items × 7 depts, last 730 days) in the *raw* M5 format (<1 MB). | Tests run the **real** ingest code on real-shaped data, so CI needs no Kaggle credentials and runs in seconds. The alternative was a mocked DB, but that wouldn't catch ingest bugs. |
