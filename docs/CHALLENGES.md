# Challenges & Lessons

Real problems hit while building, how they were solved, and known limitations.
Format: **Problem → Root cause → Fix → What I'd do at scale.**

## 1. Structural zeros in M5
- **Problem:** the raw file has a value for every item×store×day since 2011, but many items launched later. About 21% of the cells (12.3M of 59.2M) are zeros from before an item was ever on the shelf.
- **Root cause:** the wide format can't express "not listed", so missing days are stored as 0.
- **Fix:** inner-join to `sell_prices`, because an item with no price that week wasn't sold in that store. This keeps 46.9M rows.
  - **Verified:** total units before and after are identical (66,927,173), so no real sale was lost.
- **Why it matters:** the zeros would bias average demand down, inflate "zero-sales" stockout signals, and make forecasting models learn launch effects as demand.

## 2. The reshape doesn't fit in pandas
- **Problem:** `melt`-ing 30,490 × 1,941 = 59M rows in pandas needs several GB of RAM.
- **Fix:** run `UNPIVOT` plus the joins inside DuckDB, which streams the data and spills to disk if needed. It finished in 92s.
- **At scale:** the same SQL runs on Spark or Databricks, and the logic ports directly to a Delta table.

## 3. M5 has no inventory data
- **Problem:** the resume says "sales, inventory and forecast accuracy", but M5 contains sales and prices only.
- **Fix (planned):** derive inventory-health *signals* rather than stock levels:
  - zero-sales streaks while the item is listed (a likely stockout or out-of-shelf);
  - days of supply against the forecast.
  - **Interview line:** "M5 has no on-hand data, so I modelled availability from sales patterns. That's the same approach retailers use for phantom-inventory detection."
