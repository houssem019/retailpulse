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

## 4. Training/serving skew in price features
- **Problem:** `price_ratio` (price vs the series' mean price) was first computed with a window over the query's date range. Training queries span years and serving queries span 28 days, so the same row got different values.
- **Fix:** compute per-series statistics (`mean_price`, `launch_date`) over the *full* price table in a CTE, independent of the requested range.
  - The test `test_features_identical_between_train_and_serve_queries` now asserts that both query shapes return identical numbers.
- **Lesson:** skew is the #1 silent killer of deployed models. The test makes the guarantee explicit.

## 5. Leakage in a time-series split
- **Risk:** random train/test splits, or features that peek at the horizon, give great offline scores that collapse in production.
- **Fix:**
  - Use a strict time split.
  - Every feature looks ≥28 days back.
  - The test `test_no_leakage_into_horizon` checks that future rows, which have no actuals, still get every feature.
  - Early stopping uses a separate validation window, never the holdout.

## 6. What-if is correlational, not causal
- The model learned how demand moved with price *historically*. Prices weren't randomised, since promotions coincide with events and seasons, so the elasticity is confounded.
- **At scale:** use price experiments or a causal approach, such as double ML with price residualised on confounders. Also constrain the model to be monotonic in price (LightGBM `monotone_constraints`).

## 7. The what-if said "cut the price → sell less" (actually happened)
- **Symptom:** after the first training run, `/what-if` for FOODS_3_090 @ CA_1 predicted the opposite of real price behaviour:
  - a **20% price cut** gave **−53.6%** demand;
  - a **20% price increase** gave **+5.2%**.
- **Root cause: confounding.** In M5, low prices relative to an item's usual price often mean clearance markdowns on items being phased out. Their sales are already falling, so the model learned "low price_ratio → low sales". The forecasts were still accurate (WRMSSE 0.824), because the correlation is genuinely predictive. It just isn't *causal*.
- **Fix:** LightGBM `monotone_constraints = -1` on `sell_price`, `price_ratio` and `price_change_wow`, so predicted demand can never increase with price. The test `test_price_cut_never_lowers_demand` asserts it on every day of the horizon.
- **Lesson:**
  - A model that's good at *forecasting* isn't automatically good at *what-if / decision support*. Those need causal structure.
  - Always sanity-check a model's response to the levers the business will pull, not just its accuracy.
- **Result:** the monotone model (v2) is *also* slightly more accurate (WRMSSE 0.8227 vs 0.8240) and trains 2.5× faster (171 vs 466 boosting rounds), because the constraint acts as regularisation. The champion gate promoted it automatically.
  - After the fix, at CA_1, a −20% price change gives +7% to +13% demand and a +20% change gives −1% to −3%, depending on the item.
  - The asymmetry, where cuts move demand more than increases, matches what retailers typically see.
