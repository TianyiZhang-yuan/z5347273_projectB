# AI_NOTES

## 2026-07-30 — Station 1 ETL migration from Part A

Agent action:

- Inspected `fins2026/z5347273_projectA/src/etl.py` for the Part A data-cleaning logic.
- Inspected `fins2026/z5347273_projectB/src/etl.py` and confirmed Station 1 had TODO placeholders for equity and crypto cleaning.
- Migrated the relevant price-data checks into `fins2026/z5347273_projectB/src/etl.py`.
- Did not modify `news_headlines` handling; that remains outside this Station 1 change and should be handled separately in `features.py`.

Logic migrated from Part A:

- `_duplicate_check(df)`: counts duplicated `ticker`-`date` rows and records before/after row counts.
- `_missing_date_audit(df, expected_dates)`: audits per-ticker date coverage against an explicit expected calendar.
- `_outlier_screen(df)`: calculates per-ticker daily `adjClose` returns, flags returns with absolute z-score above 3, and retains flagged rows for review instead of dropping them.
- `_ohlc_consistency_check(df)`: checks impossible OHLCV relationships and invalid price/volume records so extreme-return records can be assessed against internal price consistency.
- `load_clean_equities()`: now loads equity prices, standardises dates, drops duplicated `ticker`-`date` rows, audits missing trading dates, screens extreme returns, prints checks, and returns `(df, checks)`.
- `load_clean_crypto()`: now loads crypto prices, standardises dates, filters to the 2020-01-01 through 2023-12-31 sample, drops duplicated `ticker`-`date` rows, audits the 365-day crypto calendar, screens extreme returns, prints checks, and returns `(df, checks)`.

Migration adjustments:

- Kept Part B's `src.data_access` loader interface; no direct Part A data paths or file names were copied.
- Reused Part B's hosted `equity_prices.parquet` and `crypto_prices.parquet` access through `data_access`.
- Removed Part A-only report/output generation logic, including paths under `results/tables` and `results/figures`.
- Preserved the Part B function names and Station 1 docstring style while replacing TODO placeholders with implemented logic.

Issues noticed and handled:

- Part A contained broader helper/report functions that write exhibits; these were not migrated because Station 1 only needs equity and crypto cleaning.
- The outlier logic intentionally flags and records extreme returns rather than deleting them, because plausible market-event outliers should not be blindly removed.
- No data-loading run was executed during this migration; only code inspection/editing was performed before handoff for review.

## 2026-07-30 — Station 1 news-headline cleaning and trading-day alignment

Agent action:

- Searched `fins2026/z5347273_projectA/src` for news/headline duplicate and trading-day alignment logic.
- Read `z5347273_projectA/src/etl.py` for Part A's `load_clean_news()` implementation.
- Read `z5347273_projectA/src/features.py` for the Part A forward trading-day mapping logic used by `assemble_headline_panel()`.
- Updated `z5347273_projectB/src/etl.py` to add `load_clean_news()` and `_align_to_next_trading_day()`.
- Ran `load_clean_news()` through Part B's `src.data_access` loader.

Logic migrated from Part A:

- News rows are treated as story-level observations, not ticker-date price observations.
- Exact news duplicates are identified by `ticker`, `date`, and `title`; same ticker/date with different titles is preserved.
- UTC-aware news timestamps are preserved in `date_utc_original` before the working `date` column is converted to tz-naive `datetime64[ns]`.
- News published on non-trading days is mapped forward to the next available equity trading day to avoid look-ahead.

New/changed implementation in Part B:

- `_duplicate_check(df, subset=None)` now supports a custom duplicate key. The default remains `["ticker", "date"]`, so equity and crypto calls are unchanged.
- Added `_align_to_next_trading_day(dates, trading_calendar)` to map each headline date to the same trading day if available, otherwise the next equity trading date.
- Added `load_clean_news()`, which returns `(df, checks)` and prints a compact `_print_checks("news", checks)` summary.
- Added `mapped_trading_date`, `mapping_failed`, and `aligned_to_next_trading_day` columns for downstream Station 2/3 use.
- No sentiment scoring, lexicon counting, VADER scoring, or predictive signal construction was added.

Run result summary:

- Raw news rows: 149,683.
- Rows after exact duplicate removal: 146,836.
- Exact duplicate rows removed: 2,847.
- Date dtype before conversion: `datetime64[us, UTC]`.
- Date dtype after conversion: `datetime64[ns]`, tz-naive.
- Date conversion sample:
  - `2020-01-01 00:00:00+00:00` -> `2020-01-01`
  - `2020-01-02 00:00:00+00:00` -> `2020-01-02`
  - `2020-01-02 00:00:00+00:00` -> `2020-01-02`
  - `2020-01-05 00:00:00+00:00` -> `2020-01-05`
  - `2020-01-06 00:00:00+00:00` -> `2020-01-06`
- Sample date shift detected: `False`.
- Equity trading calendar: 1,006 days, from 2020-01-02 to 2023-12-29.
- Headlines aligned to a later trading day rather than same day: 12,551.
- Headlines with no available next trading-day mapping: 6.
- Missing publisher rows: 140,255.
- Missing publisher percentage: 93.70%.
