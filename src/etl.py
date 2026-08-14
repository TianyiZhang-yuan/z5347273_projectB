"""Station 1 - your ETL: load and clean the data.

Load raw data through src.data_access (see context/DATA_GUIDE.md). Add your own
integrity checks. Do not commit data files.
"""
import pandas as pd

from src import data_access


def _print_checks(name, checks):
    """Print a compact integrity-check summary."""
    print(f"{name} checks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")


def _duplicate_check(df, subset=None):
    """Summarize duplicated rows for a uniqueness key."""
    if subset is None:
        subset = ["ticker", "date"]
    n_duplicates = int(df.duplicated(subset=subset).sum())
    return {
        "n_duplicates": n_duplicates,
        "duplicate_subset": list(subset),
        "rows_before": len(df),
        "rows_after": len(df) - n_duplicates,
    }


def _missing_date_audit(df, expected_dates):
    """Summarize missing dates against one explicit expected calendar."""
    expected_dates = pd.DatetimeIndex(expected_dates).dropna().drop_duplicates().sort_values()
    missing = {}
    for ticker, group in df.groupby("ticker"):
        actual_dates = pd.DatetimeIndex(group["date"].dropna().unique())
        missing_dates = expected_dates.difference(actual_dates)
        missing[ticker] = len(missing_dates)
    worst = max(missing, key=missing.get) if missing else None
    return {
        "total_missing_days": int(sum(missing.values())),
        "worst_ticker": worst,
        "worst_ticker_missing_days": int(missing.get(worst, 0)),
        "n_tickers_with_missing_dates": int(sum(count > 0 for count in missing.values())),
    }


def _outlier_screen(df):
    """Screen extreme daily adjusted-close returns without dropping rows."""
    screen = df.sort_values(["ticker", "date"]).copy()
    screen["return"] = screen.groupby("ticker")["adjClose"].pct_change()
    mean = screen.groupby("ticker")["return"].transform("mean")
    std = screen.groupby("ticker")["return"].transform("std")
    safe_std = std.mask(std.eq(0))
    z_score = (screen["return"] - mean) / safe_std
    is_extreme = z_score.abs().gt(3).fillna(False)

    required_ohlcv_cols = ["open", "high", "low", "close", "adjClose", "volume"]
    missing_ohlcv = screen[required_ohlcv_cols].isna().any(axis=1)
    high_violation = (
        (screen["high"] < screen["open"])
        | (screen["high"] < screen["close"])
        | (screen["high"] < screen["low"])
    )
    low_violation = (
        (screen["low"] > screen["open"])
        | (screen["low"] > screen["close"])
        | (screen["low"] > screen["high"])
    )
    volume_violation = screen["volume"] < 0
    price_nonpositive = (
        screen[["open", "high", "low", "close", "adjClose"]] <= 0
    ).any(axis=1)
    adjclose_invalid = screen["adjClose"].isna() | (screen["adjClose"] <= 0)
    internally_inconsistent = (
        high_violation
        | low_violation
        | volume_violation
        | price_nonpositive
        | missing_ohlcv
        | adjclose_invalid
    )

    outliers = screen.loc[is_extreme].copy()
    outliers["internally_consistent"] = ~internally_inconsistent.loc[outliers.index]
    n_extreme_inconsistent = int((~outliers["internally_consistent"]).sum())
    n_outliers = len(outliers)
    examples = outliers[
        ["ticker", "date", "return", "adjClose", "internally_consistent"]
    ].head(5).copy()
    examples["date"] = examples["date"].dt.strftime("%Y-%m-%d")
    return {
        "n_outliers": n_outliers,
        "n_extreme_rows_with_internal_inconsistency": n_extreme_inconsistent,
        "n_extreme_rows_internally_consistent": n_outliers - n_extreme_inconsistent,
        "extreme_action": (
            "Flagged and retained because the underlying OHLCV records were "
            "internally consistent."
            if n_extreme_inconsistent == 0
            else "Flagged for review; internally inconsistent rows require "
            "investigation before retention."
        ),
        "outlier_examples": examples.to_dict("records"),
    }


def _ohlc_consistency_check(df):
    """Check OHLCV field logic without dropping or modifying rows."""
    required_ohlcv_cols = ["open", "high", "low", "close", "adjClose", "volume"]
    missing_ohlcv = df[required_ohlcv_cols].isna().any(axis=1)
    high_violation = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
    )
    low_violation = (
        (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["low"] > df["high"])
    )
    volume_violation = df["volume"] < 0
    price_nonpositive = (
        df[["open", "high", "low", "close", "adjClose"]] <= 0
    ).any(axis=1)
    adjclose_invalid = df["adjClose"].isna() | (df["adjClose"] <= 0)
    inconsistent = (
        high_violation
        | low_violation
        | volume_violation
        | price_nonpositive
        | missing_ohlcv
        | adjclose_invalid
    )

    examples = []
    for idx, row in df.loc[inconsistent].head(5).iterrows():
        issues = []
        if high_violation.loc[idx]:
            issues.append("high_below_open_close_or_low")
        if low_violation.loc[idx]:
            issues.append("low_above_open_close_or_high")
        if volume_violation.loc[idx]:
            issues.append("negative_volume")
        if price_nonpositive.loc[idx]:
            issues.append("nonpositive_price")
        if missing_ohlcv.loc[idx]:
            issues.append("missing_ohlcv")
        if adjclose_invalid.loc[idx]:
            issues.append("invalid_adjclose")
        examples.append(
            {
                "ticker": row["ticker"],
                "date": row["date"].strftime("%Y-%m-%d"),
                "issues": issues,
            }
        )

    return {
        "n_high_violations": int(high_violation.sum()),
        "n_low_violations": int(low_violation.sum()),
        "n_volume_violations": int(volume_violation.sum()),
        "n_price_nonpositive": int(price_nonpositive.sum()),
        "n_missing_ohlcv_rows": int(missing_ohlcv.sum()),
        "n_invalid_adjclose": int(adjclose_invalid.sum()),
        "n_total_inconsistent_rows": int(inconsistent.sum()),
        "ohlc_inconsistency_examples": examples,
    }


def _align_to_next_trading_day(dates, trading_calendar):
    """Map dates to the same day if possible, otherwise the next trading day."""
    calendar = pd.DatetimeIndex(
        pd.to_datetime(pd.Series(trading_calendar).dropna().unique())
    ).sort_values().astype("datetime64[ns]")
    assert calendar.tz is None
    assert str(calendar.dtype) == "datetime64[ns]"

    date_series = pd.Series(pd.to_datetime(dates), index=getattr(dates, "index", None))
    if date_series.dt.tz is not None:
        date_series = date_series.dt.tz_convert(None)
    date_series = date_series.dt.normalize().astype("datetime64[ns]")

    positions = calendar.searchsorted(date_series, side="left")
    mapped = pd.Series(pd.NaT, index=date_series.index, dtype="datetime64[ns]")
    valid = positions < len(calendar)
    mapped.loc[valid] = calendar.take(positions[valid]).to_numpy()
    return mapped


def load_clean_equities():
    """Load equity prices and run your Station 1 integrity checks.

    Run a missing-date audit, duplicate ticker-date check, and
    outlier/extreme-value screen on returns. Return the clean frame and record
    what you found.
    """
    df = data_access.load_equity_prices().copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    assert str(df["date"].dtype) == "datetime64[ns]"
    assert df["date"].dt.tz is None

    duplicate_checks = _duplicate_check(df)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="first").copy()
    equity_expected_dates = pd.DatetimeIndex(
        df["date"].dropna().drop_duplicates().sort_values()
    )
    checks = {
        **duplicate_checks,
        **_missing_date_audit(df, equity_expected_dates),
        **_outlier_screen(df),
        **_ohlc_consistency_check(df),
        "date_dtype": str(df["date"].dtype),
    }
    _print_checks("equity", checks)
    return df, checks


def load_clean_crypto():
    """Load crypto prices (365-day calendar). Run Station 1 integrity checks."""
    df = data_access.load_crypto_prices().copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    assert str(df["date"].dtype) == "datetime64[ns]"
    assert df["date"].dt.tz is None

    raw_rows = len(df)
    in_sample = df["date"].between(pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31"))
    rows_outside_sample = int((~in_sample).sum())
    df = df[in_sample].copy()
    rows_after_date_filter = len(df)

    duplicate_checks = _duplicate_check(df)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="first").copy()
    crypto_expected_dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    checks = {
        **duplicate_checks,
        "raw_rows": raw_rows,
        "rows_outside_sample": rows_outside_sample,
        "rows_after_date_filter": rows_after_date_filter,
        **_missing_date_audit(df, crypto_expected_dates),
        **_outlier_screen(df),
        **_ohlc_consistency_check(df),
        "date_dtype": str(df["date"].dtype),
    }
    _print_checks("crypto", checks)
    print("crypto outlier method: per-ticker daily adjClose return z-score")
    print("crypto outlier threshold: abs((return - ticker_mean) / ticker_std) > 3")
    print(
        "crypto final row count check: "
        f"crypto_clean.shape[0]={df.shape[0]}, checks['rows_after']={checks['rows_after']}, "
        f"match={df.shape[0] == checks['rows_after']}"
    )
    return df, checks


def load_clean_news():
    """Load news headlines, align dates to trading days, and report checks."""
    df = data_access.load_news_headlines().copy()
    rows_before = len(df)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])

    date_timezone_before = df["date"].dt.tz
    date_dtype_before = str(df["date"].dtype)
    sample_before = df["date"].dropna().head(5).astype(str).tolist()

    df["date_utc_original"] = df["date"]
    if date_timezone_before is not None:
        converted_dates = df["date"].dt.tz_convert(None)
    else:
        converted_dates = pd.to_datetime(df["date"])
    df["date"] = converted_dates.dt.normalize().astype("datetime64[ns]")

    assert str(df["date"].dtype) == "datetime64[ns]"
    assert df["date"].dt.tz is None
    if date_timezone_before is not None:
        assert df["date_utc_original"].dt.tz is not None

    sample_after = df["date"].dropna().head(5).dt.strftime("%Y-%m-%d").tolist()
    original_utc_dates = (
        df["date_utc_original"]
        .dropna()
        .head(5)
        .dt.tz_convert("UTC")
        .dt.strftime("%Y-%m-%d")
        .tolist()
        if date_timezone_before is not None
        else sample_after
    )
    sample_date_shift_detected = sample_after != original_utc_dates

    n_missing_ticker = int(df["ticker"].isna().sum())
    n_missing_date = int(df["date"].isna().sum())
    n_missing_title = int(df["title"].isna().sum())
    n_missing_sector = int(df["sector"].isna().sum()) if "sector" in df.columns else 0
    n_missing_publisher = int(df["publisher"].isna().sum()) if "publisher" in df.columns else 0

    pct_missing_ticker = n_missing_ticker / rows_before * 100
    pct_missing_date = n_missing_date / rows_before * 100
    pct_missing_title = n_missing_title / rows_before * 100
    pct_missing_sector = n_missing_sector / rows_before * 100
    pct_missing_publisher = n_missing_publisher / rows_before * 100

    duplicate_key = ["ticker", "date", "title"]
    duplicate_checks = _duplicate_check(df, subset=duplicate_key)
    df = df.drop_duplicates(subset=duplicate_key, keep="first").copy()

    equity_clean, _equity_checks = load_clean_equities()
    trading_calendar = pd.DatetimeIndex(
        equity_clean["date"].dropna().drop_duplicates().sort_values()
    )
    df["mapped_trading_date"] = _align_to_next_trading_day(
        df["date"],
        trading_calendar,
    )
    df["mapping_failed"] = df["mapped_trading_date"].isna()
    df["aligned_to_next_trading_day"] = (
        df["mapped_trading_date"].notna()
        & df["date"].notna()
        & df["mapped_trading_date"].ne(df["date"])
    )

    rows_after_dedup = len(df)
    checks = {
        **duplicate_checks,
        "rows_after_dedup": rows_after_dedup,
        "date_dtype_before": date_dtype_before,
        "date_timezone_before": (
            str(date_timezone_before) if date_timezone_before is not None else None
        ),
        "date_dtype_after": str(df["date"].dtype),
        "date_timezone_after": df["date"].dt.tz,
        "date_sample_before": sample_before,
        "date_sample_after": sample_after,
        "original_utc_date_sample": original_utc_dates,
        "sample_date_shift_detected": sample_date_shift_detected,
        "trading_calendar_days": len(trading_calendar),
        "trading_calendar_start": trading_calendar.min().strftime("%Y-%m-%d"),
        "trading_calendar_end": trading_calendar.max().strftime("%Y-%m-%d"),
        "n_aligned_to_next_trading_day": int(df["aligned_to_next_trading_day"].sum()),
        "n_mapping_failed": int(df["mapping_failed"].sum()),
        "n_missing_ticker": n_missing_ticker,
        "n_missing_date": n_missing_date,
        "n_missing_title": n_missing_title,
        "n_missing_sector": n_missing_sector,
        "n_missing_publisher": n_missing_publisher,
        "pct_missing_ticker": pct_missing_ticker,
        "pct_missing_date": pct_missing_date,
        "pct_missing_title": pct_missing_title,
        "pct_missing_sector": pct_missing_sector,
        "pct_missing_publisher": pct_missing_publisher,
        "sentiment_scoring_performed": False,
    }

    print(f"news rows before/after duplicate removal: {rows_before} -> {rows_after_dedup}")
    print("news date timezone conversion sample:")
    for before, after in zip(sample_before, sample_after, strict=False):
        print(f"  {before} -> {after}")
    print(
        "news aligned to next trading day: "
        f"{checks['n_aligned_to_next_trading_day']}"
    )
    print(f"news publisher missing pct: {pct_missing_publisher:.2f}%")
    _print_checks("news", checks)
    return df, checks
