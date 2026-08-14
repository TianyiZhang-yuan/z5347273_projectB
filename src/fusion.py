"""Station 3 (extension) - fuse sentiment into the funds.

Tilt or factor: combine your sentiment signal with the portfolio weights,
look-ahead safe, then test whether it adds value. An honest negative result,
explained, is good work.
"""
from pathlib import Path
import math

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUND_WEIGHTS_PATH = PROJECT_ROOT / "results" / "data" / "fund_weights.csv"
BASELINE_SENTIMENT_LOOKBACK_DAYS = 60
TAIL_RISK_CVAR_LOOKBACK_DAYS = 252
TAIL_RISK_CVAR_CONFIDENCE_LEVEL = 0.95
FUSION_STRATEGY_LAMBDAS = {
    "Base Min-Variance": 0.0,
    "Momentum Tilt": 1.0,
    "Contrarian Tilt": -1.0,
}


def equity_min_variance_oos_weights(
    weights_path: str | Path = FUND_WEIGHTS_PATH,
) -> pd.DataFrame:
    """Return existing equity minimum-variance OOS weights with merge-ready keys."""
    weights = pd.read_csv(weights_path, parse_dates=["date"])
    required = {"fund_name", "asset_universe", "method", "date", "asset", "weight"}
    missing = required.difference(weights.columns)
    if missing:
        raise ValueError(f"weights file is missing columns: {sorted(missing)}")

    base = weights.loc[
        (weights["asset_universe"] == "equity")
        & (weights["method"] == "min_variance")
        & (weights["fund_name"] == "equity_min_variance"),
        ["fund_name", "asset_universe", "method", "date", "asset", "weight"],
    ].copy()
    if base.empty:
        raise ValueError("equity_min_variance weights were not found")
    return base.rename(columns={"asset": "ticker"}).sort_values(["date", "ticker"])


def stock_ticker_day_sentiment(scores: pd.DataFrame) -> pd.DataFrame:
    """Return ticker-day sentiment with date and ticker keys for fusion."""
    from src.sentiment import ticker_day_sentiment

    sentiment = ticker_day_sentiment(scores).rename(
        columns={"mapped_trading_date": "date"}
    )
    return sentiment[["date", "ticker", "sector", "ticker_day_sentiment"]].sort_values(
        ["date", "ticker"]
    )


def _add_lagged_rolling_z(
    data: pd.DataFrame,
    value_column: str,
    lag_column: str,
    z_column: str,
    lookback_days: int,
) -> pd.DataFrame:
    if lookback_days < 2:
        raise ValueError("lookback_days must be at least 2")

    out = data.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out[lag_column] = out.groupby("ticker", sort=False)[value_column].shift(1)
    rolling = out.groupby("ticker", sort=False)[lag_column].rolling(
        lookback_days,
        min_periods=lookback_days,
    )
    rolling_mean = rolling.mean().reset_index(level=0, drop=True)
    rolling_std = rolling.std().reset_index(level=0, drop=True)
    out[z_column] = (out[lag_column] - rolling_mean) / rolling_std.mask(
        rolling_std.eq(0)
    )
    zero_variance = rolling_std.eq(0) & rolling_mean.notna()
    out.loc[zero_variance, z_column] = 0.0
    return out


def stock_sentiment_signal(
    stock_sentiment: pd.DataFrame,
    lookback_days: int,
) -> pd.DataFrame:
    """Create one-trading-day-lagged rolling ticker sentiment z-scores."""
    required = {"date", "ticker", "sector", "ticker_day_sentiment"}
    missing = required.difference(stock_sentiment.columns)
    if missing:
        raise ValueError(f"stock_sentiment is missing columns: {sorted(missing)}")

    return _add_lagged_rolling_z(
        stock_sentiment,
        value_column="ticker_day_sentiment",
        lag_column="ticker_day_sentiment_lag1",
        z_column="stock_sentiment_z",
        lookback_days=lookback_days,
    )


def prepare_fusion_inputs(
    scores: pd.DataFrame,
    weights_path: str | Path = FUND_WEIGHTS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare existing base weights and stock-level sentiment for later fusion."""
    return equity_min_variance_oos_weights(weights_path), stock_ticker_day_sentiment(scores)


def apply_sentiment(
    weights: pd.DataFrame,
    sentiment: pd.DataFrame,
    sentiment_lambda: float,
    signal_column: str = "stock_sentiment_z",
) -> pd.DataFrame:
    """Apply a naive long-only sentiment tilt to existing base weights."""
    weight_column = "base_weight" if "base_weight" in weights.columns else "weight"
    weight_required = {"date", "ticker", weight_column}
    sentiment_required = {"date", "ticker", signal_column}
    missing_weights = weight_required.difference(weights.columns)
    missing_sentiment = sentiment_required.difference(sentiment.columns)
    if missing_weights:
        raise ValueError(f"weights is missing columns: {sorted(missing_weights)}")
    if missing_sentiment:
        raise ValueError(f"sentiment is missing columns: {sorted(missing_sentiment)}")

    base = weights.copy()
    signal = sentiment.copy()
    base["date"] = pd.to_datetime(base["date"])
    signal["date"] = pd.to_datetime(signal["date"])
    base = base.rename(columns={weight_column: "base_weight"})
    keep = [column for column in ["date", "ticker", "sector", signal_column] if column in signal]
    out = base.merge(signal[keep], on=["date", "ticker"], how="left")
    if out[signal_column].isna().any():
        missing = int(out[signal_column].isna().sum())
        raise ValueError(f"aligned sentiment z-scores are missing for {missing} rows")

    out["tilted_raw"] = out["base_weight"] * (
        1.0 + sentiment_lambda * out[signal_column]
    )
    out["tilted_nonnegative"] = out["tilted_raw"].clip(lower=0.0)
    totals = out.groupby("date")["tilted_nonnegative"].transform("sum")
    if totals.le(0.0).any():
        bad_dates = int(out.loc[totals.le(0.0), "date"].nunique())
        raise ValueError(f"tilted weights sum to zero for {bad_dates} dates")

    out["tilted_weight"] = out["tilted_nonnegative"] / totals
    columns = [
        column
        for column in [
            "fund_name",
            "asset_universe",
            "method",
            "date",
            "ticker",
            "sector",
            "base_weight",
            signal_column,
            "tilted_weight",
        ]
        if column in out.columns
    ]
    return out[columns].sort_values(["date", "ticker"]).reset_index(drop=True)


def apply_sector_neutral_sentiment(
    weights: pd.DataFrame,
    sentiment: pd.DataFrame,
    sentiment_lambda: float = -1.0,
    signal_column: str = "stock_sentiment_z",
) -> pd.DataFrame:
    """Apply a long-only sentiment tilt while preserving base sector weights."""
    tilted = apply_sentiment(
        weights,
        sentiment,
        sentiment_lambda=sentiment_lambda,
        signal_column=signal_column,
    ).rename(columns={"tilted_weight": "naive_tilted_weight"})
    if "sector" not in tilted.columns:
        raise ValueError("sector is required for sector-neutral sentiment fusion")

    tilted["base_sector_weight"] = tilted.groupby(["date", "sector"])[
        "base_weight"
    ].transform("sum")
    raw_sector_weight = tilted.groupby(["date", "sector"])[
        "naive_tilted_weight"
    ].transform("sum")
    scaled = tilted["naive_tilted_weight"] * tilted["base_sector_weight"] / raw_sector_weight
    tilted["sector_neutral_weight"] = scaled.where(
        raw_sector_weight.gt(0.0),
        tilted["base_weight"],
    )
    tilted["tilted_weight"] = tilted["sector_neutral_weight"]
    columns = [
        column
        for column in [
            "fund_name",
            "asset_universe",
            "method",
            "date",
            "ticker",
            "sector",
            "base_weight",
            signal_column,
            "sector_neutral_weight",
            "tilted_weight",
        ]
        if column in tilted.columns
    ]
    return tilted[columns].sort_values(["date", "ticker"]).reset_index(drop=True)


def _historical_cvar_loss(window: pd.Series, confidence_level: float) -> float:
    returns = window.dropna()
    if returns.empty:
        return float("nan")
    tail_count = max(1, math.ceil(len(returns) * (1.0 - confidence_level)))
    return max(float(-returns.nsmallest(tail_count).mean()), 0.0)


def tail_risk_adjusted_sentiment_signal(
    equity_returns: pd.DataFrame,
    sentiment_signal: pd.DataFrame,
    target_dates: pd.Series | list | None = None,
    lookback_days: int = TAIL_RISK_CVAR_LOOKBACK_DAYS,
    confidence_level: float = TAIL_RISK_CVAR_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Scale stock sentiment by each stock's historical CVaR relative to sector peers."""
    returns = (
        _return_matrix(equity_returns)
        .stack()
        .rename("return")
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "ticker"})
        .sort_values(["ticker", "date"])
    )
    returns["prior_return"] = returns.groupby("ticker", sort=False)["return"].shift(1)
    rolling = returns.groupby("ticker", sort=False)["prior_return"].rolling(
        lookback_days,
        min_periods=1,
    )
    returns["stock_cvar"] = rolling.apply(
        lambda window: _historical_cvar_loss(window, confidence_level),
        raw=False,
    ).reset_index(level=0, drop=True)
    tail_risk = returns[["date", "ticker", "stock_cvar"]]
    if target_dates is not None:
        dates = pd.to_datetime(pd.Series(target_dates).drop_duplicates())
        tail_risk = tail_risk.loc[tail_risk["date"].isin(dates)]

    signal = sentiment_signal.copy()
    signal["date"] = pd.to_datetime(signal["date"])
    out = signal.merge(tail_risk, on=["date", "ticker"], how="left")
    usable_cvar = out["stock_cvar"].where(out["stock_cvar"].gt(0.0))
    out["sector_median_cvar"] = usable_cvar.groupby(
        [out["date"], out["sector"]]
    ).transform("median")
    valid = out["stock_cvar"].gt(0.0) & out["sector_median_cvar"].gt(0.0)
    out["risk_scaler"] = 1.0
    out.loc[valid, "risk_scaler"] = (
        out.loc[valid, "sector_median_cvar"] / out.loc[valid, "stock_cvar"]
    )
    out["sector_median_cvar"] = out["sector_median_cvar"].fillna(out["stock_cvar"])
    out["tail_risk_adjusted_sentiment_z"] = (
        out["stock_sentiment_z"] * out["risk_scaler"]
    )
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def _return_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    if {"date", "ticker", "return"}.issubset(returns.columns):
        matrix = returns.pivot(index="date", columns="ticker", values="return")
    else:
        matrix = returns.copy()
    matrix.index = pd.to_datetime(matrix.index)
    return matrix.sort_index()


def target_weight_returns_and_turnover(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    start_date: str | pd.Timestamp = "2021-01-04",
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply target weights and return daily returns plus drift-adjusted turnover."""
    matrix = _return_matrix(returns)
    targets = target_weights.copy()
    targets["date"] = pd.to_datetime(targets["date"])
    targets = targets.sort_values(["date", "ticker"])
    target_dates = set(targets["date"])
    target_by_date = {
        date: group.set_index("ticker")["tilted_weight"].reindex(matrix.columns).fillna(0.0)
        for date, group in targets.groupby("date", sort=True)
    }

    current_weights = None
    daily = {}
    turnover_rows = []
    live_dates = matrix.index[matrix.index >= max(pd.Timestamp(start_date), min(target_dates))]
    for date in live_dates:
        if date in target_dates:
            target = target_by_date[date]
            turnover = (
                float("nan")
                if current_weights is None
                else 0.5 * float((target - current_weights).abs().sum())
            )
            turnover_rows.append({"date": date, "turnover": turnover})
            current_weights = target
        if current_weights is None:
            continue

        held = current_weights[current_weights.gt(1e-12)].index
        day_returns = matrix.loc[date, held]
        if day_returns.isna().any():
            missing = day_returns.index[day_returns.isna()].tolist()
            raise ValueError(f"missing live returns on {date.date()} for {missing}")

        portfolio_return = float((day_returns * current_weights.loc[held]).sum())
        daily[date] = portfolio_return
        gross_return = 1.0 + portfolio_return
        if gross_return <= 0:
            raise ValueError(f"portfolio gross return is non-positive on {date.date()}")
        drifted = current_weights.copy()
        drifted.loc[held] = current_weights.loc[held] * (1.0 + day_returns) / gross_return
        current_weights = drifted / drifted.sum()

    turnover = pd.DataFrame(turnover_rows).sort_values("date").reset_index(drop=True)
    return pd.Series(daily, name="return").sort_index(), turnover


def _target_weight_returns(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    start_date: str | pd.Timestamp = "2021-01-04",
) -> pd.Series:
    return target_weight_returns_and_turnover(returns, target_weights, start_date)[0]


def sentiment_fusion_oos_returns(
    equity_returns: pd.DataFrame,
    base_weights: pd.DataFrame,
    sentiment_signal: pd.DataFrame,
    start_date: str | pd.Timestamp = "2021-01-04",
) -> pd.DataFrame:
    """Build daily OOS returns for the baseline sentiment-fusion variants."""
    frames = []
    for strategy, sentiment_lambda in FUSION_STRATEGY_LAMBDAS.items():
        tilted = apply_sentiment(base_weights, sentiment_signal, sentiment_lambda)
        returns = _target_weight_returns(equity_returns, tilted, start_date)
        frame = returns.rename("return").reset_index()
        frame.columns = ["date", "return"]
        frame.insert(1, "strategy", strategy)
        frame.insert(2, "sentiment_lambda", sentiment_lambda)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["strategy", "date"])


def sector_neutral_fusion_oos_returns(
    equity_returns: pd.DataFrame,
    base_weights: pd.DataFrame,
    sentiment_signal: pd.DataFrame,
    start_date: str | pd.Timestamp = "2021-01-04",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build OOS returns for base, naive contrarian, and sector-neutral contrarian."""
    strategies = [
        ("Base Min-Variance", apply_sentiment(base_weights, sentiment_signal, 0.0)),
        ("Naive Contrarian", apply_sentiment(base_weights, sentiment_signal, -1.0)),
        (
            "Sector-Neutral Contrarian",
            apply_sector_neutral_sentiment(base_weights, sentiment_signal, -1.0),
        ),
    ]
    frames = []
    sector_neutral_weights = strategies[-1][1]
    for strategy, weights in strategies:
        returns = _target_weight_returns(equity_returns, weights, start_date)
        frame = returns.rename("return").reset_index()
        frame.columns = ["date", "return"]
        frame.insert(1, "strategy", strategy)
        frames.append(frame)
    returns = pd.concat(frames, ignore_index=True).sort_values(["strategy", "date"])
    return returns, sector_neutral_weights


def tail_risk_aware_fusion_oos_returns(
    equity_returns: pd.DataFrame,
    base_weights: pd.DataFrame,
    sentiment_signal: pd.DataFrame,
    start_date: str | pd.Timestamp = "2021-01-04",
    sentiment_lambda: float = -1.0,
    strategy_name: str = "Tail-Risk-Aware Sector-Neutral Contrarian",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build OOS returns for the tail-risk-aware sector-neutral innovation."""
    base_returns, _sector_weights = sector_neutral_fusion_oos_returns(
        equity_returns,
        base_weights,
        sentiment_signal,
        start_date=start_date,
    )
    target_dates = base_weights["date"].drop_duplicates()
    risk_signal = tail_risk_adjusted_sentiment_signal(
        equity_returns,
        sentiment_signal,
        target_dates=target_dates,
    )
    risk_weights = apply_sector_neutral_sentiment(
        base_weights,
        risk_signal,
        sentiment_lambda=sentiment_lambda,
        signal_column="tail_risk_adjusted_sentiment_z",
    )
    risk_returns = _target_weight_returns(equity_returns, risk_weights, start_date)
    risk_frame = risk_returns.rename("return").reset_index()
    risk_frame.columns = ["date", "return"]
    risk_frame.insert(1, "strategy", strategy_name)
    returns = pd.concat([base_returns, risk_frame], ignore_index=True).sort_values(
        ["strategy", "date"]
    )

    risk_context = risk_signal[
        [
            "date",
            "ticker",
            "sector",
            "stock_sentiment_z",
            "stock_cvar",
            "sector_median_cvar",
            "risk_scaler",
        ]
    ]
    risk_weights = risk_weights.merge(
        risk_context,
        on=["date", "ticker", "sector"],
        how="left",
    )
    risk_weights["final_weight"] = risk_weights["sector_neutral_weight"]
    return returns, risk_weights
