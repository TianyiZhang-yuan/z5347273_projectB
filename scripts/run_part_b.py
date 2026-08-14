"""Reproduce your Part B results. Run from the project root:

    python scripts/run_part_b.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter, PercentFormatter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.features import daily_returns

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "data"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

STAGE7_TAIL_RISK_DATA_PATH = DATA_DIR / "stage7.5_crypto_tail_risk.csv"
STAGE7_TAIL_RISK_SUMMARY_PATH = TABLES_DIR / "stage7_crypto_tail_risk_summary.csv"
STAGE7_TAIL_RISK_FIGURE_PATH = FIGURES_DIR / "stage7.2 _crypto_rolling_tail_risk.png"
STAGE7_RISK_SCORE_DATA_PATH = DATA_DIR / "7.3 crypto_risk_scores.csv"
STAGE7_RISK_SCORE_SUMMARY_PATH = TABLES_DIR / "crypto_risk_score_summary.csv"
STAGE7_PERSONALISED_BUDGET_DATA_PATH = DATA_DIR / "personalised_crypto_budgets.csv"
STAGE7_PERSONALISED_BUDGET_SUMMARY_PATH = (
    TABLES_DIR / "personalised_crypto_budget_summary.csv"
)
STAGE7_PERSONALISED_BUDGET_FIGURE_PATH = (
    FIGURES_DIR / "risk_score_vs_personalised_crypto_budget.png"
)
STAGE7_ADAPTIVE_BUDGET_HISTORY_PATH = DATA_DIR / "adaptive_budget_history.csv"
STAGE7_FUND_RETURNS_PATH = DATA_DIR / "stage7_fund_returns.csv"
STAGE7_FUND_WEIGHTS_PATH = DATA_DIR / "stage7_fund_weights.csv"
STAGE7_FUND_TURNOVER_PATH = DATA_DIR / "stage7_fund_turnover.csv"
STAGE7_REBALANCE_LOG_PATH = DATA_DIR / "stage7_rebalance_log.csv"
STAGE7_ALLOCATION_HISTORY_PATH = DATA_DIR / "stage7_allocation_history.csv"
STAGE7_ALLOCATION_HISTORY_LEGACY_PATH = DATA_DIR / "stage7.7_allocation_history.csv"
STAGE7_REBALANCE_EXPLANATIONS_PATH = DATA_DIR / "stage7_rebalance_explanations.csv"
STAGE7_PERFORMANCE_METRICS_PATH = TABLES_DIR / "stage7_performance_metrics.csv"
STAGE7_FIXED_VS_ADAPTIVE_PATH = TABLES_DIR / "stage7_fixed_vs_adaptive_comparison.csv"
STAGE7_PERSONALISATION_COMPARISON_PATH = (
    TABLES_DIR / "stage7_personalisation_comparison.csv"
)
STAGE7_FUND_RETURNS_NET_PATH = DATA_DIR / "stage7_fund_returns_net.csv"
STAGE7_GROSS_VS_NET_METRICS_PATH = TABLES_DIR / "stage7_gross_vs_net_metrics.csv"
STAGE7_TRANSACTION_COST_RATE = 0.001
STAGE7_RAW_ADAPTIVE_RESULTS_PATH = DATA_DIR / "stage7_raw_adaptive_results.csv"
STAGE7_SMOOTHING_COMPARISON_PATH = TABLES_DIR / "stage7_smoothing_comparison.csv"
STAGE7_NET_GROWTH_FIXED_VS_ADAPTIVE_FIGURE_PATH = (
    FIGURES_DIR / "stage7_net_growth_of_1_fixed_vs_adaptive.png"
)
STAGE7_NET_DRAWDOWN_FIXED_VS_ADAPTIVE_FIGURE_PATH = (
    FIGURES_DIR / "stage7_net_drawdown_fixed_vs_adaptive.png"
)
SECTOR_SENTIMENT_INDEX_PATH = DATA_DIR / "sector_sentiment_index.csv"
SECTOR_SENTIMENT_SUMMARY_PATH = TABLES_DIR / "sector_sentiment_summary.csv"
SECTOR_SENTIMENT_FIGURE_PATH = FIGURES_DIR / "sector_sentiment_index_timeseries.png"
SENTIMENT_FUSION_RETURNS_PATH = DATA_DIR / "sentiment_fusion_returns.csv"
SENTIMENT_FUSION_METRICS_PATH = TABLES_DIR / "sentiment_fusion_baseline_metrics.csv"
SENTIMENT_FUSION_GROWTH_FIGURE_PATH = (
    FIGURES_DIR / "sentiment_fusion_growth_of_1.png"
)
SENTIMENT_FUSION_DRAWDOWN_FIGURE_PATH = (
    FIGURES_DIR / "sentiment_fusion_drawdown.png"
)
SENTIMENT_FUSION_WEIGHTS_FIGURE_PATH = FIGURES_DIR / "sentiment_fusion_weights.png"
SECTOR_NEUTRAL_FUSION_WEIGHTS_PATH = DATA_DIR / "sector_neutral_fusion_weights.csv"
SECTOR_NEUTRAL_FUSION_RETURNS_PATH = DATA_DIR / "sector_neutral_fusion_returns.csv"
SECTOR_NEUTRAL_FUSION_METRICS_PATH = TABLES_DIR / "sector_neutral_fusion_metrics.csv"
SECTOR_NEUTRAL_FUSION_GROWTH_FIGURE_PATH = (
    FIGURES_DIR / "sector_neutral_fusion_growth_of_1.png"
)
TAIL_RISK_SENTIMENT_WEIGHTS_PATH = DATA_DIR / "tail_risk_sentiment_weights.csv"
TAIL_RISK_SENTIMENT_RETURNS_PATH = DATA_DIR / "tail_risk_sentiment_returns.csv"
TAIL_RISK_SENTIMENT_METRICS_PATH = TABLES_DIR / "tail_risk_sentiment_metrics.csv"
TAIL_RISK_SENTIMENT_GROWTH_FIGURE_PATH = (
    FIGURES_DIR / "tail_risk_sentiment_growth_of_1.png"
)
FINAL_SENTIMENT_LAMBDA_TUNING_PATH = (
    TABLES_DIR / "final_sentiment_lambda_tuning.csv"
)
FINAL_SENTIMENT_2023_RETURNS_PATH = DATA_DIR / "final_sentiment_2023_returns.csv"
FINAL_SENTIMENT_2023_METRICS_PATH = TABLES_DIR / "final_sentiment_2023_metrics.csv"
FINAL_SENTIMENT_2023_GROWTH_FIGURE_PATH = (
    FIGURES_DIR / "final_sentiment_2023_growth_of_1.png"
)
FINAL_SENTIMENT_INNOVATION_RETURNS_PATH = (
    DATA_DIR / "final_sentiment_innovation_returns.csv"
)
FINAL_SENTIMENT_INNOVATION_METRICS_PATH = (
    TABLES_DIR / "final_sentiment_innovation_metrics.csv"
)
FINAL_SENTIMENT_INNOVATION_GROWTH_FIGURE_PATH = (
    FIGURES_DIR / "final_sentiment_baseline_compare_innovation_growth_of_1.png"
)
FINAL_SENTIMENT_TURNOVER_PATH = DATA_DIR / "final_sentiment_turnover.csv"
FINAL_SENTIMENT_NET_RETURNS_PATH = DATA_DIR / "final_sentiment_net_returns.csv"
FINAL_SENTIMENT_COST_SENSITIVITY_PATH = (
    TABLES_DIR / "final_sentiment_cost_sensitivity.csv"
)
FINAL_SENTIMENT_COST_SENSITIVITY_FIGURE_PATH = (
    FIGURES_DIR / "final_sentiment_cost_sensitivity.png"
)
FINAL_SENTIMENT_NET_GROWTH_25BPS_FIGURE_PATH = (
    FIGURES_DIR / "final_sentiment_net_growth_of_1_25bps.png"
)

FINAL_SENTIMENT_FIGURE_ORDER = [
    "Base Min-Variance",
    "Naive Contrarian",
    "Final Tail-Risk-Aware Sector-Neutral Tuned",
]
FINAL_SENTIMENT_FIGURE_LABELS = {
    "Base Min-Variance": "Base Min-Variance",
    "Naive Contrarian": "Naive Contrarian",
    "Final Tail-Risk-Aware Sector-Neutral Tuned": "Final Sentiment Strategy",
}
FINAL_SENTIMENT_FIGURE_COLORS = {
    "Base Min-Variance": "#64748B",
    "Naive Contrarian": "#2563EB",
    "Final Tail-Risk-Aware Sector-Neutral Tuned": "#14866D",
}

WEIGHT_TOLERANCE = 1e-6
BOUND_TOLERANCE = 1e-8

UNIVERSE_ORDER = {"equity": 0, "crypto": 1, "combined": 2}
METHOD_ORDER = {
    "equal_weight": 0,
    "min_variance": 1,
    "max_sharpe": 2,
    "min_cvar": 3,
}
METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "min_variance": "Minimum Variance",
    "max_sharpe": "Maximum Sharpe",
    "min_cvar": "Minimum CVaR",
}
UNIVERSE_LABELS = {"equity": "Equity", "crypto": "Crypto", "combined": "Combined"}


def plot_final_sentiment_innovation_composite(
    returns: pd.DataFrame,
    metrics: pd.DataFrame,
    path: pathlib.Path,
) -> pathlib.Path:
    """Save the final sentiment innovation composite report figure."""
    plot_returns = returns.copy()
    plot_returns["date"] = pd.to_datetime(plot_returns["date"])
    min_date = plot_returns["date"].min()
    max_date = plot_returns["date"].max()
    metric_lookup = metrics.set_index("Strategy")

    fig = plt.figure(figsize=(13.2, 7.4), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.85, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.28,
        hspace=0.42,
    )
    ax_growth = fig.add_subplot(grid[:, 0])
    ax_sharpe = fig.add_subplot(grid[0, 1])
    ax_drawdown = fig.add_subplot(grid[1, 1])

    wealth_by_strategy: dict[str, pd.DataFrame] = {}
    for strategy in FINAL_SENTIMENT_FIGURE_ORDER:
        group = plot_returns.loc[plot_returns["strategy"].eq(strategy)].sort_values("date")
        if group.empty:
            continue
        growth = (1.0 + group["return"]).cumprod()
        dates = pd.concat(
            [
                pd.Series([group["date"].iloc[0] - pd.Timedelta(days=1)]),
                group["date"],
            ],
            ignore_index=True,
        )
        wealth = pd.concat([pd.Series([1.0]), growth], ignore_index=True)
        frame = pd.DataFrame({"date": dates, "wealth": wealth})
        frame["drawdown"] = frame["wealth"] / frame["wealth"].cummax() - 1.0
        wealth_by_strategy[strategy] = frame

        label = FINAL_SENTIMENT_FIGURE_LABELS[strategy]
        color = FINAL_SENTIMENT_FIGURE_COLORS[strategy]
        linewidth = 2.4 if "Final" in label else 1.9
        ax_growth.plot(frame["date"], frame["wealth"], label=label, color=color, linewidth=linewidth)
        ax_drawdown.plot(frame["date"], frame["drawdown"], label=label, color=color, linewidth=1.35)

    ax_growth.set_title("A. Growth of $1", loc="left", fontsize=12, fontweight="bold", pad=8)
    ax_growth.set_xlabel("Trading Date")
    ax_growth.set_ylabel("Growth of $1")
    ax_growth.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"${value:,.2f}"))
    ax_growth.grid(True, axis="y", alpha=0.22)
    ax_growth.legend(frameon=False, loc="upper left", fontsize=9)

    sharpe_values = [
        float(metric_lookup.loc[strategy, "Sharpe Ratio"])
        for strategy in FINAL_SENTIMENT_FIGURE_ORDER
    ]
    labels = [FINAL_SENTIMENT_FIGURE_LABELS[strategy] for strategy in FINAL_SENTIMENT_FIGURE_ORDER]
    colors = [FINAL_SENTIMENT_FIGURE_COLORS[strategy] for strategy in FINAL_SENTIMENT_FIGURE_ORDER]
    x_positions = np.arange(len(labels))
    bars = ax_sharpe.bar(x_positions, sharpe_values, color=colors, width=0.62)
    ax_sharpe.set_title("B. Sharpe Ratio", loc="left", fontsize=12, fontweight="bold", pad=8)
    ax_sharpe.set_xticks(x_positions)
    ax_sharpe.set_xticklabels(["Base", "Naive", "Final"], fontsize=9)
    ax_sharpe.set_ylabel("Sharpe")
    ax_sharpe.grid(True, axis="y", alpha=0.22)
    ax_sharpe.set_ylim(0, max(sharpe_values) * 1.24)
    for bar, value in zip(bars, sharpe_values):
        ax_sharpe.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sharpe_values) * 0.035,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#14253D",
        )

    ax_drawdown.set_title("C. Drawdown", loc="left", fontsize=12, fontweight="bold", pad=8)
    ax_drawdown.set_xlabel("Trading Date")
    ax_drawdown.set_ylabel("Drawdown")
    ax_drawdown.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_drawdown.grid(True, axis="y", alpha=0.22)
    for strategy in ["Naive Contrarian", "Final Tail-Risk-Aware Sector-Neutral Tuned"]:
        if strategy not in wealth_by_strategy or strategy not in metric_lookup.index:
            continue
        frame = wealth_by_strategy[strategy]
        max_drawdown = float(metric_lookup.loc[strategy, "Maximum Drawdown"])
        min_index = frame["drawdown"].idxmin()
        ax_drawdown.annotate(
            f"{FINAL_SENTIMENT_FIGURE_LABELS[strategy]} {max_drawdown:.1%}",
            xy=(frame.loc[min_index, "date"], frame.loc[min_index, "drawdown"]),
            xytext=(8, -4 if strategy == "Naive Contrarian" else 14),
            textcoords="offset points",
            fontsize=8,
            color=FINAL_SENTIMENT_FIGURE_COLORS[strategy],
            arrowprops={
                "arrowstyle": "-",
                "color": FINAL_SENTIMENT_FIGURE_COLORS[strategy],
                "lw": 0.8,
            },
        )

    for ax in [ax_growth, ax_drawdown]:
        ax.set_xlim(min_date, max_date)
        ax.set_xticks(
            [
                min_date,
                pd.Timestamp("2022-01-01"),
                pd.Timestamp("2023-01-01"),
            ]
        )
        ax.set_xticklabels(["2021", "2022", "2023"])
    for ax in [ax_growth, ax_sharpe, ax_drawdown]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)

    fig.suptitle(
        "Final Sentiment Innovation: Performance and Risk Comparison\nOOS 2021–2023",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.012,
        "The final strategy achieved stronger growth and Sharpe while reducing maximum drawdown relative to the naive sentiment baseline.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4D4D4D",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.13)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _wide_returns(returns_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long return output into the wide matrix used by portfolio code."""
    wide = returns_long.pivot(index="date", columns="ticker", values="return")
    wide = wide.sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    if not wide.index.is_monotonic_increasing:
        raise ValueError("return matrix index must be sorted")
    if not wide.index.is_unique:
        raise ValueError("return matrix index must contain unique dates")
    if not wide.columns.is_unique:
        raise ValueError("return matrix columns must contain unique asset labels")
    non_numeric = [
        column for column in wide.columns if not pd.api.types.is_numeric_dtype(wide[column])
    ]
    if non_numeric:
        raise TypeError(f"return columns must be numeric: {non_numeric}")
    return wide


def _build_return_universes() -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Load clean prices and construct equity, crypto and combined return panels."""
    from src.etl import load_clean_crypto, load_clean_equities

    equity_prices, _equity_checks = load_clean_equities()
    crypto_prices, _crypto_checks = load_clean_crypto()

    equity_returns = _wide_returns(daily_returns(equity_prices))
    crypto_returns_native = _wide_returns(daily_returns(crypto_prices))

    duplicate_labels = sorted(
        set(equity_returns.columns).intersection(crypto_returns_native.columns)
    )
    if duplicate_labels:
        raise ValueError(
            "equity and crypto asset labels overlap; provide explicit unique "
            f"identifiers before combining: {duplicate_labels}"
        )

    crypto_returns_on_equity_calendar = crypto_returns_native.reindex(equity_returns.index)
    combined_returns = pd.concat(
        [equity_returns, crypto_returns_on_equity_calendar],
        axis=1,
    )
    if not combined_returns.columns.is_unique:
        raise ValueError("combined return matrix columns must be unique")

    universes = {
        "equity": equity_returns,
        "crypto": crypto_returns_native,
        "combined": combined_returns,
    }
    universe_summary = {
        name: {
            "shape": returns.shape,
            "start_date": returns.index.min(),
            "end_date": returns.index.max(),
            "asset_count": returns.shape[1],
        }
        for name, returns in universes.items()
    }
    universe_summary["combined"]["equity_asset_count"] = equity_returns.shape[1]
    universe_summary["combined"]["crypto_asset_count"] = crypto_returns_native.shape[1]
    return universes, universe_summary


def _fund_configs() -> list[dict]:
    """Return the Stage 4 baseline fund run list in report order."""
    from src.portfolios import DEFAULT_CVAR_CONFIDENCE_LEVEL

    configs = []
    for universe, window_size, periods_per_year in [
        ("equity", 252, 252),
        ("crypto", 365, 365),
        ("combined", 252, 252),
    ]:
        for method in ["equal_weight", "min_variance", "max_sharpe", "min_cvar"]:
            configs.append(
                {
                    "fund_name": f"{universe}_{method}",
                    "asset_universe": universe,
                    "method": method,
                    "window_size": window_size,
                    "periods_per_year": periods_per_year,
                    "confidence_level": (
                        DEFAULT_CVAR_CONFIDENCE_LEVEL if method == "min_cvar" else np.nan
                    ),
                }
            )
    return configs


def _validate_fund_outputs(result: dict, metrics: dict) -> dict:
    """Validate one generated fund and return compact run-status diagnostics."""
    from src.portfolios import DEFAULT_MAX_ASSET_WEIGHT

    daily = result["daily_returns"]
    target = result["target_weights"]
    pretrade = result["pretrade_weights"]
    rebalance_log = result["rebalance_log"]

    if not np.isfinite(daily.to_numpy(dtype=float)).all():
        raise ValueError("daily returns contain NaN or infinity")
    if not rebalance_log["effective_date"].is_monotonic_increasing:
        raise ValueError("rebalance effective dates must be increasing")
    if not (
        pd.to_datetime(rebalance_log["estimation_end_date"])
        < pd.to_datetime(rebalance_log["effective_date"])
    ).all():
        raise ValueError("one or more effective dates are not after estimation end dates")
    if pd.Timestamp(result["first_live_date"]) != pd.Timestamp(daily.index.min()):
        raise ValueError("first_live_date does not match first daily-return date")
    if not metrics:
        raise ValueError("performance metrics were not calculated")

    target_sum_error = float((target.sum(axis=1) - 1.0).abs().max())
    pretrade_sum_error = float((pretrade.sum(axis=1) - 1.0).abs().max())
    weights_sum_valid = (
        target_sum_error <= WEIGHT_TOLERANCE
        and pretrade_sum_error <= WEIGHT_TOLERANCE
    )
    non_negative_weights = bool(
        (target >= -BOUND_TOLERANCE).all().all()
        and (pretrade >= -BOUND_TOLERANCE).all().all()
    )
    maximum_asset_weight = float(target.max().max())
    cap_series = rebalance_log["effective_max_asset_weight"].fillna(DEFAULT_MAX_ASSET_WEIGHT)
    max_weights = target.max(axis=1).to_numpy(dtype=float)
    active_caps = cap_series.to_numpy(dtype=float) + BOUND_TOLERANCE
    cap_valid = bool((max_weights <= active_caps).all())
    if not weights_sum_valid:
        raise ValueError("target or pre-trade weights do not sum to 1")
    if not non_negative_weights:
        raise ValueError("target or pre-trade weights contain negative values")
    if not cap_valid:
        raise ValueError("target weights exceed the active individual asset cap")

    return {
        "weights_sum_valid": weights_sum_valid,
        "non_negative_weights": non_negative_weights,
        "maximum_asset_weight": maximum_asset_weight,
        "max_target_weight_sum_error": target_sum_error,
        "max_pretrade_weight_sum_error": pretrade_sum_error,
        "fallback_count": int(rebalance_log["fallback_used"].fillna(False).sum()),
    }


def _tidy_fund_returns(config: dict, result: dict) -> pd.DataFrame:
    """Return one tidy daily-return table for a generated fund."""
    returns = result["daily_returns"].sort_index().rename("portfolio_return").reset_index()
    returns.columns = ["date", "portfolio_return"]
    returns.insert(0, "method", config["method"])
    returns.insert(0, "asset_universe", config["asset_universe"])
    returns.insert(0, "fund_name", config["fund_name"])
    return returns


def _tidy_target_weights(config: dict, result: dict) -> pd.DataFrame:
    """Return one tidy target-weight table for a generated fund."""
    weights = (
        result["target_weights"]
        .sort_index()
        .rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="asset", value_name="weight")
    )
    weights.insert(0, "method", config["method"])
    weights.insert(0, "asset_universe", config["asset_universe"])
    weights.insert(0, "fund_name", config["fund_name"])
    return weights.sort_values(["fund_name", "date", "asset"]).reset_index(drop=True)


def _tidy_pretrade_weights(config: dict, result: dict) -> pd.DataFrame:
    """Return one tidy pre-trade weight table for a generated fund."""
    weights = (
        result["pretrade_weights"]
        .sort_index()
        .rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="asset", value_name="pretrade_weight")
    )
    weights.insert(0, "method", config["method"])
    weights.insert(0, "asset_universe", config["asset_universe"])
    weights.insert(0, "fund_name", config["fund_name"])
    return weights


def _tidy_turnover(config: dict, result: dict) -> pd.DataFrame:
    """Return one tidy turnover table for a generated fund."""
    turnover = result["turnover"].sort_index().rename("turnover").reset_index()
    turnover.columns = ["date", "turnover"]
    turnover.insert(0, "method", config["method"])
    turnover.insert(0, "asset_universe", config["asset_universe"])
    turnover.insert(0, "fund_name", config["fund_name"])
    return turnover


def _tidy_rebalance_log(config: dict, result: dict) -> pd.DataFrame:
    """Return one rebalance log with fund-identifying columns first."""
    log = result["rebalance_log"].copy()
    if "method" not in log.columns:
        log.insert(0, "method", config["method"])
    else:
        remaining_columns = [column for column in log.columns if column != "method"]
        log = log[["method", *remaining_columns]]
    log.insert(0, "asset_universe", config["asset_universe"])
    log.insert(0, "fund_name", config["fund_name"])
    return log


def _run_one_fund(
    config: dict,
    returns: pd.DataFrame,
) -> tuple[
    dict,
    dict,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run one configured baseline fund and return summary/status rows."""
    from src.portfolios import oos_backtest, performance_metrics

    fund_name = config["fund_name"]
    backtest_kwargs = {
        "returns": returns,
        "method": config["method"],
        "window_size": config["window_size"],
        "periods_per_year": config["periods_per_year"],
        "risk_free_rate": 0.0,
    }
    if config["method"] == "min_cvar":
        backtest_kwargs["confidence_level"] = config["confidence_level"]

    result = oos_backtest(**backtest_kwargs)
    metrics = performance_metrics(
        result["daily_returns"],
        periods_per_year=config["periods_per_year"],
        risk_free_rate=0.0,
    )
    validation = _validate_fund_outputs(result, metrics)

    average_turnover = float(result["turnover"].dropna().mean())
    total_turnover = float(result["turnover"].dropna().sum())
    fallback_count = int(result["rebalance_log"]["fallback_used"].fillna(False).sum())

    performance_row = {
        **config,
        **metrics,
        "average_turnover": average_turnover,
        "total_turnover": total_turnover,
        "rebalance_count": int(result["rebalance_log"].shape[0]),
        "fallback_count": fallback_count,
    }
    status_row = {
        "fund_name": fund_name,
        "run_success": True,
        "daily_observation_count": int(result["daily_returns"].shape[0]),
        "rebalance_count": int(result["rebalance_log"].shape[0]),
        **validation,
    }
    return (
        performance_row,
        status_row,
        _tidy_fund_returns(config, result),
        _tidy_target_weights(config, result),
        _tidy_pretrade_weights(config, result),
        _tidy_turnover(config, result),
        _tidy_rebalance_log(config, result),
    )


# ============================================================================
# STAGE 6.2: BASELINE GROWTH OF $1 COMPARISON FIGURES
# Purpose:
# Create investor-friendly cumulative Growth of $1 charts for the baseline
# funds using the saved daily fund returns. This section does not rerun any
# portfolio model or backtest.
# ============================================================================
def create_baseline_growth_figures() -> list[pathlib.Path]:
    """Create one Growth of $1 comparison figure for each asset universe."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    returns = pd.read_csv(DATA_DIR / "fund_returns.csv", parse_dates=["date"])
    outputs = []
    figure_specs = {
        "equity": ("Equity Funds: How $1 Grew Over Time", "growth_of_1_equity.png"),
        "crypto": ("Crypto Funds: How $1 Grew Over Time", "growth_of_1_crypto.png"),
        "combined": ("Combined Funds: How $1 Grew Over Time", "growth_of_1_combined.png"),
    }
    events = {
        pd.Timestamp("2021-11-08"): "Nov 2021 crypto market peak",
        pd.Timestamp("2022-11-11"): "Nov 2022 FTX collapse",
    }

    for universe, (title, filename) in figure_specs.items():
        fig, ax = plt.subplots(figsize=(9, 5.2))
        universe_returns = returns.loc[returns["asset_universe"] == universe]
        final_points = []
        plotted_values = []
        for method in METHOD_ORDER:
            fund = universe_returns.loc[universe_returns["method"] == method].sort_values("date")
            growth = (1 + fund["portfolio_return"]).cumprod()
            dates = pd.concat(
                [pd.Series([fund["date"].iloc[0] - pd.Timedelta(days=1)]), fund["date"]],
                ignore_index=True,
            )
            values = pd.concat([pd.Series([1.0]), growth], ignore_index=True)
            (line,) = ax.plot(dates, values, label=METHOD_LABELS[method], linewidth=1.8)
            plotted_values.extend(values.to_list())
            final_points.append((METHOD_LABELS[method], dates.iloc[-1], values.iloc[-1], line.get_color()))

        ax.set_title(title, pad=34)
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1")
        ax.set_yscale("log")
        ax.set_ylim(min(plotted_values) * 0.92, max(plotted_values) * 1.12)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"${value:,.2f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)

        x_min = universe_returns["date"].min() - pd.Timedelta(days=1)
        x_max = universe_returns["date"].max()
        ax.set_xlim(x_min, x_max + pd.Timedelta(days=40))
        for event_date, event_label in events.items():
            if universe != "equity" and x_min <= event_date <= x_max:
                ax.axvline(event_date, color="0.5", linestyle="--", linewidth=0.9, alpha=0.45)
                ax.annotate(
                    event_label,
                    xy=(event_date, 0.92),
                    xycoords=("data", "axes fraction"),
                    xytext=(4, -4),
                    textcoords="offset points",
                    rotation=90,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="0.35",
                )

        for offset, (label, date, value, color) in zip(
            np.linspace(-18, 18, len(final_points)),
            sorted(final_points, key=lambda item: item[2]),
        ):
            ax.annotate(
                f"{label}: {value:.2f}",
                xy=(date, value),
                xytext=(12, offset),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=color,
                arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.45, "lw": 0.8},
            )

        output_path = FIGURES_DIR / filename
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output_path)
    return outputs


# ============================================================================
# STAGE 6.3: BASELINE DRAWDOWN SMALL-MULTIPLES FIGURE
# Purpose:
# Display the drawdown paths of all 12 baseline funds in one structured
# 3-by-4 small-multiples figure. This section uses saved daily fund returns only
# and does not rerun portfolio optimisation or backtesting.
# ============================================================================
def _drawdown_from_saved_returns(fund_returns: pd.DataFrame) -> tuple[pd.Series, pd.Timestamp, float, str]:
    """Return drawdown path, maximum-drawdown date/value, and recovery label."""
    fund_returns = fund_returns.sort_values("date").reset_index(drop=True)
    growth = (1 + fund_returns["portfolio_return"]).cumprod()
    growth.index = fund_returns["date"]
    running_peak = growth.cummax()
    drawdown = growth / running_peak - 1
    max_drawdown_date = drawdown.idxmin()
    max_drawdown = float(drawdown.loc[max_drawdown_date])
    peak_value = running_peak.loc[max_drawdown_date]
    after_max = growth.loc[growth.index > max_drawdown_date]
    recovered = after_max.loc[after_max >= peak_value * (1 - 1e-10)]
    if recovered.empty:
        recovery_label = "Not recovered by end date"
    else:
        recovery_days = (recovered.index[0] - max_drawdown_date).days
        recovery_label = f"Recovered in {recovery_days} days"
    return drawdown, max_drawdown_date, max_drawdown, recovery_label


def create_baseline_drawdown_small_multiples() -> pathlib.Path:
    """Create the formal 3-by-4 baseline drawdown comparison figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    returns = pd.read_csv(DATA_DIR / "fund_returns.csv", parse_dates=["date"])
    drawdowns = {}
    row_limits = {}

    for universe in UNIVERSE_ORDER:
        row_min = 0.0
        for method in METHOD_ORDER:
            fund = returns.loc[
                (returns["asset_universe"] == universe) & (returns["method"] == method)
            ]
            drawdown, max_date, max_drawdown, recovery_label = _drawdown_from_saved_returns(fund)
            drawdowns[(universe, method)] = (drawdown, max_date, max_drawdown, recovery_label)
            row_min = min(row_min, float(drawdown.min()))
        row_limits[universe] = (row_min * 1.12, 0.0)

    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
    fig.suptitle("Baseline Drawdown Comparison Across Asset Universes and Methods", y=0.98)
    x_min = returns["date"].min()
    x_max = returns["date"].max()

    for row, universe in enumerate(UNIVERSE_ORDER):
        for col, method in enumerate(METHOD_ORDER):
            ax = axes[row, col]
            drawdown, max_date, max_drawdown, recovery_label = drawdowns[(universe, method)]
            ax.plot(drawdown.index, drawdown, color="tab:blue", linewidth=1.2)
            ax.fill_between(drawdown.index, drawdown.to_numpy(), 0, color="tab:blue", alpha=0.18)
            ax.scatter(max_date, max_drawdown, color="tab:red", s=22, zorder=3)
            ax.annotate(
                f"Max DD: {max_drawdown:.2%}",
                xy=(max_date, max_drawdown),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=8,
                color="tab:red",
                arrowprops={"arrowstyle": "-", "color": "tab:red", "lw": 0.7, "alpha": 0.7},
            )
            ax.text(
                0.03,
                0.08,
                recovery_label,
                transform=ax.transAxes,
                fontsize=7.5,
                color="0.35",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(row_limits[universe])
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.grid(True, alpha=0.22)
            if row == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{UNIVERSE_LABELS[universe]}\nDrawdown")
            else:
                ax.tick_params(labelleft=False)
            if row == 2:
                ax.set_xlabel("Date")
            else:
                ax.tick_params(labelbottom=False)

    output_path = FIGURES_DIR / "baseline_drawdown_small_multiples.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================================
# STAGE 6.4: COMBINED PORTFOLIO WEIGHTS OVER TIME
# Purpose:
# Compare how the four Combined baseline portfolio methods allocated total
# portfolio weight between Equity and Crypto across monthly rebalances.
# This section uses saved target weights only and does not rerun portfolio
# optimisation or backtesting.
# ============================================================================
def _crypto_ticker_set() -> set[str]:
    """Return the explicit Crypto ticker set from the project data loader."""
    from src.data_access import load_crypto_prices

    crypto_prices = load_crypto_prices()
    return set(crypto_prices["ticker"].dropna().unique())


def create_combined_weights_over_time_figure() -> pathlib.Path:
    """Create the 2-by-2 Combined Equity/Crypto stacked-step allocation figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    crypto_tickers = _crypto_ticker_set()
    weights = pd.read_csv(DATA_DIR / "fund_weights.csv", parse_dates=["date"])
    combined_weights = weights.loc[weights["asset_universe"] == "combined"].copy()
    combined_weights["asset_class"] = np.where(
        combined_weights["asset"].isin(crypto_tickers), "Crypto", "Equity"
    )
    allocation = (
        combined_weights.groupby(["method", "date", "asset_class"], as_index=False)["weight"]
        .sum()
        .pivot_table(
            index=["method", "date"],
            columns="asset_class",
            values="weight",
            fill_value=0.0,
        )
        .reset_index()
    )
    for column in ["Equity", "Crypto"]:
        if column not in allocation:
            allocation[column] = 0.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey=True)
    fig.suptitle("Combined Portfolio Allocation Over Time Across Optimisation Methods", y=0.98)
    panel_order = [
        ("equal_weight", axes[0, 0]),
        ("min_variance", axes[0, 1]),
        ("max_sharpe", axes[1, 0]),
        ("min_cvar", axes[1, 1]),
    ]
    colors = {"Equity": "tab:blue", "Crypto": "tab:orange"}
    x_min = allocation["date"].min()
    x_max = allocation["date"].max()

    for method, ax in panel_order:
        panel = allocation.loc[allocation["method"] == method].sort_values("date")
        total_weight = panel["Equity"] + panel["Crypto"]
        average_crypto = panel["Crypto"].mean()
        latest_date = panel["date"].iloc[-1]
        latest_crypto = panel["Crypto"].iloc[-1]
        ax.fill_between(
            panel["date"],
            0,
            panel["Equity"].to_numpy(),
            step="post",
            color=colors["Equity"],
            alpha=0.78,
            label="Equity",
        )
        ax.fill_between(
            panel["date"],
            panel["Equity"].to_numpy(),
            total_weight.to_numpy(),
            step="post",
            color=colors["Crypto"],
            alpha=0.78,
            label="Crypto",
        )
        ax.step(panel["date"], panel["Equity"], where="post", color="white", linewidth=0.8, alpha=0.7)
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylim(0, 1)
        ax.set_xlim(x_min, x_max)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, alpha=0.22)
        ax.text(
            0.03,
            0.08,
            f"Average Crypto: {average_crypto:.1%}\nLatest Crypto: {latest_crypto:.1%}",
            transform=ax.transAxes,
            fontsize=8.5,
            color="0.25",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
        )

    axes[0, 0].set_ylabel("Portfolio allocation")
    axes[1, 0].set_ylabel("Portfolio allocation")
    axes[1, 0].set_xlabel("Date")
    axes[1, 1].set_xlabel("Date")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=False)
    fig.text(
        0.5,
        0.02,
        "Combined portfolios contain 50 Equity assets and 10 Crypto assets. Weights are aggregated into the two asset classes for readability.",
        ha="center",
        fontsize=9,
        color="0.35",
    )

    output_path = FIGURES_DIR / "combined_weights_over_time.png"
    fig.tight_layout(rect=[0, 0.05, 1, 0.9])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================================
# STAGE 6.5: BASELINE SHARPE RATIO BARPLOT
# Purpose:
# Create the required Sharpe-ratio comparison figure across baseline funds
# and methods using the saved performance_metrics.csv output only.
# ============================================================================
def create_baseline_sharpe_barplot() -> pathlib.Path:
    """Create the baseline Sharpe-ratio barplot from saved performance metrics."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(TABLES_DIR / "performance_metrics.csv")
    metrics["asset_universe"] = pd.Categorical(
        metrics["asset_universe"],
        categories=list(UNIVERSE_ORDER),
        ordered=True,
    )
    metrics["method"] = pd.Categorical(
        metrics["method"],
        categories=list(METHOD_ORDER),
        ordered=True,
    )
    metrics = metrics.sort_values(["asset_universe", "method"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    fig.suptitle("Baseline Fund Sharpe Ratio Comparison", y=0.98)
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHOD_ORDER)))
    y_min = min(0.0, float(metrics["sharpe_ratio"].min()))
    y_max = max(0.0, float(metrics["sharpe_ratio"].max()))
    y_pad = max(0.08, (y_max - y_min) * 0.18)

    for ax, universe in zip(axes, UNIVERSE_ORDER):
        panel = metrics.loc[metrics["asset_universe"] == universe]
        labels = [METHOD_LABELS[method] for method in panel["method"].astype(str)]
        bars = ax.bar(labels, panel["sharpe_ratio"], color=colors, alpha=0.82)
        ax.set_title(UNIVERSE_LABELS[universe])
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", rotation=30)
        for bar, value in zip(bars, panel["sharpe_ratio"]):
            offset = 4 if value >= 0 else -10
            va = "bottom" if value >= 0 else "top"
            ax.annotate(
                f"{value:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=8.5,
            )

    axes[0].set_ylabel("Sharpe Ratio")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_path = FIGURES_DIR / "baseline_sharpe_ratio_barplot.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================================
# STAGE 6.6: BASELINE RISK-RETURN COMPARISON FIGURE
# Purpose:
# Compare annualised return and annualised volatility across all 12 baseline
# funds and four portfolio methods using the saved performance metrics only.
# This section does not rerun portfolio optimisation or backtesting.
# ============================================================================
def create_baseline_risk_return_figure() -> pathlib.Path:
    """Create the formal baseline risk-return comparison figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    baseline_funds = [
        "equity_equal_weight",
        "equity_min_variance",
        "equity_max_sharpe",
        "equity_min_cvar",
        "crypto_equal_weight",
        "crypto_min_variance",
        "crypto_max_sharpe",
        "crypto_min_cvar",
        "combined_equal_weight",
        "combined_min_variance",
        "combined_max_sharpe",
        "combined_min_cvar",
    ]
    required_columns = [
        "fund_name",
        "asset_universe",
        "method",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
    ]
    metrics = pd.read_csv(TABLES_DIR / "performance_metrics.csv")
    missing_columns = [column for column in required_columns if column not in metrics.columns]
    if missing_columns:
        raise ValueError(f"performance_metrics.csv is missing columns: {missing_columns}")

    metrics = metrics.loc[metrics["fund_name"].isin(baseline_funds), required_columns].copy()
    missing_funds = sorted(set(baseline_funds) - set(metrics["fund_name"]))
    duplicate_funds = sorted(metrics.loc[metrics["fund_name"].duplicated(), "fund_name"].unique())
    if missing_funds:
        raise ValueError(f"performance_metrics.csv is missing baseline funds: {missing_funds}")
    if duplicate_funds:
        raise ValueError(f"performance_metrics.csv contains duplicate baseline funds: {duplicate_funds}")

    metrics["asset_universe"] = pd.Categorical(
        metrics["asset_universe"],
        categories=list(UNIVERSE_ORDER),
        ordered=True,
    )
    metrics["method"] = pd.Categorical(
        metrics["method"],
        categories=list(METHOD_ORDER),
        ordered=True,
    )
    metrics = metrics.sort_values(["asset_universe", "method"])

    method_styles = {
        "equal_weight": {"color": "#1f77b4", "marker": "o", "short_label": "EW"},
        "min_variance": {"color": "#ff7f0e", "marker": "s", "short_label": "Min Var"},
        "max_sharpe": {"color": "#2ca02c", "marker": "^", "short_label": "Max Sharpe"},
        "min_cvar": {"color": "#9467bd", "marker": "D", "short_label": "Min CVaR"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4))
    fig.patch.set_facecolor("#f7f5ef")
    fig.suptitle("Baseline Fund Risk-Return Comparison", y=0.98, fontsize=14)

    for ax, universe in zip(axes, UNIVERSE_ORDER):
        panel = metrics.loc[metrics["asset_universe"] == universe].copy()
        x_values = panel["annualised_volatility"].to_numpy(dtype=float)
        y_values = panel["annualised_return"].to_numpy(dtype=float)
        x_span = max(float(x_values.max() - x_values.min()), 0.02)
        y_span = max(float(y_values.max() - y_values.min()), 0.02)
        x_min = max(0.0, float(x_values.min()) - x_span * 0.22)
        x_max = float(x_values.max()) + x_span * 0.22
        y_min_observed = float(y_values.min())
        y_max_observed = float(y_values.max())
        y_min = y_min_observed - y_span * 0.26
        y_max = y_max_observed + y_span * 0.26
        if y_min_observed < 0 < y_max_observed:
            ax.axhline(0, color="0.35", linewidth=0.8, alpha=0.55)
            y_min = min(y_min, -y_span * 0.12)
            y_max = max(y_max, y_span * 0.12)

        ax.set_facecolor("#fffdf8")
        ax.set_title(UNIVERSE_LABELS[universe], fontsize=11)
        ax.set_xlabel("Annualised Volatility")
        ax.grid(True, axis="both", color="0.82", linewidth=0.8, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        for method in METHOD_ORDER:
            row = panel.loc[panel["method"] == method].iloc[0]
            style = method_styles[method]
            x_value = float(row["annualised_volatility"])
            y_value = float(row["annualised_return"])
            x_offset = 8 if x_value <= (x_min + x_max) / 2 else -8
            y_offset = 8 if y_value <= (y_min + y_max) / 2 else -8
            horizontal_alignment = "left" if x_offset > 0 else "right"
            vertical_alignment = "bottom" if y_offset > 0 else "top"
            ax.scatter(
                x_value,
                y_value,
                s=74,
                color=style["color"],
                marker=style["marker"],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label=METHOD_LABELS[method],
            )
            ax.annotate(
                style["short_label"],
                xy=(x_value, y_value),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=8.6,
                color="0.2",
                ha=horizontal_alignment,
                va=vertical_alignment,
            )

        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    axes[0].set_ylabel("Annualised Return")
    for ax in axes[1:]:
        ax.set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.025,
        "Each point represents one baseline fund. Higher positions indicate higher annualised return; points further right indicate higher annualised volatility. Point labels use EW, Min Var, Max Sharpe and Min CVaR.",
        ha="center",
        fontsize=8.8,
        color="0.35",
    )

    output_path = FIGURES_DIR / "baseline_risk_return_comparison.png"
    fig.tight_layout(rect=[0, 0.07, 1, 0.84])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================================
# STAGE 7.2: CRYPTO MARKET ROLLING TAIL-RISK EVIDENCE
# Purpose:
# Generate the equal-weight ten-Crypto market return series, rolling 60-day
# 90% realised CVaR positive loss measure, compact summary table, and
# risk-over-time figure. This section is separate from the baseline portfolio
# pipeline and does not modify the Pure 95% Minimum-CVaR optimiser.
# ============================================================================
def _long_returns_from_wide(wide_returns: pd.DataFrame) -> pd.DataFrame:
    """Convert a validated wide return panel back to long return format."""
    long_returns = (
        wide_returns.sort_index()
        .rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="return")
    )
    return long_returns.sort_values(["ticker", "date"]).reset_index(drop=True)


def _validate_stage7_tail_risk_outputs(
    tail_risk: pd.DataFrame,
    summary: pd.DataFrame,
    figure_path: pathlib.Path,
) -> None:
    """Run concise production checks for the generated Stage 7.2 artefacts."""
    if not STAGE7_TAIL_RISK_DATA_PATH.exists():
        raise FileNotFoundError(STAGE7_TAIL_RISK_DATA_PATH)
    if not STAGE7_TAIL_RISK_SUMMARY_PATH.exists():
        raise FileNotFoundError(STAGE7_TAIL_RISK_SUMMARY_PATH)
    if not figure_path.exists() or figure_path.stat().st_size == 0:
        raise FileNotFoundError(figure_path)

    if int(tail_risk["number_of_crypto_assets"].max()) != 10:
        raise ValueError("Crypto market series must be based on the ten-Crypto universe")
    date_range = pd.to_datetime(tail_risk["date"])
    if date_range.min().year != 2020 or date_range.max().year != 2023:
        raise ValueError("Stage 7.2 output should cover the intended 2020-2023 period")

    valid = tail_risk.loc[tail_risk["crypto_tail_risk_60d_90"].notna()]
    if valid.empty:
        raise ValueError("Stage 7.2 output has no valid rolling tail-risk observations")
    if not valid["rolling_window_observations"].eq(60).all():
        raise ValueError("valid tail-risk rows must use 60 valid observations")
    if not valid["tail_observations"].eq(6).all():
        raise ValueError("valid tail-risk rows must use six tail observations")
    if not (valid["crypto_tail_risk_60d_90"] >= 0).all():
        raise ValueError("positive tail-risk values must be non-negative")

    required_summary = {
        "first_valid_tail_risk_date",
        "maximum_positive_tail_risk_value",
        "date_of_maximum_tail_risk",
    }
    missing = required_summary.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing maximum tail-risk metadata: {sorted(missing)}")


def create_stage7_crypto_tail_risk_outputs(
    crypto_returns: pd.DataFrame | None = None,
) -> dict[str, pathlib.Path]:
    """Create the Stage 7.2 data file, summary table and figure."""
    from src.etl import load_clean_crypto
    from src.portfolios import (
        build_crypto_tail_risk_series,
        plot_crypto_tail_risk,
        summarize_crypto_tail_risk,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if crypto_returns is None:
        crypto_prices, _crypto_checks = load_clean_crypto()
        crypto_returns_long = daily_returns(crypto_prices)
    else:
        crypto_returns_long = _long_returns_from_wide(crypto_returns)

    tail_risk = build_crypto_tail_risk_series(crypto_returns_long)
    summary = summarize_crypto_tail_risk(tail_risk)
    tail_risk.to_csv(STAGE7_TAIL_RISK_DATA_PATH, index=False)
    summary.to_csv(STAGE7_TAIL_RISK_SUMMARY_PATH, index=False)
    figure_path = plot_crypto_tail_risk(tail_risk, STAGE7_TAIL_RISK_FIGURE_PATH)
    _validate_stage7_tail_risk_outputs(tail_risk, summary, figure_path)

    row = summary.iloc[0]
    print("\nStage 7.2 Crypto tail-risk evidence:")
    print(f"  tail-risk data: {STAGE7_TAIL_RISK_DATA_PATH}")
    print(f"  tail-risk summary: {STAGE7_TAIL_RISK_SUMMARY_PATH}")
    print(f"  tail-risk figure: {figure_path}")
    print(f"  first valid tail-risk date: {row['first_valid_tail_risk_date']}")
    print(
        "  positive tail risk min/median/mean/max: "
        f"{row['minimum_positive_tail_risk_value']:.6f}, "
        f"{row['median_positive_tail_risk_value']:.6f}, "
        f"{row['mean_positive_tail_risk_value']:.6f}, "
        f"{row['maximum_positive_tail_risk_value']:.6f}"
    )
    print(f"  date of maximum tail risk: {row['date_of_maximum_tail_risk']}")
    return {
        "data": STAGE7_TAIL_RISK_DATA_PATH,
        "summary": STAGE7_TAIL_RISK_SUMMARY_PATH,
        "figure": figure_path,
    }


# ============================================================================
# STAGE 7.3: LAYER 3A - CONTINUOUS CRYPTO RISK SCORE
# Purpose:
# Convert the completed Stage 7.2 rolling Crypto tail-risk series into a
# monthly no-look-ahead 0-100 score. This section writes only the signal and
# its audit summary; it does not create adaptive caps or new fund weights.
# ============================================================================
def _validate_stage7_crypto_risk_score_outputs(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> None:
    """Run concise production checks for the generated Stage 7.3 artefacts."""
    if not STAGE7_RISK_SCORE_DATA_PATH.exists():
        raise FileNotFoundError(STAGE7_RISK_SCORE_DATA_PATH)
    if not STAGE7_RISK_SCORE_SUMMARY_PATH.exists():
        raise FileNotFoundError(STAGE7_RISK_SCORE_SUMMARY_PATH)

    required_score_columns = {
        "date",
        "crypto_market_return",
        "rolling_tail_risk",
        "risk_score",
        "valid_history_count",
        "risk_score_available",
        "history_start_date",
        "history_end_date",
        "rebalance_month",
        "tail_risk_source_date",
    }
    missing = required_score_columns.difference(scores.columns)
    if missing:
        raise ValueError(f"Stage 7.3 scores are missing columns: {sorted(missing)}")

    dates = pd.DatetimeIndex(pd.to_datetime(scores["date"]))
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Stage 7.3 score dates must be sorted and unique")
    if not dates.equals(pd.DatetimeIndex(rebalance_dates)):
        raise ValueError("Stage 7.3 scores must be generated only on monthly rebalance dates")

    source_dates = pd.to_datetime(scores["tail_risk_source_date"])
    available_source_dates = source_dates.loc[source_dates.notna()]
    if not available_source_dates.le(dates[source_dates.notna()]).all():
        raise ValueError("Stage 7.3 tail-risk source dates must not be after score dates")

    history_end = pd.to_datetime(scores["history_end_date"])
    history_check = history_end.notna() & source_dates.notna()
    if history_check.any() and not (
        history_end.loc[history_check] < source_dates.loc[history_check]
    ).all():
        raise ValueError("Stage 7.3 history must exclude the current tail-risk observation")

    available = scores.loc[scores["risk_score_available"].astype(bool)].copy()
    unavailable = scores.loc[~scores["risk_score_available"].astype(bool)].copy()
    if not unavailable["risk_score"].isna().all():
        raise ValueError("unavailable early Stage 7.3 scores must remain missing")
    if not available.empty:
        if not available["risk_score"].between(0.0, 100.0).all():
            raise ValueError("available Stage 7.3 risk scores must be between 0 and 100")
        if not available["valid_history_count"].ge(60).all():
            raise ValueError("available Stage 7.3 scores require at least 60 prior daily values")
        relation = available[["rolling_tail_risk", "risk_score"]].corr().iloc[0, 1]
        if pd.notna(relation) and relation <= 0:
            raise ValueError("higher tail risk should generally map to higher scores")
    if not unavailable.empty and not unavailable["valid_history_count"].lt(60).all():
        raise ValueError("unavailable Stage 7.3 rows should be early rows below 60 prior values")
    if not scores["valid_history_count"].is_monotonic_increasing:
        raise ValueError("Stage 7.3 valid history count should increase over time")

    required_summary = {
        "first_monthly_score_date",
        "last_monthly_score_date",
        "total_monthly_score_date_rows",
        "minimum_required_history",
        "first_risk_score_available_date",
        "number_of_available_scores",
        "number_of_unavailable_early_scores",
        "minimum_risk_score",
        "median_risk_score",
        "mean_risk_score",
        "maximum_risk_score",
        "date_of_maximum_risk_score",
        "date_of_minimum_risk_score",
    }
    missing_summary = required_summary.difference(summary.columns)
    if missing_summary:
        raise ValueError(f"Stage 7.3 summary is missing columns: {sorted(missing_summary)}")


def create_stage7_crypto_risk_score_outputs(
    combined_returns: pd.DataFrame,
) -> dict[str, pathlib.Path]:
    """Create the Stage 7.3 monthly Crypto risk-score data and summary table."""
    from src.portfolios import (
        STAGE_7_RISK_SCORE_MIN_HISTORY,
        build_crypto_risk_score_series,
        identify_monthly_rebalance_dates,
        summarize_crypto_risk_scores,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    tail_risk = pd.read_csv(STAGE7_TAIL_RISK_DATA_PATH, parse_dates=["date"])
    rebalance_dates = identify_monthly_rebalance_dates(
        combined_returns,
        start_date=tail_risk["date"].min(),
        end_date=tail_risk["date"].max(),
    )
    scores = build_crypto_risk_score_series(
        tail_risk,
        rebalance_dates,
        min_history=STAGE_7_RISK_SCORE_MIN_HISTORY,
    )
    summary = summarize_crypto_risk_scores(
        scores,
        min_history=STAGE_7_RISK_SCORE_MIN_HISTORY,
    )

    scores.to_csv(STAGE7_RISK_SCORE_DATA_PATH, index=False)
    summary.to_csv(STAGE7_RISK_SCORE_SUMMARY_PATH, index=False)
    _validate_stage7_crypto_risk_score_outputs(scores, summary, rebalance_dates)

    row = summary.iloc[0]
    print("\nStage 7.3 Crypto risk-score evidence:")
    print(f"  risk-score data: {STAGE7_RISK_SCORE_DATA_PATH}")
    print(f"  risk-score summary: {STAGE7_RISK_SCORE_SUMMARY_PATH}")
    print(f"  first monthly score date: {row['first_monthly_score_date']}")
    print(f"  first risk-score-available date: {row['first_risk_score_available_date']}")
    print(
        "  risk score min/median/mean/max: "
        f"{row['minimum_risk_score']:.6f}, "
        f"{row['median_risk_score']:.6f}, "
        f"{row['mean_risk_score']:.6f}, "
        f"{row['maximum_risk_score']:.6f}"
    )
    return {
        "data": STAGE7_RISK_SCORE_DATA_PATH,
        "summary": STAGE7_RISK_SCORE_SUMMARY_PATH,
    }


# ============================================================================
# STAGE 7.4: LAYER 3B - CONTINUOUS PERSONALISED CRYPTO BUDGETS
# Purpose:
# Convert the existing monthly Stage 7.3 Crypto risk scores into target Crypto
# and Equity asset-class budgets. These are target sleeve budgets, not caps.
# This section does not create smoothing, portfolio weights or fund backtests.
# ============================================================================
def create_stage7_personalised_crypto_budget_outputs() -> dict[str, pathlib.Path]:
    """Create Stage 7.4 Raw Crypto Budget data, summary and design figure."""
    from src.portfolios import (
        build_personalised_crypto_budget_series,
        plot_personalised_crypto_budget_curves,
        summarize_personalised_crypto_budgets,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    risk_scores = pd.read_csv(STAGE7_RISK_SCORE_DATA_PATH, parse_dates=["date"])
    budgets = build_personalised_crypto_budget_series(risk_scores)
    summary = summarize_personalised_crypto_budgets(budgets)
    figure_path = plot_personalised_crypto_budget_curves(
        STAGE7_PERSONALISED_BUDGET_FIGURE_PATH
    )

    budgets.to_csv(STAGE7_PERSONALISED_BUDGET_DATA_PATH, index=False)
    summary.to_csv(STAGE7_PERSONALISED_BUDGET_SUMMARY_PATH, index=False)
    if not figure_path.exists() or figure_path.stat().st_size == 0:
        raise FileNotFoundError(figure_path)

    print("\nStage 7.4 personalised raw Crypto budgets:")
    print(f"  raw budget data: {STAGE7_PERSONALISED_BUDGET_DATA_PATH}")
    print(f"  raw budget summary: {STAGE7_PERSONALISED_BUDGET_SUMMARY_PATH}")
    print(f"  raw budget design figure: {figure_path}")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.profile}: valid={row.number_of_valid_raw_budgets}, "
            f"unavailable={row.number_of_unavailable_raw_budgets}, "
            f"first valid={row.first_valid_raw_budget_date}, "
            f"min/median/mean/max="
            f"{row.minimum_observed_raw_crypto_budget:.6f}, "
            f"{row.median_observed_raw_crypto_budget:.6f}, "
            f"{row.mean_observed_raw_crypto_budget:.6f}, "
            f"{row.maximum_observed_raw_crypto_budget:.6f}"
        )
    return {
        "data": STAGE7_PERSONALISED_BUDGET_DATA_PATH,
        "summary": STAGE7_PERSONALISED_BUDGET_SUMMARY_PATH,
        "figure": figure_path,
    }


# ============================================================================
# STAGE 7.6: PERSONALISED CRYPTO BUDGET SMOOTHING
# Purpose:
# Smooth monthly Raw Crypto Budgets into Applied Crypto and Equity Budget
# targets. This section does not create funds, portfolio weights or backtests.
# ============================================================================
def create_stage7_adaptive_budget_history_output() -> pathlib.Path:
    """Create Stage 7.6 Applied Budget history from Raw Budget output."""
    from src.portfolios import (
        PERSONALISED_CRYPTO_BUDGET_RANGES,
        smooth_crypto_budget_profile,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_budgets = pd.read_csv(
        STAGE7_PERSONALISED_BUDGET_DATA_PATH,
        parse_dates=["date"],
    )
    histories = []
    for profile in PERSONALISED_CRYPTO_BUDGET_RANGES:
        profile_budgets = raw_budgets.loc[
            raw_budgets["profile"].astype(str).eq(profile)
        ].copy()
        histories.append(smooth_crypto_budget_profile(profile_budgets))

    history = pd.concat(histories, ignore_index=True)
    history["profile"] = pd.Categorical(
        history["profile"],
        categories=list(PERSONALISED_CRYPTO_BUDGET_RANGES),
        ordered=True,
    )
    history = history.sort_values(["date", "profile"]).reset_index(drop=True)
    history.to_csv(STAGE7_ADAPTIVE_BUDGET_HISTORY_PATH, index=False)

    print("\nStage 7.6 smoothed Applied Crypto budgets:")
    print(f"  adaptive budget history: {STAGE7_ADAPTIVE_BUDGET_HISTORY_PATH}")
    for row in (
        history.groupby("profile", observed=True)
        .agg(
            no_adjustment_triggers=("no_adjustment_triggered", "sum"),
            monthly_limit_triggers=("monthly_limit_triggered", "sum"),
        )
        .reset_index()
        .itertuples(index=False)
    ):
        first_valid = history.loc[
            history["profile"].astype(str).eq(str(row.profile))
            & history["applied_budget_available"].astype(bool),
            "date",
        ].min()
        print(
            f"  {row.profile}: first valid={first_valid.date()}, "
            f"no-adjustment={int(row.no_adjustment_triggers)}, "
            f"monthly-limit={int(row.monthly_limit_triggers)}"
        )
    return STAGE7_ADAPTIVE_BUDGET_HISTORY_PATH


def create_stage7_crypto_budget_paths_figure() -> pathlib.Path:
    """Plot Fixed, Raw Adaptive and Smoothed Adaptive Crypto Budget paths."""
    raw = pd.read_csv(STAGE7_PERSONALISED_BUDGET_DATA_PATH, parse_dates=["date"])
    smoothed = pd.read_csv(STAGE7_ADAPTIVE_BUDGET_HISTORY_PATH, parse_dates=["date"])
    required_raw = {"date", "profile", "raw_crypto_budget"}
    required_smoothed = {"date", "profile", "applied_crypto_budget"}
    missing_raw = required_raw.difference(raw.columns)
    missing_smoothed = required_smoothed.difference(smoothed.columns)
    if missing_raw or missing_smoothed:
        raise ValueError(
            f"missing budget columns: raw={sorted(missing_raw)}, "
            f"smoothed={sorted(missing_smoothed)}"
        )

    profiles = [
        ("conservative", "Conservative", 0.10),
        ("balanced", "Balanced", 0.20),
        ("growth", "Growth", 0.30),
    ]
    if set(raw["profile"].astype(str)) != {profile for profile, _, _ in profiles}:
        raise ValueError("raw budgets must contain the three Stage 7 profiles")

    budgets = raw[list(required_raw)].merge(
        smoothed[list(required_smoothed)],
        on=["date", "profile"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["profile", "date"])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.6), sharey=True)
    line_specs = [
        ("fixed", "Fixed", "0.25", "--", 1.7, None, 2),
        ("raw_crypto_budget", "Raw Adaptive", "#ff7f0e", ":", 2.4, "o", 4),
        ("applied_crypto_budget", "Smoothed Adaptive", "#2ca02c", "-", 2.0, None, 3),
    ]
    summary_lines = []
    reductions = {}
    plotted_counts = {}

    for ax, (profile, title, fixed_budget) in zip(axes, profiles):
        panel = budgets.loc[budgets["profile"].astype(str).eq(profile)].sort_values("date")
        plotted_counts[profile] = int(len(panel))
        raw_budget = panel["raw_crypto_budget"].astype(float)
        smoothed_budget = panel["applied_crypto_budget"].astype(float)
        reductions[title] = raw_budget.diff().abs().max() - smoothed_budget.diff().abs().max()
        summary_lines.append(
            f"\u2022 {title}: Raw {raw_budget.min():.1%}\u2013{raw_budget.max():.1%}; "
            f"Smoothed {smoothed_budget.min():.1%}\u2013{smoothed_budget.max():.1%}."
        )

        for column, label, color, linestyle, linewidth, marker, zorder in line_specs:
            values = (
                np.full(len(panel), fixed_budget)
                if column == "fixed"
                else panel[column].astype(float)
            )
            plot_kwargs = {
                "label": label,
                "color": color,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "zorder": zorder,
            }
            if marker:
                plot_kwargs.update({"marker": marker, "markersize": 3.2, "markevery": 4})
            ax.plot(
                panel["date"],
                values,
                **plot_kwargs,
            )
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.02)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    largest_profile = max(reductions, key=reductions.get)
    summary_lines.append(
        "\u2022 Smoothing reduced abrupt monthly Budget movements most for "
        f"{largest_profile} and preserved the profile ordering: "
        "Conservative < Balanced < Growth."
    )

    axes[0].set_ylabel("Crypto Budget (%)")
    axes[-1].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        loc="lower right",
    )
    axes[0].set_ylim(0.0, 0.33)
    fig.suptitle("Stage 7 Crypto Budget Paths: Fixed vs Raw vs Smoothed Adaptive", y=0.97)
    fig.add_artist(
        plt.Line2D([0.05, 0.95], [0.255, 0.255], transform=fig.transFigure, color="0.75", lw=0.8)
    )
    fig.text(0.055, 0.225, "Summary", ha="left", va="top", fontsize=11, weight="bold")
    fig.text(
        0.07,
        0.19,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=9.5,
        linespacing=1.5,
    )
    fig.tight_layout(rect=[0, 0.28, 1, 0.92])
    output_path = FIGURES_DIR / "stage7_crypto_budget_paths.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_stage7_net_risk_metrics_fixed_vs_adaptive_figure() -> pathlib.Path:
    """Plot Stage 7 net risk metrics for Fixed versus Adaptive funds."""
    metrics = pd.read_csv(STAGE7_GROSS_VS_NET_METRICS_PATH)
    expected_funds = {
        "conservative_fixed",
        "conservative_adaptive",
        "balanced_fixed",
        "balanced_adaptive",
        "growth_fixed",
        "growth_adaptive",
    }
    required_columns = {
        "fund",
        "profile",
        "fund_type",
        "net_annualised_volatility",
        "net_maximum_drawdown",
        "net_realised_95_cvar",
    }
    missing_columns = required_columns.difference(metrics.columns)
    missing_funds = expected_funds.difference(set(metrics["fund"].astype(str)))
    if missing_columns or missing_funds:
        raise ValueError(
            f"missing columns={sorted(missing_columns)}, "
            f"missing funds={sorted(missing_funds)}"
        )

    profiles = ["conservative", "balanced", "growth"]
    profile_labels = ["Conservative", "Balanced", "Growth"]
    metric_specs = [
        ("net_annualised_volatility", "Net Annualised Volatility"),
        ("net_maximum_drawdown", "Net Maximum Drawdown"),
        ("net_realised_95_cvar", "Net Realised 95% CVaR"),
    ]
    fixed = metrics.loc[metrics["fund_type"].astype(str).eq("fixed")].set_index("profile")
    adaptive = metrics.loc[
        metrics["fund_type"].astype(str).eq("adaptive")
    ].set_index("profile")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.5))
    x = np.arange(len(profiles))
    width = 0.36
    colors = {"Fixed": "#1f77b4", "Adaptive": "#d62728"}

    for ax, (column, title) in zip(axes, metric_specs):
        fixed_values = fixed.loc[profiles, column].astype(float).to_numpy()
        adaptive_values = adaptive.loc[profiles, column].astype(float).to_numpy()
        ax.bar(x - width / 2, fixed_values, width, label="Fixed", color=colors["Fixed"])
        ax.bar(
            x + width / 2,
            adaptive_values,
            width,
            label="Adaptive",
            color=colors["Adaptive"],
        )
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(x, profile_labels)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Percentage")
    handles, labels = axes[0].get_legend_handles_labels()

    vol_improvement = (
        fixed.loc[profiles, "net_annualised_volatility"].astype(float)
        - adaptive.loc[profiles, "net_annualised_volatility"].astype(float)
    )
    drawdown_improvement = (
        adaptive.loc[profiles, "net_maximum_drawdown"].astype(float)
        - fixed.loc[profiles, "net_maximum_drawdown"].astype(float)
    )
    cvar_improvement = (
        adaptive.loc[profiles, "net_realised_95_cvar"].astype(float)
        - fixed.loc[profiles, "net_realised_95_cvar"].astype(float)
    )
    largest_profile = (
        vol_improvement + drawdown_improvement + cvar_improvement
    ).idxmax()
    largest_label = profile_labels[profiles.index(largest_profile)]
    summary_lines = [
        "\u2022 Adaptive had lower volatility for all profiles "
        f"(Growth: Fixed {fixed.loc['growth', 'net_annualised_volatility']:.2%} "
        f"vs Adaptive {adaptive.loc['growth', 'net_annualised_volatility']:.2%}).",
        "\u2022 Adaptive had smaller maximum drawdown for all profiles "
        f"(Growth: Fixed {fixed.loc['growth', 'net_maximum_drawdown']:.2%} "
        f"vs Adaptive {adaptive.loc['growth', 'net_maximum_drawdown']:.2%}).",
        "\u2022 Adaptive had better, less negative 95% CVaR for all profiles "
        f"(Growth: Fixed {fixed.loc['growth', 'net_realised_95_cvar']:.2%} "
        f"vs Adaptive {adaptive.loc['growth', 'net_realised_95_cvar']:.2%}).",
        "\u2022 Largest overall downside-risk improvement was "
        f"{largest_label}; Adaptive reduced downside risk across all three profiles, "
        "although earlier results showed lower terminal wealth.",
    ]

    fig.suptitle("Stage 7 Net Risk Comparison: Fixed vs Adaptive Funds", y=0.97)
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2)
    fig.add_artist(
        plt.Line2D([0.05, 0.95], [0.255, 0.255], transform=fig.transFigure, color="0.75", lw=0.8)
    )
    fig.text(0.055, 0.225, "Summary", ha="left", va="top", fontsize=11, weight="bold")
    fig.text(
        0.07,
        0.19,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=9.5,
        linespacing=1.45,
    )
    fig.tight_layout(rect=[0, 0.28, 1, 0.88])
    output_path = FIGURES_DIR / "stage7_net_risk_metrics_fixed_vs_adaptive.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================================
# STAGE 7.5: SIX OFFICIAL COMBINED STOCK-CRYPTO FUNDS
# Purpose:
# Run the six official Combined funds using separate Equity and Crypto
# Min-CVaR sleeves scaled by fixed or adaptive target budgets.
# ============================================================================
def create_stage7_official_fund_outputs(
    combined_returns: pd.DataFrame,
) -> dict[str, pathlib.Path]:
    """Create Stage 7.5 Budget-based fund returns, weights, turnover and log."""
    from src.portfolios import run_stage7_two_sleeve_backtest

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not STAGE7_ADAPTIVE_BUDGET_HISTORY_PATH.exists():
        create_stage7_adaptive_budget_history_output()

    adaptive_budgets = pd.read_csv(
        STAGE7_ADAPTIVE_BUDGET_HISTORY_PATH,
        parse_dates=["date"],
    )
    crypto_assets = sorted(_crypto_ticker_set())
    equity_assets = [
        asset for asset in combined_returns.columns if asset not in set(crypto_assets)
    ]
    outputs = run_stage7_two_sleeve_backtest(
        combined_returns,
        equity_assets=equity_assets,
        crypto_assets=crypto_assets,
        adaptive_budget_history=adaptive_budgets,
    )

    outputs["returns"].sort_values(["fund", "date"]).to_csv(
        STAGE7_FUND_RETURNS_PATH,
        index=False,
    )
    outputs["weights"].sort_values(["fund", "effective_date", "ticker"]).to_csv(
        STAGE7_FUND_WEIGHTS_PATH,
        index=False,
    )
    outputs["turnover"].sort_values(["fund", "effective_date"]).to_csv(
        STAGE7_FUND_TURNOVER_PATH,
        index=False,
    )
    outputs["rebalance_log"].sort_values(["fund", "effective_date"]).to_csv(
        STAGE7_REBALANCE_LOG_PATH,
        index=False,
    )

    for obsolete_path in [
        DATA_DIR / "stage7.5_fund_returns.csv",
        DATA_DIR / "stage7.5_fund_weights.csv",
        DATA_DIR / "stage7.5_fund_turnover.csv",
        DATA_DIR / "stage7.5_rebalance_log.csv",
    ]:
        obsolete_path.unlink(missing_ok=True)

    log = outputs["rebalance_log"]
    print("\nStage 7.5 official two-sleeve Budget funds:")
    print(
        "  common decision dates: "
        f"{pd.to_datetime(log['decision_date']).min().date()} to "
        f"{pd.to_datetime(log['decision_date']).max().date()}"
    )
    print(f"  fund returns: {STAGE7_FUND_RETURNS_PATH}")
    print(f"  fund weights: {STAGE7_FUND_WEIGHTS_PATH}")
    print(f"  fund turnover: {STAGE7_FUND_TURNOVER_PATH}")
    print(f"  rebalance log: {STAGE7_REBALANCE_LOG_PATH}")
    return {
        "returns": STAGE7_FUND_RETURNS_PATH,
        "weights": STAGE7_FUND_WEIGHTS_PATH,
        "turnover": STAGE7_FUND_TURNOVER_PATH,
        "rebalance_log": STAGE7_REBALANCE_LOG_PATH,
    }


# ============================================================================
# STAGE 7.7: RAW, APPLIED, TARGET AND MONTHLY PRE-TRADE ALLOCATION HISTORY
# Purpose:
# Reuse existing Stage 7 Budget outputs, fund weights and return panel to
# distinguish Raw Budget, Applied Budget, rebalance targets and monthly
# pre-trade drifted Equity/Crypto allocations.
# ============================================================================
def create_stage7_allocation_history_outputs(
    combined_returns: pd.DataFrame,
) -> dict[str, pathlib.Path]:
    """Create the monthly Stage 7.7 allocation-history output."""
    from src.portfolios import build_stage7_allocation_history

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    required_paths = [
        STAGE7_ADAPTIVE_BUDGET_HISTORY_PATH,
        STAGE7_FUND_WEIGHTS_PATH,
        STAGE7_FUND_TURNOVER_PATH,
        STAGE7_REBALANCE_LOG_PATH,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Stage 7.7 requires existing Stage 7 outputs: "
            f"{[str(path) for path in missing_paths]}"
        )

    adaptive_budgets = pd.read_csv(
        STAGE7_ADAPTIVE_BUDGET_HISTORY_PATH,
        parse_dates=["date"],
    )
    fund_weights = pd.read_csv(
        STAGE7_FUND_WEIGHTS_PATH,
        parse_dates=["decision_date", "effective_date"],
    )
    fund_turnover = pd.read_csv(
        STAGE7_FUND_TURNOVER_PATH,
        parse_dates=["decision_date", "effective_date"],
    )
    rebalance_log = pd.read_csv(
        STAGE7_REBALANCE_LOG_PATH,
        parse_dates=["decision_date", "effective_date"],
    )
    crypto_assets = sorted(_crypto_ticker_set())

    monthly_history = build_stage7_allocation_history(
        rebalance_log=rebalance_log,
        fund_weights=fund_weights,
        fund_turnover=fund_turnover,
        adaptive_budget_history=adaptive_budgets,
        returns=combined_returns,
        crypto_assets=crypto_assets,
    )

    monthly_history.to_csv(STAGE7_ALLOCATION_HISTORY_PATH, index=False)

    print("\nStage 7.7 monthly allocation history:")
    print(f"  monthly allocation history: {STAGE7_ALLOCATION_HISTORY_PATH}")
    return {"monthly": STAGE7_ALLOCATION_HISTORY_PATH}


# ============================================================================
# STAGE 7.8: AUTOMATIC EXPLAINABLE REBALANCE NARRATIVES
# Purpose:
# Generate deterministic technical and user-facing rebalance explanations from
# the existing monthly Stage 7.7 allocation history.
# ============================================================================
def create_stage7_rebalance_explanations_output() -> pathlib.Path:
    """Create the Stage 7.8 deterministic rebalance-explanation output."""
    from src.portfolios import build_stage7_rebalance_explanations

    allocation_path = STAGE7_ALLOCATION_HISTORY_PATH
    if not allocation_path.exists() and STAGE7_ALLOCATION_HISTORY_LEGACY_PATH.exists():
        allocation_path = STAGE7_ALLOCATION_HISTORY_LEGACY_PATH
    if not allocation_path.exists():
        raise FileNotFoundError(
            "Stage 7.8 requires existing monthly allocation history: "
            f"{STAGE7_ALLOCATION_HISTORY_PATH}"
        )

    allocation_history = pd.read_csv(
        allocation_path,
        parse_dates=["decision_date", "effective_date"],
    )
    explanations = build_stage7_rebalance_explanations(allocation_history)
    explanations.to_csv(STAGE7_REBALANCE_EXPLANATIONS_PATH, index=False)

    print("\nStage 7.8 rebalance explanations:")
    print(f"  source allocation history: {allocation_path}")
    print(f"  rebalance explanations: {STAGE7_REBALANCE_EXPLANATIONS_PATH}")
    return STAGE7_REBALANCE_EXPLANATIONS_PATH


# ============================================================================
# STAGE 7.9-7.10: FIXED/ADAPTIVE PERFORMANCE AND PERSONALISATION VALIDATION
# Purpose:
# Compare the six official Stage 7 funds without rerunning optimisation.
# ============================================================================
def create_stage7_performance_comparison_outputs() -> dict[str, pathlib.Path]:
    """Create Stage 7.9 and Stage 7.10 performance comparison tables."""
    from src.portfolios import (
        build_stage7_comparison_tables,
        calculate_stage7_fund_performance_metrics,
    )

    allocation_path = STAGE7_ALLOCATION_HISTORY_PATH
    if not allocation_path.exists() and STAGE7_ALLOCATION_HISTORY_LEGACY_PATH.exists():
        allocation_path = STAGE7_ALLOCATION_HISTORY_LEGACY_PATH
    required_paths = [
        STAGE7_FUND_RETURNS_PATH,
        STAGE7_FUND_TURNOVER_PATH,
        allocation_path,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Stage 7.9-7.10 requires existing Stage 7 outputs: "
            f"{[str(path) for path in missing_paths]}"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    fund_returns = pd.read_csv(STAGE7_FUND_RETURNS_PATH, parse_dates=["date"])
    fund_turnover = pd.read_csv(
        STAGE7_FUND_TURNOVER_PATH,
        parse_dates=["decision_date", "effective_date"],
    )
    allocation_history = pd.read_csv(
        allocation_path,
        parse_dates=["decision_date", "effective_date"],
    )

    performance = calculate_stage7_fund_performance_metrics(
        fund_returns=fund_returns,
        fund_turnover=fund_turnover,
        allocation_history=allocation_history,
        periods_per_year=252,
    )
    fixed_vs_adaptive, personalisation = build_stage7_comparison_tables(performance)

    performance.to_csv(STAGE7_PERFORMANCE_METRICS_PATH, index=False)
    fixed_vs_adaptive.to_csv(STAGE7_FIXED_VS_ADAPTIVE_PATH, index=False)
    personalisation.to_csv(STAGE7_PERSONALISATION_COMPARISON_PATH, index=False)

    print("\nStage 7.9-7.10 performance comparison tables:")
    print(f"  performance metrics: {STAGE7_PERFORMANCE_METRICS_PATH}")
    print(f"  Fixed versus Adaptive: {STAGE7_FIXED_VS_ADAPTIVE_PATH}")
    print(f"  personalisation comparison: {STAGE7_PERSONALISATION_COMPARISON_PATH}")
    return {
        "performance": STAGE7_PERFORMANCE_METRICS_PATH,
        "fixed_vs_adaptive": STAGE7_FIXED_VS_ADAPTIVE_PATH,
        "personalisation": STAGE7_PERSONALISATION_COMPARISON_PATH,
    }


# ============================================================================
# STAGE 7.13: TRANSACTION COSTS AND GROSS-VERSUS-NET PERFORMANCE
# Purpose:
# Apply a 10 bp proportional cost to saved Stage 7 turnover and compare gross
# and net performance without rerunning Stage 7.5 optimisation.
# ============================================================================
def create_stage7_transaction_cost_outputs() -> dict[str, pathlib.Path]:
    """Create Stage 7.13 net returns and gross-versus-net metrics."""
    from src.portfolios import (
        apply_stage7_transaction_costs,
        calculate_stage7_fund_performance_metrics,
    )

    allocation_path = STAGE7_ALLOCATION_HISTORY_PATH
    if not allocation_path.exists() and STAGE7_ALLOCATION_HISTORY_LEGACY_PATH.exists():
        allocation_path = STAGE7_ALLOCATION_HISTORY_LEGACY_PATH
    required_paths = [
        STAGE7_FUND_RETURNS_PATH,
        STAGE7_FUND_TURNOVER_PATH,
        allocation_path,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Stage 7.13 requires existing Stage 7 outputs: "
            f"{[str(path) for path in missing_paths]}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    fund_returns = pd.read_csv(STAGE7_FUND_RETURNS_PATH, parse_dates=["date"])
    fund_turnover = pd.read_csv(
        STAGE7_FUND_TURNOVER_PATH,
        parse_dates=["decision_date", "effective_date"],
    )
    allocation_history = pd.read_csv(
        allocation_path,
        parse_dates=["decision_date", "effective_date"],
    )

    net_returns = apply_stage7_transaction_costs(
        fund_returns=fund_returns,
        fund_turnover=fund_turnover,
        transaction_cost_rate=STAGE7_TRANSACTION_COST_RATE,
    )
    gross_metrics = calculate_stage7_fund_performance_metrics(
        fund_returns=net_returns,
        fund_turnover=fund_turnover,
        allocation_history=allocation_history,
        periods_per_year=252,
        return_column="gross_return",
        return_values_are_growth_factors=False,
    )
    net_metrics = calculate_stage7_fund_performance_metrics(
        fund_returns=net_returns,
        fund_turnover=fund_turnover,
        allocation_history=allocation_history,
        periods_per_year=252,
        return_column="net_return",
        return_values_are_growth_factors=False,
    )

    turnover_stats = fund_turnover.copy()
    turnover_stats["turnover"] = pd.to_numeric(turnover_stats["turnover"], errors="coerce")
    turnover_stats = turnover_stats.groupby("fund", as_index=False).agg(
        number_of_rebalances=("turnover", "size"),
        total_turnover=("turnover", "sum"),
        average_turnover=("turnover", "mean"),
    )
    turnover_stats["total_transaction_cost"] = (
        turnover_stats["total_turnover"] * STAGE7_TRANSACTION_COST_RATE
    )

    gross = gross_metrics.set_index("fund")
    net = net_metrics.set_index("fund")
    turnover_stats = turnover_stats.set_index("fund")
    rows = []
    for _, meta in gross_metrics[["fund", "profile", "fund_type"]].iterrows():
        fund = meta["fund"]
        rows.append(
            {
                "fund": fund,
                "profile": meta["profile"],
                "fund_type": meta["fund_type"],
                "number_of_rebalances": int(turnover_stats.loc[fund, "number_of_rebalances"]),
                "total_turnover": turnover_stats.loc[fund, "total_turnover"],
                "average_turnover": turnover_stats.loc[fund, "average_turnover"],
                "total_transaction_cost": turnover_stats.loc[
                    fund, "total_transaction_cost"
                ],
                "gross_annualised_return": gross.loc[fund, "annualised_return"],
                "net_annualised_return": net.loc[fund, "annualised_return"],
                "annualised_return_drag": (
                    net.loc[fund, "annualised_return"]
                    - gross.loc[fund, "annualised_return"]
                ),
                "gross_annualised_volatility": gross.loc[
                    fund, "annualised_volatility"
                ],
                "net_annualised_volatility": net.loc[fund, "annualised_volatility"],
                "gross_sharpe_ratio": gross.loc[fund, "sharpe_ratio"],
                "net_sharpe_ratio": net.loc[fund, "sharpe_ratio"],
                "sharpe_ratio_drag": (
                    net.loc[fund, "sharpe_ratio"] - gross.loc[fund, "sharpe_ratio"]
                ),
                "gross_maximum_drawdown": gross.loc[fund, "maximum_drawdown"],
                "net_maximum_drawdown": net.loc[fund, "maximum_drawdown"],
                "gross_realised_95_cvar": gross.loc[fund, "realised_95_cvar"],
                "net_realised_95_cvar": net.loc[fund, "realised_95_cvar"],
                "gross_worst_10_day_return": gross.loc[fund, "worst_10_day_return"],
                "net_worst_10_day_return": net.loc[fund, "worst_10_day_return"],
                "gross_ending_growth_of_1": gross.loc[fund, "ending_growth_of_1"],
                "net_ending_growth_of_1": net.loc[fund, "ending_growth_of_1"],
                "ending_wealth_drag": (
                    net.loc[fund, "ending_growth_of_1"]
                    - gross.loc[fund, "ending_growth_of_1"]
                ),
            }
        )

    gross_vs_net = pd.DataFrame(rows)
    net_returns.to_csv(STAGE7_FUND_RETURNS_NET_PATH, index=False, float_format="%.17g")
    gross_vs_net.to_csv(
        STAGE7_GROSS_VS_NET_METRICS_PATH,
        index=False,
        float_format="%.17g",
    )

    print("\nStage 7.13 transaction costs and gross-versus-net performance:")
    print(f"  net returns: {STAGE7_FUND_RETURNS_NET_PATH}")
    print(f"  gross versus net metrics: {STAGE7_GROSS_VS_NET_METRICS_PATH}")
    return {
        "net_returns": STAGE7_FUND_RETURNS_NET_PATH,
        "gross_vs_net_metrics": STAGE7_GROSS_VS_NET_METRICS_PATH,
    }


# ============================================================================
# STAGE 7.11: RAW VERSUS SMOOTHED ADAPTIVE BUDGET VALIDATION
# Purpose:
# Compare diagnostic Raw Adaptive funds against official Smoothed Adaptive
# funds using saved Stage 7 sleeve weights and no optimiser reruns.
# ============================================================================
def create_stage7_smoothing_validation_outputs(
    combined_returns: pd.DataFrame,
) -> dict[str, pathlib.Path]:
    """Create Stage 7.11 Raw Adaptive results and smoothing comparison."""
    from src.portfolios import (
        build_stage7_raw_adaptive_results,
        build_stage7_smoothing_comparison,
    )

    required_paths = [
        STAGE7_PERSONALISED_BUDGET_DATA_PATH,
        STAGE7_ADAPTIVE_BUDGET_HISTORY_PATH,
        STAGE7_FUND_WEIGHTS_PATH,
        STAGE7_FUND_RETURNS_NET_PATH,
        STAGE7_FUND_TURNOVER_PATH,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Stage 7.11 requires existing Stage 7 outputs: "
            f"{[str(path) for path in missing_paths]}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    raw_budgets = pd.read_csv(STAGE7_PERSONALISED_BUDGET_DATA_PATH, parse_dates=["date"])
    adaptive_budget_history = pd.read_csv(
        STAGE7_ADAPTIVE_BUDGET_HISTORY_PATH,
        parse_dates=["date"],
    )
    fund_weights = pd.read_csv(
        STAGE7_FUND_WEIGHTS_PATH,
        parse_dates=["decision_date", "effective_date"],
    )
    smoothed_net_returns = pd.read_csv(STAGE7_FUND_RETURNS_NET_PATH, parse_dates=["date"])
    smoothed_turnover = pd.read_csv(
        STAGE7_FUND_TURNOVER_PATH,
        parse_dates=["decision_date", "effective_date"],
    )

    raw_results = build_stage7_raw_adaptive_results(
        fund_weights=fund_weights,
        raw_budgets=raw_budgets,
        smoothed_net_returns=smoothed_net_returns,
        returns=combined_returns,
        transaction_cost_rate=STAGE7_TRANSACTION_COST_RATE,
    )
    smoothing_comparison = build_stage7_smoothing_comparison(
        raw_results=raw_results,
        smoothed_net_returns=smoothed_net_returns,
        adaptive_budget_history=adaptive_budget_history,
        smoothed_turnover=smoothed_turnover,
        transaction_cost_rate=STAGE7_TRANSACTION_COST_RATE,
        periods_per_year=252,
    )

    raw_results.to_csv(
        STAGE7_RAW_ADAPTIVE_RESULTS_PATH,
        index=False,
        float_format="%.17g",
    )
    smoothing_comparison.to_csv(
        STAGE7_SMOOTHING_COMPARISON_PATH,
        index=False,
        float_format="%.17g",
    )

    print("\nStage 7.11 Raw versus Smoothed Adaptive Budget validation:")
    print(f"  Raw Adaptive results: {STAGE7_RAW_ADAPTIVE_RESULTS_PATH}")
    print(f"  smoothing comparison: {STAGE7_SMOOTHING_COMPARISON_PATH}")
    return {
        "raw_adaptive_results": STAGE7_RAW_ADAPTIVE_RESULTS_PATH,
        "smoothing_comparison": STAGE7_SMOOTHING_COMPARISON_PATH,
    }


def create_stage7_net_growth_fixed_vs_adaptive_figure() -> pathlib.Path:
    """Plot Stage 7 Fixed versus Adaptive net Growth of $1 by profile."""
    required_funds = {
        "conservative_fixed",
        "conservative_adaptive",
        "balanced_fixed",
        "balanced_adaptive",
        "growth_fixed",
        "growth_adaptive",
    }
    returns = pd.read_csv(STAGE7_FUND_RETURNS_NET_PATH, parse_dates=["date"])
    missing_funds = required_funds.difference(set(returns["fund"].astype(str)))
    if missing_funds:
        raise ValueError(f"missing Stage 7 funds: {sorted(missing_funds)}")

    growth_column = "net_growth_of_1"
    if growth_column not in returns.columns:
        returns = returns.sort_values(["fund", "date"]).reset_index(drop=True)
        returns[growth_column] = returns.groupby("fund")["net_return"].transform(
            lambda series: (1.0 + series.astype(float)).cumprod()
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.4), sharey=True)
    profile_specs = [
        ("conservative", "Conservative"),
        ("balanced", "Balanced"),
        ("growth", "Growth"),
    ]
    line_specs = [("fixed", "Fixed", "#1f77b4"), ("adaptive", "Adaptive", "#d62728")]
    plotted_values = []

    for ax, (profile, title) in zip(axes, profile_specs):
        for fund_type, label, color in line_specs:
            fund = f"{profile}_{fund_type}"
            panel = returns.loc[returns["fund"].astype(str).eq(fund)].sort_values("date")
            ax.plot(
                panel["date"],
                panel[growth_column].astype(float),
                label=label,
                color=color,
                linewidth=1.8,
            )
            plotted_values.extend(panel[growth_column].astype(float).tolist())
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"${value:,.2f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())

    axes[0].set_ylabel("Net Growth of $1")
    axes[-1].legend(frameon=False, loc="lower right")
    axes[0].set_ylim(min(plotted_values) * 0.96, max(plotted_values) * 1.04)
    fig.suptitle("Stage 7 Net Growth of $1: Fixed vs Adaptive Funds", y=0.97)
    ending_values = {}
    for profile, title in profile_specs:
        fixed_panel = returns.loc[
            returns["fund"].astype(str).eq(f"{profile}_fixed")
        ].sort_values("date")
        adaptive_panel = returns.loc[
            returns["fund"].astype(str).eq(f"{profile}_adaptive")
        ].sort_values("date")
        fixed_ending = float(fixed_panel[growth_column].iloc[-1])
        adaptive_ending = float(adaptive_panel[growth_column].iloc[-1])
        ending_values[title] = (fixed_ending, adaptive_ending)
    largest_gap_profile = max(
        ending_values,
        key=lambda label: ending_values[label][0] - ending_values[label][1],
    )
    summary_lines = [
        f"\u2022 {label}: Fixed {fixed:.3f} vs Adaptive {adaptive:.3f}."
        for label, (fixed, adaptive) in ending_values.items()
    ]
    summary_lines.append(
        "\u2022 Adaptive finished below Fixed in all three profiles after transaction "
        f"costs; the largest ending-wealth gap was in {largest_gap_profile}."
    )
    fig.add_artist(
        plt.Line2D([0.05, 0.95], [0.255, 0.255], transform=fig.transFigure, color="0.75", lw=0.8)
    )
    fig.text(0.055, 0.225, "Summary", ha="left", va="top", fontsize=11, weight="bold")
    fig.text(
        0.07,
        0.19,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=9.5,
        linespacing=1.5,
    )
    fig.tight_layout(rect=[0, 0.28, 1, 0.92])
    fig.savefig(STAGE7_NET_GROWTH_FIXED_VS_ADAPTIVE_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return STAGE7_NET_GROWTH_FIXED_VS_ADAPTIVE_FIGURE_PATH


def create_stage7_net_drawdown_fixed_vs_adaptive_figure() -> pathlib.Path:
    """Plot Stage 7 Fixed versus Adaptive net drawdown by profile."""
    required_funds = {
        "conservative_fixed",
        "conservative_adaptive",
        "balanced_fixed",
        "balanced_adaptive",
        "growth_fixed",
        "growth_adaptive",
    }
    returns = pd.read_csv(STAGE7_FUND_RETURNS_NET_PATH, parse_dates=["date"])
    missing_funds = required_funds.difference(set(returns["fund"].astype(str)))
    if missing_funds:
        raise ValueError(f"missing Stage 7 funds: {sorted(missing_funds)}")

    returns = returns.sort_values(["fund", "date"]).reset_index(drop=True)
    if "drawdown" not in returns.columns:
        if "net_growth_of_1" not in returns.columns:
            returns["net_growth_of_1"] = returns.groupby("fund")["net_return"].transform(
                lambda series: (1.0 + series.astype(float)).cumprod()
            )
        returns["drawdown"] = returns.groupby("fund")["net_growth_of_1"].transform(
            lambda wealth: wealth.astype(float) / wealth.astype(float).cummax() - 1.0
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    profile_specs = [
        ("conservative", "Conservative"),
        ("balanced", "Balanced"),
        ("growth", "Growth"),
    ]
    line_specs = [("fixed", "Fixed", "#1f77b4"), ("adaptive", "Adaptive", "#d62728")]
    plotted_values = []

    for ax, (profile, title) in zip(axes, profile_specs):
        for fund_type, label, color in line_specs:
            fund = f"{profile}_{fund_type}"
            panel = returns.loc[returns["fund"].astype(str).eq(fund)].sort_values("date")
            ax.plot(
                panel["date"],
                panel["drawdown"].astype(float),
                label=label,
                color=color,
                linewidth=1.6,
            )
            plotted_values.extend(panel["drawdown"].astype(float).tolist())
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    axes[0].set_ylabel("Drawdown")
    axes[-1].legend(frameon=False, loc="lower right")
    axes[0].set_ylim(min(plotted_values) * 1.08, 0.02)
    fig.suptitle("Stage 7 Net Drawdown: Fixed vs Adaptive Funds", y=0.97)
    max_drawdowns = returns.groupby("fund")["drawdown"].min()
    summary_lines = []
    improvements = {}
    for profile, title in profile_specs:
        fixed = float(max_drawdowns[f"{profile}_fixed"])
        adaptive = float(max_drawdowns[f"{profile}_adaptive"])
        improvements[title] = adaptive - fixed
        summary_lines.append(
            f"\u2022 {title}: Fixed {fixed:.2%} vs Adaptive {adaptive:.2%}."
        )
    largest_profile = max(improvements, key=improvements.get)
    summary_lines.append(
        "\u2022 Adaptive had smaller maximum drawdown across all profiles; "
        f"largest improvement was {largest_profile} "
        f"({improvements[largest_profile] * 100:.2f} percentage points), "
        "although earlier results showed lower terminal wealth."
    )
    fig.add_artist(
        plt.Line2D([0.05, 0.95], [0.255, 0.255], transform=fig.transFigure, color="0.75", lw=0.8)
    )
    fig.text(0.055, 0.225, "Summary", ha="left", va="top", fontsize=11, weight="bold")
    fig.text(
        0.07,
        0.19,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=9.5,
        linespacing=1.5,
    )
    fig.tight_layout(rect=[0, 0.28, 1, 0.92])
    fig.savefig(
        STAGE7_NET_DRAWDOWN_FIXED_VS_ADAPTIVE_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    return STAGE7_NET_DRAWDOWN_FIXED_VS_ADAPTIVE_FIGURE_PATH


def create_sector_sentiment_index_outputs() -> dict[str, pathlib.Path]:
    """Save the baseline sector sentiment index, summary table, and figure."""
    from src.etl import load_clean_news
    from src.sentiment import score_headlines, sector_sentiment_index

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    headlines, _checks = load_clean_news()
    sector_sentiment = sector_sentiment_index(score_headlines(headlines)).sort_values(
        ["sector", "mapped_trading_date"]
    )
    sector_sentiment.to_csv(SECTOR_SENTIMENT_INDEX_PATH, index=False)

    summary = (
        sector_sentiment.groupby("sector", as_index=False)["sector_sentiment"]
        .agg(["mean", "std", "min", "max", "count"])
        .rename(
            columns={
                "mean": "mean_sector_sentiment",
                "std": "std_sector_sentiment",
                "min": "min_sector_sentiment",
                "max": "max_sector_sentiment",
                "count": "n_trading_day_observations",
            }
        )
    )
    summary.to_csv(SECTOR_SENTIMENT_SUMMARY_PATH, index=False)

    fig, axes = plt.subplots(5, 2, figsize=(13, 12), sharex=True, sharey=True)
    for ax, (sector, group) in zip(axes.ravel(), sector_sentiment.groupby("sector", sort=True)):
        group = group.sort_values("mapped_trading_date").copy()
        group["rolling_30d"] = group["sector_sentiment"].rolling(
            30, min_periods=1
        ).mean()
        ax.plot(
            group["mapped_trading_date"],
            group["sector_sentiment"],
            color="#6baed6",
            linewidth=0.7,
            alpha=0.45,
            label="Daily",
        )
        ax.plot(
            group["mapped_trading_date"],
            group["rolling_30d"],
            color="#08519c",
            linewidth=1.6,
            label="30-day mean",
        )
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        ax.set_title(sector, fontsize=11, pad=6)
        ax.grid(True, axis="y", alpha=0.25)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Sector Sentiment Index Over Time\n"
        "Thin line = daily sentiment; bold line = 30-trading-day rolling mean",
        y=0.995,
    )
    fig.supxlabel("Trading Date", y=0.045)
    fig.supylabel("Sector Sentiment (finVADER compound)", x=0.02)
    fig.text(
        0.5,
        0.008,
        "Daily sentiment is noisy, while the 30-day mean shows clearer sector trends.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4D4D4D",
    )
    fig.tight_layout(rect=[0.03, 0.075, 1, 0.96])
    fig.savefig(SECTOR_SENTIMENT_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "data": SECTOR_SENTIMENT_INDEX_PATH,
        "summary": SECTOR_SENTIMENT_SUMMARY_PATH,
        "figure": SECTOR_SENTIMENT_FIGURE_PATH,
    }


def create_sentiment_fusion_baseline_outputs(
    equity_returns: pd.DataFrame,
) -> dict[str, pathlib.Path]:
    """Create baseline sentiment-fusion returns, metrics, and growth figure."""
    from src.etl import load_clean_news
    from src.fusion import (
        BASELINE_SENTIMENT_LOOKBACK_DAYS,
        equity_min_variance_oos_weights,
        sentiment_fusion_oos_returns,
        stock_sentiment_signal,
        stock_ticker_day_sentiment,
    )
    from src.portfolios import performance_metrics
    from src.sentiment import score_headlines

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    headlines, _checks = load_clean_news()
    stock_sentiment = stock_ticker_day_sentiment(score_headlines(headlines))
    sentiment_signal = stock_sentiment_signal(
        stock_sentiment,
        lookback_days=BASELINE_SENTIMENT_LOOKBACK_DAYS,
    )
    fusion_returns = sentiment_fusion_oos_returns(
        equity_returns=equity_returns,
        base_weights=equity_min_variance_oos_weights(),
        sentiment_signal=sentiment_signal,
        start_date="2021-01-04",
    )
    fusion_returns = fusion_returns.sort_values(["strategy", "date"]).reset_index(
        drop=True
    )
    fusion_returns.to_csv(SENTIMENT_FUSION_RETURNS_PATH, index=False)

    metric_rows = []
    for strategy, group in fusion_returns.groupby("strategy", sort=False):
        metrics = performance_metrics(
            group.sort_values("date").set_index("date")["return"],
            periods_per_year=252,
            risk_free_rate=0.0,
        )
        metric_rows.append(
            {
                "strategy": strategy,
                "sentiment_lambda": group["sentiment_lambda"].iloc[0],
                "annualised_return": metrics["annualised_return"],
                "annualised_volatility": metrics["annualised_volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "maximum_drawdown": metrics["maximum_drawdown"],
            }
        )
    metrics_table = pd.DataFrame(metric_rows)
    strategy_order = ["Base Min-Variance", "Momentum Tilt", "Contrarian Tilt"]
    metrics_table["strategy"] = pd.Categorical(
        metrics_table["strategy"],
        categories=strategy_order,
        ordered=True,
    )
    metrics_table = metrics_table.sort_values("strategy").reset_index(drop=True)
    base = metrics_table.loc[metrics_table["strategy"] == "Base Min-Variance"].iloc[0]
    for column in [
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
    ]:
        metrics_table[f"delta_{column}"] = metrics_table[column] - base[column]
    metrics_table.to_csv(SENTIMENT_FUSION_METRICS_PATH, index=False)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for strategy in strategy_order:
        group = fusion_returns.loc[fusion_returns["strategy"] == strategy].sort_values(
            "date"
        )
        growth = (1.0 + group["return"]).cumprod()
        dates = pd.concat(
            [pd.Series([group["date"].iloc[0] - pd.Timedelta(days=1)]), group["date"]],
            ignore_index=True,
        )
        values = pd.concat([pd.Series([1.0]), growth], ignore_index=True)
        ax.plot(dates, values, label=strategy, linewidth=1.8)

    ax.set_title(
        "Sentiment Fusion: Base vs Naive Sentiment Tilts\n"
        "Equity Minimum-Variance Fund, OOS 2021-2023, before transaction costs",
        pad=16,
    )
    ax.set_xlabel("Trading Date")
    ax.set_ylabel("Growth of $1")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"${value:,.2f}"))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SENTIMENT_FUSION_GROWTH_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "returns": SENTIMENT_FUSION_RETURNS_PATH,
        "metrics": SENTIMENT_FUSION_METRICS_PATH,
        "figure": SENTIMENT_FUSION_GROWTH_FIGURE_PATH,
    }


def create_sector_neutral_fusion_outputs(
    equity_returns: pd.DataFrame,
) -> dict[str, pathlib.Path]:
    """Create sector-neutral sentiment-fusion weights, returns, metrics, and growth figure."""
    from src.etl import load_clean_news
    from src.fusion import (
        BASELINE_SENTIMENT_LOOKBACK_DAYS,
        equity_min_variance_oos_weights,
        sector_neutral_fusion_oos_returns,
        stock_sentiment_signal,
        stock_ticker_day_sentiment,
    )
    from src.portfolios import performance_metrics
    from src.sentiment import score_headlines

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    headlines, _checks = load_clean_news()
    stock_sentiment = stock_ticker_day_sentiment(score_headlines(headlines))
    sentiment_signal = stock_sentiment_signal(
        stock_sentiment,
        lookback_days=BASELINE_SENTIMENT_LOOKBACK_DAYS,
    )
    fusion_returns, sector_neutral_weights = sector_neutral_fusion_oos_returns(
        equity_returns=equity_returns,
        base_weights=equity_min_variance_oos_weights(),
        sentiment_signal=sentiment_signal,
        start_date="2021-01-04",
    )
    strategy_order = [
        "Base Min-Variance",
        "Naive Contrarian",
        "Sector-Neutral Contrarian",
    ]
    fusion_returns = fusion_returns.sort_values(["strategy", "date"]).reset_index(
        drop=True
    )
    fusion_returns.to_csv(SECTOR_NEUTRAL_FUSION_RETURNS_PATH, index=False)

    sector_neutral_weights[
        [
            "date",
            "ticker",
            "sector",
            "base_weight",
            "stock_sentiment_z",
            "sector_neutral_weight",
        ]
    ].to_csv(SECTOR_NEUTRAL_FUSION_WEIGHTS_PATH, index=False)

    metric_rows = []
    for strategy, group in fusion_returns.groupby("strategy", sort=False):
        metrics = performance_metrics(
            group.sort_values("date").set_index("date")["return"],
            periods_per_year=252,
            risk_free_rate=0.0,
        )
        metric_rows.append(
            {
                "Strategy": strategy,
                "Annualized Return": metrics["annualised_return"],
                "Annualized Volatility": metrics["annualised_volatility"],
                "Sharpe Ratio": metrics["sharpe_ratio"],
                "Maximum Drawdown": metrics["maximum_drawdown"],
            }
        )
    metrics_table = pd.DataFrame(metric_rows)
    metrics_table["Strategy"] = pd.Categorical(
        metrics_table["Strategy"],
        categories=strategy_order,
        ordered=True,
    )
    metrics_table = metrics_table.sort_values("Strategy").reset_index(drop=True)
    metrics_table.to_csv(SECTOR_NEUTRAL_FUSION_METRICS_PATH, index=False)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for strategy in strategy_order:
        group = fusion_returns.loc[fusion_returns["strategy"] == strategy].sort_values(
            "date"
        )
        growth = (1.0 + group["return"]).cumprod()
        dates = pd.concat(
            [pd.Series([group["date"].iloc[0] - pd.Timedelta(days=1)]), group["date"]],
            ignore_index=True,
        )
        values = pd.concat([pd.Series([1.0]), growth], ignore_index=True)
        ax.plot(dates, values, label=strategy, linewidth=1.8)

    ax.set_title(
        "Sector-Neutral Sentiment Fusion\n"
        "Equity Minimum-Variance Fund, contrarian tilt, before transaction costs",
        pad=16,
    )
    ax.set_xlabel("Trading Date")
    ax.set_ylabel("Growth of $1")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"${value:,.2f}"))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SECTOR_NEUTRAL_FUSION_GROWTH_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "weights": SECTOR_NEUTRAL_FUSION_WEIGHTS_PATH,
        "returns": SECTOR_NEUTRAL_FUSION_RETURNS_PATH,
        "metrics": SECTOR_NEUTRAL_FUSION_METRICS_PATH,
        "figure": SECTOR_NEUTRAL_FUSION_GROWTH_FIGURE_PATH,
    }


def create_tail_risk_sentiment_outputs(
    equity_returns: pd.DataFrame,
) -> dict[str, pathlib.Path]:
    """Create tail-risk-aware sector-neutral sentiment fusion outputs."""
    from src.etl import load_clean_news
    from src.fusion import (
        BASELINE_SENTIMENT_LOOKBACK_DAYS,
        TAIL_RISK_CVAR_CONFIDENCE_LEVEL,
        TAIL_RISK_CVAR_LOOKBACK_DAYS,
        equity_min_variance_oos_weights,
        stock_sentiment_signal,
        stock_ticker_day_sentiment,
        tail_risk_aware_fusion_oos_returns,
    )
    from src.portfolios import performance_metrics
    from src.sentiment import score_headlines

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    headlines, _checks = load_clean_news()
    stock_sentiment = stock_ticker_day_sentiment(score_headlines(headlines))
    sentiment_signal = stock_sentiment_signal(
        stock_sentiment,
        lookback_days=BASELINE_SENTIMENT_LOOKBACK_DAYS,
    )
    fusion_returns, risk_weights = tail_risk_aware_fusion_oos_returns(
        equity_returns=equity_returns,
        base_weights=equity_min_variance_oos_weights(),
        sentiment_signal=sentiment_signal,
        start_date="2021-01-04",
    )
    strategy_order = [
        "Base Min-Variance",
        "Naive Contrarian",
        "Sector-Neutral Contrarian",
        "Tail-Risk-Aware Sector-Neutral Contrarian",
    ]
    fusion_returns = fusion_returns.sort_values(["strategy", "date"]).reset_index(
        drop=True
    )
    fusion_returns.to_csv(TAIL_RISK_SENTIMENT_RETURNS_PATH, index=False)

    risk_weights[
        [
            "date",
            "ticker",
            "sector",
            "base_weight",
            "stock_sentiment_z",
            "stock_cvar",
            "sector_median_cvar",
            "risk_scaler",
            "tail_risk_adjusted_sentiment_z",
            "final_weight",
        ]
    ].to_csv(TAIL_RISK_SENTIMENT_WEIGHTS_PATH, index=False)

    metric_rows = []
    for strategy, group in fusion_returns.groupby("strategy", sort=False):
        metrics = performance_metrics(
            group.sort_values("date").set_index("date")["return"],
            periods_per_year=252,
            risk_free_rate=0.0,
        )
        metric_rows.append(
            {
                "Strategy": strategy,
                "Annualized Return": metrics["annualised_return"],
                "Annualized Volatility": metrics["annualised_volatility"],
                "Sharpe Ratio": metrics["sharpe_ratio"],
                "Maximum Drawdown": metrics["maximum_drawdown"],
            }
        )
    metrics_table = pd.DataFrame(metric_rows)
    metrics_table["Strategy"] = pd.Categorical(
        metrics_table["Strategy"],
        categories=strategy_order,
        ordered=True,
    )
    metrics_table = metrics_table.sort_values("Strategy").reset_index(drop=True)
    metrics_table.to_csv(TAIL_RISK_SENTIMENT_METRICS_PATH, index=False)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for strategy in strategy_order:
        group = fusion_returns.loc[fusion_returns["strategy"] == strategy].sort_values(
            "date"
        )
        growth = (1.0 + group["return"]).cumprod()
        dates = pd.concat(
            [pd.Series([group["date"].iloc[0] - pd.Timedelta(days=1)]), group["date"]],
            ignore_index=True,
        )
        values = pd.concat([pd.Series([1.0]), growth], ignore_index=True)
        ax.plot(dates, values, label=strategy, linewidth=1.8)

    ax.set_title(
        "Tail-Risk-Aware Sector-Neutral Sentiment Fusion\n"
        "252-day historical 95% CVaR scaler, before transaction costs",
        pad=16,
    )
    ax.set_xlabel("Trading Date")
    ax.set_ylabel("Growth of $1")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"${value:,.2f}"))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(TAIL_RISK_SENTIMENT_GROWTH_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "weights": TAIL_RISK_SENTIMENT_WEIGHTS_PATH,
        "returns": TAIL_RISK_SENTIMENT_RETURNS_PATH,
        "metrics": TAIL_RISK_SENTIMENT_METRICS_PATH,
        "figure": TAIL_RISK_SENTIMENT_GROWTH_FIGURE_PATH,
        "cvar_lookback_days": TAIL_RISK_CVAR_LOOKBACK_DAYS,
        "cvar_confidence_level": TAIL_RISK_CVAR_CONFIDENCE_LEVEL,
    }


def create_final_sentiment_lambda_selection_outputs(
    equity_returns: pd.DataFrame,
) -> dict[str, pathlib.Path]:
    """Select final tail-risk-aware lambda on 2021-2022 and save OOS outputs."""
    from src.etl import load_clean_news
    from src.fusion import (
        BASELINE_SENTIMENT_LOOKBACK_DAYS,
        equity_min_variance_oos_weights,
        stock_sentiment_signal,
        stock_ticker_day_sentiment,
        tail_risk_aware_fusion_oos_returns,
    )
    from src.portfolios import performance_metrics
    from src.sentiment import score_headlines

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def metric_row(
        strategy: str,
        returns: pd.Series,
        sentiment_lambda: float | None = None,
    ) -> dict:
        metrics = performance_metrics(
            returns,
            periods_per_year=252,
            risk_free_rate=0.0,
        )
        row = {
            "Strategy": strategy,
            "Annualized Return": metrics["annualised_return"],
            "Annualized Volatility": metrics["annualised_volatility"],
            "Sharpe Ratio": metrics["sharpe_ratio"],
            "Maximum Drawdown": metrics["maximum_drawdown"],
        }
        if sentiment_lambda is not None:
            row["Lambda"] = sentiment_lambda
        return row

    def plot_growth(
        returns: pd.DataFrame,
        strategy_order: list[str],
        title: str,
        path: pathlib.Path,
    ) -> pathlib.Path:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for strategy in strategy_order:
            group = returns.loc[returns["strategy"] == strategy].sort_values("date")
            growth = (1.0 + group["return"]).cumprod()
            dates = pd.concat(
                [
                    pd.Series([group["date"].iloc[0] - pd.Timedelta(days=1)]),
                    group["date"],
                ],
                ignore_index=True,
            )
            values = pd.concat([pd.Series([1.0]), growth], ignore_index=True)
            ax.plot(dates, values, label=strategy, linewidth=1.8)
        ax.set_title(title, pad=16)
        ax.set_xlabel("Trading Date")
        ax.set_ylabel("Growth of $1")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"${value:,.2f}"))
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    headlines, _checks = load_clean_news()
    stock_sentiment = stock_ticker_day_sentiment(score_headlines(headlines))
    sentiment_signal = stock_sentiment_signal(
        stock_sentiment,
        lookback_days=BASELINE_SENTIMENT_LOOKBACK_DAYS,
    )
    base_weights = equity_min_variance_oos_weights()
    lambda_grid = [-1.0, -0.5, 0.0, 0.5, 1.0]

    tuning_rows = []
    full_returns_by_lambda = {}
    for candidate in lambda_grid:
        strategy = f"lambda={candidate:+.1f}"
        returns, _weights = tail_risk_aware_fusion_oos_returns(
            equity_returns=equity_returns,
            base_weights=base_weights,
            sentiment_signal=sentiment_signal,
            start_date="2021-01-04",
            sentiment_lambda=candidate,
            strategy_name=strategy,
        )
        full_returns_by_lambda[candidate] = returns
        discovery = returns.loc[
            returns["strategy"].eq(strategy)
            & returns["date"].between("2021-01-04", "2022-12-30")
        ].sort_values("date")
        row = metric_row(
            strategy,
            discovery.set_index("date")["return"],
            sentiment_lambda=candidate,
        )
        row["direction"] = (
            "Contrarian" if candidate < 0 else "Momentum" if candidate > 0 else "Base"
        )
        tuning_rows.append(row)

    tuning_table = pd.DataFrame(tuning_rows).rename(columns={"Lambda": "lambda"})
    selected_lambda = float(
        tuning_table.sort_values(["Sharpe Ratio", "lambda"], ascending=[False, True])
        .iloc[0]["lambda"]
    )
    tuning_table["selected"] = tuning_table["lambda"].eq(selected_lambda)
    tuning_table = tuning_table[
        [
            "lambda",
            "direction",
            "Annualized Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
            "selected",
        ]
    ]
    tuning_table.to_csv(FINAL_SENTIMENT_LAMBDA_TUNING_PATH, index=False)

    fixed_returns = full_returns_by_lambda[-1.0].replace(
        {
            "lambda=-1.0": "Tail-Risk-Aware Sector-Neutral Fixed"
        }
    )
    tuned_returns = full_returns_by_lambda[selected_lambda].replace(
        {
            f"lambda={selected_lambda:+.1f}": (
                "Tail-Risk-Aware Sector-Neutral Tuned"
            )
        }
    )
    comparison_2023_order = [
        "Base Min-Variance",
        "Naive Contrarian",
        "Sector-Neutral Contrarian",
        "Tail-Risk-Aware Sector-Neutral Fixed",
        "Tail-Risk-Aware Sector-Neutral Tuned",
    ]
    comparison_2023 = pd.concat(
        [
            fixed_returns.loc[
                fixed_returns["strategy"].isin(comparison_2023_order[:-1])
            ],
            tuned_returns.loc[
                tuned_returns["strategy"].eq(
                    "Tail-Risk-Aware Sector-Neutral Tuned"
                )
            ],
        ],
        ignore_index=True,
    )
    comparison_2023 = comparison_2023.loc[
        comparison_2023["date"].between("2023-01-03", "2023-12-29")
    ].sort_values(["strategy", "date"])
    comparison_2023.to_csv(FINAL_SENTIMENT_2023_RETURNS_PATH, index=False)

    lambda_by_strategy = {
        "Base Min-Variance": 0.0,
        "Naive Contrarian": -1.0,
        "Sector-Neutral Contrarian": -1.0,
        "Tail-Risk-Aware Sector-Neutral Fixed": -1.0,
        "Tail-Risk-Aware Sector-Neutral Tuned": selected_lambda,
    }
    metrics_2023 = []
    for strategy in comparison_2023_order:
        group = comparison_2023.loc[comparison_2023["strategy"] == strategy]
        metrics_2023.append(
            metric_row(
                strategy,
                group.sort_values("date").set_index("date")["return"],
                sentiment_lambda=lambda_by_strategy[strategy],
            )
        )
    metrics_2023 = pd.DataFrame(metrics_2023)[
        [
            "Strategy",
            "Lambda",
            "Annualized Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
        ]
    ]
    metrics_2023.to_csv(FINAL_SENTIMENT_2023_METRICS_PATH, index=False)
    plot_growth(
        comparison_2023,
        comparison_2023_order,
        "Final Sentiment Innovation: 2023 OOS Robustness\n"
        "Lambda selected using 2021-2022 discovery period",
        FINAL_SENTIMENT_2023_GROWTH_FIGURE_PATH,
    )

    final_strategy = "Final Tail-Risk-Aware Sector-Neutral Tuned"
    final_full = tuned_returns.replace(
        {"Tail-Risk-Aware Sector-Neutral Tuned": final_strategy}
    )
    full_order = [
        "Base Min-Variance",
        "Naive Contrarian",
        final_strategy,
    ]
    final_full = final_full.loc[final_full["strategy"].isin(full_order)].sort_values(
        ["strategy", "date"]
    )
    final_full.to_csv(FINAL_SENTIMENT_INNOVATION_RETURNS_PATH, index=False)

    full_metrics = []
    for strategy in full_order:
        group = final_full.loc[final_full["strategy"] == strategy]
        full_metrics.append(
            metric_row(
                strategy,
                group.sort_values("date").set_index("date")["return"],
                sentiment_lambda=(
                    selected_lambda
                    if strategy == final_strategy
                    else lambda_by_strategy[strategy]
                ),
            )
        )
    full_metrics = pd.DataFrame(full_metrics)[
        [
            "Strategy",
            "Lambda",
            "Annualized Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
        ]
    ]
    full_metrics.to_csv(FINAL_SENTIMENT_INNOVATION_METRICS_PATH, index=False)
    plot_final_sentiment_innovation_composite(
        final_full,
        full_metrics,
        FINAL_SENTIMENT_INNOVATION_GROWTH_FIGURE_PATH,
    )

    return {
        "tuning": FINAL_SENTIMENT_LAMBDA_TUNING_PATH,
        "returns_2023": FINAL_SENTIMENT_2023_RETURNS_PATH,
        "metrics_2023": FINAL_SENTIMENT_2023_METRICS_PATH,
        "figure_2023": FINAL_SENTIMENT_2023_GROWTH_FIGURE_PATH,
        "returns_full": FINAL_SENTIMENT_INNOVATION_RETURNS_PATH,
        "metrics_full": FINAL_SENTIMENT_INNOVATION_METRICS_PATH,
        "figure_full": FINAL_SENTIMENT_INNOVATION_GROWTH_FIGURE_PATH,
        "selected_lambda": selected_lambda,
    }


def create_final_sentiment_cost_robustness_outputs(
    equity_returns: pd.DataFrame,
) -> dict[str, pathlib.Path]:
    """Evaluate final sentiment innovation turnover and transaction-cost robustness."""
    from src.etl import load_clean_news
    from src.fusion import (
        BASELINE_SENTIMENT_LOOKBACK_DAYS,
        apply_sector_neutral_sentiment,
        apply_sentiment,
        equity_min_variance_oos_weights,
        stock_sentiment_signal,
        stock_ticker_day_sentiment,
        tail_risk_adjusted_sentiment_signal,
        target_weight_returns_and_turnover,
    )
    from src.portfolios import performance_metrics
    from src.sentiment import score_headlines

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    headlines, _checks = load_clean_news()
    stock_sentiment = stock_ticker_day_sentiment(score_headlines(headlines))
    sentiment_signal = stock_sentiment_signal(
        stock_sentiment,
        lookback_days=BASELINE_SENTIMENT_LOOKBACK_DAYS,
    )
    base_weights = equity_min_variance_oos_weights()
    risk_signal = tail_risk_adjusted_sentiment_signal(
        equity_returns,
        sentiment_signal,
        target_dates=base_weights["date"].drop_duplicates(),
    )
    target_weights = {
        "Base Min-Variance": apply_sentiment(base_weights, sentiment_signal, 0.0),
        "Naive Contrarian": apply_sentiment(base_weights, sentiment_signal, -1.0),
        "Final Tail-Risk-Aware Sector-Neutral Tuned": apply_sector_neutral_sentiment(
            base_weights,
            risk_signal,
            sentiment_lambda=-1.0,
            signal_column="tail_risk_adjusted_sentiment_z",
        ),
    }
    strategy_order = list(target_weights)

    gross_frames = []
    turnover_frames = []
    for strategy, weights in target_weights.items():
        gross, turnover = target_weight_returns_and_turnover(
            equity_returns,
            weights,
            start_date="2021-01-04",
        )
        frame = gross.rename("return").reset_index()
        frame.columns = ["date", "return"]
        frame.insert(1, "strategy", strategy)
        gross_frames.append(frame)
        turnover = turnover.copy()
        turnover.insert(1, "strategy", strategy)
        turnover_frames.append(turnover)

    gross_returns = pd.concat(gross_frames, ignore_index=True)
    turnover = pd.concat(turnover_frames, ignore_index=True).sort_values(
        ["strategy", "date"]
    )
    turnover.to_csv(FINAL_SENTIMENT_TURNOVER_PATH, index=False)

    cost_bps_grid = [0, 10, 25, 50]
    net_frames = []
    for cost_bps in cost_bps_grid:
        cost_rate = cost_bps / 10000.0
        panel = gross_returns.merge(turnover, on=["strategy", "date"], how="left")
        panel["turnover"] = panel["turnover"].fillna(0.0)
        panel["transaction_cost_bps"] = cost_bps
        panel["return_net"] = panel["return"] - panel["turnover"] * cost_rate
        net_frames.append(
            panel[["date", "strategy", "transaction_cost_bps", "return_net"]]
        )
    net_returns = pd.concat(net_frames, ignore_index=True).sort_values(
        ["strategy", "transaction_cost_bps", "date"]
    )
    net_returns.to_csv(FINAL_SENTIMENT_NET_RETURNS_PATH, index=False)

    average_turnover = turnover.groupby("strategy")["turnover"].mean()
    metric_rows = []
    for (strategy, cost_bps), group in net_returns.groupby(
        ["strategy", "transaction_cost_bps"],
        sort=False,
    ):
        metrics = performance_metrics(
            group.sort_values("date").set_index("date")["return_net"],
            periods_per_year=252,
            risk_free_rate=0.0,
        )
        metric_rows.append(
            {
                "Strategy": strategy,
                "Transaction Cost (bps)": cost_bps,
                "Average Turnover": average_turnover[strategy],
                "Annualized Return": metrics["annualised_return"],
                "Annualized Volatility": metrics["annualised_volatility"],
                "Sharpe Ratio": metrics["sharpe_ratio"],
                "Maximum Drawdown": metrics["maximum_drawdown"],
            }
        )
    metrics_table = pd.DataFrame(metric_rows)
    metrics_table["Strategy"] = pd.Categorical(
        metrics_table["Strategy"],
        categories=strategy_order,
        ordered=True,
    )
    metrics_table = metrics_table.sort_values(
        ["Strategy", "Transaction Cost (bps)"]
    ).reset_index(drop=True)
    metrics_table.to_csv(FINAL_SENTIMENT_COST_SENSITIVITY_PATH, index=False)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for strategy in strategy_order:
        panel = metrics_table.loc[metrics_table["Strategy"].eq(strategy)]
        ax.plot(
            panel["Transaction Cost (bps)"],
            panel["Sharpe Ratio"],
            marker="o",
            linewidth=1.8,
            label=strategy,
        )
    ax.set_title(
        "Final Sentiment Innovation: Transaction-Cost Sensitivity\n"
        "Sharpe ratio after turnover-based costs, OOS 2021-2023",
        pad=16,
    )
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xticks(cost_bps_grid)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        FINAL_SENTIMENT_COST_SENSITIVITY_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    net_25 = net_returns.loc[net_returns["transaction_cost_bps"].eq(25)]
    for strategy in strategy_order:
        group = net_25.loc[net_25["strategy"] == strategy].sort_values("date")
        growth = (1.0 + group["return_net"]).cumprod()
        dates = pd.concat(
            [pd.Series([group["date"].iloc[0] - pd.Timedelta(days=1)]), group["date"]],
            ignore_index=True,
        )
        values = pd.concat([pd.Series([1.0]), growth], ignore_index=True)
        ax.plot(dates, values, label=strategy, linewidth=1.8)
    ax.set_title(
        "Final Sentiment Innovation: Net Growth of $1 at 25 bps\n"
        "Turnover-based transaction costs, OOS 2021-2023",
        pad=16,
    )
    ax.set_xlabel("Trading Date")
    ax.set_ylabel("Growth of $1")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"${value:,.2f}"))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        FINAL_SENTIMENT_NET_GROWTH_25BPS_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "turnover": FINAL_SENTIMENT_TURNOVER_PATH,
        "net_returns": FINAL_SENTIMENT_NET_RETURNS_PATH,
        "cost_sensitivity": FINAL_SENTIMENT_COST_SENSITIVITY_PATH,
        "cost_sensitivity_figure": FINAL_SENTIMENT_COST_SENSITIVITY_FIGURE_PATH,
        "net_growth_25bps_figure": FINAL_SENTIMENT_NET_GROWTH_25BPS_FIGURE_PATH,
    }


def create_sentiment_fusion_remaining_figures() -> dict[str, pathlib.Path]:
    """Create drawdown and real target-weight figures for baseline fusion."""
    from src.etl import load_clean_news
    from src.fusion import (
        BASELINE_SENTIMENT_LOOKBACK_DAYS,
        FUSION_STRATEGY_LAMBDAS,
        apply_sentiment,
        equity_min_variance_oos_weights,
        stock_sentiment_signal,
        stock_ticker_day_sentiment,
    )
    from src.sentiment import score_headlines

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    strategy_order = ["Base Min-Variance", "Momentum Tilt", "Contrarian Tilt"]
    fusion_returns = pd.read_csv(SENTIMENT_FUSION_RETURNS_PATH, parse_dates=["date"])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for strategy in strategy_order:
        group = fusion_returns.loc[fusion_returns["strategy"] == strategy].sort_values(
            "date"
        )
        growth = (1.0 + group["return"]).cumprod()
        drawdown = growth / growth.cummax() - 1.0
        ax.plot(group["date"], drawdown, label=strategy, linewidth=1.6)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title(
        "Sentiment Fusion Drawdown Comparison\n"
        "Equity Minimum-Variance Fund, OOS 2021-2023, before transaction costs",
        pad=16,
    )
    ax.set_xlabel("Trading Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SENTIMENT_FUSION_DRAWDOWN_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    headlines, _checks = load_clean_news()
    stock_sentiment = stock_ticker_day_sentiment(score_headlines(headlines))
    sentiment_signal = stock_sentiment_signal(
        stock_sentiment,
        lookback_days=BASELINE_SENTIMENT_LOOKBACK_DAYS,
    )
    base_weights = equity_min_variance_oos_weights()
    weight_panels = []
    for strategy in strategy_order:
        panel = apply_sentiment(
            base_weights,
            sentiment_signal,
            sentiment_lambda=FUSION_STRATEGY_LAMBDAS[strategy],
        )
        panel.insert(0, "strategy", strategy)
        weight_panels.append(panel)
    weights = pd.concat(weight_panels, ignore_index=True)
    tilt_effect = (
        weights.loc[weights["strategy"] != "Base Min-Variance"]
        .assign(abs_delta=lambda frame: (frame["tilted_weight"] - frame["base_weight"]).abs())
        .groupby("ticker")["abs_delta"]
        .mean()
        .sort_values(ascending=False)
    )
    selected_tickers = tilt_effect.head(6).index.tolist()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True, sharey=True)
    for ax, strategy in zip(axes, strategy_order):
        panel = weights.loc[
            weights["strategy"].eq(strategy) & weights["ticker"].isin(selected_tickers)
        ]
        for ticker, group in panel.groupby("ticker", sort=True):
            ax.step(
                group["date"],
                group["tilted_weight"],
                where="post",
                label=ticker,
                linewidth=1.3,
            )
        ax.set_title(strategy, fontsize=11)
        ax.set_ylabel("Weight")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
    )
    axes[-1].set_xlabel("Rebalance Date")
    fig.suptitle(
        "Sentiment Fusion Weights Over Time\n"
        "Top six tickers by average absolute sentiment-tilt effect",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(SENTIMENT_FUSION_WEIGHTS_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "drawdown": SENTIMENT_FUSION_DRAWDOWN_FIGURE_PATH,
        "weights": SENTIMENT_FUSION_WEIGHTS_FIGURE_PATH,
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    universes, universe_summary = _build_return_universes()
    performance_rows = []
    status_rows = []
    fund_return_tables = []
    fund_weight_tables = []
    pretrade_weight_tables = []
    turnover_tables = []
    rebalance_log_tables = []

    for config in _fund_configs():
        fund_name = config["fund_name"]
        try:
            (
                performance_row,
                status_row,
                fund_returns,
                fund_weights,
                pretrade_weights,
                turnover,
                rebalance_log,
            ) = _run_one_fund(config, universes[config["asset_universe"]])
            performance_rows.append(performance_row)
            status_rows.append(status_row)
            fund_return_tables.append(fund_returns)
            fund_weight_tables.append(fund_weights)
            pretrade_weight_tables.append(pretrade_weights)
            turnover_tables.append(turnover)
            rebalance_log_tables.append(rebalance_log)
            print(f"generated {fund_name}")
        except Exception as exc:
            status_rows.append(
                {
                    "fund_name": fund_name,
                    "run_success": False,
                    "error": str(exc),
                    "daily_observation_count": np.nan,
                    "rebalance_count": np.nan,
                    "weights_sum_valid": False,
                    "non_negative_weights": False,
                    "maximum_asset_weight": np.nan,
                    "fallback_count": np.nan,
                }
            )
            print(f"failed {fund_name}: {exc}")

    performance_summary = pd.DataFrame(performance_rows)
    if not performance_summary.empty:
        performance_summary = performance_summary.sort_values(
            by=["asset_universe", "method"],
            key=lambda column: column.map(
                UNIVERSE_ORDER if column.name == "asset_universe" else METHOD_ORDER
            ),
        )
    status_summary = pd.DataFrame(status_rows)
    fund_returns = pd.concat(fund_return_tables, ignore_index=True)
    fund_weights = pd.concat(fund_weight_tables, ignore_index=True)
    pretrade_weights = pd.concat(pretrade_weight_tables, ignore_index=True).sort_values(
        ["asset_universe", "method", "fund_name", "date", "asset"]
    )
    turnover = pd.concat(turnover_tables, ignore_index=True).sort_values(
        ["asset_universe", "method", "fund_name", "date"]
    )
    rebalance_log = pd.concat(rebalance_log_tables, ignore_index=True).sort_values(
        ["asset_universe", "method", "fund_name", "effective_date"]
    )

    returns_path = DATA_DIR / "fund_returns.csv"
    weights_path = DATA_DIR / "fund_weights.csv"
    pretrade_path = DATA_DIR / "baseline_pretrade_weights.csv"
    turnover_path = DATA_DIR / "baseline_turnover.csv"
    rebalance_log_path = DATA_DIR / "baseline_rebalance_log.csv"
    performance_path = TABLES_DIR / "R1.performance_metrics.csv"
    status_path = TABLES_DIR / "baseline_fund_run_status.csv"
    fund_returns.to_csv(returns_path, index=False)
    fund_weights.to_csv(weights_path, index=False)
    pretrade_weights.to_csv(pretrade_path, index=False)
    turnover.to_csv(turnover_path, index=False)
    rebalance_log.to_csv(rebalance_log_path, index=False)
    performance_summary.to_csv(performance_path, index=False)
    status_summary.to_csv(status_path, index=False)

    # ============================================================================
    # STAGE 6.1: FINAL BASELINE PERFORMANCE METRICS TABLE
    # Purpose:
    # Create the final concise cross-fund performance table required for Part B
    # from the existing Stage 4 performance results.
    # ============================================================================
    final_performance = pd.read_csv(performance_path)
    final_performance = final_performance[
        [
            "fund_name",
            "asset_universe",
            "method",
            "annualised_return",
            "annualised_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
        ]
    ].sort_values(
        by=["asset_universe", "method"],
        key=lambda column: column.map(
            UNIVERSE_ORDER if column.name == "asset_universe" else METHOD_ORDER
        ),
    )
    final_performance.to_csv(TABLES_DIR / "performance_metrics.csv", index=False)
    growth_figure_paths = create_baseline_growth_figures()
    drawdown_figure_path = create_baseline_drawdown_small_multiples()
    combined_weights_figure_path = create_combined_weights_over_time_figure()
    sharpe_barplot_path = create_baseline_sharpe_barplot()
    risk_return_figure_path = create_baseline_risk_return_figure()
    sector_sentiment_paths = create_sector_sentiment_index_outputs()
    sentiment_fusion_paths = create_sentiment_fusion_baseline_outputs(
        universes["equity"].rename_axis("date").reset_index().melt(
            id_vars="date",
            var_name="ticker",
            value_name="return",
        )
    )
    sentiment_fusion_figure_paths = create_sentiment_fusion_remaining_figures()
    sector_neutral_fusion_paths = create_sector_neutral_fusion_outputs(
        universes["equity"].rename_axis("date").reset_index().melt(
            id_vars="date",
            var_name="ticker",
            value_name="return",
        )
    )
    tail_risk_sentiment_paths = create_tail_risk_sentiment_outputs(
        universes["equity"].rename_axis("date").reset_index().melt(
            id_vars="date",
            var_name="ticker",
            value_name="return",
        )
    )
    final_sentiment_paths = create_final_sentiment_lambda_selection_outputs(
        universes["equity"].rename_axis("date").reset_index().melt(
            id_vars="date",
            var_name="ticker",
            value_name="return",
        )
    )
    final_sentiment_cost_paths = create_final_sentiment_cost_robustness_outputs(
        universes["equity"].rename_axis("date").reset_index().melt(
            id_vars="date",
            var_name="ticker",
            value_name="return",
        )
    )
    stage7_tail_risk_paths = create_stage7_crypto_tail_risk_outputs(universes["crypto"])
    stage7_risk_score_paths = create_stage7_crypto_risk_score_outputs(
        universes["combined"]
    )
    stage7_personalised_budget_paths = (
        create_stage7_personalised_crypto_budget_outputs()
    )
    stage7_adaptive_budget_history_path = (
        create_stage7_adaptive_budget_history_output()
    )
    stage7_official_fund_paths = create_stage7_official_fund_outputs(
        universes["combined"]
    )
    stage7_allocation_history_paths = create_stage7_allocation_history_outputs(
        universes["combined"]
    )
    stage7_rebalance_explanations_path = create_stage7_rebalance_explanations_output()
    stage7_performance_comparison_paths = create_stage7_performance_comparison_outputs()
    stage7_transaction_cost_paths = create_stage7_transaction_cost_outputs()
    stage7_smoothing_validation_paths = create_stage7_smoothing_validation_outputs(
        universes["combined"]
    )

    print("\nreturn universes:")
    for universe, info in universe_summary.items():
        print(
            f"  {universe}: shape={info['shape']}, "
            f"date_range={info['start_date'].date()} to {info['end_date'].date()}, "
            f"assets={info['asset_count']}"
        )
    print(f"\nfund returns: {returns_path}")
    print(f"fund weights: {weights_path}")
    print(f"baseline pre-trade weights: {pretrade_path}")
    print(f"baseline turnover: {turnover_path}")
    print(f"baseline rebalance log: {rebalance_log_path}")
    print(f"performance metrics: {performance_path}")
    for path in growth_figure_paths:
        print(f"growth figure: {path}")
    print(f"drawdown figure: {drawdown_figure_path}")
    print(f"combined weights figure: {combined_weights_figure_path}")
    print(f"sharpe barplot: {sharpe_barplot_path}")
    print(f"risk-return figure: {risk_return_figure_path}")
    print(f"sector sentiment index: {sector_sentiment_paths['data']}")
    print(f"sector sentiment summary: {sector_sentiment_paths['summary']}")
    print(f"sector sentiment figure: {sector_sentiment_paths['figure']}")
    print(f"sentiment fusion returns: {sentiment_fusion_paths['returns']}")
    print(f"sentiment fusion metrics: {sentiment_fusion_paths['metrics']}")
    print(f"sentiment fusion growth figure: {sentiment_fusion_paths['figure']}")
    print(f"sentiment fusion drawdown figure: {sentiment_fusion_figure_paths['drawdown']}")
    print(f"sentiment fusion weights figure: {sentiment_fusion_figure_paths['weights']}")
    print(f"sector-neutral fusion weights: {sector_neutral_fusion_paths['weights']}")
    print(f"sector-neutral fusion returns: {sector_neutral_fusion_paths['returns']}")
    print(f"sector-neutral fusion metrics: {sector_neutral_fusion_paths['metrics']}")
    print(f"sector-neutral fusion figure: {sector_neutral_fusion_paths['figure']}")
    print(f"tail-risk sentiment weights: {tail_risk_sentiment_paths['weights']}")
    print(f"tail-risk sentiment returns: {tail_risk_sentiment_paths['returns']}")
    print(f"tail-risk sentiment metrics: {tail_risk_sentiment_paths['metrics']}")
    print(f"tail-risk sentiment figure: {tail_risk_sentiment_paths['figure']}")
    print(f"final sentiment lambda tuning: {final_sentiment_paths['tuning']}")
    print(f"final sentiment 2023 returns: {final_sentiment_paths['returns_2023']}")
    print(f"final sentiment 2023 metrics: {final_sentiment_paths['metrics_2023']}")
    print(f"final sentiment 2023 figure: {final_sentiment_paths['figure_2023']}")
    print(f"final sentiment full-OOS returns: {final_sentiment_paths['returns_full']}")
    print(f"final sentiment full-OOS metrics: {final_sentiment_paths['metrics_full']}")
    print(f"final sentiment full-OOS figure: {final_sentiment_paths['figure_full']}")
    print(f"final sentiment turnover: {final_sentiment_cost_paths['turnover']}")
    print(f"final sentiment net returns: {final_sentiment_cost_paths['net_returns']}")
    print(
        "final sentiment cost sensitivity: "
        f"{final_sentiment_cost_paths['cost_sensitivity']}"
    )
    print(
        "final sentiment cost sensitivity figure: "
        f"{final_sentiment_cost_paths['cost_sensitivity_figure']}"
    )
    print(
        "final sentiment 25 bps net growth figure: "
        f"{final_sentiment_cost_paths['net_growth_25bps_figure']}"
    )
    print(f"stage 7.2 tail-risk data: {stage7_tail_risk_paths['data']}")
    print(f"stage 7.2 tail-risk summary: {stage7_tail_risk_paths['summary']}")
    print(f"stage 7.2 tail-risk figure: {stage7_tail_risk_paths['figure']}")
    print(f"stage 7.3 risk-score data: {stage7_risk_score_paths['data']}")
    print(f"stage 7.3 risk-score summary: {stage7_risk_score_paths['summary']}")
    print(
        "stage 7.4 personalised raw budget data: "
        f"{stage7_personalised_budget_paths['data']}"
    )
    print(
        "stage 7.4 personalised raw budget summary: "
        f"{stage7_personalised_budget_paths['summary']}"
    )
    print(
        "stage 7.4 personalised raw budget figure: "
        f"{stage7_personalised_budget_paths['figure']}"
    )
    print(f"stage 7.6 adaptive budget history: {stage7_adaptive_budget_history_path}")
    print(f"stage 7.5 fund returns: {stage7_official_fund_paths['returns']}")
    print(f"stage 7.5 fund weights: {stage7_official_fund_paths['weights']}")
    print(f"stage 7.5 fund turnover: {stage7_official_fund_paths['turnover']}")
    print(f"stage 7.5 rebalance log: {stage7_official_fund_paths['rebalance_log']}")
    print(f"stage 7.7 monthly allocation history: {stage7_allocation_history_paths['monthly']}")
    print(f"stage 7.8 rebalance explanations: {stage7_rebalance_explanations_path}")
    print(f"stage 7.9 performance metrics: {stage7_performance_comparison_paths['performance']}")
    print(
        "stage 7.9 Fixed versus Adaptive comparison: "
        f"{stage7_performance_comparison_paths['fixed_vs_adaptive']}"
    )
    print(
        "stage 7.10 personalisation comparison: "
        f"{stage7_performance_comparison_paths['personalisation']}"
    )
    print(f"stage 7.13 net returns: {stage7_transaction_cost_paths['net_returns']}")
    print(
        "stage 7.13 gross-versus-net metrics: "
        f"{stage7_transaction_cost_paths['gross_vs_net_metrics']}"
    )
    print(
        "stage 7.11 Raw Adaptive results: "
        f"{stage7_smoothing_validation_paths['raw_adaptive_results']}"
    )
    print(
        "stage 7.11 smoothing comparison: "
        f"{stage7_smoothing_validation_paths['smoothing_comparison']}"
    )
    print(f"run status summary: {status_path}")


if __name__ == "__main__":
    main()
