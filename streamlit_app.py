"""PulseAlloc Streamlit app shell and Market Pulse page."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
SENTIMENT_INDEX_PATH = RESULTS_DIR / "data" / "sector_sentiment_index.csv"
NEWS_EXPLORER_PATH = RESULTS_DIR / "data" / "sector_news_explorer.csv"
TAIL_METRICS_PATH = RESULTS_DIR / "tables" / "tail_risk_sentiment_metrics.csv"
COST_SENSITIVITY_PATH = RESULTS_DIR / "tables" / "final_sentiment_cost_sensitivity.csv"
NET_RETURNS_PATH = RESULTS_DIR / "data" / "final_sentiment_net_returns.csv"
TAIL_WEIGHTS_PATH = RESULTS_DIR / "data" / "tail_risk_sentiment_weights.csv"
REBALANCE_EXPLANATIONS_PATH = RESULTS_DIR / "data" / "stage7_rebalance_explanations.csv"
CRYPTO_BUDGETS_PATH = RESULTS_DIR / "data" / "personalised_crypto_budgets.csv"
CRYPTO_RISK_SCORES_PATH = RESULTS_DIR / "data" / "7.3 crypto_risk_scores.csv"
PERSONALISATION_COMPARISON_PATH = (
    RESULTS_DIR / "tables" / "stage7_personalisation_comparison.csv"
)
STAGE7_FUND_RETURNS_NET_PATH = RESULTS_DIR / "data" / "stage7_fund_returns_net.csv"
STAGE7_FIXED_VS_ADAPTIVE_PATH = (
    RESULTS_DIR / "tables" / "stage7_fixed_vs_adaptive_comparison.csv"
)
FUND_RETURNS_PATH = RESULTS_DIR / "data" / "fund_returns.csv"
FUND_WEIGHTS_PATH = RESULTS_DIR / "data" / "fund_weights.csv"

NAV_ITEMS = [
    "Home",
    "Explore Funds",
    "Build Portfolio",
    "Adaptive Allocation",
    "Sentiment Signals",
]
SENTIMENT_SUBNAV = ["Market Pulse", "Sentiment Strategy"]
RANGE_OPTIONS = ["90D", "1Y", "Full History"]
COST_CONTEXT_BPS = 25

STRATEGY_LABELS = {
    "Base Min-Variance": "Base Min-Variance",
    "Naive Contrarian": "Naive Sentiment Tilt",
    "Sector-Neutral Contrarian": "Sector-Neutral",
    "Tail-Risk-Aware Sector-Neutral Contrarian": "Final Innovation",
    "Final Tail-Risk-Aware Sector-Neutral Tuned": "Final Strategy",
}

STRATEGY_COLORS = {
    "Base Min-Variance": "#64748B",
    "Naive Sentiment Tilt": "#2563EB",
    "Sector-Neutral": "#1D4ED8",
    "Final Innovation": "#14866D",
    "Final Strategy": "#14866D",
}


def apply_shell_css() -> None:
    """Apply the single PulseAlloc CSS block."""
    st.markdown(
        """
        <style>
        :root {
            --pa-navy-dark: #061E36;
            --pa-navy: #0B2A4A;
            --pa-blue: #2563EB;
            --pa-soft-blue: #EEF4FF;
            --pa-green: #14866D;
            --pa-soft-green: #EFF9F5;
            --pa-red: #C94A4A;
            --pa-text: #14253D;
            --pa-muted: #66758A;
            --pa-border: #D9E2EC;
            --pa-background: #F6F8FB;
            --pa-surface: #FFFFFF;
            --pa-radius: 12px;
        }

        .stApp {
            background: var(--pa-background);
            color: var(--pa-text);
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI",
                sans-serif;
        }

        .block-container {
            max-width: 1480px;
            padding: 1.05rem 1.8rem 1.2rem;
        }

        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stToolbarActions"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        div[data-testid="stDeployButton"],
        button[data-testid="baseButton-headerNoPadding"],
        button[data-testid="stBaseButton-headerNoPadding"],
        button[data-testid="collapsedControl"],
        [data-testid="collapsedControl"],
        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
        }

        h1, h2, h3, h4, h5, h6, p, li, label, span {
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI",
                sans-serif;
        }

        p {
            color: var(--pa-muted);
            font-size: 0.88rem;
            line-height: 1.42;
        }

        section[data-testid="stSidebar"] {
            width: 252px !important;
            background: linear-gradient(
                180deg,
                var(--pa-navy-dark) 0%,
                var(--pa-navy) 100%
            );
            border-right: 0;
        }

        section[data-testid="stSidebar"] > div {
            width: 252px !important;
            padding: 1.35rem 0.86rem 0.95rem;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
            color: rgba(255, 255, 255, 0.86) !important;
        }

        .pa-brand {
            display: flex;
            align-items: center;
            gap: 0.62rem;
            margin: 0.1rem 0 1.4rem;
            padding: 0 0.15rem;
        }

        .pa-brand-mark {
            width: 34px;
            height: 34px;
            border-radius: 11px;
            background: linear-gradient(135deg, #43B7FF 0%, #2563EB 55%,
                #14866D 100%);
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.18);
        }

        .pa-brand-name {
            color: #FFFFFF;
            font-size: 1.25rem;
            line-height: 1;
            font-weight: 820;
            letter-spacing: -0.02em;
        }

        .pa-sidebar-line {
            height: 1px;
            background: rgba(255, 255, 255, 0.16);
            margin: 0 0.15rem 1.1rem;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.16rem;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            min-height: 40px;
            border: 0;
            border-radius: var(--pa-radius);
            padding: 0.54rem 0.72rem;
            margin: 0.08rem 0;
            background: transparent;
            transition: background 140ms ease, color 140ms ease;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.075);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: var(--pa-blue);
            box-shadow: 0 9px 20px rgba(37, 99, 235, 0.25);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label p {
            color: rgba(255, 255, 255, 0.78) !important;
            font-size: 0.91rem;
            font-weight: 690;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"]
        label:has(input:checked) p {
            color: #FFFFFF !important;
            font-weight: 760;
        }

        .pa-sidebar-spacer {
            height: 28vh;
            min-height: 130px;
        }

        .pa-sidebar-footer {
            padding: 0.86rem 0.88rem;
            border-radius: var(--pa-radius);
            background: rgba(255, 255, 255, 0.065);
            border: 1px solid rgba(255, 255, 255, 0.14);
            margin: 0 0.1rem;
        }

        .pa-sidebar-footer-label {
            color: rgba(255, 255, 255, 0.54);
            font-size: 0.68rem;
            font-weight: 760;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .pa-sidebar-footer-value {
            color: rgba(255, 255, 255, 0.90);
            font-size: 0.84rem;
            font-weight: 690;
            line-height: 1.35;
            margin-bottom: 0.7rem;
        }

        .pa-sidebar-button {
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 8px;
            color: rgba(255, 255, 255, 0.78);
            text-align: center;
            padding: 0.36rem 0.4rem;
            font-size: 0.76rem;
            font-weight: 720;
            margin-bottom: 0.75rem;
        }

        .pa-sidebar-data {
            color: rgba(255, 255, 255, 0.62);
            font-size: 0.76rem;
            line-height: 1.35;
            padding: 0 0.2rem;
            margin-top: 0.9rem;
        }

        .pa-topbar {
            margin-bottom: 0.7rem;
        }

        .pa-title {
            color: var(--pa-text);
            font-size: 2rem;
            line-height: 1.03;
            font-weight: 840;
            letter-spacing: -0.024em;
            margin-bottom: 0.14rem;
        }

        .pa-subtitle {
            color: var(--pa-muted);
            font-size: 0.91rem;
            line-height: 1.25;
            font-weight: 560;
            margin-bottom: 0.36rem;
        }

        .pa-history {
            color: #4272B8;
            font-size: 0.78rem;
            font-weight: 720;
        }

        .pa-subnav-wrap {
            display: inline-flex;
            margin: 0.08rem 0 0.65rem;
        }

        div[data-testid="stRadio"] > label {
            display: none;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 0.95rem;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label {
            border: 0;
            border-radius: 0;
            background: transparent;
            padding: 0.08rem 0 0.25rem;
            min-height: 24px;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label p {
            color: var(--pa-muted) !important;
            font-size: 0.84rem;
            font-weight: 760;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            border-bottom: 2px solid var(--pa-blue);
        }

        div[data-testid="stRadio"] div[role="radiogroup"]
        label:has(input:checked) p {
            color: var(--pa-blue) !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"]
        div[role="radiogroup"] {
            gap: 0.16rem;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"]
        div[role="radiogroup"] label {
            border: 0;
            border-radius: var(--pa-radius);
            background: transparent;
            padding: 0.54rem 0.72rem;
            min-height: 40px;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"]
        div[role="radiogroup"] label:has(input:checked) {
            background: var(--pa-blue);
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"]
        div[role="radiogroup"] label p {
            color: rgba(255, 255, 255, 0.82) !important;
            font-size: 0.91rem;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"]
        div[role="radiogroup"] label:has(input:checked) p {
            color: #FFFFFF !important;
        }

        .pa-grid-card {
            background: var(--pa-surface);
            border: 1px solid var(--pa-border);
            border-radius: var(--pa-radius);
            padding: 0.95rem 1rem 0.85rem;
        }

        .pa-section-title {
            color: var(--pa-text);
            font-size: 1rem;
            line-height: 1.15;
            font-weight: 820;
            letter-spacing: -0.014em;
            margin: 0.7rem 0 0.48rem;
        }

        .pa-section-title.first {
            margin-top: 0;
        }

        .pa-label {
            color: var(--pa-text);
            font-size: 0.78rem;
            font-weight: 800;
            margin: 0 0 0.38rem;
        }

        div[data-testid="stSelectbox"] label {
            color: var(--pa-text) !important;
            font-size: 0.76rem !important;
            font-weight: 800 !important;
        }

        div[data-baseweb="select"] > div {
            background: var(--pa-surface);
            border-color: var(--pa-border);
            border-radius: 9px;
            min-height: 36px;
        }

        div[data-baseweb="select"] span {
            color: var(--pa-text);
            font-size: 0.83rem;
            font-weight: 660;
        }

        div[data-testid="stSegmentedControl"] button {
            border-color: var(--pa-border) !important;
            background: var(--pa-surface) !important;
            color: var(--pa-muted) !important;
            min-height: 32px;
        }

        div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
            border-color: var(--pa-blue) !important;
            background: var(--pa-blue) !important;
            color: #FFFFFF !important;
        }

        div[data-testid="stSegmentedControl"] button[aria-pressed="true"] p,
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"] span {
            color: #FFFFFF !important;
        }

        .pa-snapshot-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.62rem;
        }

        .pa-kpi-card {
            background: var(--pa-surface);
            border: 1px solid var(--pa-border);
            border-radius: var(--pa-radius);
            padding: 0.75rem 0.78rem;
            min-height: 96px;
        }

        .pa-kpi-label,
        .pa-mini-label {
            color: var(--pa-muted);
            font-size: 0.70rem;
            font-weight: 790;
            margin-bottom: 0.52rem;
        }

        .pa-kpi-value {
            color: var(--pa-text);
            font-size: 1.42rem;
            line-height: 1.1;
            font-weight: 840;
            letter-spacing: -0.02em;
        }

        .pa-kpi-support,
        .pa-mini-support {
            color: var(--pa-muted);
            font-size: 0.70rem;
            line-height: 1.35;
            margin-top: 0.48rem;
            font-weight: 620;
        }

        .pa-green { color: var(--pa-green) !important; }
        .pa-red { color: var(--pa-red) !important; }
        .pa-muted { color: var(--pa-muted) !important; }
        .pa-blue { color: var(--pa-blue) !important; }

        .pa-insight {
            display: grid;
            grid-template-columns: 34px 1fr;
            gap: 0.78rem;
            align-items: start;
            background: var(--pa-surface);
            border: 1px solid var(--pa-border);
            border-radius: var(--pa-radius);
            padding: 0.78rem 0.86rem;
            margin-top: 0.72rem;
        }

        .pa-icon {
            width: 28px;
            height: 28px;
            border-radius: 9px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--pa-blue);
            background: var(--pa-soft-blue);
            font-size: 0.96rem;
        }

        .pa-insight-title {
            color: var(--pa-text);
            font-size: 0.92rem;
            font-weight: 830;
            margin-bottom: 0.18rem;
        }

        .pa-insight-body {
            color: var(--pa-muted);
            font-size: 0.82rem;
            line-height: 1.34;
            font-weight: 610;
        }

        .pa-two-card-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.72rem;
        }

        .pa-market-card,
        .pa-watch-card,
        .pa-monitor-card {
            background: var(--pa-surface);
            border: 1px solid var(--pa-border);
            border-radius: var(--pa-radius);
            padding: 0.74rem 0.84rem;
        }

        .pa-market-value {
            color: var(--pa-text);
            font-size: 1.32rem;
            font-weight: 850;
            line-height: 1.1;
        }

        .pa-watch-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.52rem;
        }

        .pa-watch-value {
            color: var(--pa-text);
            font-size: 1.05rem;
            line-height: 1.12;
            font-weight: 840;
            white-space: normal;
            overflow-wrap: anywhere;
        }

        .pa-pill {
            display: inline-block;
            margin-top: 0.34rem;
            border-radius: 999px;
            padding: 0.12rem 0.4rem;
            background: var(--pa-soft-blue);
            color: var(--pa-blue);
            font-size: 0.68rem;
            font-weight: 820;
        }

        .pa-pill.green {
            background: var(--pa-soft-green);
            color: var(--pa-green);
        }

        .pa-pill.red {
            background: #FFF2F2;
            color: var(--pa-red);
        }

        .pa-range-label {
            color: var(--pa-muted);
            font-size: 0.76rem;
            font-weight: 780;
            margin: 0.1rem 0 0.18rem;
        }

        div[data-testid="stPlotlyChart"] {
            border: 0;
            background: transparent;
        }

        .pa-divider {
            height: 1px;
            background: var(--pa-border);
            margin: 0.74rem 0 0.65rem;
        }

        .pa-news-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: var(--pa-surface);
            border: 1px solid var(--pa-border);
            border-radius: 10px;
            overflow: hidden;
            font-size: 0.76rem;
        }

        .pa-news-table th {
            text-align: left;
            color: var(--pa-muted);
            font-weight: 820;
            background: #F8FAFD;
            padding: 0.48rem 0.55rem;
            border-bottom: 1px solid var(--pa-border);
        }

        .pa-news-table td {
            color: var(--pa-text);
            padding: 0.43rem 0.55rem;
            border-bottom: 1px solid #EAF0F6;
            vertical-align: top;
            line-height: 1.28;
        }

        .pa-news-table tr:last-child td {
            border-bottom: 0;
        }

        .pa-news-table th:nth-child(1),
        .pa-news-table td:nth-child(1) {
            width: 10%;
            font-weight: 760;
        }

        .pa-news-table th:nth-child(2),
        .pa-news-table td:nth-child(2) {
            width: 63%;
        }

        .pa-news-table th:nth-child(3),
        .pa-news-table td:nth-child(3) {
            width: 13%;
            text-align: right;
        }

        .pa-news-table th:nth-child(4),
        .pa-news-table td:nth-child(4) {
            width: 14%;
            font-weight: 780;
        }

        .pa-monitor-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 0.7rem;
        }

        .pa-monitor-card {
            display: grid;
            grid-template-columns: 36px 1fr;
            gap: 0.7rem;
            align-items: start;
            min-height: 72px;
        }

        .pa-monitor-title {
            color: var(--pa-text);
            font-size: 0.92rem;
            font-weight: 830;
            line-height: 1.2;
            margin-bottom: 0.16rem;
        }

        .pa-monitor-body {
            color: var(--pa-muted);
            font-size: 0.78rem;
            line-height: 1.32;
            font-weight: 610;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--pa-border);
            border-radius: var(--pa-radius);
            background: var(--pa-surface);
            margin-top: 0.62rem;
        }

        div[data-testid="stExpander"] summary p {
            color: var(--pa-text) !important;
            font-size: 0.84rem;
            font-weight: 820;
        }

        .pa-kpi-row,
        .pa-takeaway-row {
            display: grid;
            gap: 0.75rem;
        }

        .pa-kpi-row {
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin: 0.62rem 0 0.9rem;
        }

        .pa-takeaway-row {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }

        .pa-takeaway-row.compact {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .pa-signal-card,
        .pa-takeaway-card,
        .pa-small-card,
        .pa-cost-matrix {
            background: var(--pa-surface);
            border: 1px solid var(--pa-border);
            border-radius: var(--pa-radius);
            padding: 0.78rem 0.82rem;
        }

        .pa-card-title {
            color: var(--pa-text);
            font-size: 0.88rem;
            font-weight: 840;
            line-height: 1.2;
            margin-bottom: 0.28rem;
        }

        .pa-card-copy {
            color: var(--pa-muted);
            font-size: 0.72rem;
            line-height: 1.32;
            min-height: 2.1rem;
        }

        .pa-section-panel {
            background: var(--pa-surface);
            border: 1px solid var(--pa-border);
            border-radius: var(--pa-radius);
            padding: 0.88rem 0.95rem;
            margin-bottom: 0.82rem;
        }

        .pa-section-subtitle {
            color: var(--pa-muted);
            font-size: 0.76rem;
            line-height: 1.35;
            margin-top: -0.22rem;
            margin-bottom: 0.7rem;
        }

        .pa-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.72rem;
        }

        .pa-table th {
            color: var(--pa-muted);
            background: #F8FAFD;
            font-weight: 830;
            text-align: left;
            padding: 0.44rem 0.48rem;
            border-bottom: 1px solid var(--pa-border);
        }

        .pa-table td {
            color: var(--pa-text);
            padding: 0.43rem 0.48rem;
            border-bottom: 1px solid #EAF0F6;
            vertical-align: top;
        }

        .pa-table tr:last-child td {
            border-bottom: 0;
        }

        .pa-inline-insight {
            background: var(--pa-soft-green);
            border: 1px solid rgba(20, 134, 109, 0.18);
            border-radius: 10px;
            color: var(--pa-green);
            font-size: 0.76rem;
            font-weight: 690;
            line-height: 1.35;
            padding: 0.58rem 0.68rem;
            margin-top: 0.58rem;
        }

        .pa-control-panel {
            background: var(--pa-surface);
            border: 1px solid var(--pa-border);
            border-radius: var(--pa-radius);
            padding: 0.78rem 0.9rem 0.25rem;
            margin: 0.45rem 0 0.75rem;
        }

        .pa-control-title {
            color: var(--pa-text);
            font-size: 0.84rem;
            font-weight: 840;
            margin-bottom: 0.45rem;
        }

        .pa-engine-row,
        .pa-scenario-row,
        .pa-current-state-row {
            display: grid;
            gap: 0.66rem;
        }

        .pa-engine-row {
            grid-template-columns: repeat(5, minmax(0, 1fr));
            align-items: stretch;
        }

        .pa-scenario-row,
        .pa-current-state-row {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .pa-ending-stack {
            display: grid;
            gap: 0.55rem;
        }

        .pa-engine-card,
        .pa-status-card {
            background: var(--pa-surface);
            border: 1px solid var(--pa-border);
            border-radius: var(--pa-radius);
            padding: 0.74rem 0.78rem;
            min-width: 0;
        }

        .pa-engine-card.final {
            background: var(--pa-soft-green);
            border-color: rgba(20, 134, 109, 0.30);
        }

        .pa-card-badge {
            display: inline-block;
            color: var(--pa-blue);
            background: var(--pa-soft-blue);
            border-radius: 999px;
            padding: 0.12rem 0.38rem;
            font-size: 0.58rem;
            line-height: 1.2;
            font-weight: 840;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-top: 0.48rem;
        }

        .pa-engine-value {
            color: var(--pa-text);
            font-size: 1.12rem;
            line-height: 1.12;
            font-weight: 850;
            letter-spacing: -0.02em;
            margin: 0.18rem 0 0.22rem;
        }

        .pa-engine-support {
            color: var(--pa-muted);
            font-size: 0.70rem;
            line-height: 1.3;
            font-weight: 640;
        }

        .pa-context-line {
            color: var(--pa-muted);
            font-size: 0.78rem;
            font-weight: 700;
            margin: -0.38rem 0 0.62rem;
        }

        .pa-research-label {
            color: var(--pa-muted);
            font-size: 0.70rem;
            font-weight: 760;
            margin: 0.15rem 0 0.45rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .pa-note {
            color: var(--pa-muted);
            font-size: 0.72rem;
            line-height: 1.35;
            margin-top: 0.55rem;
        }

        .pa-scale-row,
        .pa-risk-marker {
            display: flex;
            justify-content: space-between;
            color: var(--pa-muted);
            font-size: 0.68rem;
            font-weight: 760;
            margin: -0.25rem 0 0.45rem;
        }

        .pa-risk-marker {
            gap: 0.4rem;
            justify-content: flex-start;
            flex-wrap: wrap;
            margin-top: 0.55rem;
        }

        .pa-pill {
            display: inline-flex;
            border: 1px solid var(--pa-border);
            border-radius: 999px;
            padding: 0.18rem 0.46rem;
            background: #F8FAFD;
            color: var(--pa-muted);
        }

        .pa-pill.active {
            color: var(--pa-blue);
            background: var(--pa-soft-blue);
            border-color: rgba(37, 99, 235, 0.24);
        }

        .pa-table tr.pa-active-regime td {
            background: #FFF7ED;
            font-weight: 800;
        }

        .pa-table th.pa-selected-cost,
        .pa-table td.pa-selected-cost {
            background: var(--pa-soft-blue);
            color: var(--pa-blue);
            font-weight: 840;
        }

        .pa-table td.pa-final-value {
            color: var(--pa-green);
            font-weight: 850;
        }

        .pa-home-snapshot,
        .pa-home-status-grid,
        .pa-home-insight-row,
        .pa-home-journey {
            display: grid;
            gap: 0.72rem;
        }

        .pa-home-snapshot {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }

        .pa-home-status-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .pa-home-insight-row {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .pa-home-journey {
            grid-template-columns: repeat(4, minmax(0, 1fr));
            align-items: stretch;
        }

        .pa-home-metric,
        .pa-home-panel,
        .pa-home-status,
        .pa-home-insight,
        .pa-home-step {
            background: var(--pa-surface);
            border: 1px solid var(--pa-border);
            border-radius: 16px;
        }

        .pa-home-metric {
            display: grid;
            grid-template-columns: 42px 1fr;
            gap: 0.72rem;
            align-items: center;
            padding: 0.78rem 0.86rem;
            min-height: 96px;
        }

        .pa-home-icon {
            width: 38px;
            height: 38px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            font-weight: 840;
        }

        .pa-home-icon.blue { color: var(--pa-blue); background: var(--pa-soft-blue); }
        .pa-home-icon.green { color: var(--pa-green); background: var(--pa-soft-green); }
        .pa-home-icon.purple { color: #6D5BD0; background: #F1EFFF; }
        .pa-home-icon.orange { color: #B45309; background: #FFF7ED; }

        .pa-home-panel {
            padding: 0.88rem 0.95rem;
            min-height: 330px;
        }

        .pa-home-status,
        .pa-home-insight,
        .pa-home-step {
            padding: 0.72rem 0.78rem;
        }

        .pa-home-status {
            min-height: 112px;
        }

        .pa-home-insight {
            min-height: 132px;
        }

        .pa-home-action {
            display: block;
            width: 100%;
            border: 1px solid rgba(37, 99, 235, 0.20);
            border-radius: 11px;
            color: var(--pa-blue);
            background: var(--pa-soft-blue);
            text-align: center;
            font-size: 0.78rem;
            font-weight: 820;
            padding: 0.55rem 0.6rem;
            margin-top: 0.72rem;
        }

        .pa-compare-selector,
        .pa-selected-funds {
            background: var(--pa-surface);
            border: 1px solid var(--pa-border);
            border-radius: 16px;
        }

        .pa-compare-selector {
            padding: 0.82rem 0.95rem 0.28rem;
            margin-bottom: 0.82rem;
        }

        .pa-compare-accent {
            height: 4px;
            border-radius: 999px;
            background: var(--pa-blue);
            margin-bottom: 0.72rem;
        }

        .pa-compare-accent.fund-2 { background: var(--pa-green); }
        .pa-compare-accent.fund-3 { background: #6D5BD0; }

        .pa-fund-card-body {
            min-height: 138px;
            display: flex;
            flex-direction: column;
        }

        .pa-fund-title {
            color: var(--pa-navy);
            font-size: 1.08rem;
            font-weight: 860;
            letter-spacing: -0.02em;
            margin-bottom: 0.08rem;
        }

        .pa-fund-subtitle {
            color: var(--pa-muted);
            font-size: 0.76rem;
            font-weight: 650;
            margin-bottom: 0.72rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .pa-fund-metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.35rem;
            margin-top: auto;
        }

        .pa-fund-mini-label {
            color: var(--pa-muted);
            font-size: 0.63rem;
            font-weight: 780;
            line-height: 1.15;
            margin-bottom: 0.12rem;
        }

        .pa-fund-mini-value {
            color: var(--pa-navy);
            font-size: 0.94rem;
            font-weight: 840;
        }

        .pa-fund-action-divider {
            height: 1px;
            background: var(--pa-border);
            margin: 0.78rem 0 0.55rem;
        }

        .pa-compare-native-card-title {
            color: var(--pa-text);
            font-size: 0.92rem;
            font-weight: 840;
            line-height: 1.2;
            margin-bottom: 0.28rem;
        }

        .pa-compare-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            overflow: hidden;
            border: 1px solid var(--pa-border);
            border-radius: 13px;
            font-size: 0.80rem;
        }

        .pa-compare-table th,
        .pa-compare-table td {
            padding: 0.58rem 0.62rem;
            border-bottom: 1px solid #E8EEF6;
            text-align: left;
            vertical-align: middle;
        }

        .pa-compare-table th {
            background: #F8FAFD;
            color: var(--pa-muted);
            font-size: 0.68rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .pa-compare-table tr:last-child td {
            border-bottom: 0;
        }

        .pa-best-cell {
            color: var(--pa-green);
            font-weight: 840;
        }

        .pa-selected-funds {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.78rem 0.9rem;
            margin-top: 0.9rem;
            flex-wrap: wrap;
        }

        .pa-fund-chip-row {
            display: inline-flex;
            gap: 0.42rem;
            flex-wrap: wrap;
            margin-left: 0.4rem;
        }

        .pa-fund-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.28rem 0.54rem;
            background: var(--pa-soft-blue);
            color: var(--pa-blue);
            font-size: 0.76rem;
            font-weight: 780;
        }

        .pa-compare-insight {
            color: var(--pa-text);
            font-size: 0.82rem;
            line-height: 1.42;
        }

        .pa-fact-identity {
            display: grid;
            grid-template-columns: 1.25fr repeat(3, minmax(0, 1fr));
            gap: 0.72rem;
            align-items: stretch;
        }

        .pa-fact-main,
        .pa-fact-mini,
        .pa-fact-insight {
            background: var(--pa-surface);
            border: 1px solid var(--pa-border);
            border-radius: 16px;
            padding: 0.82rem 0.9rem;
        }

        .pa-fact-main {
            border-color: rgba(37, 99, 235, 0.22);
            background: linear-gradient(135deg, #FFFFFF 0%, #F7FAFF 100%);
        }

        .pa-fact-name {
            color: var(--pa-navy);
            font-size: 1.12rem;
            line-height: 1.12;
            font-weight: 860;
            letter-spacing: -0.02em;
            margin-bottom: 0.3rem;
        }

        .pa-fact-summary {
            color: var(--pa-muted);
            font-size: 0.75rem;
            line-height: 1.35;
            font-weight: 640;
        }

        .pa-fact-label {
            color: var(--pa-muted);
            font-size: 0.68rem;
            font-weight: 820;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .pa-fact-value {
            color: var(--pa-text);
            font-size: 0.96rem;
            line-height: 1.18;
            font-weight: 840;
        }

        .pa-fact-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.72rem;
            margin: 0.78rem 0;
        }

        .pa-fact-section-title {
            color: var(--pa-text);
            font-size: 0.94rem;
            font-weight: 850;
            letter-spacing: -0.01em;
            margin-bottom: 0.2rem;
        }

        .pa-fact-insight {
            background: var(--pa-soft-green);
            border-color: rgba(20, 134, 109, 0.20);
        }

        .pa-fact-insight .pa-fact-value {
            color: var(--pa-green);
            font-size: 0.86rem;
            line-height: 1.35;
        }

        .pa-info-banner {
            background: var(--pa-soft-blue);
            border: 1px solid rgba(37, 99, 235, 0.18);
            border-radius: 14px;
            color: var(--pa-blue);
            font-size: 0.82rem;
            font-weight: 720;
            line-height: 1.35;
            padding: 0.72rem 0.86rem;
            margin: 0.15rem 0 0.78rem;
        }

        .pa-build-section-title {
            color: var(--pa-text);
            font-size: 1rem;
            font-weight: 860;
            letter-spacing: -0.01em;
            margin-bottom: 0.18rem;
        }

        .pa-build-section-copy {
            color: var(--pa-muted);
            font-size: 0.76rem;
            line-height: 1.32;
            font-weight: 640;
            margin-bottom: 0.7rem;
        }

        .pa-allocation-status {
            border-radius: 999px;
            display: inline-flex;
            padding: 0.22rem 0.55rem;
            font-size: 0.74rem;
            font-weight: 820;
            background: #FFF7ED;
            color: #B45309;
        }

        .pa-allocation-status.valid {
            background: var(--pa-soft-green);
            color: var(--pa-green);
        }

        .pa-profile-card {
            border: 1px solid var(--pa-border);
            border-radius: 14px;
            background: #FFFFFF;
            padding: 0.68rem 0.72rem;
            min-height: 88px;
        }

        .pa-profile-card.active {
            border-color: rgba(37, 99, 235, 0.38);
            background: var(--pa-soft-blue);
        }

        .aa-headline {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            flex-wrap: wrap;
        }

        .aa-module-title {
            color: var(--pa-text);
            font-size: 1.08rem;
            font-weight: 880;
            letter-spacing: -0.015em;
            margin-bottom: 0.12rem;
        }

        .aa-flow {
            display: grid;
            grid-template-columns: 0.9fr 34px 1.4fr 34px 0.9fr;
            gap: 0.55rem;
            align-items: stretch;
            margin-top: 0.75rem;
        }

        .aa-zone {
            border: 1px solid var(--pa-border);
            border-radius: 14px;
            background: #F8FAFD;
            padding: 0.78rem 0.82rem;
        }

        .aa-zone.engine {
            background: #FFFFFF;
            border-color: rgba(37, 99, 235, 0.24);
        }

        .aa-zone-label {
            color: var(--pa-muted);
            font-size: 0.68rem;
            font-weight: 860;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
        }

        .aa-big {
            color: var(--pa-text);
            font-size: 1.65rem;
            font-weight: 900;
            line-height: 1.05;
            letter-spacing: -0.025em;
        }

        .aa-arrow {
            align-self: center;
            justify-self: center;
            color: var(--pa-blue);
            font-size: 1.5rem;
            font-weight: 900;
        }

        .aa-signal {
            margin-bottom: 0.58rem;
        }

        .aa-rowline {
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            align-items: baseline;
        }

        .aa-meter {
            height: 7px;
            border-radius: 999px;
            overflow: hidden;
            background: #EAF0F6;
            margin-top: 0.25rem;
        }

        .aa-meter > span {
            display: block;
            height: 100%;
            border-radius: 999px;
            background: var(--pa-blue);
        }

        .aa-meter > span.green { background: var(--pa-green); }
        .aa-meter > span.amber { background: #D97706; }
        .aa-meter > span.red { background: var(--pa-red); }

        .aa-bar-row {
            margin: 0.62rem 0;
        }

        .aa-bar-label {
            display: flex;
            justify-content: space-between;
            color: var(--pa-text);
            font-size: 0.8rem;
            font-weight: 820;
            margin-bottom: 0.2rem;
        }

        .aa-bar {
            height: 9px;
            border-radius: 999px;
            background: #EAF0F6;
            overflow: hidden;
        }

        .aa-bar > span {
            display: block;
            height: 100%;
            border-radius: 999px;
            background: #64748B;
        }

        .aa-bar > span.adaptive {
            background: var(--pa-green);
        }

        .aa-evidence-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.55rem;
        }

        .aa-mini-metric {
            border: 1px solid var(--pa-border);
            border-radius: 12px;
            padding: 0.62rem 0.66rem;
            background: #FFFFFF;
        }

        .pa-profile-title {
            color: var(--pa-text);
            font-size: 0.86rem;
            font-weight: 850;
            margin-bottom: 0.16rem;
        }

        .pa-profile-copy {
            color: var(--pa-muted);
            font-size: 0.70rem;
            line-height: 1.28;
            font-weight: 640;
        }

        .pa-action-bar {
            display: grid;
            grid-template-columns: 1fr 1fr 1.55fr;
            gap: 0.8rem;
            align-items: center;
            background: #FFFFFF;
            border: 1px solid var(--pa-border);
            border-radius: 16px;
            padding: 0.78rem 0.9rem;
            margin-top: 0.85rem;
        }

        .pa-allocation-legend {
            display: grid;
            gap: 0.52rem;
            margin-top: 0.28rem;
        }

        .pa-allocation-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 0.5rem;
            color: var(--pa-text);
            font-size: 0.78rem;
            font-weight: 760;
            border-bottom: 1px solid #EAF0F6;
            padding-bottom: 0.42rem;
        }

        .pa-allocation-row:last-child {
            border-bottom: 0;
        }

        .pa-home-step {
            position: relative;
            min-height: 96px;
        }

        .pa-step-number {
            width: 26px;
            height: 26px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #FFFFFF;
            background: var(--pa-blue);
            font-size: 0.72rem;
            font-weight: 840;
            margin-bottom: 0.42rem;
        }

        @media (max-width: 1100px) {
            .pa-snapshot-grid,
            .pa-watch-grid,
            .pa-monitor-grid,
            .pa-kpi-row,
            .pa-takeaway-row,
            .pa-engine-row,
            .pa-scenario-row,
            .pa-current-state-row,
            .pa-home-snapshot,
            .pa-home-insight-row,
            .pa-home-journey {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=86_400, show_spinner="Loading saved sentiment artifacts...")
def load_sentiment_artifacts() -> dict[str, pd.DataFrame]:
    """Load saved sentiment artifacts. No modelling or scoring is run here."""
    sentiment = pd.read_csv(
        SENTIMENT_INDEX_PATH,
        parse_dates=["mapped_trading_date"],
    )
    news = pd.read_csv(
        NEWS_EXPLORER_PATH,
        parse_dates=["news_date", "mapped_trading_date"],
    )
    sentiment = sentiment.sort_values(["sector", "mapped_trading_date"])
    news = news.sort_values(["mapped_trading_date", "sector", "ticker"])
    tail_metrics = pd.read_csv(TAIL_METRICS_PATH)
    cost = pd.read_csv(COST_SENSITIVITY_PATH)
    net_returns = pd.read_csv(NET_RETURNS_PATH, parse_dates=["date"])
    weights = pd.read_csv(TAIL_WEIGHTS_PATH, parse_dates=["date"])
    rebalance = (
        pd.read_csv(
            REBALANCE_EXPLANATIONS_PATH,
            parse_dates=["decision_date", "effective_date"],
        )
        if REBALANCE_EXPLANATIONS_PATH.exists()
        else pd.DataFrame()
    )
    crypto_budgets = (
        pd.read_csv(CRYPTO_BUDGETS_PATH, parse_dates=["date"])
        if CRYPTO_BUDGETS_PATH.exists()
        else pd.DataFrame()
    )
    crypto_risk = (
        pd.read_csv(CRYPTO_RISK_SCORES_PATH, parse_dates=["date"])
        if CRYPTO_RISK_SCORES_PATH.exists()
        else pd.DataFrame()
    )
    personalisation = (
        pd.read_csv(PERSONALISATION_COMPARISON_PATH)
        if PERSONALISATION_COMPARISON_PATH.exists()
        else pd.DataFrame()
    )
    stage7_returns_net = (
        pd.read_csv(STAGE7_FUND_RETURNS_NET_PATH, parse_dates=["date"])
        if STAGE7_FUND_RETURNS_NET_PATH.exists()
        else pd.DataFrame()
    )
    fixed_vs_adaptive = (
        pd.read_csv(STAGE7_FIXED_VS_ADAPTIVE_PATH)
        if STAGE7_FIXED_VS_ADAPTIVE_PATH.exists()
        else pd.DataFrame()
    )
    fund_returns = (
        pd.read_csv(FUND_RETURNS_PATH, parse_dates=["date"])
        if FUND_RETURNS_PATH.exists()
        else pd.DataFrame()
    )
    if not fund_returns.empty:
        fund_returns = fund_returns.sort_values(["fund_name", "date"])
    fund_weights = (
        pd.read_csv(FUND_WEIGHTS_PATH, parse_dates=["date"])
        if FUND_WEIGHTS_PATH.exists()
        else pd.DataFrame()
    )
    if not fund_weights.empty:
        fund_weights = fund_weights.sort_values(["fund_name", "date", "asset"])
    return {
        "sentiment": sentiment,
        "news": news,
        "tail_metrics": tail_metrics,
        "cost": cost,
        "net_returns": net_returns,
        "weights": weights,
        "rebalance": rebalance,
        "crypto_budgets": crypto_budgets,
        "crypto_risk": crypto_risk,
        "personalisation": personalisation,
        "stage7_returns_net": stage7_returns_net,
        "fixed_vs_adaptive": fixed_vs_adaptive,
        "fund_returns": fund_returns,
        "fund_weights": fund_weights,
    }


def tone(score: float) -> str:
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


def signed_value(value: float) -> str:
    return f"{value:+.3f}"


def format_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def format_pp(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def value_class(value: float) -> str:
    if value > 0:
        return "pa-green"
    if value < 0:
        return "pa-red"
    return "pa-muted"


def direction_label(delta: float) -> tuple[str, str]:
    if delta > 0.0025:
        return "Strengthening", "pa-green"
    if delta < -0.0025:
        return "Weakening", "pa-red"
    return "Stable", "pa-muted"


def standard_strategy_name(name: str) -> str:
    return STRATEGY_LABELS.get(name, name)


def apply_plotly_theme(fig: go.Figure, height: int = 280) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": "#14253D", "size": 11},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 10},
        },
        xaxis={
            "gridcolor": "rgba(217,226,236,0.18)",
            "showline": False,
        },
        yaxis={"gridcolor": "#EAF0F6", "zeroline": False},
    )
    return fig


def allocation_donut_chart(allocation: pd.DataFrame, centre_label: str) -> go.Figure:
    colors = ["#2563EB", "#14866D", "#6D5BD0", "#F59E0B", "#64748B"]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=allocation["asset"],
                values=allocation["weight"],
                hole=0.64,
                marker={"colors": colors[: len(allocation)]},
                textinfo="none",
                hovertemplate="%{label}<br>Weight: %{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=235,
        margin={"l": 0, "r": 0, "t": 4, "b": 4},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[
            {
                "text": centre_label,
                "x": 0.5,
                "y": 0.5,
                "font": {"size": 13, "color": "#14253D"},
                "showarrow": False,
            }
        ],
    )
    return fig


def recent_windows(
    sentiment: pd.DataFrame,
    latest_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    dates = sorted(sentiment["mapped_trading_date"].dropna().unique())
    latest_index = dates.index(latest_date)
    recent_dates = dates[max(0, latest_index - 29) : latest_index + 1]
    previous_dates = dates[max(0, latest_index - 59) : max(0, latest_index - 29)]
    recent = sentiment[sentiment["mapped_trading_date"].isin(recent_dates)]
    previous = sentiment[sentiment["mapped_trading_date"].isin(previous_dates)]
    return recent, previous, recent_dates


def render_sidebar(
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    latest_date: pd.Timestamp,
) -> str:
    """Render permanent PulseAlloc navigation."""
    with st.sidebar:
        st.markdown(
            """
            <div class="pa-brand">
                <div class="pa-brand-mark"></div>
                <div class="pa-brand-name">PulseAlloc</div>
            </div>
            <div class="pa-sidebar-line"></div>
            """,
            unsafe_allow_html=True,
        )
        nav_target = st.session_state.pop("_nav_target", None)
        if nav_target in NAV_ITEMS:
            st.session_state["selected_main_page_v2"] = nav_target
        current_page = st.session_state.get("selected_main_page_v2", "Home")
        default_index = (
            NAV_ITEMS.index(current_page) if current_page in NAV_ITEMS else NAV_ITEMS.index("Home")
        )
        selected_page = st.radio(
            "Navigation",
            NAV_ITEMS,
            index=default_index,
            key="selected_main_page_v2",
            label_visibility="collapsed",
        )
        st.markdown(
            f"""
            <div class="pa-sidebar-spacer"></div>
            <div class="pa-sidebar-footer">
                <div class="pa-sidebar-footer-label">Selected Period</div>
                <div class="pa-sidebar-footer-value">
                    {period_start.date().isoformat()}<br>
                    {period_end.date().isoformat()}
                </div>
                <div class="pa-sidebar-button">Change Period</div>
                <div class="pa-sidebar-footer-label">Data as of</div>
                <div class="pa-sidebar-footer-value">
                    {latest_date.date().isoformat()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return selected_page


def render_topbar(latest_date: pd.Timestamp, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="pa-topbar">
            <div>
                <div class="pa-title">{escape(title)}</div>
                <div class="pa-subtitle">
                    {escape(subtitle)}
                </div>
                <div class="pa-history">
                    Historical data through {latest_date.date().isoformat()}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sentiment_subnav() -> str:
    selected = st.radio(
        "Sentiment section",
        SENTIMENT_SUBNAV,
        index=SENTIMENT_SUBNAV.index("Market Pulse"),
        horizontal=True,
        key="selected_sentiment_section_v2",
        label_visibility="collapsed",
    )
    return selected


def render_kpi_card(label: str, value: str, support: str, css_class: str = "") -> str:
    return (
        '<div class="pa-kpi-card">'
        f'<div class="pa-kpi-label">{escape(label)}</div>'
        f'<div class="pa-kpi-value {css_class}">{escape(value)}</div>'
        f'<div class="pa-kpi-support">{escape(support)}</div>'
        "</div>"
    )


def render_home_metric(
    icon: str,
    label: str,
    value: str,
    support: str,
    color: str = "blue",
    css_class: str = "",
) -> str:
    return (
        '<div class="pa-home-metric">'
        f'<div class="pa-home-icon {color}">{escape(icon)}</div>'
        "<div>"
        f'<div class="pa-kpi-label">{escape(label)}</div>'
        f'<div class="pa-kpi-value {css_class}">{escape(value)}</div>'
        f'<div class="pa-kpi-support">{escape(support)}</div>'
        "</div></div>"
    )


def render_home_status(label: str, value: str, support: str, css_class: str = "") -> str:
    return (
        '<div class="pa-home-status">'
        f'<div class="pa-mini-label">{escape(label)}</div>'
        f'<div class="pa-engine-value {css_class}">{escape(value)}</div>'
        f'<div class="pa-engine-support">{escape(support)}</div>'
        "</div>"
    )


def render_home_insight(
    category: str,
    headline: str,
    body: str,
    action: str,
) -> str:
    return (
        '<div class="pa-home-insight">'
        f'<div class="pa-mini-label">{escape(category)}</div>'
        f'<div class="pa-card-title">{escape(headline)}</div>'
        f'<div class="pa-card-copy">{escape(body)}</div>'
        f'<div class="pa-history">{escape(action)}</div>'
        "</div>"
    )


def home_empty_allocation() -> str:
    return (
        '<div class="pa-home-panel">'
        + section_header_html("Current Allocation")
        + '<div class="pa-small-card"><div class="pa-card-title">No portfolio built yet</div>'
        '<div class="pa-card-copy">Create a portfolio to see allocation, risk and '
        'adaptive insights here.</div></div></div>'
    )


def render_market_pair(label: str, sector: str, value: float, weakest: bool = False) -> str:
    css_class = value_class(value)
    if weakest and value > 0:
        css_class = "pa-green"
    return (
        '<div class="pa-market-card">'
        f'<div class="pa-mini-label">{escape(label)}</div>'
        f'<div class="pa-market-value {css_class}">{escape(sector)}</div>'
        f'<div class="pa-mini-support">{signed_value(value)} 30D sentiment</div>'
        "</div>"
    )


def render_watch_card(label: str, value: str, pill: str, pill_class: str = "") -> str:
    return (
        '<div class="pa-watch-card">'
        f'<div class="pa-mini-label">{escape(label)}</div>'
        f'<div class="pa-watch-value">{escape(value)}</div>'
        f'<span class="pa-pill {pill_class}">{escape(pill)}</span>'
        "</div>"
    )


def build_news_table(news: pd.DataFrame) -> str:
    if news.empty:
        rows = (
            '<tr><td colspan="4" class="pa-muted">'
            "No saved headlines match the selected filters."
            "</td></tr>"
        )
    else:
        rows = []
        for _, row in news.head(8).iterrows():
            score = float(row["sentiment_score"])
            row_tone = tone(score)
            tone_class = (
                "pa-green"
                if row_tone == "Positive"
                else "pa-red"
                if row_tone == "Negative"
                else "pa-muted"
            )
            rows.append(
                "<tr>"
                f"<td>{escape(str(row['ticker']))}</td>"
                f"<td>{escape(str(row['title']))}</td>"
                f"<td>{score:.2f}</td>"
                f'<td class="{tone_class}">{row_tone}</td>'
                "</tr>"
            )
        rows = "".join(rows)
    return (
        '<table class="pa-news-table">'
        "<thead><tr>"
        "<th>Ticker</th><th>Headline</th><th>Sentiment</th><th>Tone</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def trend_window(
    sector_history: pd.DataFrame,
    latest_date: pd.Timestamp,
    display_range: str,
) -> pd.DataFrame:
    if display_range == "90D":
        start = latest_date - pd.DateOffset(days=120)
        window = sector_history[sector_history["mapped_trading_date"] >= start]
        return window.tail(90)
    if display_range == "1Y":
        start = latest_date - pd.DateOffset(years=1)
        return sector_history[sector_history["mapped_trading_date"] >= start]
    return sector_history


def render_trend_chart(chart_data: pd.DataFrame) -> None:
    chart_data = chart_data.sort_values("mapped_trading_date").copy()
    chart_data["sentiment_30d"] = chart_data["sector_sentiment"].rolling(
        30,
        min_periods=5,
    ).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_data["mapped_trading_date"],
            y=chart_data["sector_sentiment"],
            mode="lines",
            name="Daily sentiment",
            line={"color": "rgba(98, 146, 204, 0.32)", "width": 1.3},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Daily: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_data["mapped_trading_date"],
            y=chart_data["sentiment_30d"],
            mode="lines",
            name="30-day average",
            line={"color": "#0B2A4A", "width": 2.5},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>30D avg: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#D9E2EC", line_width=1)
    fig = apply_plotly_theme(fig, height=286)
    fig.update_xaxes(title="Trading Date")
    fig.update_yaxes(title="Sentiment")
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )


def render_monitor_card(icon: str, title: str, body: str) -> str:
    return (
        '<div class="pa-monitor-card">'
        f'<div class="pa-icon">{escape(icon)}</div>'
        "<div>"
        f'<div class="pa-monitor-title">{escape(title)}</div>'
        f'<div class="pa-monitor-body">{escape(body)}</div>'
        "</div></div>"
    )


def signal_card(label: str, value: str, support: str, css_class: str = "") -> str:
    return (
        '<div class="pa-signal-card">'
        f'<div class="pa-mini-label">{escape(label)}</div>'
        f'<div class="pa-watch-value {css_class}">{escape(value)}</div>'
        f'<div class="pa-mini-support">{escape(support)}</div>'
        "</div>"
    )


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(section_header_html(title, subtitle), unsafe_allow_html=True)


def section_header_html(title: str, subtitle: str = "") -> str:
    subtitle_html = (
        f'<div class="pa-section-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    )
    return f'<div class="pa-section-title first">{escape(title)}</div>{subtitle_html}'


def prepare_strategy_data(artifacts: dict[str, pd.DataFrame]) -> dict[str, object]:
    tail_metrics = artifacts["tail_metrics"].copy()
    tail_metrics["Display"] = tail_metrics["Strategy"].map(standard_strategy_name)
    metric_lookup = tail_metrics.set_index("Display")

    cost = artifacts["cost"].copy()
    cost["Display"] = cost["Strategy"].map(standard_strategy_name)
    net = artifacts["net_returns"].copy()
    net["Display"] = net["strategy"].map(standard_strategy_name)
    weights = artifacts["weights"].copy()

    final_full = metric_lookup.loc["Final Innovation"]
    base_full = metric_lookup.loc["Base Min-Variance"]
    naive_full = metric_lookup.loc["Naive Sentiment Tilt"]

    cost_context = cost[cost["Transaction Cost (bps)"].eq(COST_CONTEXT_BPS)]
    cost_lookup = cost_context.set_index("Display")
    final_net = cost_lookup.loc["Final Strategy"]
    base_net = cost_lookup.loc["Base Min-Variance"]
    naive_net = cost_lookup.loc["Naive Sentiment Tilt"]

    latest_weights = weights[weights["date"].eq(weights["date"].max())].copy()
    latest_weights["delta"] = (
        latest_weights["final_weight"] - latest_weights["base_weight"]
    )
    current = latest_weights.loc[latest_weights["delta"].abs().idxmax()]

    cvar_low, cvar_high = weights["stock_cvar"].quantile([1 / 3, 2 / 3])
    tail_status = (
        "Low"
        if current["stock_cvar"] <= cvar_low
        else "High"
        if current["stock_cvar"] >= cvar_high
        else "Moderate"
    )
    influence = (
        "Reduced"
        if current["risk_scaler"] < 0.95
        else "Full"
        if current["risk_scaler"] >= 1.0
        else "Moderate"
    )
    delta = float(current["delta"])
    action = (
        "Overweight"
        if delta > 0.05
        else "Mild Overweight"
        if delta > 0.01
        else "Underweight"
        if delta < -0.05
        else "Mild Underweight"
        if delta < -0.01
        else "Neutral"
    )
    z_value = float(current["stock_sentiment_z"])
    signal_state = (
        "Unusually Positive"
        if z_value >= 1.0
        else "Unusually Negative"
        if z_value <= -1.0
        else "Normal"
    )

    news = artifacts["news"]
    sentiment = artifacts["sentiment"]
    current_date = current["date"]
    ticker_news = news[
        news["mapped_trading_date"].eq(current_date)
        & news["ticker"].eq(current["ticker"])
    ]
    ticker_sentiment = (
        float(ticker_news["sentiment_score"].mean())
        if not ticker_news.empty
        else float(current["stock_sentiment_z"])
    )
    sector_match = sentiment[
        sentiment["mapped_trading_date"].eq(current_date)
        & sentiment["sector"].eq(current["sector"])
    ]
    sector_sentiment = (
        float(sector_match["sector_sentiment"].iloc[0])
        if not sector_match.empty
        else 0.0
    )
    relative = ticker_sentiment - sector_sentiment

    regimes = weights.copy()
    regimes["Market Stress Regime"] = pd.qcut(
        regimes["stock_cvar"],
        q=3,
        labels=["Low Stress", "Medium Stress", "High Stress"],
    )
    regime_table = regimes.groupby("Market Stress Regime", observed=True).agg(
        avg_tail_risk=("stock_cvar", "mean"),
        avg_scaler=("risk_scaler", "mean"),
    )
    regime_table["reduction"] = 1.0 - regime_table["avg_scaler"]

    return {
        "metric_lookup": metric_lookup,
        "cost": cost,
        "net_returns": net,
        "final_full": final_full,
        "base_full": base_full,
        "naive_full": naive_full,
        "final_net": final_net,
        "base_net": base_net,
        "naive_net": naive_net,
        "current": current,
        "relative": relative,
        "signal_state": signal_state,
        "tail_status": tail_status,
        "influence": influence,
        "action": action,
        "regime_table": regime_table,
    }


def signal_strength_label(z_value: float) -> str:
    if z_value <= -1.0:
        return "Unusually Negative"
    if z_value <= -0.5:
        return "Moderately Negative"
    if z_value >= 1.0:
        return "Unusually Positive"
    if z_value >= 0.5:
        return "Moderately Positive"
    return "Normal"


def tail_status_from_value(value: float, weights: pd.DataFrame) -> str:
    cvar_low, cvar_high = weights["stock_cvar"].quantile([1 / 3, 2 / 3])
    if value <= cvar_low:
        return "Low"
    if value >= cvar_high:
        return "High"
    return "Moderate"


def influence_from_scaler(scaler: float) -> str:
    if scaler < 0.95:
        return "Reduced"
    if scaler >= 1.0:
        return "Full"
    return "Moderate"


def action_from_delta(delta: float) -> str:
    if delta > 0.05:
        return "Overweight"
    if delta > 0.01:
        return "Mild Overweight"
    if delta < -0.05:
        return "Underweight"
    if delta < -0.01:
        return "Mild Underweight"
    return "Neutral"


def relative_sentiment_for_state(
    row: pd.Series,
    news: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> float:
    current_date = row["date"]
    ticker_news = news[
        news["mapped_trading_date"].eq(current_date)
        & news["ticker"].eq(row["ticker"])
    ]
    ticker_sentiment = (
        float(ticker_news["sentiment_score"].mean())
        if not ticker_news.empty
        else float(row["stock_sentiment_z"])
    )
    sector_match = sentiment[
        sentiment["mapped_trading_date"].eq(current_date)
        & sentiment["sector"].eq(row["sector"])
    ]
    sector_sentiment = (
        float(sector_match["sector_sentiment"].iloc[0])
        if not sector_match.empty
        else 0.0
    )
    return ticker_sentiment - sector_sentiment


def build_current_signal_state(
    row: pd.Series,
    artifacts: dict[str, pd.DataFrame],
) -> dict[str, object]:
    weights = artifacts["weights"]
    delta = float(row["final_weight"] - row["base_weight"])
    stock_cvar = float(row["stock_cvar"])
    risk_scaler = float(row["risk_scaler"])
    z_value = float(row["stock_sentiment_z"])
    tail_percentile = float(weights["stock_cvar"].rank(pct=True).loc[row.name])
    return {
        "date": pd.Timestamp(row["date"]),
        "ticker": str(row["ticker"]),
        "sector": str(row["sector"]),
        "relative": relative_sentiment_for_state(
            row,
            artifacts["news"],
            artifacts["sentiment"],
        ),
        "signal_z": z_value,
        "signal_state": signal_strength_label(z_value),
        "tail_risk": stock_cvar,
        "tail_percentile": tail_percentile,
        "tail_status": tail_status_from_value(stock_cvar, weights),
        "risk_scaler": risk_scaler,
        "influence": influence_from_scaler(risk_scaler),
        "delta": delta,
        "action": action_from_delta(delta),
        "base_weight": float(row["base_weight"]),
        "final_weight": float(row["final_weight"]),
        "sector_median_cvar": float(row["sector_median_cvar"]),
        "adjusted_signal": float(row["tail_risk_adjusted_sentiment_z"]),
    }


def build_scenario_state(
    current: dict[str, object],
    scenario_signal: float,
    scenario_tail_risk: float,
) -> dict[str, object]:
    risk_scaler = float(current["sector_median_cvar"]) / scenario_tail_risk
    adjusted_signal = scenario_signal * risk_scaler
    observed_adjusted = float(current["adjusted_signal"])
    observed_delta = float(current["delta"])
    scenario_delta = (
        observed_delta
        if abs(observed_adjusted) < 1e-9
        else observed_delta * (adjusted_signal / observed_adjusted)
    )
    return {
        "risk_scaler": risk_scaler,
        "adjusted_signal": adjusted_signal,
        "influence": influence_from_scaler(risk_scaler),
        "action": action_from_delta(scenario_delta),
        "delta": scenario_delta,
    }


def risk_regime_label(tail_status: str) -> str:
    return {
        "Low": "Low Stress",
        "Moderate": "Medium Stress",
        "High": "High Stress",
    }.get(tail_status, "Medium Stress")


def scenario_tail_status(value: float, weights: pd.DataFrame) -> str:
    return tail_status_from_value(value, weights)


def window_return_metrics(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {"return": 0.0, "volatility": 0.0, "sharpe": 0.0, "drawdown": 0.0}
    growth = (1 + returns).cumprod()
    ann_return = growth.iloc[-1] ** (252 / len(returns)) - 1
    ann_vol = returns.std() * (252**0.5)
    sharpe = ann_return / ann_vol if ann_vol else 0.0
    drawdown = growth / growth.cummax() - 1
    return {
        "return": ann_return,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "drawdown": float(drawdown.min()),
    }


def performance_summary(
    net_returns: pd.DataFrame,
    transaction_cost: int,
    window: str,
) -> dict[str, dict[str, float]]:
    selected = filter_performance_window(net_returns, window)
    selected = selected[selected["transaction_cost_bps"].eq(transaction_cost)]
    matrix = selected.pivot(index="date", columns="Display", values="return_net")
    return {
        strategy: window_return_metrics(matrix[strategy])
        for strategy in ["Final Strategy", "Naive Sentiment Tilt", "Base Min-Variance"]
        if strategy in matrix
    }


def performance_insight(summary: dict[str, dict[str, float]], window: str) -> str:
    final = summary["Final Strategy"]
    naive = summary["Naive Sentiment Tilt"]
    ending_leader = max(summary, key=lambda name: summary[name]["return"])
    drawdown_better = final["drawdown"] > naive["drawdown"]
    leader_phrase = (
        "delivered the highest annualised return"
        if ending_leader == "Final Strategy"
        else f"trailed {ending_leader} on annualised return"
    )
    risk_phrase = (
        "with better downside control than Naive Sentiment Tilt"
        if drawdown_better
        else "but did not improve drawdown versus Naive Sentiment Tilt"
    )
    return f"Over the selected {window} window, the Final Strategy {leader_phrase}, {risk_phrase}."


def risk_explorer_insight(
    current_status: str,
    scenario_status: str,
    influence: str,
) -> str:
    base = (
        f"The selected observation is in a {current_status.lower()}-risk state, "
        f"so the saved risk scaler sets {influence.lower()} sentiment influence."
    )
    if scenario_status != current_status:
        return (
            f"{base} Scenario state: {scenario_status}. "
            "The scenario marker changes only the decision preview, not the historical regime table."
        )
    return f"{base} The scenario remains in the same risk regime."


def engine_card(
    label: str,
    value: str,
    support: str,
    badge: str,
    css_class: str = "",
    final: bool = False,
) -> str:
    final_class = " final" if final else ""
    value_class_name = f" {css_class}" if css_class else ""
    return (
        f'<div class="pa-engine-card{final_class}">'
        f'<div class="pa-mini-label">{escape(label)}</div>'
        f'<div class="pa-engine-value{value_class_name}">{escape(value)}</div>'
        f'<div class="pa-engine-support">{escape(support)}</div>'
        f'<div class="pa-card-badge">{escape(badge)}</div>'
        "</div>"
    )


def status_card(label: str, value: str, support: str, css_class: str = "") -> str:
    value_class_name = f" {css_class}" if css_class else ""
    return (
        '<div class="pa-status-card">'
        f'<div class="pa-mini-label">{escape(label)}</div>'
        f'<div class="pa-engine-value{value_class_name}">{escape(value)}</div>'
        f'<div class="pa-engine-support">{escape(support)}</div>'
        "</div>"
    )


def why_action_text(current: dict[str, object]) -> str:
    ticker = str(current["ticker"])
    sector = str(current["sector"])
    relative = float(current["relative"])
    tail_status = str(current["tail_status"]).lower()
    signal_state = str(current["signal_state"]).lower()
    influence = str(current["influence"]).lower()
    action = str(current["action"]).lower()
    relative_phrase = (
        "stronger than"
        if relative > 0.01
        else "weaker than"
        if relative < -0.01
        else "close to"
    )
    return (
        f"{ticker}'s sentiment is {relative_phrase} the broader {sector} sector. "
        f"The rolling signal is {signal_state}; {tail_status} downside tail risk "
        f"sets {influence} sentiment influence, producing a {action} decision "
        "versus the base portfolio."
    )


def available_windows(net_returns: pd.DataFrame) -> list[str]:
    days = (net_returns["date"].max() - net_returns["date"].min()).days
    choices = ["1Y"]
    if days >= 730:
        choices.append("2Y")
    choices.append("Full OOS")
    return choices


def filter_performance_window(net_returns: pd.DataFrame, window: str) -> pd.DataFrame:
    if window == "Full OOS":
        return net_returns
    years = 2 if window == "2Y" else 1
    start = net_returns["date"].max() - pd.DateOffset(years=years)
    return net_returns[net_returns["date"].ge(start)]


def strategy_progression_chart(stages: list[dict[str, str | float | bool]]) -> go.Figure:
    fig = go.Figure()
    names = [str(stage["short"]) for stage in stages]
    values = [float(stage["sharpe"]) for stage in stages]
    colors = [
        STRATEGY_COLORS.get(str(stage["display"]), "#2563EB") for stage in stages
    ]
    fig.add_trace(
        go.Scatter(
            x=names,
            y=values,
            mode="lines+markers+text",
            text=[format_number(value) for value in values],
            textposition="top center",
            marker={"size": 9, "color": colors},
            line={"color": "#2563EB", "width": 2.2},
            hovertemplate="%{x}<br>Sharpe: %{y:.3f}<extra></extra>",
            name="Sharpe Ratio",
        )
    )
    fig = apply_plotly_theme(fig, height=230)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(
        title="Sharpe Ratio",
        range=[max(0, min(values) - 0.10), max(values) + 0.10],
    )
    fig.update_xaxes(title=None)
    return fig


def net_growth_chart(
    net_returns: pd.DataFrame,
    transaction_cost: int,
    window: str,
    strategies: list[str],
) -> tuple[go.Figure, pd.Series]:
    windowed = filter_performance_window(net_returns, window)
    selected = net_returns[
        net_returns["transaction_cost_bps"].eq(transaction_cost)
        & net_returns["Display"].isin(strategies)
    ].copy()
    selected = selected[selected["date"].isin(windowed["date"])]
    matrix = selected.pivot(index="date", columns="Display", values="return_net")
    matrix = matrix[[strategy for strategy in strategies if strategy in matrix]].dropna()
    growth = (1 + matrix).cumprod()
    fig = go.Figure()
    for strategy in growth.columns:
        fig.add_trace(
            go.Scatter(
                x=growth.index,
                y=growth[strategy],
                mode="lines",
                name=strategy,
                line={
                    "color": STRATEGY_COLORS.get(strategy, "#2563EB"),
                    "width": 2.4 if strategy == "Final Strategy" else 1.8,
                },
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.3f}<extra></extra>",
            )
        )
    fig = apply_plotly_theme(fig, height=260)
    fig.update_yaxes(title="Growth of $1")
    fig.update_xaxes(title="Trading Date")
    return fig, growth.iloc[-1]


def cost_robustness_chart(cost: pd.DataFrame, selected_cost: int) -> go.Figure:
    fig = go.Figure()
    chart_data = cost[
        cost["Display"].isin(
            ["Base Min-Variance", "Naive Sentiment Tilt", "Final Strategy"]
        )
    ]
    for strategy, group in chart_data.groupby("Display", sort=False):
        group = group.sort_values("Transaction Cost (bps)")
        fig.add_trace(
            go.Scatter(
                x=group["Transaction Cost (bps)"],
                y=group["Sharpe Ratio"],
                mode="lines+markers",
                name=strategy,
                line={"color": STRATEGY_COLORS.get(strategy, "#2563EB"), "width": 2},
                hovertemplate="Cost: %{x} bps<br>Sharpe: %{y:.3f}<extra></extra>",
            )
        )
        selected = group[group["Transaction Cost (bps)"].eq(selected_cost)]
        if not selected.empty:
            fig.add_trace(
                go.Scatter(
                    x=selected["Transaction Cost (bps)"],
                    y=selected["Sharpe Ratio"],
                    mode="markers",
                    name=f"{strategy} selected cost",
                    marker={
                        "size": 11,
                        "color": STRATEGY_COLORS.get(strategy, "#2563EB"),
                        "line": {"color": "#FFFFFF", "width": 2},
                    },
                    showlegend=False,
                    hovertemplate=(
                        "Selected cost: %{x} bps<br>Sharpe: %{y:.3f}<extra></extra>"
                    ),
                )
            )
    fig.add_vline(
        x=selected_cost,
        line_dash="dash",
        line_color="rgba(37, 99, 235, 0.36)",
        line_width=1,
    )
    fig = apply_plotly_theme(fig, height=260)
    fig.update_xaxes(title="Transaction Cost (bps)")
    fig.update_yaxes(title="Sharpe Ratio (Net)")
    return fig


def cost_matrix_html(cost: pd.DataFrame, selected_cost: int) -> str:
    matrix = cost.pivot(
        index="Display",
        columns="Transaction Cost (bps)",
        values="Sharpe Ratio",
    ).reindex(["Base Min-Variance", "Naive Sentiment Tilt", "Final Strategy"])
    header = "".join(
        f'<th class="{"pa-selected-cost" if int(col) == selected_cost else ""}">'
        f"{int(col)}</th>"
        for col in matrix.columns
    )
    rows = []
    for strategy, row in matrix.iterrows():
        values = ""
        for col, value in row.items():
            classes = []
            if int(col) == selected_cost:
                classes.append("pa-selected-cost")
            if strategy == "Final Strategy":
                classes.append("pa-final-value")
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            values += f"<td{class_attr}>{value:.3f}</td>"
        rows.append(f"<tr><td>{escape(strategy)}</td>{values}</tr>")
    return (
        '<table class="pa-table"><thead><tr><th>Strategy</th>'
        f"{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


FUND_COLORS = ["#2563EB", "#14866D", "#6D5BD0"]
FUND_UNIVERSE_ORDER = ["combined", "equity", "crypto"]
FUND_METHOD_ORDER = ["equal_weight", "max_sharpe", "min_cvar", "min_variance"]
DEFAULT_COMPARE_FUNDS = [
    "combined_min_variance",
    "combined_max_sharpe",
    "combined_min_cvar",
]


def fund_label(fund_name: str) -> str:
    for universe in FUND_UNIVERSE_ORDER:
        prefix = f"{universe}_"
        if fund_name.startswith(prefix):
            method = fund_name.removeprefix(prefix)
            if method in FUND_METHOD_ORDER:
                return f"{universe.title()} {method_label(method)}"
    return fund_name.replace("_", " ").title()


def short_fund_label(fund_name: str) -> str:
    for universe in FUND_UNIVERSE_ORDER:
        prefix = f"{universe}_"
        if fund_name.startswith(prefix):
            method = fund_name.removeprefix(prefix)
            if method in FUND_METHOD_ORDER:
                return method_label(method)
    return fund_label(fund_name)


def method_label(method: str) -> str:
    labels = {
        "equal_weight": "Equal Weight",
        "max_sharpe": "Max Sharpe",
        "min_cvar": "Min CVaR",
        "min_variance": "Min Variance",
    }
    return labels.get(method, method.replace("_", " ").title())


def asset_universe_label(asset_universe: str) -> str:
    labels = {
        "combined": "Combined Assets",
        "equity": "Equity",
        "crypto": "Crypto",
    }
    return labels.get(asset_universe, asset_universe.replace("_", " ").title())


def ordered_model_funds(fund_returns: pd.DataFrame) -> list[str]:
    if fund_returns.empty:
        return []
    entities = fund_returns[["fund_name", "asset_universe", "method"]].drop_duplicates()
    universe_rank = {name: index for index, name in enumerate(FUND_UNIVERSE_ORDER)}
    method_rank = {name: index for index, name in enumerate(FUND_METHOD_ORDER)}
    entities["_universe_rank"] = (
        entities["asset_universe"].astype(str).map(universe_rank).fillna(99)
    )
    entities["_method_rank"] = entities["method"].astype(str).map(method_rank).fillna(99)
    entities = entities.sort_values(
        ["_universe_rank", "_method_rank", "asset_universe", "method", "fund_name"]
    )
    return entities["fund_name"].astype(str).tolist()


def fund_metadata(fund_returns: pd.DataFrame) -> dict[str, dict[str, str]]:
    if fund_returns.empty:
        return {}
    meta = (
        fund_returns[["fund_name", "asset_universe", "method"]]
        .drop_duplicates("fund_name")
        .set_index("fund_name")
    )
    return {
        fund: {
            "name": fund_label(fund),
            "asset_class": asset_universe_label(str(row["asset_universe"])),
            "method": method_label(str(row["method"])),
        }
        for fund, row in meta.iterrows()
    }


def comparison_window_data(
    fund_returns: pd.DataFrame,
    selected_funds: list[str],
    period: str,
) -> pd.DataFrame:
    data = fund_returns[fund_returns["fund_name"].isin(selected_funds)].copy()
    if data.empty:
        return data
    latest_date = data["date"].max()
    if period == "1Y":
        data = data[data["date"].ge(latest_date - pd.DateOffset(years=1))]
    elif period == "2Y":
        data = data[data["date"].ge(latest_date - pd.DateOffset(years=2))]
    return data.sort_values(["fund_name", "date"])


def calculate_period_metrics(
    fund_returns: pd.DataFrame,
    selected_funds: list[str],
    period: str,
) -> pd.DataFrame:
    data = comparison_window_data(fund_returns, selected_funds, period)
    rows = []
    for fund in selected_funds:
        returns = data.loc[data["fund_name"].eq(fund), "portfolio_return"].dropna()
        if returns.empty:
            rows.append(
                {
                    "fund_name": fund,
                    "annualised_return": None,
                    "annualised_volatility": None,
                    "sharpe_ratio": None,
                    "maximum_drawdown": None,
                    "observation_count": 0,
                }
            )
            continue
        growth = (1 + returns).cumprod()
        ann_return = growth.iloc[-1] ** (252 / len(returns)) - 1
        ann_vol = returns.std() * (252**0.5)
        drawdown = growth / growth.cummax() - 1
        rows.append(
            {
                "fund_name": fund,
                "annualised_return": float(ann_return),
                "annualised_volatility": float(ann_vol),
                "sharpe_ratio": float(ann_return / ann_vol) if ann_vol else 0.0,
                "maximum_drawdown": float(drawdown.min()),
                "observation_count": int(len(returns)),
            }
        )
    return pd.DataFrame(rows)


def fund_growth_frame(
    fund_returns: pd.DataFrame,
    selected_funds: list[str],
    period: str,
) -> pd.DataFrame:
    data = comparison_window_data(fund_returns, selected_funds, period)
    frames = []
    for fund in selected_funds:
        fund_data = data.loc[data["fund_name"].eq(fund), ["date", "portfolio_return"]].copy()
        if fund_data.empty:
            continue
        fund_data["growth"] = (1 + fund_data["portfolio_return"]).cumprod()
        fund_data["growth"] = fund_data["growth"] / fund_data["growth"].iloc[0]
        fund_data["fund_name"] = fund
        frames.append(fund_data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def select_unique_fund(
    label: str,
    key: str,
    all_funds: list[str],
    selected: list[str],
    default: str,
    optional: bool = False,
) -> str | None:
    current = st.session_state.get(key, default)
    blocked = {fund for fund in selected if fund != current}
    options = [fund for fund in all_funds if fund not in blocked]
    if optional:
        options = ["None"] + options
        if current not in options:
            current = "None"
    elif current not in options:
        current = options[0] if options else ""
    index = options.index(current) if current in options else 0
    value = st.selectbox(
        label,
        options,
        index=index,
        key=key,
        format_func=lambda fund: "None" if fund == "None" else fund_label(str(fund)),
    )
    return None if value == "None" else str(value)


def initialise_compare_selector_state(
    all_funds: list[str],
    defaults: list[str],
) -> dict[str, str]:
    selector_specs = [
        ("compare_fund_1", defaults[0], False),
        ("compare_fund_2", defaults[1] if len(defaults) > 1 else defaults[0], False),
        ("compare_fund_3", defaults[2] if len(defaults) > 2 else "None", True),
    ]
    used: list[str] = []
    values: dict[str, str] = {}
    for key, default, optional in selector_specs:
        current = st.session_state.get(key, default)
        if current == "None" and optional:
            values[key] = "None"
            continue
        if current not in all_funds or current in used:
            replacement = next((fund for fund in all_funds if fund not in used), None)
            current = replacement if replacement is not None else "None"
            if key in st.session_state:
                st.session_state[key] = current
        values[key] = str(current)
        if current != "None":
            used.append(str(current))
    return values


def render_fund_compare_card(
    fund: str,
    metrics: pd.Series,
    meta: dict[str, str],
    index: int,
) -> str:
    def metric_value(column: str, pct: bool = True) -> str:
        value = metrics.get(column)
        if pd.isna(value):
            return "Not available"
        return format_pct(float(value)) if pct else format_number(float(value), 2)

    mini_metrics = [
        ("Return", metric_value("annualised_return")),
        ("Volatility", metric_value("annualised_volatility")),
        ("Sharpe", metric_value("sharpe_ratio", pct=False)),
        ("Max DD", metric_value("maximum_drawdown")),
    ]
    mini_html = "".join(
        '<div>'
        f'<div class="pa-fund-mini-value">{escape(value)}</div>'
        f'<div class="pa-fund-mini-label">{escape(label)}</div>'
        "</div>"
        for label, value in mini_metrics
    )
    css = "" if index == 0 else f" fund-{index + 1}"
    return (
        f'<div class="pa-compare-accent{css}"></div>'
        '<div class="pa-fund-card-body">'
        f'<div class="pa-mini-label">Fund {index + 1}</div>'
        f'<div class="pa-fund-title">{escape(fund_label(fund))}</div>'
        f'<div class="pa-fund-subtitle">{escape(meta.get("asset_class", "Not available"))}'
        f' · {escape(meta.get("method", "Not available"))}</div>'
        f'<div class="pa-fund-metric-grid">{mini_html}</div>'
        "</div>"
    )


def risk_return_chart(metrics: pd.DataFrame, meta: dict[str, dict[str, str]]) -> go.Figure:
    fig = go.Figure()
    for index, row in metrics.iterrows():
        fund = str(row["fund_name"])
        fig.add_trace(
            go.Scatter(
                x=[float(row["annualised_volatility"])],
                y=[float(row["annualised_return"])],
                mode="markers+text",
                text=[short_fund_label(fund)],
                textposition="top center",
                marker={"size": 14, "color": FUND_COLORS[index % len(FUND_COLORS)]},
                name=fund_label(fund),
                customdata=[[float(row["sharpe_ratio"])]],
                hovertemplate=(
                    f"Fund: {escape(meta.get(fund, {}).get('name', fund_label(fund)))}"
                    "<br>Annualised Return: %{y:.2%}"
                    "<br>Annualised Volatility: %{x:.2%}"
                    "<br>Sharpe Ratio: %{customdata[0]:.2f}<extra></extra>"
                ),
            )
        )
    fig = apply_plotly_theme(fig, height=310)
    fig.update_xaxes(title="Annualised Volatility", tickformat=".0%")
    fig.update_yaxes(title="Annualised Return", tickformat=".0%")
    clean_metrics = metrics.dropna(
        subset=["annualised_volatility", "annualised_return"]
    )
    if not clean_metrics.empty:
        x_values = clean_metrics["annualised_volatility"].astype(float)
        y_values = clean_metrics["annualised_return"].astype(float)
        x_padding = max((x_values.max() - x_values.min()) * 0.22, 0.015)
        y_padding = max((y_values.max() - y_values.min()) * 0.22, 0.015)
        fig.update_xaxes(range=[x_values.min() - x_padding, x_values.max() + x_padding])
        fig.update_yaxes(range=[y_values.min() - y_padding, y_values.max() + y_padding])
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin={"l": 34, "r": 24, "t": 22, "b": 42},
    )
    return fig


def growth_comparison_chart(
    growth: pd.DataFrame,
    selected_funds: list[str],
) -> go.Figure:
    fig = go.Figure()
    for index, fund in enumerate(selected_funds):
        fund_data = growth[growth["fund_name"].eq(fund)]
        if fund_data.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=fund_data["date"],
                y=fund_data["growth"],
                mode="lines",
                name=fund_label(fund),
                line={"color": FUND_COLORS[index % len(FUND_COLORS)], "width": 2.3},
                hovertemplate="%{x|%Y-%m-%d}<br>Growth: %{y:.3f}<extra></extra>",
            )
        )
    fig = apply_plotly_theme(fig, height=310)
    fig.update_yaxes(title="Growth of $1")
    return fig


def comparison_table_html(metrics: pd.DataFrame, meta: dict[str, dict[str, str]]) -> str:
    selected_funds = metrics["fund_name"].astype(str).tolist()
    headers = "".join(
        f"<th>Fund {index + 1}</th>" for index, _ in enumerate(selected_funds)
    )

    metric_specs = [
        ("Annualised Return", "annualised_return", "max", True),
        ("Annualised Volatility", "annualised_volatility", "min", True),
        ("Sharpe Ratio", "sharpe_ratio", "max", False),
        ("Max Drawdown", "maximum_drawdown", "max", True),
    ]
    rows = []
    for label, column, direction, pct in metric_specs:
        valid = metrics.dropna(subset=[column])
        best_fund = (
            str(valid.loc[valid[column].idxmax(), "fund_name"])
            if direction == "max" and not valid.empty
            else str(valid.loc[valid[column].idxmin(), "fund_name"])
            if not valid.empty
            else "Not available"
        )
        values = ""
        for fund in selected_funds:
            value = metrics.loc[metrics["fund_name"].eq(fund), column].iloc[0]
            display = "Not available" if pd.isna(value) else (
                format_pct(float(value)) if pct else format_number(float(value), 2)
            )
            css_class = ' class="pa-best-cell"' if fund == best_fund else ""
            values += f"<td{css_class}>{escape(display)}</td>"
        rows.append(
            f"<tr><td>{escape(label)}</td>{values}"
            f'<td class="pa-best-cell">{escape(fund_label(best_fund))}</td></tr>'
        )

    if any(meta.get(fund, {}).get("asset_class") for fund in selected_funds):
        values = "".join(
            f'<td>{escape(meta.get(fund, {}).get("asset_class", "Not available"))}</td>'
            for fund in selected_funds
        )
        rows.append(f"<tr><td>Asset Class</td>{values}<td></td></tr>")

    return (
        '<table class="pa-compare-table"><thead><tr><th>Metric</th>'
        f"{headers}<th>Best</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def horizontal_metric_chart(
    metrics: pd.DataFrame,
    column: str,
    title: str,
    pct: bool = True,
) -> go.Figure:
    plot_data = metrics.dropna(subset=[column]).copy()
    values = plot_data[column].astype(float)
    if column == "maximum_drawdown":
        values = values.abs()
    fig = go.Figure(
        go.Bar(
            x=values,
            y=[short_fund_label(str(fund)) for fund in plot_data["fund_name"]],
            orientation="h",
            marker={"color": FUND_COLORS[: len(plot_data)]},
            hovertemplate=f"{title}: %{{x:.2%}}<extra></extra>" if pct else f"{title}: %{{x:.2f}}<extra></extra>",
        )
    )
    fig = apply_plotly_theme(fig, height=205)
    fig.update_layout(showlegend=False, title={"text": title, "font": {"size": 12}})
    fig.update_xaxes(tickformat=".0%" if pct else None)
    return fig


def comparison_insight(metrics: pd.DataFrame, period: str) -> str:
    valid = metrics.dropna(
        subset=[
            "annualised_return",
            "annualised_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
        ]
    )
    if valid.empty:
        return "Saved return data is not available for the selected comparison."
    high_return = str(valid.loc[valid["annualised_return"].idxmax(), "fund_name"])
    best_sharpe = str(valid.loc[valid["sharpe_ratio"].idxmax(), "fund_name"])
    low_vol = str(valid.loc[valid["annualised_volatility"].idxmin(), "fund_name"])
    small_drawdown = str(valid.loc[valid["maximum_drawdown"].idxmax(), "fund_name"])
    winners = {
        "highest annualised return": high_return,
        "strongest Sharpe ratio": best_sharpe,
        "lowest volatility": low_vol,
        "smallest maximum drawdown": small_drawdown,
    }
    grouped: dict[str, list[str]] = {}
    for description, fund in winners.items():
        grouped.setdefault(fund, []).append(description)

    sentences = []
    for fund, descriptions in grouped.items():
        if len(descriptions) == 1:
            metric_text = descriptions[0]
        elif len(descriptions) == 2:
            metric_text = " and ".join(descriptions)
        else:
            metric_text = ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"
        sentences.append(f"{fund_label(fund)} recorded the {metric_text}")

    return f"{'; '.join(sentences)} over the selected {period} period."


def fund_objective(meta: dict[str, str]) -> str:
    method = meta.get("method", "model fund")
    if method == "Max Sharpe":
        return "Return-focused construction with explicit risk adjustment."
    if method == "Min CVaR":
        return "Tail-risk-aware construction focused on downside control."
    if method == "Min Variance":
        return "Risk-controlled construction designed to reduce volatility."
    if method == "Equal Weight":
        return "Simple diversified exposure across the selected asset universe."
    return "Model portfolio construction using saved PulseAlloc backtest data."


def single_fund_return_window(
    fund_returns: pd.DataFrame,
    fund: str,
    period: str,
) -> pd.DataFrame:
    data = comparison_window_data(fund_returns, [fund], period)
    return data.loc[data["fund_name"].eq(fund)].sort_values("date").copy()


def single_fund_growth_drawdown(fund_data: pd.DataFrame) -> pd.DataFrame:
    if fund_data.empty or "portfolio_return" not in fund_data.columns:
        return pd.DataFrame()
    frame = fund_data[["date", "portfolio_return"]].dropna().sort_values("date").copy()
    if frame.empty:
        return pd.DataFrame()
    frame["growth"] = (1 + frame["portfolio_return"]).cumprod()
    frame["growth"] = frame["growth"] / frame["growth"].iloc[0]
    frame["drawdown"] = frame["growth"] / frame["growth"].cummax() - 1
    return frame


def growth_of_one_chart(frame: pd.DataFrame, fund: str) -> go.Figure:
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["growth"],
                mode="lines",
                name=fund_label(fund),
                line={"color": "#2563EB", "width": 2.5},
                hovertemplate="%{x|%Y-%m-%d}<br>Growth: %{y:.3f}<extra></extra>",
            )
        )
    fig = apply_plotly_theme(fig, height=340)
    fig.update_layout(showlegend=False, margin={"l": 34, "r": 18, "t": 12, "b": 38})
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Growth of $1", tickformat=".2f")
    return fig


def drawdown_chart(frame: pd.DataFrame, fund: str) -> go.Figure:
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["drawdown"],
                mode="lines",
                name=fund_label(fund),
                line={"color": "#C94A4A", "width": 1.7},
                fill="tozeroy",
                fillcolor="rgba(201, 74, 74, 0.16)",
                hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>",
            )
        )
    fig = apply_plotly_theme(fig, height=340)
    fig.update_layout(showlegend=False, margin={"l": 34, "r": 18, "t": 12, "b": 38})
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Drawdown", tickformat=".0%")
    return fig


def latest_fund_holdings(
    fund_weights: pd.DataFrame,
    fund: str,
    top_n: int,
) -> tuple[pd.DataFrame, pd.Timestamp | None, str]:
    required = {"fund_name", "date", "asset", "weight"}
    if fund_weights.empty or not required.issubset(fund_weights.columns):
        return pd.DataFrame(), None, "No saved fund weights were available."
    fund_data = fund_weights.loc[fund_weights["fund_name"].eq(fund)].copy()
    if fund_data.empty:
        return pd.DataFrame(), None, "No saved weights matched the selected fund."
    latest_date = fund_data["date"].max()
    latest = fund_data.loc[fund_data["date"].eq(latest_date), ["asset", "weight"]].copy()
    latest = (
        latest.groupby("asset", as_index=False)["weight"]
        .sum()
        .assign(abs_weight=lambda data: data["weight"].abs())
        .sort_values("abs_weight", ascending=False)
        .head(top_n)
        .sort_values("weight", ascending=True)
    )
    return latest, latest_date, "Latest portfolio weights used as holdings."


def holdings_bar_chart(holdings: pd.DataFrame, top_n: int) -> go.Figure:
    fig = go.Figure()
    if not holdings.empty:
        colors = [
            "#14866D" if float(weight) >= 0 else "#C94A4A"
            for weight in holdings["weight"]
        ]
        fig.add_trace(
            go.Bar(
                x=holdings["weight"],
                y=holdings["asset"],
                orientation="h",
                marker={"color": colors},
                text=[format_pct(float(weight)) for weight in holdings["weight"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>Weight: %{x:.2%}<extra></extra>",
            )
        )
    fig = apply_plotly_theme(fig, height=max(300, 34 * top_n + 110))
    fig.update_layout(showlegend=False, margin={"l": 72, "r": 52, "t": 10, "b": 34})
    fig.update_xaxes(title="Portfolio Weight", tickformat=".0%")
    fig.update_yaxes(title="")
    return fig


def fact_sheet_insight(metrics: pd.Series, period: str) -> str:
    if metrics.empty or pd.isna(metrics.get("sharpe_ratio")):
        return "Saved return data is not available for the selected period."
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    drawdown = float(metrics.get("maximum_drawdown", 0.0))
    ann_return = float(metrics.get("annualised_return", 0.0))
    if sharpe >= 1:
        sharpe_text = "strong risk-adjusted performance"
    elif sharpe >= 0.5:
        sharpe_text = "moderate risk-adjusted performance"
    else:
        sharpe_text = "weaker risk-adjusted performance"
    if drawdown <= -0.25:
        drawdown_text = "a severe drawdown profile"
    elif drawdown <= -0.15:
        drawdown_text = "a moderate drawdown profile"
    else:
        drawdown_text = "a contained drawdown profile"
    direction = "positive" if ann_return >= 0 else "negative"
    return (
        f"Over the selected {period} window, this fund delivered {direction} "
        f"annualised return, {sharpe_text}, and {drawdown_text}."
    )


def add_model_fund_to_portfolio(fund: str) -> None:
    selected = st.session_state.setdefault("portfolio_selected_funds", [])
    if fund not in selected:
        selected.append(fund)
        st.session_state["portfolio_selected_funds"] = selected


def render_selected_funds_strip(selected_for_portfolio: list[str]) -> None:
    with st.container(border=True):
        bar_cols = st.columns([0.22, 0.53, 0.25], gap="medium", vertical_alignment="center")
        with bar_cols[0]:
            st.markdown(
                f'<div class="pa-card-title">{len(selected_for_portfolio)} funds selected</div>',
                unsafe_allow_html=True,
            )
        with bar_cols[1]:
            if selected_for_portfolio:
                chip_cols = st.columns(len(selected_for_portfolio), gap="small")
                for col, fund in zip(chip_cols, selected_for_portfolio):
                    with col:
                        if st.button(
                            f"{fund_label(fund)} ×",
                            key=f"remove_{fund}",
                            use_container_width=True,
                        ):
                            st.session_state["portfolio_selected_funds"] = [
                                item for item in selected_for_portfolio if item != fund
                            ]
                            st.rerun()
            else:
                st.markdown(
                    '<div class="pa-mini-support">Select model funds above to build a portfolio.</div>',
                    unsafe_allow_html=True,
                )
        with bar_cols[2]:
            if st.button(
                "Continue to Build Portfolio →",
                key="continue_to_build_portfolio",
                disabled=not bool(selected_for_portfolio),
                use_container_width=True,
            ):
                st.session_state["build_portfolio_selected_funds"] = selected_for_portfolio
                st.session_state["_nav_target"] = "Build Portfolio"
                st.rerun()


def selected_build_portfolio_funds() -> list[str]:
    selected = st.session_state.get(
        "portfolio_selected_funds",
        st.session_state.get("build_portfolio_selected_funds", []),
    )
    return [str(fund) for fund in selected if str(fund)]


def equal_weight_percentages(funds: list[str]) -> dict[str, float]:
    if not funds:
        return {}
    base = round(100.0 / len(funds), 2)
    weights = {fund: base for fund in funds}
    weights[funds[-1]] = round(100.0 - base * (len(funds) - 1), 2)
    return weights


def initialise_build_weights(selected_funds: list[str]) -> dict[str, float]:
    stored = st.session_state.get("build_portfolio_weights")
    if not stored:
        weights = equal_weight_percentages(selected_funds)
        st.session_state["build_portfolio_weights"] = weights
        st.session_state["build_portfolio_initial_weights"] = weights.copy()
        return weights

    weights = {
        fund: float(stored.get(fund, 0.0))
        for fund in selected_funds
    }
    if selected_funds and not any(abs(value) > 1e-9 for value in weights.values()):
        weights = equal_weight_percentages(selected_funds)
    initial = st.session_state.get("build_portfolio_initial_weights", {}).copy()
    for fund in selected_funds:
        initial.setdefault(fund, weights.get(fund, 0.0))
    st.session_state["build_portfolio_initial_weights"] = {
        fund: float(initial.get(fund, 0.0))
        for fund in selected_funds
    }
    st.session_state["build_portfolio_weights"] = weights
    return weights


def build_window_returns(
    fund_returns: pd.DataFrame,
    selected_funds: list[str],
    period: str,
) -> pd.DataFrame:
    return comparison_window_data(fund_returns, selected_funds, period)


def calculate_custom_portfolio_returns(
    fund_returns: pd.DataFrame,
    selected_funds: list[str],
    weights_pct: dict[str, float],
    period: str,
) -> pd.DataFrame:
    data = build_window_returns(fund_returns, selected_funds, period)
    if data.empty:
        return pd.DataFrame()
    wide = data.pivot(index="date", columns="fund_name", values="portfolio_return")
    available_funds = [fund for fund in selected_funds if fund in wide.columns]
    if not available_funds:
        return pd.DataFrame()
    wide = wide[available_funds].dropna()
    if wide.empty:
        return pd.DataFrame()
    weights = pd.Series(
        {fund: float(weights_pct.get(fund, 0.0)) / 100.0 for fund in available_funds}
    )
    portfolio_return = wide.mul(weights, axis=1).sum(axis=1)
    return pd.DataFrame(
        {
            "date": portfolio_return.index,
            "portfolio_return": portfolio_return.values,
        }
    )


def calculate_custom_portfolio_metrics(portfolio_returns: pd.DataFrame) -> dict[str, float | None]:
    if portfolio_returns.empty:
        return {
            "annualised_return": None,
            "annualised_volatility": None,
            "sharpe_ratio": None,
            "maximum_drawdown": None,
        }
    returns = portfolio_returns["portfolio_return"].dropna()
    if returns.empty:
        return {
            "annualised_return": None,
            "annualised_volatility": None,
            "sharpe_ratio": None,
            "maximum_drawdown": None,
        }
    growth = (1 + returns).cumprod()
    annualised_return = growth.iloc[-1] ** (252 / len(returns)) - 1
    annualised_volatility = returns.std() * (252**0.5)
    drawdown = growth / growth.cummax() - 1
    return {
        "annualised_return": float(annualised_return),
        "annualised_volatility": float(annualised_volatility),
        "sharpe_ratio": float(annualised_return / annualised_volatility)
        if annualised_volatility
        else 0.0,
        "maximum_drawdown": float(drawdown.min()),
    }


def custom_portfolio_growth_frame(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    if portfolio_returns.empty:
        return pd.DataFrame()
    frame = portfolio_returns.sort_values("date").copy()
    frame["growth"] = (1 + frame["portfolio_return"]).cumprod()
    frame["growth"] = frame["growth"] / frame["growth"].iloc[0]
    return frame


def custom_growth_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["growth"],
                mode="lines",
                line={"color": "#2563EB", "width": 2.2},
                hovertemplate="%{x|%Y-%m-%d}<br>Growth: %{y:.3f}<extra></extra>",
                name="Custom Portfolio",
            )
        )
    fig = apply_plotly_theme(fig, height=210)
    fig.update_layout(showlegend=False, margin={"l": 28, "r": 14, "t": 6, "b": 32})
    fig.update_xaxes(title="Trading Date")
    fig.update_yaxes(title="Growth of $1")
    return fig


def allocation_donut_for_funds(
    selected_funds: list[str],
    weights_pct: dict[str, float],
    total_pct: float,
) -> go.Figure:
    labels = [fund_label(fund) for fund in selected_funds]
    values = [max(float(weights_pct.get(fund, 0.0)), 0.0) for fund in selected_funds]
    colors = ["#2563EB", "#6D5BD0", "#14866D", "#F59E0B", "#64748B"][: len(labels)]
    if total_pct < 100.0:
        labels.append("Unallocated")
        values.append(max(100.0 - float(total_pct), 0.0))
        colors.append("#E5EAF1")
    allocation = pd.DataFrame(
        {
            "fund": labels,
            "weight": values,
        }
    )
    fig = go.Figure(
        go.Pie(
            labels=allocation["fund"],
            values=allocation["weight"],
            hole=0.66,
            marker={"colors": colors},
            textinfo="none",
            hovertemplate="%{label}<br>Allocation: %{value:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=245,
        margin={"l": 0, "r": 0, "t": 4, "b": 4},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[
            {
                "text": f"{total_pct:.0f}%<br>Total",
                "x": 0.5,
                "y": 0.5,
                "font": {"size": 14, "color": "#14253D"},
                "showarrow": False,
            }
        ],
    )
    return fig


def asset_class_from_asset(asset: str) -> str:
    return "Crypto" if str(asset).endswith("-USD") else "Equity"


def asset_exposure_frame(
    fund_weights: pd.DataFrame,
    selected_funds: list[str],
    weights_pct: dict[str, float],
) -> pd.DataFrame:
    rows = []
    if not fund_weights.empty and {"fund_name", "date", "asset", "weight"}.issubset(
        fund_weights.columns
    ):
        for fund in selected_funds:
            fund_data = fund_weights.loc[fund_weights["fund_name"].eq(fund)].copy()
            if fund_data.empty:
                continue
            latest_date = fund_data["date"].max()
            latest = fund_data.loc[fund_data["date"].eq(latest_date)].copy()
            latest["asset_class"] = latest["asset"].map(asset_class_from_asset)
            class_weights = latest.groupby("asset_class")["weight"].sum()
            for asset_class, fund_asset_weight in class_weights.items():
                rows.append(
                    {
                        "asset_class": asset_class,
                        "weight": float(weights_pct.get(fund, 0.0))
                        * float(fund_asset_weight)
                        / 100.0,
                    }
                )
    if not rows:
        return pd.DataFrame(
            {
                "asset_class": ["Fund Universe Mix"],
                "weight": [sum(float(weights_pct.get(fund, 0.0)) for fund in selected_funds) / 100.0],
            }
        )
    exposure = pd.DataFrame(rows).groupby("asset_class", as_index=False)["weight"].sum()
    return exposure.sort_values("weight", ascending=True)


def exposure_bar_chart(exposure: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not exposure.empty:
        fig.add_trace(
            go.Bar(
                x=exposure["weight"],
                y=exposure["asset_class"],
                orientation="h",
                marker={"color": ["#14866D" if item == "Equity" else "#6D5BD0" for item in exposure["asset_class"]]},
                text=[format_pct(float(value)) for value in exposure["weight"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>Exposure: %{x:.2%}<extra></extra>",
            )
        )
    fig = apply_plotly_theme(fig, height=210)
    fig.update_layout(showlegend=False, margin={"l": 62, "r": 42, "t": 6, "b": 28})
    fig.update_xaxes(title="Exposure", tickformat=".0%")
    fig.update_yaxes(title="")
    return fig


def build_allocation_status(total_pct: float) -> tuple[str, bool, str]:
    if abs(total_pct - 100.0) <= 0.05:
        return "100% allocated ✓", True, "valid"
    if total_pct < 100:
        return f"{total_pct:.0f}% allocated — {100 - total_pct:.0f}% remaining", False, ""
    return f"{total_pct:.0f}% allocated — reduce by {total_pct - 100:.0f}%", False, ""


def ui_risk_alignment(
    metrics: dict[str, float | None],
    exposure: pd.DataFrame,
    selected_profile: str,
) -> tuple[str, str]:
    volatility = metrics.get("annualised_volatility")
    crypto_exposure = 0.0
    if not exposure.empty and exposure["asset_class"].eq("Crypto").any():
        crypto_exposure = float(exposure.loc[exposure["asset_class"].eq("Crypto"), "weight"].sum())
    if volatility is None:
        return "Profile alignment unavailable until return data is available.", "pa-muted"

    score = 0
    if float(volatility) > 0.18:
        score += 2
    elif float(volatility) > 0.13:
        score += 1
    if crypto_exposure > 0.25:
        score += 1
    observed = "Growth" if score >= 3 else "Balanced" if score >= 1 else "Conservative"
    if observed == selected_profile:
        return f"Your current allocation is broadly aligned with a {selected_profile} profile.", "pa-green"
    if selected_profile == "Conservative" and observed != "Conservative":
        return f"Your current allocation carries more historical risk than the selected {selected_profile} profile.", "pa-red"
    return f"Your current allocation looks closer to a {observed} profile than the selected {selected_profile} profile.", "pa-muted"


def build_portfolio_insight(
    selected_funds: list[str],
    weights_pct: dict[str, float],
    metrics: dict[str, float | None],
    exposure: pd.DataFrame,
    selected_profile: str,
    period: str,
) -> str:
    volatility = metrics.get("annualised_volatility")
    crypto_exposure = 0.0
    equity_exposure = 0.0
    if not exposure.empty:
        crypto_exposure = float(exposure.loc[exposure["asset_class"].eq("Crypto"), "weight"].sum())
        equity_exposure = float(exposure.loc[exposure["asset_class"].eq("Equity"), "weight"].sum())
    concentration = max((float(weights_pct.get(fund, 0.0)) for fund in selected_funds), default=0.0)
    if volatility is None:
        return "Historical return data is unavailable for the selected fund mix."
    exposure_text = (
        "crypto exposure"
        if crypto_exposure >= 0.20
        else "mostly equity-linked exposure"
        if equity_exposure >= 0.70
        else "mixed asset-universe exposure"
    )
    concentration_text = (
        "concentrated in one model fund"
        if concentration >= 60
        else f"diversified across {len(selected_funds)} model funds"
    )
    return (
        f"Over the selected {period} window, the current mix is {concentration_text}, "
        f"has {exposure_text}, and shows {format_pct(float(volatility))} historical annualised volatility "
        f"against the selected {selected_profile} profile."
    )


def readiness_risk_status(profile_message: str, selected_profile: str) -> tuple[str, str, str]:
    if "broadly aligned" in profile_message:
        return selected_profile, f"Selected profile: {selected_profile}", "Aligned ✓"
    if "carries more historical risk" in profile_message:
        return "Growth-leaning", f"Selected profile: {selected_profile}", "Above selected risk"
    if "closer to a Conservative" in profile_message:
        return "Conservative-leaning", f"Selected profile: {selected_profile}", "Partially aligned"
    if "closer to a Balanced" in profile_message:
        return "Balanced-leaning", f"Selected profile: {selected_profile}", "Partially aligned"
    if "closer to a Growth" in profile_message:
        return "Growth-leaning", f"Selected profile: {selected_profile}", "Partially aligned"
    return "Unavailable", f"Selected profile: {selected_profile}", "Check required"


def readiness_summary(total_pct: float, selected_count: int, allocation_valid: bool) -> str:
    if total_pct > 100.05:
        return "Reduce the allocation to 100% before saving this baseline portfolio."
    if not allocation_valid:
        return (
            "Complete the allocation to 100% and include more model funds to build "
            "a broader baseline. Then save this portfolio to continue to Adaptive Allocation."
        )
    if selected_count <= 1:
        return (
            "Allocation is complete. Consider whether the current single-fund baseline "
            "provides the diversification you want before continuing."
        )
    return "Baseline ready. Save this portfolio to continue to Adaptive Allocation."


def render_compare_funds_content(fund_returns: pd.DataFrame) -> None:
    all_funds = ordered_model_funds(fund_returns)
    meta = fund_metadata(fund_returns)
    defaults = [fund for fund in DEFAULT_COMPARE_FUNDS if fund in all_funds]
    defaults.extend([fund for fund in all_funds if fund not in defaults])
    defaults = defaults[:3]
    selector_values = initialise_compare_selector_state(all_funds, defaults)

    selected_funds: list[str] = []
    with st.container(border=True):
        selector_cols = st.columns([1, 1, 1, 1.15], gap="medium")
        with selector_cols[0]:
            fund_1 = select_unique_fund(
                "Fund 1",
                "compare_fund_1",
                all_funds,
                [
                    fund
                    for fund in [
                        selector_values.get("compare_fund_2"),
                        selector_values.get("compare_fund_3"),
                    ]
                    if fund and fund != "None"
                ],
                defaults[0],
            )
            if fund_1:
                selected_funds.append(fund_1)
        with selector_cols[1]:
            fund_2 = select_unique_fund(
                "Fund 2",
                "compare_fund_2",
                all_funds,
                [
                    fund
                    for fund in [
                        selector_values.get("compare_fund_1"),
                        selector_values.get("compare_fund_3"),
                    ]
                    if fund and fund != "None"
                ],
                defaults[1] if len(defaults) > 1 else defaults[0],
            )
            if fund_2:
                selected_funds.append(fund_2)
        with selector_cols[2]:
            fund_3 = select_unique_fund(
                "Fund 3 (Optional)",
                "compare_fund_3",
                all_funds,
                [
                    fund
                    for fund in [
                        selector_values.get("compare_fund_1"),
                        selector_values.get("compare_fund_2"),
                    ]
                    if fund and fund != "None"
                ],
                defaults[2] if len(defaults) > 2 else defaults[0],
                optional=True,
            )
            if fund_3:
                selected_funds.append(fund_3)
        with selector_cols[3]:
            period = st.segmented_control(
                "Comparison Period",
                ["1Y", "2Y", "Full History"],
                default=st.session_state.get("compare_period", "1Y"),
                key="compare_period",
            )

    metrics = calculate_period_metrics(fund_returns, selected_funds, period)
    growth = fund_growth_frame(fund_returns, selected_funds, period)

    card_cols = st.columns(len(selected_funds), gap="medium")
    for index, (col, fund) in enumerate(zip(card_cols, selected_funds)):
        fund_metrics = metrics.loc[metrics["fund_name"].eq(fund)].iloc[0]
        with col:
            with st.container(border=True):
                st.markdown(
                    render_fund_compare_card(fund, fund_metrics, meta.get(fund, {}), index),
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="pa-fund-action-divider"></div>', unsafe_allow_html=True)
                action_cols = st.columns([1, 1], gap="small")
                with action_cols[0]:
                    if st.button(
                        "Add to Portfolio",
                        key=f"add_compare_{fund}",
                        use_container_width=True,
                    ):
                        add_model_fund_to_portfolio(fund)
                        st.rerun()
                with action_cols[1]:
                    if st.button(
                        "View Fact Sheet →",
                        key=f"factsheet_compare_{fund}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_model_fund_for_fact_sheet"] = fund
                        st.session_state["selected_fact_sheet_fund"] = fund
                        st.session_state["selected_explore_section_v2"] = "Fund Fact Sheet"
                        st.rerun()

    upper_cols = st.columns([0.44, 0.56], gap="medium")
    with upper_cols[0]:
        with st.container(border=True):
            st.markdown(
                '<div class="pa-compare-native-card-title">Risk vs Return</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                risk_return_chart(metrics, meta),
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )
    with upper_cols[1]:
        with st.container(border=True):
            st.markdown(
                '<div class="pa-compare-native-card-title">Growth of $1</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                growth_comparison_chart(growth, selected_funds),
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )

    lower_cols = st.columns([0.45, 0.55], gap="medium")
    with lower_cols[0]:
        with st.container(border=True):
            st.markdown(
                '<div class="pa-compare-native-card-title">Key Metrics Comparison</div>'
                f"{comparison_table_html(metrics, meta)}",
                unsafe_allow_html=True,
            )
    with lower_cols[1]:
        with st.container(border=True):
            st.markdown(
                '<div class="pa-compare-native-card-title">Risk Comparison</div>',
                unsafe_allow_html=True,
            )
            risk_cols = st.columns(2, gap="small")
            with risk_cols[0]:
                st.plotly_chart(
                    horizontal_metric_chart(metrics, "maximum_drawdown", "Maximum Drawdown"),
                    width="stretch",
                    config={"displayModeBar": False, "responsive": True},
                )
            with risk_cols[1]:
                st.plotly_chart(
                    horizontal_metric_chart(
                        metrics,
                        "annualised_volatility",
                        "Annualised Volatility",
                    ),
                    width="stretch",
                    config={"displayModeBar": False, "responsive": True},
                )
        with st.container(border=True):
            st.markdown(
                '<div class="pa-compare-native-card-title">Comparison Insight</div>'
                f'<div class="pa-compare-insight">{escape(comparison_insight(metrics, period))}</div>',
                unsafe_allow_html=True,
            )

    selected_for_portfolio = st.session_state.get("portfolio_selected_funds", [])
    render_selected_funds_strip(selected_for_portfolio)


def render_fund_fact_sheet_content(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
) -> None:
    all_funds = ordered_model_funds(fund_returns)
    if not all_funds:
        st.warning("Saved fund return data is not available.")
        return

    meta = fund_metadata(fund_returns)
    default_fund = st.session_state.get(
        "selected_fact_sheet_fund",
        st.session_state.get("selected_model_fund_for_fact_sheet", all_funds[0]),
    )
    if default_fund not in all_funds:
        default_fund = all_funds[0]

    with st.container(border=True):
        controls = st.columns([1.45, 0.95, 0.85, 0.95], gap="medium")
        with controls[0]:
            selected_fund = st.selectbox(
                "Fund",
                all_funds,
                index=all_funds.index(default_fund),
                key="selected_fact_sheet_fund",
                format_func=lambda fund: fund_label(str(fund)),
            )
        with controls[1]:
            period = st.segmented_control(
                "Performance Window",
                ["1Y", "2Y", "Full History"],
                default=st.session_state.get("fact_sheet_period", "1Y"),
                key="fact_sheet_period",
            )
        with controls[2]:
            top_n_label = st.selectbox(
                "Top Holdings",
                ["Top 5", "Top 10", "Top 15"],
                index=1,
                key="fact_sheet_top_holdings",
            )
        with controls[3]:
            st.markdown('<div style="height: 1.72rem;"></div>', unsafe_allow_html=True)
            if st.button(
                "Add to Portfolio",
                key=f"add_fact_sheet_{selected_fund}",
                use_container_width=True,
            ):
                add_model_fund_to_portfolio(str(selected_fund))
                st.rerun()

    selected_fund = str(selected_fund)
    period = str(period or "1Y")
    top_n = int(str(top_n_label).replace("Top ", ""))
    fund_meta = meta.get(selected_fund, {})
    fund_data = single_fund_return_window(fund_returns, selected_fund, period)
    metrics = calculate_period_metrics(fund_returns, [selected_fund], period).iloc[0]
    growth_drawdown = single_fund_growth_drawdown(fund_data)
    holdings, holdings_date, holdings_fallback = latest_fund_holdings(
        fund_weights,
        selected_fund,
        top_n,
    )
    latest_return_date = fund_data["date"].max() if not fund_data.empty else None
    first_return_date = fund_data["date"].min() if not fund_data.empty else None

    identity_html = (
        '<div class="pa-fact-identity">'
        '<div class="pa-fact-main">'
        '<div class="pa-fact-label">Fund</div>'
        f'<div class="pa-fact-name">{escape(fund_label(selected_fund))}</div>'
        f'<div class="pa-fact-summary">{escape(fund_objective(fund_meta))}</div>'
        "</div>"
        '<div class="pa-fact-mini">'
        '<div class="pa-fact-label">Universe</div>'
        f'<div class="pa-fact-value">{escape(fund_meta.get("asset_class", "Not available"))}</div>'
        "</div>"
        '<div class="pa-fact-mini">'
        '<div class="pa-fact-label">Method</div>'
        f'<div class="pa-fact-value">{escape(fund_meta.get("method", "Not available"))}</div>'
        "</div>"
        '<div class="pa-fact-mini">'
        '<div class="pa-fact-label">As of</div>'
        f'<div class="pa-fact-value">{escape(latest_return_date.date().isoformat() if latest_return_date is not None else "Not available")}</div>'
        "</div>"
        "</div>"
    )
    st.markdown(identity_html, unsafe_allow_html=True)

    metric_specs = [
        ("Annualised Return", "annualised_return", "Compounded from selected window", True),
        ("Annualised Volatility", "annualised_volatility", "Daily return volatility × √252", True),
        ("Sharpe Ratio", "sharpe_ratio", "Return per unit of volatility", False),
        ("Max Drawdown", "maximum_drawdown", "Worst peak-to-trough decline", True),
    ]
    kpi_html = '<div class="pa-fact-kpi-grid">'
    for label, column, support, pct in metric_specs:
        value = metrics.get(column)
        display = "Not available" if pd.isna(value) else (
            format_pct(float(value)) if pct else format_number(float(value), 2)
        )
        css = value_class(float(value)) if column in {"annualised_return", "maximum_drawdown"} and not pd.isna(value) else ""
        kpi_html += render_kpi_card(label, display, support, css)
    kpi_html += "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)

    chart_cols = st.columns(2, gap="medium")
    with chart_cols[0]:
        with st.container(border=True):
            st.markdown(
                '<div class="pa-compare-native-card-title">Growth of $1</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                growth_of_one_chart(growth_drawdown, selected_fund),
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )
    with chart_cols[1]:
        with st.container(border=True):
            st.markdown(
                '<div class="pa-compare-native-card-title">Drawdown</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                drawdown_chart(growth_drawdown, selected_fund),
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )

    lower_cols = st.columns([0.64, 0.36], gap="medium")
    with lower_cols[0]:
        with st.container(border=True):
            st.markdown(
                '<div class="pa-compare-native-card-title">Current Holdings</div>',
                unsafe_allow_html=True,
            )
            if holdings.empty:
                st.info(holdings_fallback)
            else:
                st.plotly_chart(
                    holdings_bar_chart(holdings, top_n),
                    width="stretch",
                    config={"displayModeBar": False, "responsive": True},
                )
                st.caption(
                    f"{holdings_fallback} As of "
                    f"{holdings_date.date().isoformat() if holdings_date is not None else 'latest available date'}."
                )
    with lower_cols[1]:
        holdings_count = (
            int(
                fund_weights.loc[fund_weights["fund_name"].eq(selected_fund), "asset"]
                .dropna()
                .nunique()
            )
            if not fund_weights.empty and "fund_name" in fund_weights.columns
            else 0
        )
        latest_rebalance = (
            holdings_date.date().isoformat() if holdings_date is not None else "Not available"
        )
        summary_html = (
            '<div class="pa-fact-mini">'
            '<div class="pa-fact-section-title">Key Facts</div>'
            f'<div class="pa-fact-label">Asset class</div><div class="pa-fact-value">{escape(fund_meta.get("asset_class", "Not available"))}</div>'
            '<div style="height:0.55rem"></div>'
            f'<div class="pa-fact-label">Number of holdings</div><div class="pa-fact-value">{holdings_count or "Not available"}</div>'
            '<div style="height:0.55rem"></div>'
            f'<div class="pa-fact-label">Latest rebalance date</div><div class="pa-fact-value">{escape(latest_rebalance)}</div>'
            '<div style="height:0.55rem"></div>'
            f'<div class="pa-fact-label">Transaction cost assumption</div><div class="pa-fact-value">{COST_CONTEXT_BPS} bps context</div>'
            "</div>"
        )
        st.markdown(summary_html, unsafe_allow_html=True)
        st.markdown(
            '<div style="height:0.72rem"></div>'
            '<div class="pa-fact-insight">'
            '<div class="pa-fact-label">Portfolio Insight</div>'
            f'<div class="pa-fact-value">{escape(fact_sheet_insight(metrics, period))}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        if first_return_date is not None and latest_return_date is not None:
            st.caption(
                f"Selected window: {first_return_date.date().isoformat()} to "
                f"{latest_return_date.date().isoformat()}."
            )


def render_build_portfolio(artifacts: dict[str, pd.DataFrame]) -> None:
    fund_returns = artifacts["fund_returns"]
    fund_weights = artifacts.get("fund_weights", pd.DataFrame())
    latest_date = (
        fund_returns["date"].max()
        if not fund_returns.empty
        else artifacts["sentiment"]["mapped_trading_date"].max()
    )
    render_topbar(
        latest_date,
        "Build Portfolio",
        "Create your custom portfolio by allocating across selected model funds.",
    )
    st.markdown(
        '<div class="pa-info-banner">'
        "Build your baseline portfolio below. This allocation becomes the starting point "
        "for PulseAlloc's adaptive risk engine."
        '<span class="pa-pill active" style="margin-left:0.65rem;">Historical Research Mode</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    selected_funds = selected_build_portfolio_funds()
    available_funds = set(ordered_model_funds(fund_returns))
    selected_funds = [fund for fund in selected_funds if fund in available_funds]
    st.session_state["portfolio_selected_funds"] = selected_funds
    st.session_state["build_portfolio_selected_funds"] = selected_funds

    if not selected_funds:
        with st.container(border=True):
            st.markdown(
                '<div class="pa-build-section-title">No model funds selected yet.</div>'
                '<div class="pa-build-section-copy">Select funds in Explore Funds before building your portfolio.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Go to Explore Funds →", key="build_empty_go_explore"):
                st.session_state["_nav_target"] = "Explore Funds"
                st.rerun()
        return

    meta = fund_metadata(fund_returns)
    weights_pct = initialise_build_weights(selected_funds)
    if st.session_state.pop("build_reset_requested", False):
        reset_weights = st.session_state.get(
            "build_portfolio_initial_weights",
            equal_weight_percentages(selected_funds),
        )
        weights_pct = {
            fund: float(reset_weights.get(fund, 0.0))
            for fund in selected_funds
        }
        st.session_state["build_portfolio_weights"] = weights_pct
        for fund in selected_funds:
            st.session_state[f"build_weight_slider_{fund}"] = float(
                weights_pct.get(fund, 0.0)
            )

    left_col, right_col = st.columns([0.46, 0.54], gap="medium")

    with left_col:
        with st.container(border=True):
            st.markdown(
                '<div class="pa-build-section-title">1. Allocate Your Portfolio</div>'
                '<div class="pa-build-section-copy">Adjust fund weights below. Total allocation must equal 100%.</div>',
                unsafe_allow_html=True,
            )
            for fund in list(selected_funds):
                row = st.columns([0.39, 0.43, 0.11, 0.07], gap="small", vertical_alignment="center")
                with row[0]:
                    fund_meta = meta.get(fund, {})
                    st.markdown(
                        f'<div class="pa-card-title">{escape(fund_label(fund))}</div>'
                        f'<div class="pa-mini-support">{escape(fund_meta.get("asset_class", "Not available"))} · '
                        f'{escape(fund_meta.get("method", "Not available"))}</div>',
                        unsafe_allow_html=True,
                    )
                slider_key = f"build_weight_slider_{fund}"
                if slider_key not in st.session_state:
                    st.session_state[slider_key] = float(weights_pct.get(fund, 0.0))
                with row[1]:
                    value = st.slider(
                        f"{fund_label(fund)} allocation",
                        0.0,
                        100.0,
                        step=1.0,
                        key=slider_key,
                        label_visibility="collapsed",
                    )
                with row[2]:
                    st.markdown(
                        f'<div class="pa-fact-value">{float(value):.0f}%</div>',
                        unsafe_allow_html=True,
                    )
                with row[3]:
                    if st.button("×", key=f"remove_build_{fund}", help="Remove fund"):
                        updated = [item for item in selected_funds if item != fund]
                        st.session_state["portfolio_selected_funds"] = updated
                        st.session_state["build_portfolio_selected_funds"] = updated
                        current_weights = st.session_state.get("build_portfolio_weights", {}).copy()
                        current_weights.pop(fund, None)
                        st.session_state["build_portfolio_weights"] = current_weights
                        st.rerun()
                st.session_state["build_portfolio_weights"][fund] = float(value)
                st.markdown('<div class="pa-divider"></div>', unsafe_allow_html=True)

            total_pct = sum(
                float(st.session_state["build_portfolio_weights"].get(fund, 0.0))
                for fund in selected_funds
            )
            status_text, allocation_valid, status_class = build_allocation_status(total_pct)
            st.markdown(
                f'<span class="pa-allocation-status {status_class}">{escape(status_text)}</span>',
                unsafe_allow_html=True,
            )
            if st.button("+ Add Another Fund", key="build_add_another_fund"):
                st.session_state["_nav_target"] = "Explore Funds"
                st.rerun()

        with st.container(border=True):
            st.markdown(
                '<div class="pa-build-section-title">2. Choose Your Risk Profile</div>'
                "<div class=\"pa-build-section-copy\">We'll help you understand how your portfolio aligns.</div>",
                unsafe_allow_html=True,
            )
            selected_profile = st.segmented_control(
                "Risk Profile",
                ["Conservative", "Balanced", "Growth"],
                default=st.session_state.get("build_risk_profile", "Balanced"),
                key="build_risk_profile",
                label_visibility="collapsed",
            )
            profile_cols = st.columns(3, gap="small")
            profile_copy = {
                "Conservative": ("Lower risk", "Lower return potential"),
                "Balanced": ("Moderate risk", "Moderate return potential"),
                "Growth": ("Higher risk", "Higher return potential"),
            }
            for col, profile in zip(profile_cols, profile_copy):
                active = " active" if profile == selected_profile else ""
                with col:
                    first, second = profile_copy[profile]
                    st.markdown(
                        f'<div class="pa-profile-card{active}">'
                        f'<div class="pa-profile-title">{escape(profile)}</div>'
                        f'<div class="pa-profile-copy">{escape(first)}<br>{escape(second)}</div>'
                        "</div>",
                        unsafe_allow_html=True,
                    )

    current_weights = {
        fund: float(st.session_state["build_portfolio_weights"].get(fund, 0.0))
        for fund in selected_funds
    }
    selected_profile = str(st.session_state.get("build_risk_profile", "Balanced"))
    selected_window = st.session_state.get("build_performance_window", "1Y")
    portfolio_returns = calculate_custom_portfolio_returns(
        fund_returns,
        selected_funds,
        current_weights,
        str(selected_window),
    )
    portfolio_metrics = calculate_custom_portfolio_metrics(portfolio_returns)
    exposure = asset_exposure_frame(fund_weights, selected_funds, current_weights)
    profile_message, profile_class = ui_risk_alignment(
        portfolio_metrics,
        exposure,
        selected_profile,
    )

    with left_col:
        with st.container(border=True):
            st.markdown(
                '<div class="pa-build-section-title">Historical Portfolio Performance</div>'
                '<div class="pa-build-section-copy">Historical growth of the current baseline allocation over the selected performance window.</div>',
                unsafe_allow_html=True,
            )
            growth_frame = custom_portfolio_growth_frame(portfolio_returns)
            if not growth_frame.empty:
                st.markdown(
                    '<div class="pa-compare-native-card-title">Growth of $1</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    custom_growth_chart(growth_frame),
                    width="stretch",
                    config={"displayModeBar": False, "responsive": True},
                )

    with right_col:
        with st.container(border=True):
            preview_header = st.columns([0.62, 0.38], gap="medium", vertical_alignment="center")
            with preview_header[0]:
                st.markdown(
                    '<div class="pa-build-section-title">3. Portfolio Preview</div>'
                    '<div class="pa-build-section-copy">Performance statistics based on your current allocation.</div>',
                    unsafe_allow_html=True,
                )
            with preview_header[1]:
                selected_window = st.segmented_control(
                    "Performance Window",
                    ["1Y", "2Y", "Full History"],
                    default=str(selected_window),
                    key="build_performance_window",
                    label_visibility="collapsed",
                )

            portfolio_returns = calculate_custom_portfolio_returns(
                fund_returns,
                selected_funds,
                current_weights,
                str(selected_window),
            )
            portfolio_metrics = calculate_custom_portfolio_metrics(portfolio_returns)
            exposure = asset_exposure_frame(fund_weights, selected_funds, current_weights)
            metric_specs = [
                ("Annualised Return", "annualised_return", True),
                ("Annualised Volatility", "annualised_volatility", True),
                ("Sharpe Ratio", "sharpe_ratio", False),
                ("Max Drawdown", "maximum_drawdown", True),
            ]
            kpi_cols = st.columns(4, gap="small")
            for col, (label, column, pct) in zip(kpi_cols, metric_specs):
                value = portfolio_metrics.get(column)
                display = "Not available" if value is None else (
                    format_pct(float(value)) if pct else format_number(float(value), 2)
                )
                css = value_class(float(value)) if value is not None and column in {"annualised_return", "maximum_drawdown"} else ""
                with col:
                    st.markdown(
                        render_kpi_card(label, display, str(selected_window), css),
                        unsafe_allow_html=True,
                    )

            chart_cols = st.columns([0.48, 0.52], gap="medium")
            with chart_cols[0]:
                st.markdown(
                    '<div class="pa-compare-native-card-title">Portfolio Allocation</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    allocation_donut_for_funds(selected_funds, current_weights, total_pct),
                    width="stretch",
                    config={"displayModeBar": False, "responsive": True},
                )
            with chart_cols[1]:
                st.markdown(
                    '<div class="pa-compare-native-card-title">Asset Class Exposure</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    exposure_bar_chart(exposure),
                    width="stretch",
                    config={"displayModeBar": False, "responsive": True},
                )

        with st.container(border=True):
            st.markdown(
                '<div class="pa-build-section-title">4. Portfolio Insight</div>'
                f'<div class="pa-inline-insight {profile_class}">{escape(profile_message)}</div>'
                f'<div class="pa-note">{escape(build_portfolio_insight(selected_funds, current_weights, portfolio_metrics, exposure, selected_profile, str(selected_window)))}</div>',
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown(
                '<div class="pa-build-section-title">5. Portfolio Readiness</div>'
                '<div class="pa-build-section-copy">Check whether your baseline portfolio is ready for the adaptive risk engine.</div>',
                unsafe_allow_html=True,
            )
            if total_pct > 100.05:
                allocation_secondary = f"Reduce allocation by {total_pct - 100:.0f}%"
                allocation_status = "Overallocated"
                allocation_class = "pa-red"
            elif allocation_valid:
                allocation_secondary = "Fully allocated"
                allocation_status = "Complete ✓"
                allocation_class = "pa-green"
            else:
                allocation_secondary = f"{100 - total_pct:.0f}% still unallocated"
                allocation_status = "Incomplete"
                allocation_class = "pa-muted"

            selected_count = len(selected_funds)
            if selected_count <= 1:
                diversification_secondary = "Limited diversification"
                diversification_status = "Needs more funds"
                diversification_class = "pa-muted"
            elif selected_count == 2:
                diversification_secondary = "Moderate diversification"
                diversification_status = "Two model funds"
                diversification_class = "pa-muted"
            else:
                diversification_secondary = "Broader diversification"
                diversification_status = "Three or more funds"
                diversification_class = "pa-green"

            risk_headline, risk_secondary, risk_status = readiness_risk_status(
                profile_message,
                selected_profile,
            )
            risk_class = (
                "pa-green"
                if "Aligned" in risk_status
                else "pa-red"
                if "Above" in risk_status
                else "pa-muted"
            )

            readiness_cards = st.columns(3, gap="small")
            card_specs = [
                (
                    "Allocation",
                    f"{total_pct:.0f}% / 100%",
                    allocation_secondary,
                    allocation_status,
                    allocation_class,
                ),
                (
                    "Diversification",
                    f"{selected_count} model fund" + ("" if selected_count == 1 else "s"),
                    diversification_secondary,
                    diversification_status,
                    diversification_class,
                ),
                (
                    "Risk Profile Alignment",
                    risk_headline,
                    risk_secondary,
                    risk_status,
                    risk_class,
                ),
            ]
            for col, (label, value, secondary, status, css_class) in zip(
                readiness_cards,
                card_specs,
            ):
                with col:
                    st.markdown(
                        '<div class="pa-profile-card">'
                        f'<div class="pa-fact-label">{escape(label)}</div>'
                        f'<div class="pa-fact-value">{escape(value)}</div>'
                        f'<div class="pa-mini-support">{escape(secondary)}</div>'
                        f'<div class="pa-card-badge {css_class}">{escape(status)}</div>'
                        "</div>",
                        unsafe_allow_html=True,
                    )
            st.markdown(
                f'<div class="pa-inline-insight">{escape(readiness_summary(total_pct, selected_count, allocation_valid))}</div>',
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        action_cols = st.columns([0.28, 0.24, 0.48], gap="medium", vertical_alignment="center")
        with action_cols[0]:
            st.markdown(
                f'<div class="pa-mini-label">Selected Funds</div>'
                f'<div class="pa-fact-value">{len(selected_funds)} model funds</div>',
                unsafe_allow_html=True,
            )
        with action_cols[1]:
            st.markdown(
                f'<div class="pa-mini-label">Total Allocation</div>'
                f'<div class="pa-fact-value {("pa-green" if allocation_valid else "pa-red")}">{total_pct:.0f}%</div>'
                f'<div class="pa-mini-support">{escape(status_text)}</div>',
                unsafe_allow_html=True,
            )
        with action_cols[2]:
            reset_col, save_col = st.columns([0.42, 0.58], gap="small")
            with reset_col:
                if st.button("Reset Allocation", key="build_reset_allocation", use_container_width=True):
                    st.session_state["build_reset_requested"] = True
                    st.rerun()
            with save_col:
                if st.button(
                    "Save Portfolio — Use in Adaptive Allocation",
                    key="build_save_portfolio",
                    disabled=not allocation_valid,
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state["baseline_portfolio"] = {
                        "selected_funds": selected_funds,
                        "fund_weights": {
                            fund: float(current_weights.get(fund, 0.0)) / 100.0
                            for fund in selected_funds
                        },
                        "risk_profile": selected_profile,
                        "performance_window": str(selected_window),
                        "data_as_of": latest_date.date().isoformat(),
                        "source": "Build Portfolio",
                    }
                    st.session_state["build_portfolio_saved"] = True
                    st.success("Baseline portfolio saved for Adaptive Allocation.")
                if st.session_state.get("build_portfolio_saved"):
                    if st.button(
                        "Continue to Adaptive Allocation →",
                        key="build_continue_adaptive",
                        use_container_width=True,
                    ):
                        st.session_state["_nav_target"] = "Adaptive Allocation"
                        st.rerun()


def valid_baseline_portfolio() -> dict | None:
    baseline = st.session_state.get("baseline_portfolio")
    if not isinstance(baseline, dict):
        return None
    selected_funds = baseline.get("selected_funds")
    fund_weights = baseline.get("fund_weights")
    if not selected_funds or not isinstance(fund_weights, dict):
        return None
    return baseline


def adaptive_level(score: float | None) -> tuple[str, str, str]:
    if score is None or pd.isna(score):
        return "Not available", "pa-muted", "amber"
    if score < 25:
        return "Low", "pa-green", "green"
    if score < 50:
        return "Moderate", "pa-muted", "amber"
    if score < 75:
        return "Elevated", "pa-red", "red"
    return "High", "pa-red", "red"


def adaptive_series(stage7_returns: pd.DataFrame, profile: str, window: str) -> pd.DataFrame:
    frame = stage7_returns[
        stage7_returns["profile"].astype(str).str.lower().eq(profile)
        & stage7_returns["fund_type"].astype(str).str.lower().isin(["fixed", "adaptive"])
    ].sort_values(["fund_type", "date"]).copy()
    if frame.empty:
        return frame
    if window != "Full History":
        years = 1 if window == "1Y" else 2
        frame = frame[frame["date"] >= frame["date"].max() - pd.DateOffset(years=years)]
    frame["display_growth"] = frame.groupby("fund_type")["net_return"].transform(
        lambda x: (1 + x.astype(float)).cumprod()
    )
    frame["display_growth"] = frame.groupby("fund_type")["display_growth"].transform(
        lambda x: x / x.iloc[0]
    )
    frame["drawdown"] = frame.groupby("fund_type")["display_growth"].transform(
        lambda x: x / x.cummax() - 1
    )
    return frame


def adaptive_evidence_metrics(series: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for fund_type, group in series.groupby("fund_type"):
        returns = group.sort_values("date")["net_return"].astype(float)
        if returns.empty:
            continue
        growth = (1 + returns).cumprod()
        drawdown = growth / growth.cummax() - 1
        tail = returns[returns <= returns.quantile(0.05)]
        annual_return = growth.iloc[-1] ** (252 / len(returns)) - 1
        annual_vol = returns.std() * (252 ** 0.5)
        metrics[str(fund_type).lower()] = {
            "Annualised Return": float(annual_return),
            "Annualised Volatility": float(annual_vol),
            "Max Drawdown": float(drawdown.min()),
            "Realised 95% CVaR": float(tail.mean()) if not tail.empty else 0.0,
            "Sharpe Ratio": float(annual_return / annual_vol) if annual_vol else 0.0,
        }
    return metrics


def adaptive_line_chart(series: pd.DataFrame, view: str) -> go.Figure:
    fig = go.Figure()
    labels = {"fixed": "Fixed Baseline", "adaptive": "Adaptive Portfolio"}
    colors = {"fixed": "#64748B", "adaptive": "#14866D"}
    y_col = "display_growth" if view == "Growth of $1" else "drawdown"
    for fund_type in ["fixed", "adaptive"]:
        data = series[series["fund_type"].astype(str).str.lower().eq(fund_type)]
        if data.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data[y_col],
                mode="lines",
                name=labels[fund_type],
                line={"color": colors[fund_type], "width": 2.2},
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>",
            )
        )
    fig = apply_plotly_theme(fig, height=310)
    fig.update_layout(margin={"l": 34, "r": 12, "t": 8, "b": 32})
    fig.update_yaxes(title=view, tickformat=".0%" if view == "Drawdown" else None)
    return fig


def render_adaptive_allocation(artifacts: dict[str, pd.DataFrame]) -> None:
    stage7_returns = artifacts.get("stage7_returns_net", pd.DataFrame())
    rebalance = artifacts.get("rebalance", pd.DataFrame())
    fund_returns = artifacts.get("fund_returns", pd.DataFrame())
    latest_date = (
        stage7_returns["date"].max()
        if not stage7_returns.empty and "date" in stage7_returns
        else fund_returns["date"].max()
        if not fund_returns.empty and "date" in fund_returns
        else artifacts["sentiment"]["mapped_trading_date"].max()
    )
    st.markdown(
        f"""
        <div class="pa-topbar">
            <div>
                <div class="aa-headline">
                    <div class="pa-title">Adaptive Allocation</div>
                    <span class="pa-pill active">✦ Innovation Core</span>
                </div>
                <div class="pa-subtitle">Personalised downside-risk-aware portfolio adjustment.</div>
                <div class="pa-history">
                    Historical data through {latest_date.date().isoformat()}
                    <span class="pa-pill" style="margin-left:0.55rem;">Historical Research Mode</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    baseline = valid_baseline_portfolio()
    if baseline is None:
        with st.container(border=True):
            st.markdown(
                '<div class="aa-module-title">No saved baseline portfolio.</div>'
                '<div class="pa-build-section-copy">Build and save a portfolio first.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Go to Build Portfolio →", key="adaptive_empty_go_build"):
                st.session_state["_nav_target"] = "Build Portfolio"
                st.rerun()
        return
    if rebalance.empty or stage7_returns.empty:
        st.info("Saved adaptive-allocation research outputs are not available.")
        return

    available = set(rebalance["profile"].astype(str).str.lower())
    profiles = [name for name in ["Conservative", "Balanced", "Growth"] if name.lower() in available]
    default_profile = str(baseline.get("risk_profile", "Balanced")).title()
    if default_profile not in profiles:
        default_profile = profiles[0]

    with st.container(border=True):
        st.markdown(
            '<div class="aa-module-title">Adaptive Allocation Engine</div>'
            '<div class="pa-build-section-copy">Explore how investor profile and saved downside-risk conditions change the portfolio.</div>',
            unsafe_allow_html=True,
        )
        control_cols = st.columns(2, gap="medium")
        with control_cols[0]:
            st.markdown('<div class="pa-mini-label">Investor Profile</div>', unsafe_allow_html=True)
            selected_profile_display = st.segmented_control(
                "Investor Profile",
                profiles,
                default=default_profile,
                key="adaptive_engine_profile",
                label_visibility="collapsed",
            )
        selected_profile = str(selected_profile_display).lower()
        profile_rows = rebalance[rebalance["profile"].astype(str).str.lower().eq(selected_profile)]
        adaptive_rows = profile_rows[profile_rows["fund_type"].astype(str).str.lower().eq("adaptive")].sort_values("decision_date")
        fixed_rows = profile_rows[profile_rows["fund_type"].astype(str).str.lower().eq("fixed")].sort_values("decision_date")
        date_options = list(adaptive_rows["decision_date"].dropna().sort_values().unique())
        with control_cols[1]:
            st.markdown('<div class="pa-mini-label">Historical Risk Observation</div>', unsafe_allow_html=True)
            selected_date = st.selectbox(
                "As of Date",
                date_options,
                index=len(date_options) - 1,
                format_func=lambda value: pd.Timestamp(value).date().isoformat(),
                key="adaptive_engine_date",
                label_visibility="collapsed",
            )

        adaptive_row = adaptive_rows[adaptive_rows["decision_date"].eq(selected_date)].tail(1).iloc[0]
        fixed_row = fixed_rows[fixed_rows["decision_date"].eq(selected_date)].tail(1).iloc[0]
        base_crypto = float(fixed_row["target_crypto_budget"])
        base_equity = float(fixed_row["target_equity_budget"])
        adaptive_crypto = float(adaptive_row["target_crypto_budget"])
        adaptive_equity = float(adaptive_row["target_equity_budget"])
        risk_score = float(adaptive_row["risk_score"])
        downside_risk = float(adaptive_row["rolling_tail_risk"])
        risk_scaler = adaptive_crypto / base_crypto if base_crypto else 0.0
        prev_score = adaptive_row.get("previous_risk_score")
        trend_delta = None if pd.isna(prev_score) else risk_score - float(prev_score)
        risk_trend = "Rising" if trend_delta is not None and trend_delta > 1 else "Falling" if trend_delta is not None and trend_delta < -1 else "Stable"
        trend_icon = "↑" if risk_trend == "Rising" else "↓" if risk_trend == "Falling" else "→"
        risk_level, risk_class, risk_color = adaptive_level(risk_score)
        change = adaptive_crypto - base_crypto
        action = "Reduce Crypto Exposure" if change < -0.005 else "Allow Higher Crypto Exposure" if change > 0.005 else "Maintain Crypto Exposure"
        scaler_direction = "Exposure scaled down" if risk_scaler < 0.99 else "Exposure maintained" if risk_scaler <= 1.01 else "Exposure scaled up"
        max_downside = max(float(adaptive_rows["rolling_tail_risk"].max()), downside_risk, 0.001)
        downside_width = min(max(downside_risk / max_downside, 0), 1) * 100
        st.markdown(
            f"""
            <div class="aa-flow">
                <div class="aa-zone">
                    <div class="aa-zone-label">BASELINE</div>
                    <div class="pa-mini-support">{escape(str(selected_profile_display))}</div>
                    <div class="aa-big">{format_pct(base_crypto)}</div>
                    <div class="pa-mini-label">Base Crypto Budget</div>
                    <div class="pa-mini-support">Equity {format_pct(base_equity)}</div>
                </div>
                <div class="aa-arrow">→</div>
                <div class="aa-zone engine">
                    <div class="aa-zone-label">RISK ENGINE</div>
                    <div class="aa-signal">
                        <div class="aa-rowline"><div class="pa-card-title">Crypto Risk Score</div><span class="pa-pill {risk_class}">{escape(risk_level)}</span></div>
                        <div class="pa-fact-value">{risk_score:.1f} / 100</div>
                        <div class="aa-meter"><span class="{risk_color}" style="width:{min(max(risk_score, 0), 100):.1f}%;"></span></div>
                    </div>
                    <div class="aa-signal">
                        <div class="aa-rowline"><div class="pa-card-title">Downside Risk</div><div class="pa-fact-value">{format_pct(downside_risk)}</div></div>
                        <div class="aa-meter"><span class="{risk_color}" style="width:{downside_width:.1f}%;"></span></div>
                    </div>
                    <div class="aa-rowline"><div class="pa-card-title">Risk Trend</div><div class="pa-fact-value">{trend_icon} {escape(risk_trend)}</div></div>
                    <div class="aa-rowline"><div class="pa-card-title">Risk Scaler</div><div class="pa-fact-value">{risk_scaler:.2f}×</div></div>
                    <div class="pa-mini-support">{escape(scaler_direction)}</div>
                </div>
                <div class="aa-arrow">→</div>
                <div class="aa-zone">
                    <div class="aa-zone-label">ADAPTIVE</div>
                    <div class="aa-big pa-green">{format_pct(adaptive_crypto)}</div>
                    <div class="pa-mini-label">Adaptive Crypto Budget</div>
                    <div class="{value_class(change)}">{format_pp(change)}</div>
                    <div class="pa-mini-support">Adaptive Equity {format_pct(adaptive_equity)}</div>
                </div>
            </div>
            <div class="pa-inline-insight">PulseAlloc applies a {risk_scaler:.2f}× risk scaler to the {escape(str(selected_profile_display))} portfolio's {format_pct(base_crypto)} base crypto budget, producing a {format_pct(adaptive_crypto)} adaptive crypto target.</div>
            """,
            unsafe_allow_html=True,
        )

    response_col, why_col = st.columns(2, gap="medium")

    def bar_pair(asset: str, base: float, adaptive: float) -> str:
        return (
            f'<div class="aa-bar-row"><div class="pa-card-title">{escape(asset)}</div>'
            f'<div class="aa-bar-label"><span>Baseline</span><span>{format_pct(base)}</span></div>'
            f'<div class="aa-bar"><span style="width:{base * 100:.1f}%;"></span></div>'
            f'<div class="aa-bar-label"><span>Adaptive</span><span>{format_pct(adaptive)}</span></div>'
            f'<div class="aa-bar"><span class="adaptive" style="width:{adaptive * 100:.1f}%;"></span></div></div>'
        )

    with response_col:
        with st.container(border=True):
            exposure_text = f"Crypto exposure ↓ {abs(change) * 100:.2f} pp" if change < -0.0005 else f"Crypto exposure ↑ {change * 100:.2f} pp" if change > 0.0005 else "Crypto exposure maintained"
            st.markdown(
                '<div class="aa-module-title">Allocation Response</div>'
                '<div class="pa-build-section-copy">See how the baseline portfolio changes after risk scaling.</div>'
                + bar_pair("Equity", base_equity, adaptive_equity)
                + bar_pair("Crypto", base_crypto, adaptive_crypto)
                + f'<div class="pa-inline-insight">{escape(exposure_text)}</div>',
                unsafe_allow_html=True,
            )

    with why_col:
        with st.container(border=True):
            direction_word = "reduces" if change < -0.0005 else "increases" if change > 0.0005 else "maintains"
            explanation = f"Historical risk state produces a {risk_scaler:.2f}× scaler, so PulseAlloc {direction_word} the {selected_profile_display} portfolio's crypto budget."
            st.markdown(
                '<div class="aa-module-title">Why Did PulseAlloc Change It?</div>'
                '<div class="pa-build-section-copy">Understand the saved risk state behind the allocation decision.</div>'
                f'<div class="aa-rowline"><div><b>Crypto Risk Score</b><div class="pa-mini-support">{risk_score:.1f} / 100</div></div><span class="pa-pill {risk_class}">{escape(risk_level)}</span></div>'
                f'<div class="aa-rowline"><div><b>Downside Risk</b><div class="pa-mini-support">{format_pct(downside_risk)}</div></div><span class="pa-pill {risk_class}">{escape(risk_level)}</span></div>'
                f'<div class="aa-rowline"><div><b>Risk Trend</b><div class="pa-mini-support">{trend_icon} {escape(risk_trend)}</div></div></div>'
                f'<div class="aa-rowline"><div><b>Risk Scaler</b><div class="pa-mini-support">{risk_scaler:.2f}×</div></div></div>'
                '<div class="pa-divider"></div>'
                f'<div class="pa-inline-insight"><b>{escape(action)}</b><br>{escape(explanation)}</div>',
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.markdown(
            '<div class="aa-module-title">Historical Evidence</div>'
            '<div class="pa-build-section-copy">Test whether adaptive allocation improved historical portfolio outcomes.</div>',
            unsafe_allow_html=True,
        )
        evidence_cols = st.columns([0.58, 0.42], gap="medium")
        with evidence_cols[0]:
            evidence_view = st.segmented_control(
                "Evidence View",
                ["Risk & Return", "Growth of $1", "Drawdown"],
                default="Risk & Return",
                key="adaptive_evidence_view",
            )
        with evidence_cols[1]:
            performance_window = st.segmented_control(
                "Performance Window",
                ["1Y", "2Y", "Full History"],
                default="Full History",
                key="adaptive_evidence_window",
            )
        series = adaptive_series(stage7_returns, selected_profile, str(performance_window))
        if evidence_view == "Risk & Return":
            metrics = adaptive_evidence_metrics(series)
            metric_html = '<div class="aa-evidence-grid">'
            for label in ["Annualised Return", "Annualised Volatility", "Max Drawdown", "Realised 95% CVaR", "Sharpe Ratio"]:
                fixed = metrics.get("fixed", {}).get(label)
                adaptive = metrics.get("adaptive", {}).get(label)
                if fixed is None or adaptive is None:
                    continue
                diff = adaptive - fixed
                pct_metric = label != "Sharpe Ratio"
                fixed_display = format_pct(fixed) if pct_metric else f"{fixed:.3f}"
                adaptive_display = format_pct(adaptive) if pct_metric else f"{adaptive:.3f}"
                if label == "Annualised Return" and diff < 0:
                    css = "pa-muted"
                elif label == "Annualised Volatility" and diff < 0:
                    css = "pa-green"
                elif label in {"Max Drawdown", "Realised 95% CVaR"} and diff > 0:
                    css = "pa-green"
                else:
                    css = value_class(diff)
                change_text = format_pp(diff) if pct_metric else f"{diff:+.3f}"
                metric_html += (
                    '<div class="aa-mini-metric">'
                    f'<div class="pa-mini-label">{escape(label)}</div>'
                    f'<div class="pa-mini-support">Fixed {escape(fixed_display)}</div>'
                    f'<div class="pa-mini-support">Adaptive {escape(adaptive_display)}</div>'
                    f'<div class="{css}">{escape(change_text)}</div></div>'
                )
            metric_html += "</div>"
            tradeoff = (
                "Adaptive allocation reduced historical downside risk and volatility, although this came with lower historical return."
                if metrics.get("adaptive", {}).get("Annualised Return", 0) < metrics.get("fixed", {}).get("Annualised Return", 0)
                else "Adaptive allocation changed historical risk and return relative to the fixed baseline."
            )
            st.markdown(metric_html + f'<div class="pa-inline-insight">{escape(tradeoff)}</div>', unsafe_allow_html=True)
        else:
            st.plotly_chart(
                adaptive_line_chart(series, str(evidence_view)),
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )

    st.markdown(
        '<div class="pa-note">Adaptive Allocation uses saved historical research results. '
        'It is not live market data and does not guarantee future performance.</div>',
        unsafe_allow_html=True,
    )


def render_explore_funds(artifacts: dict[str, pd.DataFrame]) -> None:
    fund_returns = artifacts["fund_returns"]
    latest_date = (
        fund_returns["date"].max()
        if not fund_returns.empty
        else artifacts["sentiment"]["mapped_trading_date"].max()
    )
    selected_section = st.radio(
        "Explore Funds section",
        ["Compare Funds", "Fund Fact Sheet"],
        index=["Compare Funds", "Fund Fact Sheet"].index(
            st.session_state.get("selected_explore_section_v2", "Compare Funds")
        ),
        horizontal=True,
        key="selected_explore_section_v2",
        label_visibility="collapsed",
    )
    if selected_section == "Fund Fact Sheet":
        render_topbar(
            latest_date,
            "Fund Fact Sheet",
            "Review historical performance, drawdown and current holdings for a selected model fund.",
        )
    else:
        render_topbar(
            latest_date,
            "Compare Funds",
            "Compare risk, return and portfolio characteristics side by side.",
        )

    if fund_returns.empty:
        st.warning("Saved fund return data is not available.")
        return

    if selected_section == "Fund Fact Sheet":
        render_fund_fact_sheet_content(fund_returns, artifacts.get("fund_weights", pd.DataFrame()))
    else:
        render_compare_funds_content(fund_returns)


def home_profile_options(artifacts: dict[str, pd.DataFrame]) -> list[str]:
    profiles: set[str] = set()
    for key in ("personalisation", "rebalance", "crypto_budgets"):
        data = artifacts.get(key, pd.DataFrame())
        if not data.empty and "profile" in data.columns:
            profiles.update(
                data["profile"].dropna().astype(str).str.lower().str.strip().tolist()
            )
    preferred = ["conservative", "balanced", "growth"]
    ordered = [profile for profile in preferred if profile in profiles]
    ordered.extend(sorted(profiles.difference(ordered)))
    return ordered


def build_home_state(
    artifacts: dict[str, pd.DataFrame],
    selected_profile: str,
) -> dict[str, object]:
    profile = selected_profile.lower().strip()
    personalisation = artifacts["personalisation"]
    rebalance = artifacts["rebalance"]
    budgets = artifacts["crypto_budgets"]
    crypto_risk = artifacts["crypto_risk"]
    sentiment = artifacts["sentiment"]

    comparison_row = (
        personalisation.loc[
            personalisation["profile"].astype(str).str.lower().eq(profile)
        ].iloc[-1]
        if not personalisation.empty
        and personalisation["profile"].astype(str).str.lower().eq(profile).any()
        else None
    )
    adaptive_rows = (
        rebalance.loc[
            rebalance["profile"].astype(str).str.lower().eq(profile)
            & rebalance["fund_type"].astype(str).str.lower().eq("adaptive")
        ].sort_values("effective_date")
        if not rebalance.empty
        else pd.DataFrame()
    )
    latest_rebalance = adaptive_rows.iloc[-1] if not adaptive_rows.empty else None
    latest_budget = (
        budgets.loc[budgets["profile"].astype(str).str.lower().eq(profile)]
        .sort_values("date")
        .iloc[-1]
        if not budgets.empty
        and budgets["profile"].astype(str).str.lower().eq(profile).any()
        else None
    )
    latest_risk = crypto_risk.sort_values("date").iloc[-1] if not crypto_risk.empty else None
    latest_sentiment_date = sentiment["mapped_trading_date"].max()
    recent, _, _ = recent_windows(sentiment, latest_sentiment_date)
    strongest_sector = (
        str(recent.groupby("sector")["sector_sentiment"].mean().idxmax())
        if not recent.empty
        else "Not available"
    )

    if latest_rebalance is not None:
        crypto_weight = float(latest_rebalance["target_crypto_budget"])
        equity_weight = float(latest_rebalance["target_equity_budget"])
        user_explanation = str(latest_rebalance.get("user_explanation", ""))
        no_adjustment = str(latest_rebalance.get("no_adjustment_triggered", "")).lower()
        adaptive_status = (
            "Maintain exposure"
            if no_adjustment == "true"
            else "Reduce exposure"
            if "reduced" in user_explanation.lower()
            else "Increase cautiously"
            if "increased" in user_explanation.lower()
            else "Within target"
        )
    elif latest_budget is not None:
        crypto_weight = float(latest_budget["raw_crypto_budget"])
        equity_weight = float(latest_budget["raw_equity_budget"])
        adaptive_status = "Build portfolio first"
        user_explanation = ""
    else:
        crypto_weight = equity_weight = None
        adaptive_status = "Not available"
        user_explanation = ""

    risk_score = float(latest_risk["risk_score"]) if latest_risk is not None else None
    if risk_score is None:
        crypto_risk_label = "Not available"
        risk_class = "pa-muted"
    elif risk_score >= 75:
        crypto_risk_label = "High"
        risk_class = "pa-red"
    elif risk_score >= 50:
        crypto_risk_label = "Elevated"
        risk_class = "pa-red"
    elif risk_score >= 25:
        crypto_risk_label = "Moderate"
        risk_class = "pa-muted"
    else:
        crypto_risk_label = "Low"
        risk_class = "pa-green"

    return {
        "profile": profile.title(),
        "profile_key": profile,
        "comparison": comparison_row,
        "latest_rebalance": latest_rebalance,
        "latest_budget": latest_budget,
        "latest_risk": latest_risk,
        "crypto_weight": crypto_weight,
        "equity_weight": equity_weight,
        "adaptive_status": adaptive_status,
        "user_explanation": user_explanation,
        "risk_score": risk_score,
        "crypto_risk_label": crypto_risk_label,
        "risk_class": risk_class,
        "strongest_sector": strongest_sector,
        "latest_sentiment_date": latest_sentiment_date,
    }


def render_home(artifacts: dict[str, pd.DataFrame]) -> None:
    latest_date = artifacts["sentiment"]["mapped_trading_date"].max()
    profile_options = home_profile_options(artifacts)
    default_profile = "balanced" if "balanced" in profile_options else (
        profile_options[0] if profile_options else ""
    )
    if "home_risk_profile" not in st.session_state:
        st.session_state["home_risk_profile"] = default_profile
    if st.session_state["home_risk_profile"] not in profile_options and default_profile:
        st.session_state["home_risk_profile"] = default_profile
    selected_profile = st.session_state.get("home_risk_profile", default_profile)
    state = build_home_state(artifacts, selected_profile)
    comparison = state["comparison"]

    render_topbar(
        latest_date,
        "Home",
        "Your portfolio intelligence, in one place.",
    )

    if comparison is not None:
        value_label = "Model Portfolio"
        value = f'{float(comparison["ending_growth_of_1"]):.3f}×'
        expected_return = format_pct(float(comparison["annualised_return"]))
        portfolio_risk = format_pct(float(comparison["annualised_volatility"]))
    else:
        value_label = "Model Portfolio"
        value = "Not available"
        expected_return = "Not available"
        portfolio_risk = "Not available"

    st.markdown('<div class="pa-section-title first">Portfolio Snapshot</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4, gap="small")
    metric_cards = [
        render_home_metric("◼", value_label, value, "Growth of $1", "blue"),
        render_home_metric("↗", "Expected Return", expected_return, "Annualised", "green", "pa-green"),
        render_home_metric("◇", "Portfolio Risk", portfolio_risk, "Annualised volatility", "purple"),
    ]
    for col, card in zip(metric_cols[:3], metric_cards):
        with col:
            st.markdown(card, unsafe_allow_html=True)
    with metric_cols[3]:
        with st.container(border=True):
            icon_col, profile_col = st.columns([0.22, 0.78], gap="small")
            with icon_col:
                st.markdown('<div class="pa-home-icon orange">●</div>', unsafe_allow_html=True)
            with profile_col:
                st.markdown('<div class="pa-kpi-label">Risk Profile</div>', unsafe_allow_html=True)
                st.selectbox(
                    "Risk Profile",
                    profile_options if profile_options else ["not selected"],
                    key="home_risk_profile",
                    format_func=lambda value: str(value).title(),
                    label_visibility="collapsed",
                )
                st.markdown(
                    '<div class="pa-kpi-support">Investor profile</div>',
                    unsafe_allow_html=True,
                )

    allocation_col, status_col = st.columns([0.50, 0.50], gap="medium")
    with allocation_col:
        if state["crypto_weight"] is None:
            st.markdown(home_empty_allocation(), unsafe_allow_html=True)
        else:
            allocation = pd.DataFrame(
                {
                    "asset": ["Equity", "Crypto"],
                    "weight": [state["equity_weight"], state["crypto_weight"]],
                }
            )
            section_header("Current Allocation")
            chart_col, legend_col = st.columns([0.50, 0.50], gap="small")
            with chart_col:
                st.plotly_chart(
                    allocation_donut_chart(allocation, str(state["profile"])),
                    width="stretch",
                    config={"displayModeBar": False},
                )
            with legend_col:
                rows = "".join(
                    f'<div class="pa-allocation-row"><span>{escape(row.asset)}</span>'
                    f"<span>{format_pct(float(row.weight))}</span></div>"
                    for row in allocation.itertuples()
                )
                st.markdown(
                    f'<div class="pa-allocation-legend">{rows}</div>',
                    unsafe_allow_html=True,
                )
            if st.button("View full allocation →", key="home_view_full_allocation"):
                st.session_state["_nav_target"] = "Build Portfolio"
                st.rerun()
    with status_col:
        crypto_budget = (
            format_pct(float(state["crypto_weight"]))
            if state["crypto_weight"] is not None
            else "Not available"
        )
        risk_alignment = "Within target" if state["crypto_weight"] is not None else "Not available"
        risk_score_support = (
            f'Risk score {float(state["risk_score"]):.1f}/100'
            if state["risk_score"] is not None
            else "No current score"
        )
        st.markdown(
            '<div class="pa-home-panel">'
            + section_header_html("Allocation Status")
            + '<div class="pa-home-status-grid">'
            + render_home_status("Risk Alignment", risk_alignment, "Current target weights")
            + render_home_status("Crypto Budget", crypto_budget, "of portfolio", "pa-green")
            + render_home_status(
                "Current Crypto Risk",
                str(state["crypto_risk_label"]),
                risk_score_support,
                str(state["risk_class"]),
            )
            + render_home_status("Adaptive Status", str(state["adaptive_status"]), "System response")
            + "</div></div>",
            unsafe_allow_html=True,
        )
        if st.button("View Adaptive Allocation →", key="home_view_adaptive_allocation"):
            st.session_state["_nav_target"] = "Adaptive Allocation"
            st.rerun()

    st.markdown('<div class="pa-section-title">PulseAlloc Insights</div>', unsafe_allow_html=True)
    portfolio_headline = (
        f'Portfolio remains aligned with the {str(state["profile"]).lower()} target'
        if state["crypto_weight"] is not None
        else "Build a portfolio to activate Home insights"
    )
    adaptive_headline = (
        f'Crypto downside risk is {str(state["crypto_risk_label"]).lower()}'
        if state["risk_score"] is not None
        else "Crypto risk signal is not available"
    )
    sentiment_headline = (
        f'{state["strongest_sector"]} currently shows the strongest sentiment'
        if state["strongest_sector"] != "Not available"
        else "Market sentiment is not available"
    )
    st.markdown(
        '<div class="pa-home-insight-row">'
        + render_home_insight(
            "PORTFOLIO",
            portfolio_headline,
            "Latest saved adaptive weights drive the current allocation view.",
            "View Portfolio →",
        )
        + render_home_insight(
            "ADAPTIVE RISK",
            adaptive_headline,
            str(state["user_explanation"])[:118] if state["user_explanation"] else "Adaptive response uses saved crypto-risk outputs.",
            "View Allocation →",
        )
        + render_home_insight(
            "SENTIMENT",
            sentiment_headline,
            "Based on the latest saved sector sentiment window.",
            "View Sentiment →",
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    insight_nav = st.columns(3, gap="small")
    insight_targets = [
        ("View Portfolio →", "Build Portfolio", "home_insight_portfolio"),
        ("View Allocation →", "Adaptive Allocation", "home_insight_adaptive"),
        ("View Sentiment →", "Sentiment Signals", "home_insight_sentiment"),
    ]
    for col, (label, target, key) in zip(insight_nav, insight_targets):
        with col:
            if st.button(label, key=key):
                st.session_state["_nav_target"] = target
                st.rerun()

    st.markdown('<div class="pa-section-title">Your PulseAlloc Journey</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Explore", "Compare funds", "Explore Funds"),
        ("2", "Build", "Create your portfolio", "Build Portfolio"),
        ("3", "Adapt", "Respond to changing risk", "Adaptive Allocation"),
        ("4", "Monitor", "Track market sentiment", "Sentiment Signals"),
    ]
    st.markdown(
        '<div class="pa-home-journey">'
        + "".join(
            '<div class="pa-home-step">'
            f'<div class="pa-step-number">{escape(num)}</div>'
            f'<div class="pa-card-title">{escape(title)}</div>'
            f'<div class="pa-card-copy">{escape(body)}</div>'
            f'<div class="pa-history">{escape(target)} →</div></div>'
            for num, title, body, target in steps
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    journey_nav = st.columns(4, gap="small")
    for col, (_, title, _, target) in zip(journey_nav, steps):
        with col:
            if st.button(f"{title} →", key=f"home_journey_{target}"):
                st.session_state["_nav_target"] = target
                st.rerun()


def render_sentiment_strategy(artifacts: dict[str, pd.DataFrame]) -> None:
    data = prepare_strategy_data(artifacts)
    metric_lookup = data["metric_lookup"]
    latest_date = artifacts["net_returns"]["date"].max()
    weights = artifacts["weights"].copy()
    weights["delta"] = weights["final_weight"] - weights["base_weight"]

    render_topbar(
        latest_date,
        "Sentiment Strategy",
        "How does sentiment change the portfolio decision?",
    )
    if render_sentiment_subnav() != "Sentiment Strategy":
        return

    cost_levels = sorted(int(x) for x in data["cost"]["Transaction Cost (bps)"].unique())
    default_cost = COST_CONTEXT_BPS if COST_CONTEXT_BPS in cost_levels else cost_levels[0]
    selected_cost_state = st.session_state.get("selected_transaction_cost", default_cost)
    if selected_cost_state not in cost_levels:
        selected_cost_state = default_cost

    ticker_options = sorted(weights["ticker"].dropna().astype(str).unique())
    latest_strategy = weights[weights["date"].eq(weights["date"].max())]
    default_ticker = str(latest_strategy.loc[latest_strategy["delta"].abs().idxmax(), "ticker"])
    inherited_ticker = st.session_state.get("news_ticker_v3")
    if inherited_ticker in ticker_options:
        default_ticker = str(inherited_ticker)

    selected_ticker = st.session_state.get("selected_sentiment_ticker", default_ticker)
    if selected_ticker not in ticker_options:
        selected_ticker = default_ticker
    ticker_dates = [
        pd.Timestamp(date)
        for date in sorted(weights.loc[weights["ticker"].eq(selected_ticker), "date"].unique())
    ]
    if "selected_sentiment_date" not in st.session_state:
        st.session_state["selected_sentiment_date"] = pd.Timestamp(ticker_dates[-1])
    selected_date = pd.Timestamp(st.session_state["selected_sentiment_date"])
    valid_prior_dates = [date for date in ticker_dates if pd.Timestamp(date) <= selected_date]
    selected_date = pd.Timestamp(valid_prior_dates[-1] if valid_prior_dates else ticker_dates[-1])
    st.session_state["selected_sentiment_date"] = selected_date

    window_options = available_windows(data["net_returns"])
    selected_window_state = st.session_state.get("selected_performance_window", "Full OOS")
    if selected_window_state not in window_options:
        selected_window_state = window_options[-1]

    st.markdown(
        '<div class="pa-control-panel"><div class="pa-control-title">Strategy Context</div></div>',
        unsafe_allow_html=True,
    )
    date_col, ticker_col, cost_col, window_col = st.columns(4, gap="small")
    with ticker_col:
        selected_ticker = st.selectbox(
            "Ticker",
            ticker_options,
            index=ticker_options.index(selected_ticker),
            key="selected_sentiment_ticker",
        )
    if selected_ticker != st.session_state.get("_last_sentiment_ticker"):
        ticker_dates = [
            pd.Timestamp(date)
            for date in sorted(weights.loc[weights["ticker"].eq(selected_ticker), "date"].unique())
        ]
        selected_date = pd.Timestamp(ticker_dates[-1])
        st.session_state["selected_sentiment_date"] = selected_date
        st.session_state["_last_sentiment_ticker"] = selected_ticker
    with date_col:
        selected_date = st.selectbox(
            "As of Date",
            ticker_dates,
            index=ticker_dates.index(pd.Timestamp(st.session_state["selected_sentiment_date"])),
            format_func=lambda value: pd.Timestamp(value).date().isoformat(),
        )
        st.session_state["selected_sentiment_date"] = pd.Timestamp(selected_date)
    with cost_col:
        selected_cost = st.selectbox(
            "Transaction Cost",
            cost_levels,
            index=cost_levels.index(int(selected_cost_state)),
            format_func=lambda value: f"{int(value)} bps",
            key="selected_transaction_cost",
        )
    with window_col:
        selected_window = st.selectbox(
            "Performance Window",
            window_options,
            index=window_options.index(selected_window_state),
            key="selected_performance_window",
        )

    current_row = weights[
        weights["ticker"].eq(selected_ticker) & weights["date"].eq(pd.Timestamp(selected_date))
    ]
    if current_row.empty:
        ticker_history = weights[weights["ticker"].eq(selected_ticker)]
        current_row = ticker_history[ticker_history["date"].le(pd.Timestamp(selected_date))]
        current_row = current_row.tail(1) if not current_row.empty else ticker_history.tail(1)
        st.info(
            "Using latest available strategy observation: "
            f"{pd.Timestamp(current_row.iloc[0]['date']).date().isoformat()}"
        )
    current = build_current_signal_state(current_row.iloc[0], artifacts)
    obs_key = f"{current['ticker']}|{pd.Timestamp(current['date']).date().isoformat()}"
    if st.session_state.get("_sentiment_obs_key") != obs_key:
        st.session_state["_sentiment_obs_key"] = obs_key
        st.session_state["_scenario_reset_nonce"] = 0
    if "_scenario_reset_nonce" not in st.session_state:
        st.session_state["_scenario_reset_nonce"] = 0

    selected_cost_data = data["cost"][
        data["cost"]["Transaction Cost (bps)"].eq(int(selected_cost))
    ].set_index("Display")
    final_net = selected_cost_data.loc["Final Strategy"]
    base_net = selected_cost_data.loc["Base Min-Variance"]
    naive_net = selected_cost_data.loc["Naive Sentiment Tilt"]
    final_beats_cost = (
        data["cost"].sort_values("Sharpe Ratio")
        .groupby("Transaction Cost (bps)")
        .tail(1)["Display"]
        .eq("Final Strategy")
        .all()
    )
    top_cards = [
        (
            "Final Sharpe (Net)",
            format_number(final_net["Sharpe Ratio"]),
            f"{int(selected_cost)} bps transaction cost",
        ),
        (
            "Sharpe Improvement",
            signed_value(final_net["Sharpe Ratio"] - base_net["Sharpe Ratio"]),
            "vs Base Min-Variance",
        ),
        (
            "Max Drawdown Improvement",
            format_pp(final_net["Maximum Drawdown"] - naive_net["Maximum Drawdown"]),
            f"vs Naive Sentiment Tilt • {int(selected_cost)} bps",
        ),
        (
            "Cost Robustness",
            "Strong" if final_beats_cost else "Mixed",
            "Outperforms across tested cost levels"
            if final_beats_cost
            else "Ranking varies by cost level",
        ),
    ]
    st.markdown(
        '<div class="pa-kpi-row">'
        + "".join(
            render_kpi_card(
                label,
                value,
                support,
                "pa-green" if i in {0, 1, 3} else "pa-red",
            )
            for i, (label, value, support) in enumerate(top_cards)
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    section_header(
        "1. Current Strategy Engine",
        "See how the selected sentiment signal becomes a portfolio decision.",
    )
    st.markdown(
        f'<div class="pa-context-line">{escape(str(current["ticker"]))} • '
        f'{escape(str(current["sector"]))} • '
        f'{pd.Timestamp(current["date"]).date().strftime("%d %b %Y")} • Observed</div>',
        unsafe_allow_html=True,
    )
    relative_support = (
        "stronger than sector"
        if float(current["relative"]) > 0
        else "weaker than sector"
        if float(current["relative"]) < 0
        else "in line with sector"
    )
    engine_cards = [
        engine_card(
            "Relative Sentiment",
            signed_value(float(current["relative"])),
            relative_support,
            "Sector-neutral",
            value_class(float(current["relative"])),
        ),
        engine_card(
            "Signal Strength",
            f'{float(current["signal_z"]):+.1f}σ',
            str(current["signal_state"]),
            "Rolling signal",
            value_class(float(current["signal_z"])),
        ),
        engine_card(
            "Tail Risk",
            str(current["tail_status"]),
            f'{float(current["tail_percentile"]) * 100:.0f}th percentile',
            "Risk-aware",
            "pa-red" if current["tail_status"] == "High" else "pa-green",
        ),
        engine_card(
            "Sentiment Influence",
            str(current["influence"]),
            f'{float(current["risk_scaler"]):.2f}× risk scaler',
            "Tuned response",
            "pa-green" if current["influence"] == "Full" else "pa-red",
        ),
        engine_card(
            "Portfolio Action",
            str(current["action"]),
            f'Weight change: {float(current["delta"]) * 100:+.1f} pp',
            "Observed weights",
            "pa-red" if "Underweight" in str(current["action"]) else "pa-green",
            final=True,
        ),
    ]
    st.markdown(
        '<div class="pa-engine-row">' + "".join(engine_cards) + "</div>"
        + f'<div class="pa-inline-insight"><b>Why this action?</b> '
        f'{escape(why_action_text(current))}</div>'
        '<div class="pa-note">Innovation layers: Sector-neutral signal • Rolling '
        "standardisation • Tuned response • Tail-risk control</div>",
        unsafe_allow_html=True,
    )

    signal_min, signal_max = weights["stock_sentiment_z"].quantile([0.01, 0.99])
    tail_min, tail_max = weights["stock_cvar"].quantile([0.01, 0.99])
    scenario_suffix = f"{obs_key}|{st.session_state['_scenario_reset_nonce']}"
    stages = [
        {
            "badge": "0  BASELINE",
            "title": "Base Min-Variance",
            "short": "Base Min-Variance",
            "display": "Base Min-Variance",
            "copy": "No sentiment. Market-only portfolio benchmark.",
            "sharpe": metric_lookup.loc["Base Min-Variance", "Sharpe Ratio"],
            "drawdown": metric_lookup.loc["Base Min-Variance", "Maximum Drawdown"],
        },
        {
            "badge": "1  BASELINE",
            "title": "Naive Sentiment Tilt",
            "short": "Naive Sentiment Tilt",
            "display": "Naive Sentiment Tilt",
            "copy": "Direct sentiment input without the final risk-control innovations.",
            "sharpe": metric_lookup.loc["Naive Sentiment Tilt", "Sharpe Ratio"],
            "drawdown": metric_lookup.loc["Naive Sentiment Tilt", "Maximum Drawdown"],
        },
        {
            "badge": "2  INNOVATION",
            "title": "Sector-Neutral Relative Signal",
            "short": "Sector-Neutral",
            "display": "Sector-Neutral",
            "copy": "Remove broad sector mood and focus on stock-specific signal.",
            "sharpe": metric_lookup.loc["Sector-Neutral", "Sharpe Ratio"],
            "drawdown": metric_lookup.loc["Sector-Neutral", "Maximum Drawdown"],
        },
        {
            "badge": "3  FINAL INNOVATION",
            "title": "Tail-Risk-Aware Final Strategy",
            "short": "Final Innovation",
            "display": "Final Innovation",
            "copy": (
                "Translate the improved signal into controlled portfolio response "
                "under stress."
            ),
            "sharpe": metric_lookup.loc["Final Innovation", "Sharpe Ratio"],
            "drawdown": metric_lookup.loc["Final Innovation", "Maximum Drawdown"],
            "final": True,
        },
    ]

    explore_col, research_col = st.columns([0.59, 0.41], gap="medium")
    with explore_col:
        section_header(
            "2. Explore the Strategy",
            "Test how signal strength and tail risk change the portfolio response.",
        )
        st.markdown(
            '<div class="pa-research-label">Scenario / What-if</div>'
            f'<div class="pa-scale-row"><span>{float(signal_min):+.1f}σ</span>'
            '<span>0</span>'
            f'<span>{float(signal_max):+.1f}σ</span></div>',
            unsafe_allow_html=True,
        )
        scenario_signal = st.slider(
            "Signal Strength",
            min_value=float(signal_min),
            max_value=float(signal_max),
            value=float(current["signal_z"]),
            step=0.1,
            format="%.1fσ",
            key=f"scenario_signal_strength_{scenario_suffix}",
        )
        st.caption(f'Observed: {float(current["signal_z"]):+.1f}σ')
        st.markdown(
            '<div class="pa-scale-row"><span>Low</span><span>Moderate</span><span>High</span></div>',
            unsafe_allow_html=True,
        )
        scenario_tail = st.slider(
            "Tail Risk",
            min_value=float(tail_min),
            max_value=float(tail_max),
            value=float(current["tail_risk"]),
            step=0.001,
            format="%.3f",
            key=f"scenario_tail_risk_{scenario_suffix}",
        )
        st.caption(f'Observed downside tail risk: {float(current["tail_risk"]):.3f}')
        if st.button("Reset to observed state", key="reset_sentiment_scenario"):
            st.session_state["_scenario_reset_nonce"] += 1
            st.rerun()
        scenario = build_scenario_state(current, scenario_signal, scenario_tail)
        st.markdown(
            '<div class="pa-scenario-row">'
            + status_card(
                "Tuned Response",
                f'{float(scenario["adjusted_signal"]):+.2f}',
                "Scenario signal × risk scaler",
                value_class(float(scenario["adjusted_signal"])),
            )
            + status_card(
                "Risk Scaler",
                f'{float(scenario["risk_scaler"]):.2f}×',
                str(scenario["influence"]),
                "pa-green" if scenario["influence"] == "Full" else "pa-red",
            )
            + status_card(
                "Scenario Portfolio Action",
                str(scenario["action"]),
                f'Preview change: {float(scenario["delta"]) * 100:+.1f} pp',
                "pa-red" if "Underweight" in str(scenario["action"]) else "pa-green",
            )
            + "</div>"
            '<div class="pa-note">Scenario exploration changes the current decision '
            "preview only. Historical OOS research results remain unchanged.</div>",
            unsafe_allow_html=True,
        )
    with research_col:
        section_header(
            "3. Research Evidence",
            "Did each innovation improve the historical strategy?",
        )
        st.markdown(
            '<div class="pa-section-title">Sentiment Innovation Progression</div>'
            '<div class="pa-section-subtitle">Sharpe Ratio • Full OOS 2021-2023</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            strategy_progression_chart(stages),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.markdown(
            '<div class="pa-note">Fixed OOS evidence: '
            f'{format_number(float(metric_lookup.loc["Base Min-Variance", "Sharpe Ratio"]))} → '
            f'{format_number(float(metric_lookup.loc["Naive Sentiment Tilt", "Sharpe Ratio"]))} → '
            f'{format_number(float(metric_lookup.loc["Sector-Neutral", "Sharpe Ratio"]))} → '
            f'{format_number(float(metric_lookup.loc["Final Innovation", "Sharpe Ratio"]))}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="pa-small-card"><div class="pa-card-title">Key Insight</div>'
            '<div class="pa-card-copy">Each innovation layer improves risk-adjusted performance. '
            "Sector neutralisation removes broad sector noise, while the final "
            "tail-risk-aware strategy improves downside resilience.</div></div>",
            unsafe_allow_html=True,
        )
        with st.expander("View research breakdown", expanded=False):
            st.write(
                "Base Min-Variance: no-sentiment benchmark. "
                "Naive Sentiment Tilt: direct sentiment response. "
                "Sector-Neutral Relative Signal: removes broad sector mood. "
                "Final Tail-Risk-Aware Strategy: uses the frozen tuned response with "
                "dynamic downside-risk scaling."
            )

    scenario_status = scenario_tail_status(float(scenario_tail), weights)
    perf_col, risk_col = st.columns([0.58, 0.42], gap="medium")
    with perf_col:
        section_header(
            "4. Performance Explorer",
            f"Growth of $1 • saved net returns • {int(selected_cost)} bps • {selected_window}",
        )
        strategy_order = ["Base Min-Variance", "Naive Sentiment Tilt", "Final Strategy"]
        st.markdown('<div class="pa-research-label">Compare</div>', unsafe_allow_html=True)
        compare_cols = st.columns(3, gap="small")
        visible_strategies = []
        for compare_col, strategy in zip(compare_cols, strategy_order):
            with compare_col:
                if st.checkbox(
                    strategy,
                    value=True,
                    key=f"compare_{strategy}",
                ):
                    visible_strategies.append(strategy)
        if not visible_strategies:
            visible_strategies = ["Final Strategy"]
        fig, _ = net_growth_chart(
            data["net_returns"],
            int(selected_cost),
            str(selected_window),
            visible_strategies,
        )
        perf_selected = filter_performance_window(data["net_returns"], str(selected_window))
        perf_selected = perf_selected[perf_selected["transaction_cost_bps"].eq(int(selected_cost))]
        perf_matrix = perf_selected.pivot(index="date", columns="Display", values="return_net")
        growth_all = (1 + perf_matrix[strategy_order].dropna()).cumprod()
        ending = growth_all.iloc[-1]
        perf_summary = performance_summary(
            data["net_returns"],
            int(selected_cost),
            str(selected_window),
        )
        chart_col, value_col = st.columns([0.70, 0.30], gap="small")
        with chart_col:
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with value_col:
            st.markdown(
                '<div class="pa-ending-stack">'
                + "".join(
                    signal_card(name, f"$1 → ${ending[name]:.3f}", "Ending value")
                    for name in [
                        "Final Strategy",
                        "Naive Sentiment Tilt",
                        "Base Min-Variance",
                    ]
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        final_summary = perf_summary["Final Strategy"]
        naive_summary = perf_summary["Naive Sentiment Tilt"]
        st.markdown(
            '<div class="pa-scenario-row">'
            + status_card("Final Return", format_pct(final_summary["return"]), str(selected_window))
            + status_card("Final Volatility", format_pct(final_summary["volatility"]), "annualised")
            + status_card("Final Sharpe", format_number(final_summary["sharpe"]), "selected window")
            + "</div>"
            '<div class="pa-scenario-row" style="margin-top:0.55rem;">'
            + status_card("Final Drawdown", format_pct(final_summary["drawdown"]), "selected window")
            + status_card(
                "Return vs Naive",
                format_pp(final_summary["return"] - naive_summary["return"]),
                "annualised difference",
                value_class(final_summary["return"] - naive_summary["return"]),
            )
            + status_card(
                "Drawdown vs Naive",
                format_pp(final_summary["drawdown"] - naive_summary["drawdown"]),
                "less negative is better",
                value_class(final_summary["drawdown"] - naive_summary["drawdown"]),
            )
            + "</div>"
            f'<div class="pa-inline-insight">{escape(performance_insight(perf_summary, str(selected_window)))}</div>',
            unsafe_allow_html=True,
        )
    with risk_col:
        current_regime = risk_regime_label(str(current["tail_status"]))
        scenario_regime = risk_regime_label(scenario_status)
        section_header(
            "5. Risk Control Explorer",
            "Current tail-risk state and historical regime evidence.",
        )
        st.markdown(
            '<div class="pa-current-state-row">'
            + status_card(
                "Current Tail Risk",
                str(current["tail_status"]).upper(),
                f'{float(current["tail_percentile"]) * 100:.0f}th percentile',
                "pa-red" if current["tail_status"] == "High" else "pa-green",
            )
            + status_card(
                "Current Risk Scaler",
                f'{float(current["risk_scaler"]):.2f}×',
                "Saved strategy observation",
            )
            + status_card(
                "Current Influence",
                str(current["influence"]).upper(),
                "Observed historical state",
                "pa-green" if current["influence"] == "Full" else "pa-red",
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="pa-risk-marker">'
            f'<span class="pa-pill active">Observed: {escape(current_regime)}</span>'
            f'<span class="pa-pill">Scenario: {escape(scenario_regime)}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        rows = []
        for regime, row in data["regime_table"].iterrows():
            active_class = ' class="pa-active-regime"' if str(regime) == current_regime else ""
            rows.append(
                f"<tr{active_class}>"
                f"<td>{escape(str(regime))}</td>"
                f"<td>{format_pct(row['avg_tail_risk'])}</td>"
                f"<td>{row['avg_scaler']:.2f}x</td>"
                f"<td>{format_pct(row['reduction'])}</td>"
                "</tr>"
            )
        st.markdown(
            '<div class="pa-section-panel">'
            + '<table class="pa-table"><thead><tr><th>Market Stress Regime</th>'
            "<th>Avg. Downside Tail Risk</th><th>Avg. Influence</th>"
            f"<th>Reduction vs Naive</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
            f'<div class="pa-inline-insight">{escape(risk_explorer_insight(str(current["tail_status"]), scenario_status, str(current["influence"])))}</div></div>',
            unsafe_allow_html=True,
        )

    cost_col, takeaway_col = st.columns([0.62, 0.38], gap="medium")
    with cost_col:
        section_header(
            "6. Cost Stress Test",
            "Test whether the strategy remains competitive as implementation costs rise.",
        )
        cost_chart_col, matrix_col, cost_insight_col = st.columns(
            [0.47, 0.31, 0.22],
            gap="small",
        )
        with cost_chart_col:
            st.plotly_chart(
                cost_robustness_chart(data["cost"], int(selected_cost)),
                width="stretch",
                config={"displayModeBar": False},
            )
        with matrix_col:
            st.markdown(cost_matrix_html(data["cost"], int(selected_cost)), unsafe_allow_html=True)
        with cost_insight_col:
            final_vs_base = final_net["Sharpe Ratio"] - base_net["Sharpe Ratio"]
            final_vs_naive = final_net["Sharpe Ratio"] - naive_net["Sharpe Ratio"]
            selected_rank = int(
                selected_cost_data["Sharpe Ratio"]
                .rank(ascending=False, method="min")
                .loc["Final Strategy"]
            )
            level_message = (
                "Final Strategy remains the strongest strategy at this cost level."
                if selected_rank == 1
                else "Final Strategy is competitive but not first at this cost level."
            )
            all_cost_message = (
                "Final Strategy ranks first across every tested transaction-cost assumption."
                if final_beats_cost
                else "Strategy ranking varies across the tested cost assumptions."
            )
            st.markdown(
                f'<div class="pa-small-card"><div class="pa-card-title">At {int(selected_cost)} bps</div>'
                f'<div class="pa-card-copy">Base Min-Variance: {base_net["Sharpe Ratio"]:.3f}<br>'
                f'Naive Sentiment Tilt: {naive_net["Sharpe Ratio"]:.3f}<br>'
                f'Final Strategy: <span class="pa-green">{final_net["Sharpe Ratio"]:.3f}</span><br>'
                f'Final vs Base: {final_vs_base:+.3f}<br>'
                f'Final vs Naive: {final_vs_naive:+.3f}<br>'
                f'Rank: #{selected_rank} of 3</div></div>'
                f'<div class="pa-inline-insight">{escape(level_message)} {escape(all_cost_message)}</div>',
                unsafe_allow_html=True,
            )
    with takeaway_col:
        message = (
            "Final strategy remains the strongest even with higher trading costs."
            if final_beats_cost
            else "Cost sensitivity changes the strategy ranking at some tested levels."
        )
        takeaways = [
            (
                "Better Signal",
                "Sector-neutral construction focuses on stock-specific information.",
            ),
            (
                "Smarter Response",
                "Rolling signal strength and frozen tuned sensitivity avoid overreaction.",
            ),
            ("Lower Risk", "Tail-risk-aware control reduces sentiment influence when downside risk rises."),
            (
                "Stronger Results",
                "Historical evidence shows stronger risk-adjusted and cost-robust outcomes.",
            ),
        ]
        st.markdown(
            '<div class="pa-section-panel">'
            + section_header_html("7. Investor Takeaway")
            + '<div class="pa-takeaway-row compact">'
            + "".join(
                f'<div class="pa-takeaway-card"><div class="pa-card-title">{escape(title)}</div>'
                f'<div class="pa-card-copy">{escape(copy)}</div></div>'
                for title, copy in takeaways
            )
            + "</div>"
            f'<div class="pa-inline-insight">{escape(message)}</div></div>',
            unsafe_allow_html=True,
        )

    with st.expander("Methodology", expanded=False):
        st.write(
            "Sector-relative sentiment compares stock sentiment against sector context. "
            "Rolling standardisation uses lagged rolling history to assess unusualness. "
            "The sentiment tilt was selected on discovery data and frozen for OOS use. "
            "Tail-risk scaling reduces sentiment influence when downside tail risk rises. "
            "OOS performance is evaluated on held-out 2021-2023 data, with transaction "
            "costs applied using saved turnover-based assumptions."
        )


def render_market_pulse(sentiment: pd.DataFrame, news: pd.DataFrame) -> None:
    latest_date = sentiment["mapped_trading_date"].max()
    sectors = sorted(sentiment["sector"].dropna().unique())

    recent, previous, recent_dates = recent_windows(sentiment, latest_date)
    recent_by_sector = recent.groupby("sector")["sector_sentiment"].mean()
    previous_by_sector = previous.groupby("sector")["sector_sentiment"].mean()
    activity_by_sector = (
        news[news["mapped_trading_date"].isin(recent_dates)]
        .groupby("sector")
        .size()
    )

    render_topbar(
        latest_date,
        "Market Pulse",
        "News-driven market intelligence for equity investors.",
    )
    selected_subnav = render_sentiment_subnav()
    if selected_subnav != "Market Pulse":
        return

    left_col, right_col = st.columns([0.42, 0.58], gap="medium")
    with left_col:
        st.markdown('<div class="pa-label">Explore Sector</div>', unsafe_allow_html=True)
        selected_sector = st.selectbox(
            "Explore Sector",
            sectors,
            index=sectors.index("Comm") if "Comm" in sectors else 0,
            label_visibility="collapsed",
            key="market_sector_v2",
        )

        sector_recent = float(recent_by_sector[selected_sector])
        sector_previous = float(previous_by_sector.get(selected_sector, sector_recent))
        sector_delta = sector_recent - sector_previous
        sector_direction, direction_class = direction_label(sector_delta)
        sector_rank = int(recent_by_sector.rank(ascending=False, method="min")[selected_sector])
        sector_news_count = int(activity_by_sector.get(selected_sector, 0))

        st.markdown('<div class="pa-section-title">Market Snapshot</div>', unsafe_allow_html=True)
        snapshot_html = (
            '<div class="pa-snapshot-grid">'
            + render_kpi_card(
                "Recent Sentiment",
                signed_value(sector_recent),
                "30-day average",
                value_class(sector_recent),
            )
            + render_kpi_card(
                "Direction",
                sector_direction,
                "vs previous 30-day average",
                direction_class,
            )
            + render_kpi_card(
                "Sector Rank",
                f"{sector_rank} / {len(sectors)}",
                "recent sentiment",
            )
            + render_kpi_card(
                "News Activity",
                f"{sector_news_count}",
                "headlines, last 30 trading days",
            )
            + "</div>"
        )
        st.markdown(snapshot_html, unsafe_allow_html=True)

        sentiment_word = (
            "positive" if sector_recent > 0 else "negative" if sector_recent < 0 else "neutral"
        )
        direction_word = (
            "strengthened"
            if sector_delta > 0
            else "weakened"
            if sector_delta < 0
            else "held steady"
        )
        insight_body = (
            f"{selected_sector} sentiment is {sentiment_word} on a 30-day basis "
            f"and has {direction_word} relative to the previous window. It "
            f"currently ranks {sector_rank} of {len(sectors)} equity sectors."
        )
        st.markdown(
            f"""
            <div class="pa-insight">
                <div class="pa-icon">◌</div>
                <div>
                    <div class="pa-insight-title">What this means</div>
                    <div class="pa-insight-body">{escape(insight_body)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        strongest_sector = str(recent_by_sector.idxmax())
        weakest_sector = str(recent_by_sector.idxmin())
        st.markdown('<div class="pa-section-title">Across the Market</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pa-two-card-grid">'
            + render_market_pair(
                "Strongest Sentiment",
                strongest_sector,
                float(recent_by_sector[strongest_sector]),
            )
            + render_market_pair(
                "Weakest Sentiment",
                weakest_sector,
                float(recent_by_sector[weakest_sector]),
                weakest=True,
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="pa-section-title">News Explorer</div>', unsafe_allow_html=True)
        news_dates = sorted(news["mapped_trading_date"].dt.date.dropna().unique())
        date_col, sector_col, ticker_col = st.columns([0.30, 0.35, 0.35], gap="small")
        with date_col:
            selected_news_date = st.selectbox(
                "Date",
                news_dates,
                index=len(news_dates) - 1,
                key="news_date_v3",
            )
        with sector_col:
            selected_news_sector_filter = st.selectbox(
                "Sector",
                ["All sectors", *sectors],
                index=0,
                key="news_sector_v3",
            )
        selected_date_ts = pd.Timestamp(selected_news_date)
        news_filter_base = news[news["mapped_trading_date"].eq(selected_date_ts)].copy()
        if selected_news_sector_filter != "All sectors":
            news_filter_base = news_filter_base[
                news_filter_base["sector"].eq(selected_news_sector_filter)
            ]
        tickers = sorted(
            news_filter_base["ticker"]
            .dropna()
            .astype(str)
            .unique()
        )
        with ticker_col:
            selected_ticker = st.selectbox(
                "Ticker",
                ["All", *tickers],
                key="news_ticker_v3",
            )

        selected_news = news_filter_base.copy()
        if selected_ticker != "All":
            selected_news = selected_news[selected_news["ticker"].eq(selected_ticker)]

        sector_recent_news = news[
            news["mapped_trading_date"].isin(recent_dates)
            & news["sector"].eq(selected_sector)
        ]
        ticker_stats = sector_recent_news.groupby("ticker").agg(
            avg_sentiment=("sentiment_score", "mean"),
            headlines=("title", "count"),
        )
        if ticker_stats.empty:
            strong_ticker = weak_ticker = active_ticker = "—"
            strong_value = weak_value = 0.0
            active_count = 0
        else:
            strong_ticker = str(ticker_stats["avg_sentiment"].idxmax())
            weak_ticker = str(ticker_stats["avg_sentiment"].idxmin())
            active_ticker = str(ticker_stats["headlines"].idxmax())
            strong_value = float(ticker_stats.loc[strong_ticker, "avg_sentiment"])
            weak_value = float(ticker_stats.loc[weak_ticker, "avg_sentiment"])
            active_count = int(ticker_stats.loc[active_ticker, "headlines"])

        trend_phrase = (
            "Above recent trend"
            if sector_delta > 0.0025
            else "Below recent trend"
            if sector_delta < -0.0025
            else "Near recent trend"
        )
        st.markdown('<div class="pa-section-title">What to Watch</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pa-watch-grid">'
            + render_watch_card(
                "Strongest Sentiment",
                strong_ticker,
                signed_value(strong_value),
                "green" if strong_value > 0 else "red" if strong_value < 0 else "",
            )
            + render_watch_card(
                "Weakest Sentiment",
                weak_ticker,
                signed_value(weak_value),
                "green" if weak_value > 0 else "red" if weak_value < 0 else "",
            )
            + render_watch_card(
                "Highest News Activity",
                active_ticker,
                f"{active_count} headlines",
                "green" if active_count > 0 else "",
            )
            + render_watch_card(
                "Sector vs Recent Trend",
                trend_phrase,
                signed_value(sector_delta),
                "green" if sector_delta > 0 else "red" if sector_delta < 0 else "",
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            '<div class="pa-section-title first">Sentiment Trend</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="pa-range-label">Display range</div>', unsafe_allow_html=True)
        display_range = st.radio(
            "Display range",
            RANGE_OPTIONS,
            index=RANGE_OPTIONS.index("1Y"),
            horizontal=True,
            key="display_range_v2",
            label_visibility="collapsed",
        )
        sector_history = sentiment[sentiment["sector"].eq(selected_sector)]
        chart_data = trend_window(sector_history, latest_date, display_range)
        render_trend_chart(chart_data)

        st.markdown('<div class="pa-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pa-section-title first">Relevant News</div>',
            unsafe_allow_html=True,
        )
        st.markdown(build_news_table(selected_news), unsafe_allow_html=True)

    positive_news = selected_news["sentiment_score"].gt(0).any()
    negative_news = selected_news["sentiment_score"].lt(0).any()
    mixed_title = (
        "Mixed headline signals"
        if positive_news and negative_news
        else "Consistent headline tone"
    )
    mixed_body = (
        f"Both positive and negative {selected_news_sector_filter} headlines were "
        "present on the selected date."
        if positive_news and negative_news
        else f"Saved {selected_news_sector_filter} headlines were not mixed on the selected date."
    )
    monitor_html = (
        '<div class="pa-monitor-grid">'
        + render_monitor_card(
            "▥",
            f"{active_ticker} — elevated attention",
            f"{active_ticker} had the highest recent headline count in "
            f"{selected_sector}.",
        )
        + render_monitor_card(
            "↗",
            f"{selected_sector} — {trend_phrase.lower()}",
            "Sector sentiment was compared with its recent 30-trading-day average.",
        )
        + render_monitor_card("▤", mixed_title, mixed_body)
        + "</div>"
    )
    st.markdown(monitor_html, unsafe_allow_html=True)

    with st.expander("About this signal", expanded=False):
        st.write(
            "PulseAlloc's sector sentiment signal is constructed from saved "
            "financial news headlines. Sector sentiment summarises the tone of "
            "equity-sector news, while the 30-trading-day average reduces daily "
            "noise. No-news observations follow the project's existing neutral "
            "treatment."
        )


def main() -> None:
    st.set_page_config(
        page_title="PulseAlloc Market Pulse",
        page_icon="◼",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_shell_css()

    try:
        artifacts = load_sentiment_artifacts()
    except FileNotFoundError as exc:
        st.error(f"Missing saved artifact: {exc.filename}")
        st.stop()

    sentiment = artifacts["sentiment"]
    latest_date = sentiment["mapped_trading_date"].max()
    default_start = latest_date - pd.DateOffset(years=1)
    selected_page = render_sidebar(default_start, latest_date, latest_date)

    if selected_page == "Home":
        render_home(artifacts)
    elif selected_page == "Explore Funds":
        render_explore_funds(artifacts)
    elif selected_page == "Build Portfolio":
        render_build_portfolio(artifacts)
    elif selected_page == "Adaptive Allocation":
        render_adaptive_allocation(artifacts)
    elif selected_page == "Sentiment Signals":
        selected_sentiment = st.session_state.get(
            "selected_sentiment_section_v2",
            "Market Pulse",
        )
        if selected_sentiment == "Sentiment Strategy":
            render_sentiment_strategy(artifacts)
        else:
            render_market_pulse(artifacts["sentiment"], artifacts["news"])


if __name__ == "__main__":
    main()
