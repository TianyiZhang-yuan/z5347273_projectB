"""Station 3 - your sentiment model and index from news headlines.

This is the model step: score each headline, aggregate to a daily per-ticker score,
then to an equal-weight sector index. Headlines are a noisy proxy, so lag to avoid
look-ahead.
"""
import pandas as pd


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Score each preserved headline with finVADER compound sentiment."""
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    if "title" not in panel.columns:
        raise ValueError("panel is missing required column: title")

    nltk_download = nltk.download
    nltk.download = lambda *args, **kwargs: True
    try:
        from finvader.Henry import lexicon2
        from finvader.SentiBignomics import lexicon1
    finally:
        nltk.download = nltk_download

    sentibignomics = {term: score * 0.1 for term, score in lexicon1().items()}
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update({**sentibignomics, **lexicon2()})

    out = panel.copy()
    headline_text = out["title"].fillna("").astype(str)
    scores = {
        text: analyzer.polarity_scores(text)["compound"]
        for text in headline_text.drop_duplicates()
    }
    out["finvader_compound"] = headline_text.map(scores)
    return out


def ticker_day_sentiment(scores: pd.DataFrame) -> pd.DataFrame:
    """Build ticker-day headline sentiment on the full equity trading grid."""
    required = {"mapped_trading_date", "ticker", "sector", "finvader_compound"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"scores is missing required columns: {sorted(missing)}")

    from src.data_access import load_equity_prices

    equities = load_equity_prices()
    equity_required = {"date", "ticker", "sector"}
    equity_missing = equity_required.difference(equities.columns)
    if equity_missing:
        raise ValueError(f"equities is missing required columns: {sorted(equity_missing)}")

    ticker_day = (
        scores.groupby(["mapped_trading_date", "ticker", "sector"], as_index=False)[
            "finvader_compound"
        ]
        .mean()
        .rename(columns={"finvader_compound": "ticker_day_sentiment"})
    )
    full_grid = (
        equities[["ticker", "sector"]]
        .drop_duplicates()
        .merge(
            pd.DataFrame(
                {"mapped_trading_date": equities["date"].drop_duplicates().sort_values()}
            ),
            how="cross",
        )
    )
    ticker_day = full_grid.merge(
        ticker_day,
        on=["mapped_trading_date", "ticker", "sector"],
        how="left",
    )
    ticker_day["ticker_day_sentiment"] = ticker_day["ticker_day_sentiment"].fillna(0.0)
    return ticker_day.sort_values(["ticker", "mapped_trading_date"]).reset_index(drop=True)


def sector_sentiment_index(scores: pd.DataFrame) -> pd.DataFrame:
    """Build equal-weight sector sentiment from headline finVADER scores."""
    ticker_day = ticker_day_sentiment(scores)
    sector_day = (
        ticker_day.groupby(["mapped_trading_date", "sector"], as_index=False)[
            "ticker_day_sentiment"
        ]
        .mean()
        .rename(columns={"ticker_day_sentiment": "sector_sentiment"})
    )
    sector_day = sector_day.sort_values(["sector", "mapped_trading_date"]).reset_index(
        drop=True
    )
    sector_day["sector_sentiment_lag1"] = sector_day.groupby("sector")[
        "sector_sentiment"
    ].shift(1)
    return sector_day
