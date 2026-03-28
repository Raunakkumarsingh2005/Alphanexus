"""
Feature Engineering
=====================
Computes derived features that the ML model needs but are not raw from APIs.

These are Raunak's responsibility per the workflow doc:
  - volume_zscore
  - trade_value_zscore
  - days_before_bill_vote
  - trader_historical_frequency
  - avg_sentiment_score (7-day window around trade date)
  - price_on_date
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Hardcoded bill passage dates for the three demo acts
# ML friend uses these as reference points for days_before_bill_vote
BILL_DATES = {
    # Act 1 — CHIPS and Science Act
    "CHIPS_ACT": "2022-08-09",
    # Act 2 — NDAA 2023 (broad tech/defense provisions)
    "NDAA_2023": "2022-12-23",
    # Act 3 — AI regulation bills (approximate)
    "AI_EXEC_ORDER": "2023-10-30",
}


class FeatureEngineer:
    """Computes all derived features on the joined trades + market + news DataFrame."""

    def compute_all(
        self,
        trades_df: pd.DataFrame,
        market_df: Optional[pd.DataFrame] = None,
        news_df: Optional[pd.DataFrame] = None,
        bill_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Master feature computation. Call this after joining all source tables.

        Parameters
        ----------
        trades_df : DataFrame
            Combined political + insider trades (must have: ticker, trade_date, trade_value, trader_id)
        market_df : DataFrame, optional
            OHLCV data (ticker, date, close_price, volume)
        news_df : DataFrame, optional
            News sentiment data (ticker, date, sentiment_score)
        bill_date : str, optional
            Override bill date for days_before_bill_vote calculation.

        Returns
        -------
        DataFrame
            Enriched trades_df with all derived feature columns.
        """
        df = trades_df.copy()

        if df.empty:
            return df

        # Ensure date types
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        if "trade_value" not in df.columns:
            df["trade_value"] = df.get("trade_value_min", pd.Series(0, index=df.index))

        df = self._compute_volume_zscore(df, market_df)
        df = self._compute_trade_value_zscore(df)
        df = self._compute_days_before_bill_vote(df, bill_date)
        df = self._compute_trader_historical_frequency(df)
        df = self._compute_avg_sentiment_score(df, news_df)
        df = self._compute_price_on_date(df, market_df)

        return df

    # ──────────────────────────────────────────────────────────
    # Individual feature computations
    # ──────────────────────────────────────────────────────────

    def _compute_volume_zscore(
        self, df: pd.DataFrame, market_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """volume_zscore: how anomalous was volume on the trade date."""
        if market_df is None or market_df.empty or "volume" not in market_df.columns:
            df["volume_zscore"] = 0.0
            return df

        # Compute zscore per ticker
        market_df = market_df.copy()
        market_df["date"] = pd.to_datetime(market_df["date"]).dt.date

        vol_stats = market_df.groupby("ticker")["volume"].agg(["mean", "std"]).reset_index()
        vol_stats.columns = ["ticker", "vol_mean", "vol_std"]

        market_df = market_df.merge(vol_stats, on="ticker", how="left")
        market_df["volume_zscore"] = np.where(
            market_df["vol_std"] > 0,
            (market_df["volume"] - market_df["vol_mean"]) / market_df["vol_std"],
            0.0,
        )

        # Join back to trades on (ticker, date)
        vol_join = market_df[["ticker", "date", "volume_zscore"]]
        df = df.merge(
            vol_join, left_on=["ticker", "trade_date"],
            right_on=["ticker", "date"], how="left"
        )
        df["volume_zscore"] = df["volume_zscore"].fillna(0.0)
        if "date" in df.columns:
            df.drop(columns=["date"], inplace=True)
        return df

    def _compute_trade_value_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """trade_value_zscore: how anomalous is this trade relative to this trader's history."""
        def _zscore_group(x):
            mean, std = x.mean(), x.std()
            if std == 0 or pd.isna(std):
                return pd.Series(0.0, index=x.index)
            return (x - mean) / std

        if "trader_id" in df.columns and "trade_value" in df.columns:
            df["trade_value_zscore"] = df.groupby("trader_id")["trade_value"].transform(_zscore_group)
            df["trade_value_zscore"] = df["trade_value_zscore"].fillna(0.0)
        else:
            df["trade_value_zscore"] = 0.0
        return df

    def _compute_days_before_bill_vote(
        self, df: pd.DataFrame, bill_date: Optional[str]
    ) -> pd.DataFrame:
        """days_before_bill_vote: negative = after the bill, positive = before."""
        if bill_date is None:
            # Pick the most relevant bill date based on the ticker
            bill_date = BILL_DATES.get("CHIPS_ACT", "2022-08-09")

        try:
            bill_dt = pd.to_datetime(bill_date).date()
            df["days_before_bill_vote"] = df["trade_date"].apply(
                lambda d: (bill_dt - d).days if pd.notna(d) else 0
            )
        except Exception as e:
            logger.warning(f"Could not compute days_before_bill_vote: {e}")
            df["days_before_bill_vote"] = 0

        return df

    def _compute_trader_historical_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Number of trades per trader in the dataset."""
        if "trader_id" in df.columns:
            freq = df.groupby("trader_id")["trader_id"].transform("count")
            df["trader_historical_frequency"] = freq
        else:
            df["trader_historical_frequency"] = 1
        return df

    def _compute_avg_sentiment_score(
        self, df: pd.DataFrame, news_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """7-day average sentiment score around trade date for the same ticker."""
        if news_df is None or news_df.empty or "sentiment_score" not in news_df.columns:
            df["avg_sentiment_score"] = 0.0
            return df

        news_df = news_df.copy()
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.date

        def _avg_sentiment(row):
            t = row["ticker"]
            d = row["trade_date"]
            window = news_df[
                (news_df["ticker"] == t) &
                (news_df["date"] >= pd.Timestamp(d) - pd.Timedelta(days=7)) &
                (news_df["date"] <= pd.Timestamp(d) + pd.Timedelta(days=7))
            ]
            scores = window["sentiment_score"].dropna()
            return float(scores.mean()) if not scores.empty else 0.0

        df["avg_sentiment_score"] = df.apply(_avg_sentiment, axis=1)
        return df

    def _compute_price_on_date(
        self, df: pd.DataFrame, market_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Join close price on trade date."""
        if market_df is None or market_df.empty or "close_price" not in market_df.columns:
            df["price_on_date"] = None
            return df

        market_df = market_df.copy()
        market_df["date"] = pd.to_datetime(market_df["date"]).dt.date
        price_join = market_df[["ticker", "date", "close_price"]].rename(
            columns={"close_price": "price_on_date"}
        )
        df = df.merge(
            price_join, left_on=["ticker", "trade_date"],
            right_on=["ticker", "date"], how="left"
        )
        if "date" in df.columns:
            df.drop(columns=["date"], inplace=True)
        return df
