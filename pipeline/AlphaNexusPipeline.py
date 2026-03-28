"""
AlphaNexus Full Data Pipeline
================================
Orchestrates: ingest → entity resolution → feature engineering → PostgreSQL write

Usage:
    # Full pipeline for a specific ticker
    python -m pipeline.AlphaNexusPipeline --ticker NVDA --from 2021-01-01 --to 2024-12-31

    # As a module (called by analysis_service.py)
    from pipeline.AlphaNexusPipeline import AlphaNexusPipeline
    pipeline = AlphaNexusPipeline()
    df = pipeline.run(ticker="NVDA", date_from="2021-01-01", date_to="2024-12-31")

    # Export FinBERT corpus for ML friend
    python -m pipeline.AlphaNexusPipeline --export-corpus
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

# Add project root to path when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from pipeline.ingest.edgar import EdgarIngester, DEMO_CIK_MAP
from pipeline.ingest.finnhub_ingest import MarketDataIngester
from pipeline.ingest.newsapi_ingest import NewsIngester
from pipeline.ingest.quiver import QuiverIngester
from pipeline.clean.entity_resolution import EntityResolver
from pipeline.clean.feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AlphaNexusPipeline")


class AlphaNexusPipeline:
    """
    Master pipeline: pull data from all sources, clean, enrich, and write to PostgreSQL.
    Returns a clean DataFrame for the ML friend.
    """

    def __init__(self):
        self.edgar = EdgarIngester()
        self.market = MarketDataIngester(finnhub_api_key=settings.FINNHUB_API_KEY)
        self.news = NewsIngester(newsapi_key=settings.NEWSAPI_KEY)
        self.quiver = QuiverIngester(api_key=settings.QUIVER_API_KEY)
        self.resolver = EntityResolver()
        self.engineer = FeatureEngineer()

    def run(
        self,
        ticker: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        save_to_db: bool = True,
        export_csv: bool = False,
    ) -> pd.DataFrame:
        """
        Full pipeline run. Returns the clean, enriched DataFrame.

        The returned DataFrame has the exact columns the ML model expects:
            trader_id, trader_name, ticker, trade_date, trade_value, direction,
            price_on_date, volume_zscore, avg_sentiment_score,
            days_before_bill_vote, trader_historical_frequency, trade_value_zscore
        """
        ticker = ticker.upper()
        logger.info(f"Starting pipeline for {ticker} ({date_from} → {date_to})")

        # ── Step 1: Ingest Market Data ─────────────────────────
        logger.info("Fetching OHLCV data...")
        market_df = self.market.fetch_ohlcv(ticker, date_from or "2021-01-01", date_to or datetime.now().strftime("%Y-%m-%d"))
        logger.info(f"  → {len(market_df)} market data rows")

        # ── Step 2: Ingest Insider Trades (SEC EDGAR) ──────────
        logger.info("Fetching SEC EDGAR insider trades...")
        cik = DEMO_CIK_MAP.get(ticker)
        insider_df = self.edgar.fetch_insider_trades(ticker, cik, date_from, date_to)
        insider_df["trader_type"] = "corporate_insider"
        logger.info(f"  → {len(insider_df)} insider trade rows")

        # ── Step 3: Ingest Congressional Trades (Quiver / Capitol Trades) ──
        logger.info("Fetching congressional trades...")
        political_df = self.quiver.fetch_congressional_trades(ticker, date_from, date_to)
        logger.info(f"  → {len(political_df)} congressional trade rows")

        # ── Step 4: Ingest News Sentiment ─────────────────────
        logger.info("Fetching news articles...")
        news_df = self.news.fetch_news(ticker, date_from, date_to)
        logger.info(f"  → {len(news_df)} news articles")

        # ── Step 5: Combine trade sources ─────────────────────
        all_trades = self._combine_trades(insider_df, political_df, ticker)
        if all_trades.empty:
            logger.warning(f"No trade data found for {ticker}. Pipeline complete with empty result.")
            return pd.DataFrame()

        # ── Step 6: Entity Resolution ─────────────────────────
        logger.info("Resolving entity IDs...")
        all_trades = self.resolver.enrich_dataframe(all_trades)

        # ── Step 7: Feature Engineering ───────────────────────
        logger.info("Computing derived features...")
        enriched_df = self.engineer.compute_all(
            trades_df=all_trades,
            market_df=market_df if not market_df.empty else None,
            news_df=news_df if not news_df.empty else None,
        )

        # ── Step 8: Write to PostgreSQL ────────────────────────
        if save_to_db:
            self._write_to_db(enriched_df, market_df, news_df, ticker)

        # ── Step 9: Export CSV for ML friend ──────────────────
        if export_csv:
            csv_path = f"data/{ticker}_clean_data.csv"
            os.makedirs("data", exist_ok=True)
            enriched_df.to_csv(csv_path, index=False)
            logger.info(f"Clean dataframe exported → {csv_path}")

        logger.info(f"Pipeline complete. {len(enriched_df)} enriched trade rows ready.")
        return enriched_df

    def export_finbert_corpus(self) -> str:
        """Build and save the FinBERT training corpus CSV for the ML friend."""
        logger.info("Exporting FinBERT corpus...")
        date_ranges = [
            {"label": "pelosi_nvda", "ticker": "NVDA", "from": "2021-01-01", "to": "2021-06-30"},
            {"label": "tech_committee", "ticker": "NVDA", "from": "2022-01-01", "to": "2023-12-31"},
            {"label": "ai_chip_2024", "ticker": "NVDA", "from": "2024-01-01", "to": "2024-12-31"},
        ]
        tickers = ["NVDA", "AMD", "INTC", "TSMC"]
        corpus_df = self.news.build_finbert_corpus(tickers, date_ranges)

        os.makedirs("corpus", exist_ok=True)
        out_path = "corpus/finbert_corpus.csv"
        corpus_df.to_csv(out_path, index=False)
        logger.info(f"FinBERT corpus saved → {out_path} ({len(corpus_df)} headlines)")
        return out_path

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _combine_trades(
        self, insider_df: pd.DataFrame, political_df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """Merge insider and political trades into a single unified DataFrame."""
        frames = []

        if not insider_df.empty:
            insider_df = insider_df.copy()
            insider_df.setdefault("trade_value", insider_df.get("exact_value", 0))
            frames.append(insider_df)

        if not political_df.empty:
            frames.append(political_df.copy())

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined["ticker"] = ticker
        return combined

    def _write_to_db(
        self,
        trades_df: pd.DataFrame,
        market_df: pd.DataFrame,
        news_df: pd.DataFrame,
        ticker: str,
    ) -> None:
        """Persist all cleaned data to PostgreSQL (upsert on unique constraints)."""
        try:
            from database import engine

            # Market data
            if not market_df.empty:
                market_df.to_sql(
                    "market_data", engine, if_exists="append",
                    index=False, method="multi",
                )
                logger.info(f"  → Wrote {len(market_df)} market rows to DB")

            # News
            if not news_df.empty:
                news_df.to_sql(
                    "news_sentiment", engine, if_exists="append",
                    index=False, method="multi",
                )
                logger.info(f"  → Wrote {len(news_df)} news rows to DB")

            # Trades — split by type
            if not trades_df.empty:
                political = trades_df[trades_df.get("trader_type", "").eq("politician")] if "trader_type" in trades_df else pd.DataFrame()
                insider = trades_df[trades_df.get("trader_type", "").eq("corporate_insider")] if "trader_type" in trades_df else trades_df

                if not political.empty:
                    political_cols = [c for c in [
                        "trader_id", "trader_name", "ticker", "trade_date",
                        "trade_value_min", "trade_value_max", "trade_value", "direction",
                        "days_before_bill_vote", "volume_zscore", "trade_value_zscore",
                        "trader_historical_frequency"
                    ] if c in political.columns]
                    political[political_cols].to_sql(
                        "political_trades", engine, if_exists="append", index=False
                    )

                if not insider.empty:
                    insider_cols = [c for c in [
                        "cik", "trader_name", "trader_title", "ticker", "trade_date",
                        "exact_value", "shares", "price_per_share", "direction", "filing_date",
                        "conviction_score", "anomaly_score"
                    ] if c in insider.columns]
                    insider[insider_cols].to_sql(
                        "insider_trades", engine, if_exists="append", index=False
                    )

            logger.info("Database write complete.")
        except Exception as e:
            logger.error(f"Database write failed: {e}")
            raise


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaNexus Data Pipeline")
    parser.add_argument("--ticker", type=str, help="Ticker symbol, e.g. NVDA")
    parser.add_argument("--from", dest="date_from", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--export-csv", action="store_true", help="Export clean CSV for ML friend")
    parser.add_argument("--export-corpus", action="store_true", help="Export FinBERT corpus CSV")
    parser.add_argument("--no-db", action="store_true", help="Skip database write (dry run)")
    args = parser.parse_args()

    pipeline = AlphaNexusPipeline()

    if args.export_corpus:
        pipeline.export_finbert_corpus()
    elif args.ticker:
        pipeline.run(
            ticker=args.ticker,
            date_from=args.date_from,
            date_to=args.date_to,
            save_to_db=not args.no_db,
            export_csv=args.export_csv,
        )
    else:
        parser.print_help()
