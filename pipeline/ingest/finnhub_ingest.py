"""
Finnhub + yFinance Market Data Ingester
=========================================
Primary: Finnhub API (OHLCV candles)
Fallback: yFinance (no rate limit, no key required)

Usage:
    from pipeline.ingest.finnhub_ingest import MarketDataIngester
    ingester = MarketDataIngester()
    df = ingester.fetch_ohlcv("NVDA", "2021-01-01", "2024-12-31")
"""

import logging
from datetime import datetime, date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MarketDataIngester:
    """Fetches OHLCV data. Tries Finnhub first, falls back to yFinance silently."""

    def __init__(self, finnhub_api_key: Optional[str] = None):
        self.finnhub_api_key = finnhub_api_key
        self._finnhub_client = None
        if finnhub_api_key:
            try:
                import finnhub
                self._finnhub_client = finnhub.Client(api_key=finnhub_api_key)
            except ImportError:
                logger.warning("finnhub-python not installed, will use yFinance only.")

    def fetch_ohlcv(
        self, ticker: str, date_from: str, date_to: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a ticker between two dates (inclusive).
        Returns DataFrame: ticker, date, open, high, low, close, volume, source
        """
        df = None
        if self._finnhub_client:
            df = self._fetch_finnhub(ticker, date_from, date_to)
        if df is None or df.empty:
            logger.info(f"Falling back to yFinance for {ticker}")
            df = self._fetch_yfinance(ticker, date_from, date_to)
        return df

    def fetch_ticker_summary(self, ticker: str) -> dict:
        """Return current price/52w high-low for ticker detail endpoint."""
        if self._finnhub_client:
            try:
                quote = self._finnhub_client.quote(ticker)
                profile = self._finnhub_client.company_profile2(symbol=ticker)
                return {
                    "symbol": ticker.upper(),
                    "name": profile.get("name", ticker),
                    "price": quote.get("c"),
                    "change": quote.get("d"),
                    "changePercent": quote.get("dp"),
                    "volume": None,
                    "marketCap": profile.get("marketCapitalization"),
                    "high52w": quote.get("h"),
                    "low52w": quote.get("l"),
                }
            except Exception as e:
                logger.warning(f"Finnhub summary failed for {ticker}: {e}")

        # yFinance fallback
        return self._yfinance_summary(ticker)

    def search_tickers(self, query: str) -> list:
        """Autocomplete ticker search."""
        if self._finnhub_client:
            try:
                result = self._finnhub_client.symbol_lookup(query)
                hits = result.get("result", [])
                return [
                    {"symbol": r.get("symbol"), "name": r.get("description"), "type": r.get("type")}
                    for r in hits[:10]
                ]
            except Exception as e:
                logger.warning(f"Finnhub search failed: {e}")
        # Simple static fallback for demo
        return [{"symbol": query.upper(), "name": query.upper(), "type": "Common Stock"}]

    # ──────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────

    def _fetch_finnhub(self, ticker: str, date_from: str, date_to: str) -> Optional[pd.DataFrame]:
        try:
            from_ts = int(datetime.strptime(date_from, "%Y-%m-%d").timestamp())
            to_ts = int(datetime.strptime(date_to, "%Y-%m-%d").timestamp())
            candles = self._finnhub_client.stock_candles(ticker, "D", from_ts, to_ts)
            if candles.get("s") != "ok" or not candles.get("t"):
                return None
            df = pd.DataFrame({
                "date": [datetime.fromtimestamp(t).date() for t in candles["t"]],
                "open_price": candles["o"],
                "high_price": candles["h"],
                "low_price": candles["l"],
                "close_price": candles["c"],
                "volume": candles["v"],
            })
            df["ticker"] = ticker.upper()
            df["source"] = "finnhub"
            return df
        except Exception as e:
            logger.warning(f"Finnhub OHLCV failed for {ticker}: {e}")
            return None

    def _fetch_yfinance(self, ticker: str, date_from: str, date_to: str) -> pd.DataFrame:
        try:
            import yfinance as yf
            raw = yf.download(ticker, start=date_from, end=date_to,
                              progress=False, auto_adjust=True)
            if raw.empty:
                return pd.DataFrame()
            # Handle multi-level columns from yfinance
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            df = raw.reset_index()
            df = df.rename(columns={
                "Date": "date",
                "Open": "open_price",
                "High": "high_price",
                "Low": "low_price",
                "Close": "close_price",
                "Volume": "volume",
            })
            df["ticker"] = ticker.upper()
            df["source"] = "yfinance"
            keep = ["ticker", "date", "open_price", "high_price",
                    "low_price", "close_price", "volume", "source"]
            return df[[c for c in keep if c in df.columns]]
        except Exception as e:
            logger.error(f"yFinance fetch failed for {ticker}: {e}")
            return pd.DataFrame()

    def _yfinance_summary(self, ticker: str) -> dict:
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            return {
                "symbol": ticker.upper(),
                "name": info.get("longName", ticker),
                "price": info.get("currentPrice"),
                "change": None,
                "changePercent": None,
                "volume": info.get("volume"),
                "marketCap": info.get("marketCap"),
                "high52w": info.get("fiftyTwoWeekHigh"),
                "low52w": info.get("fiftyTwoWeekLow"),
            }
        except Exception as e:
            logger.error(f"yFinance summary failed for {ticker}: {e}")
            return {"symbol": ticker.upper(), "name": ticker}
