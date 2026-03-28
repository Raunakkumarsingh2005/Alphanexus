"""
NewsAPI Ingester
=================
Primary source for recent news headlines and sentiment corpus.
Falls back to GDELT for historical data (>1 month).

Usage:
    from pipeline.ingest.newsapi_ingest import NewsIngester
    ingester = NewsIngester(api_key="your_key")
    df = ingester.fetch_news("NVDA", "2024-01-01", "2024-12-31")
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict

import requests
import pandas as pd

logger = logging.getLogger(__name__)

GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"


class NewsIngester:
    """Fetches news articles from NewsAPI (recent) and GDELT (historical)."""

    def __init__(self, newsapi_key: Optional[str] = None):
        self.newsapi_key = newsapi_key
        self._newsapi_client = None
        if newsapi_key:
            try:
                from newsapi import NewsApiClient
                self._newsapi_client = NewsApiClient(api_key=newsapi_key)
            except ImportError:
                logger.warning("newsapi-python not installed, falling back to GDELT only.")

    def fetch_news(
        self,
        ticker: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch news articles for a ticker.
        Returns DataFrame: ticker, date, headline, source, url, summary, published_at, data_source
        """
        # Decide source based on how historical the request is
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        use_gdelt = date_from and date_from < cutoff

        articles: List[Dict] = []
        if not use_gdelt and self._newsapi_client:
            articles = self._fetch_newsapi(ticker, date_from, date_to, page, page_size)
        if not articles:
            logger.info(f"Using GDELT for {ticker} ({date_from} → {date_to})")
            articles = self._fetch_gdelt(ticker, date_from, date_to)

        if not articles:
            return pd.DataFrame()

        df = pd.DataFrame(articles)
        df["ticker"] = ticker.upper()
        return df

    def fetch_trending(self) -> pd.DataFrame:
        """Fetch trending news across all tracked tickers."""
        if not self._newsapi_client:
            return pd.DataFrame()
        try:
            result = self._newsapi_client.get_top_headlines(
                category="business", language="en", page_size=50
            )
            articles = result.get("articles", [])
            return self._parse_newsapi_articles(articles, ticker="TRENDING")
        except Exception as e:
            logger.error(f"Trending news fetch failed: {e}")
            return pd.DataFrame()

    # ──────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────

    def _fetch_newsapi(
        self, ticker: str, date_from: Optional[str], date_to: Optional[str],
        page: int, page_size: int
    ) -> List[Dict]:
        try:
            result = self._newsapi_client.get_everything(
                q=f"{ticker} stock insider trade",
                from_param=date_from,
                to=date_to,
                language="en",
                sort_by="relevancy",
                page=page,
                page_size=min(page_size, 100),
            )
            return self._parse_newsapi_articles(result.get("articles", []), ticker)
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed for {ticker}: {e}")
            return []

    def _parse_newsapi_articles(self, articles: list, ticker: str) -> List[Dict]:
        records = []
        for a in articles:
            published = a.get("publishedAt", "")
            try:
                pub_dt = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                pub_dt = None
            records.append({
                "headline": a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "url": a.get("url", ""),
                "summary": a.get("description", ""),
                "published_at": pub_dt,
                "date": pub_dt.date() if pub_dt else None,
                "sentiment_label": None,    # filled by FinBERT later
                "sentiment_score": None,
                "related_tickers": [ticker],
                "data_source": "newsapi",
            })
        return records

    def _fetch_gdelt(
        self, ticker: str, date_from: Optional[str], date_to: Optional[str]
    ) -> List[Dict]:
        """GDELT API — no key, historical data, free."""
        try:
            params = {
                "query": f"{ticker} congress trade insider legislation",
                "mode": "artlist",
                "format": "json",
                "maxrecords": 250,
            }
            if date_from:
                params["startdatetime"] = date_from.replace("-", "") + "000000"
            if date_to:
                params["enddatetime"] = date_to.replace("-", "") + "235959"

            resp = requests.get(GDELT_BASE, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])

            records = []
            for a in articles:
                seen_date_str = a.get("seendate", "")
                try:
                    pub_dt = datetime.strptime(seen_date_str, "%Y%m%dT%H%M%SZ")
                except Exception:
                    pub_dt = None
                records.append({
                    "headline": a.get("title", ""),
                    "source": a.get("domain", ""),
                    "url": a.get("url", ""),
                    "summary": None,
                    "published_at": pub_dt,
                    "date": pub_dt.date() if pub_dt else None,
                    "sentiment_label": None,
                    "sentiment_score": None,
                    "related_tickers": [ticker],
                    "data_source": "gdelt",
                })
            return records
        except Exception as e:
            logger.error(f"GDELT fetch failed for {ticker}: {e}")
            return []

    def build_finbert_corpus(
        self, tickers: List[str], date_ranges: List[Dict]
    ) -> pd.DataFrame:
        """
        Build the FinBERT training corpus CSV.
        Output matches finbert_corpus.csv contract for the ML friend.
        """
        all_articles = []
        for r in date_ranges:
            for ticker in tickers if isinstance(r.get("ticker"), list) else [r.get("ticker")]:
                articles = self._fetch_gdelt(ticker, r.get("from"), r.get("to"))
                for a in articles:
                    a["label"] = r.get("label", "")
                all_articles.extend(articles)

        df = pd.DataFrame(all_articles)
        if df.empty:
            return df
        df = df.drop_duplicates(subset=["headline"])
        return df[["label", "ticker", "headline", "published_at", "source"]].rename(
            columns={"published_at": "date"}
        )
