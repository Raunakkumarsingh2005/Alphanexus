"""Ticker service — wraps market data ingester for API layer."""
import logging
import concurrent.futures
from typing import Optional, List
from datetime import date

from sqlalchemy.orm import Session

from pipeline.ingest.finnhub_ingest import MarketDataIngester
from pipeline.ingest.edgar import EdgarIngester, DEMO_CIK_MAP
from config import settings

logger = logging.getLogger(__name__)

_market_ingester = MarketDataIngester(finnhub_api_key=settings.FINNHUB_API_KEY)
_edgar_ingester = EdgarIngester()


def search_tickers(query: str) -> List[dict]:
    return _market_ingester.search_tickers(query)


def get_ticker_summary(symbol: str) -> dict:
    return _market_ingester.fetch_ticker_summary(symbol)


def get_ticker_history(symbol: str, date_from: str, date_to: str) -> dict:
    df = _market_ingester.fetch_ohlcv(symbol, date_from, date_to)
    prices = []
    if not df.empty:
        for _, row in df.iterrows():
            prices.append({
                "date": str(row.get("date", row.get("date", ""))),
                "close": float(row.get("close_price", 0)),
                "volume": int(row.get("volume", 0)) if row.get("volume") else None,
            })
    return {"symbol": symbol.upper(), "from_date": date_from, "to_date": date_to, "prices": prices}


def get_insider_trades(symbol: str, db: Session) -> List[dict]:
    """Check DB first; fall back to live EDGAR fetch (non-blocking, 10s cap)."""
    from models.trade_models import InsiderTrade
    rows = db.query(InsiderTrade).filter(InsiderTrade.ticker == symbol.upper()).limit(100).all()

    if rows:
        return [
            {
                "id": str(r.id),
                "traderName": r.trader_name,
                "traderTitle": r.trader_title,
                "tradeDate": str(r.trade_date) if r.trade_date else None,
                "tradeType": r.direction,
                "shares": float(r.shares) if r.shares else None,
                "pricePerShare": float(r.price_per_share) if r.price_per_share else None,
                "totalValue": float(r.exact_value) if r.exact_value else None,
                "filingDate": str(r.filing_date) if r.filing_date else None,
                "source": "SEC Form-4",
            }
            for r in rows
        ]

    # Live fallback — capped at 10s so the API never hangs
    def _live_fetch():
        cik = DEMO_CIK_MAP.get(symbol.upper())
        df = _edgar_ingester.fetch_insider_trades(symbol, cik)
        return [] if df.empty else df.to_dict(orient="records")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_live_fetch)
            return future.result(timeout=10)
    except concurrent.futures.TimeoutError:
        logger.warning(f"EDGAR live fetch timed out for {symbol} — returning empty")
        return []
    except Exception as e:
        logger.error(f"EDGAR live fetch failed for {symbol}: {e}")
        return []
