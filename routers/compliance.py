"""
Compliance Auditor Router
==========================
Role: "analyst"
Endpoints purpose: fetch all news, trade activity, and flagged items for a ticker.
No ML analysis is triggered — this is a pure data-review/audit flow.

Routes:
  GET /api/compliance/news/{ticker}       — paginated news for a ticker
  GET /api/compliance/trades/{ticker}     — insider + political trades summary
  GET /api/compliance/summary/{ticker}    — aggregated overview card
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from database import get_db
from routers.deps import get_current_user
import services.news_service as news_svc
import services.ticker_service as ticker_svc
from models.trade_models import PoliticalTrade, InsiderTrade
from models.market_models import NewsSentiment
from pipeline.ingest.newsapi_ingest import NewsIngester
from config import settings

router = APIRouter(prefix="/api/compliance", tags=["Compliance Auditor"])

_news_ingester = NewsIngester(newsapi_key=settings.NEWSAPI_KEY)


@router.get("/news/{ticker}")
def get_compliance_news(
    ticker: str,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Compliance Auditor: Fetch all news articles for a given ticker.
    Returns paginated news with sentiment scores.
    Uses news_svc which has a 10s-capped live fallback.
    """
    result = news_svc.get_news(db, ticker=ticker.upper(), page=page, limit=limit)
    result["ticker"] = ticker.upper()
    result["source"] = "database" if result.get("articles") else "live-or-empty"
    return result


@router.get("/trades/{ticker}")
def get_compliance_trades(
    ticker: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Compliance Auditor: Return all insider and political trades for a ticker.
    """
    ticker = ticker.upper()

    # Insider trades
    insider_rows = (
        db.query(InsiderTrade)
        .filter(InsiderTrade.ticker == ticker)
        .order_by(InsiderTrade.trade_date.desc())
        .limit(100)
        .all()
    )
    insider = [
        {
            "source": "SEC EDGAR",
            "tradeDate": str(r.trade_date) if r.trade_date else None,
            "traderName": r.trader_name,
            "traderTitle": r.trader_title,
            "direction": r.direction,
            "shares": float(r.shares) if r.shares else None,
            "value": float(r.exact_value) if r.exact_value else None,
        }
        for r in insider_rows
    ]

    # Political trades
    political_rows = (
        db.query(PoliticalTrade)
        .filter(PoliticalTrade.ticker == ticker)
        .order_by(PoliticalTrade.trade_date.desc())
        .limit(100)
        .all()
    )
    political = [
        {
            "source": "Congressional (Quiver/Capitol Trades)",
            "tradeDate": str(r.trade_date) if r.trade_date else None,
            "traderName": r.trader_name,
            "direction": r.direction,
            "valueMin": float(r.trade_value_min) if r.trade_value_min else None,
            "valueMax": float(r.trade_value_max) if r.trade_value_max else None,
        }
        for r in political_rows
    ]

    # If DB empty, do live fetch
    if not insider:
        live = ticker_svc.get_insider_trades(ticker, db)
        insider = live if live else []

    return {
        "ticker": ticker,
        "insiderTrades": insider,
        "politicalTrades": political,
        "totalInsider": len(insider),
        "totalPolitical": len(political),
    }


@router.get("/summary/{ticker}")
def get_compliance_summary(
    ticker: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Compliance Auditor: Overview card aggregating news count, trade count, 
    and average sentiment for a ticker in one shot.
    """
    ticker = ticker.upper()

    news_count = db.query(NewsSentiment).filter(NewsSentiment.ticker == ticker).count()
    insider_count = db.query(InsiderTrade).filter(InsiderTrade.ticker == ticker).count()
    political_count = db.query(PoliticalTrade).filter(PoliticalTrade.ticker == ticker).count()

    # Average sentiment
    from sqlalchemy import func
    avg_sent = (
        db.query(func.avg(NewsSentiment.sentiment_score))
        .filter(NewsSentiment.ticker == ticker)
        .scalar()
    )

    # Live ticker summary
    market = ticker_svc.get_ticker_summary(ticker)

    return {
        "ticker": ticker,
        "marketSummary": market,
        "newsCount": news_count,
        "insiderTradeCount": insider_count,
        "politicalTradeCount": political_count,
        "avgSentimentScore": float(avg_sent) if avg_sent else None,
        "riskSignal": (
            "elevated" if (insider_count + political_count) > 10 else
            "moderate" if (insider_count + political_count) > 3 else
            "low"
        ),
    }
