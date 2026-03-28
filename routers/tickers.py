"""Tickers router — 4 endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database import get_db
from routers.deps import get_current_user
import services.ticker_service as ticker_svc

router = APIRouter(prefix="/api/tickers", tags=["Tickers"])


@router.get("/search")
def search_tickers(
    q: str = Query(..., min_length=1, description="Partial ticker symbol or company name"),
    current_user: dict = Depends(get_current_user),
):
    return {"results": ticker_svc.search_tickers(q)}


@router.get("/{symbol}")
def get_ticker(symbol: str, current_user: dict = Depends(get_current_user)):
    data = ticker_svc.get_ticker_summary(symbol.upper())
    return data


@router.get("/{symbol}/history")
def get_ticker_history(
    symbol: str,
    from_date: str = Query(alias="from"),
    to_date: str = Query(alias="to"),
    current_user: dict = Depends(get_current_user),
):
    return ticker_svc.get_ticker_history(symbol.upper(), from_date, to_date)


@router.get("/{symbol}/insider-trades")
def get_insider_trades(
    symbol: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    trades = ticker_svc.get_insider_trades(symbol.upper(), db)
    return {"symbol": symbol.upper(), "trades": trades}
