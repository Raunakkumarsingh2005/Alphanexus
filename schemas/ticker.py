from pydantic import BaseModel
from typing import Optional, List
from datetime import date, datetime
import uuid


class TickerSummary(BaseModel):
    symbol: str
    name: Optional[str]
    price: Optional[float]
    change: Optional[float]
    changePercent: Optional[float]
    volume: Optional[int]
    marketCap: Optional[float]
    high52w: Optional[float]
    low52w: Optional[float]


class PricePoint(BaseModel):
    date: date
    close: float
    volume: Optional[int]


class TickerHistory(BaseModel):
    symbol: str
    from_date: date
    to_date: date
    prices: List[PricePoint]


class InsiderTradeOut(BaseModel):
    id: uuid.UUID
    traderName: Optional[str]
    traderTitle: Optional[str]
    tradeDate: Optional[date]
    tradeType: Optional[str]
    shares: Optional[float]
    pricePerShare: Optional[float]
    totalValue: Optional[float]
    filingDate: Optional[date]
    source: str = "SEC Form-4"

    class Config:
        from_attributes = True


class InsiderTradeListOut(BaseModel):
    symbol: str
    trades: List[InsiderTradeOut]


class TickerSearchResult(BaseModel):
    symbol: str
    name: Optional[str]
    type: Optional[str]
