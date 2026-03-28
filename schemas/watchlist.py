from pydantic import BaseModel
from typing import List
from datetime import datetime
import uuid


class WatchlistAddRequest(BaseModel):
    symbol: str


class WatchlistItem(BaseModel):
    id: uuid.UUID
    ticker: str
    createdAt: datetime

    class Config:
        from_attributes = True


class WatchlistOut(BaseModel):
    tickers: List[WatchlistItem]
