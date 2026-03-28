from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid


class ArticleOut(BaseModel):
    id: uuid.UUID
    headline: Optional[str]
    source: Optional[str]
    publishedAt: Optional[datetime]
    url: Optional[str]
    sentiment: Optional[str]
    sentimentScore: Optional[float]
    relatedTickers: Optional[List[str]]
    summary: Optional[str]

    class Config:
        from_attributes = True


class Pagination(BaseModel):
    page: int
    limit: int
    total: int


class NewsListOut(BaseModel):
    articles: List[ArticleOut]
    pagination: Pagination


class ArticleDetail(ArticleOut):
    """Same as ArticleOut — extended when FinBERT breakdown is added later."""
    pass
