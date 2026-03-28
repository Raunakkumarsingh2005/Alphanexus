"""News router — 3 endpoints."""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from routers.deps import get_current_user
import services.news_service as news_svc

router = APIRouter(prefix="/api/news", tags=["News"])


@router.get("")
def get_news(
    ticker: str = Query(None, description="Filter by ticker symbol"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return news_svc.get_news(db, ticker, page, limit)


@router.get("/trending")
def get_trending(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return news_svc.get_trending(db)


@router.get("/{article_id}")
def get_article(
    article_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    article = news_svc.get_article(db, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article
