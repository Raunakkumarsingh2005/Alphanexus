"""News service — loads from DB or fetches live from NewsAPI/GDELT."""
import logging
import uuid
import concurrent.futures
from typing import Optional

from sqlalchemy.orm import Session

from models.market_models import NewsSentiment
from pipeline.ingest.newsapi_ingest import NewsIngester
from config import settings

logger = logging.getLogger(__name__)

_news_ingester = NewsIngester(newsapi_key=settings.NEWSAPI_KEY)


def get_news(
    db: Session, ticker: Optional[str], page: int = 1, limit: int = 20
) -> dict:
    query = db.query(NewsSentiment)
    if ticker:
        query = query.filter(NewsSentiment.ticker == ticker.upper())
    total = query.count()
    rows = query.order_by(NewsSentiment.published_at.desc()).offset((page - 1) * limit).limit(limit).all()

    articles = [_row_to_dict(r) for r in rows]

    # If DB is empty, do a capped live fetch (10s timeout)
    if not articles and ticker:
        def _live():
            df = _news_ingester.fetch_news(ticker, page_size=limit)
            if not df.empty:
                return df.head(limit).to_dict(orient="records")
            return []
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                articles = ex.submit(_live).result(timeout=10)
                total = len(articles)
        except (concurrent.futures.TimeoutError, Exception) as e:
            logger.warning(f"Live news fetch timed out/failed for {ticker}: {e}")

    return {
        "articles": articles,
        "pagination": {"page": page, "limit": limit, "total": total},
    }


def get_article(db: Session, article_id: str) -> Optional[dict]:
    try:
        row = db.query(NewsSentiment).filter(NewsSentiment.id == uuid.UUID(article_id)).first()
        return _row_to_dict(row) if row else None
    except Exception as e:
        logger.error(f"get_article failed: {e}")
        return None


def get_trending(db: Session) -> dict:
    rows = (
        db.query(NewsSentiment)
        .order_by(NewsSentiment.published_at.desc())
        .limit(50)
        .all()
    )
    articles = [_row_to_dict(r) for r in rows]
    if not articles:
        df = _news_ingester.fetch_trending()
        if not df.empty:
            articles = df.head(50).to_dict(orient="records")
    return {"articles": articles}


def _row_to_dict(row: NewsSentiment) -> dict:
    return {
        "id": str(row.id),
        "headline": row.headline,
        "source": row.source,
        "publishedAt": row.published_at.isoformat() if row.published_at else None,
        "url": row.url,
        "sentiment": row.sentiment_label,
        "sentimentScore": float(row.sentiment_score) if row.sentiment_score else None,
        "relatedTickers": row.related_tickers or [],
        "summary": row.summary,
    }
