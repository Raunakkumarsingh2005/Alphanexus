"""Graph service — formats D3 graph data from DB for the graph router."""
import logging
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from models.job_models import GraphResult, AnalysisJob
from models.trade_models import PoliticalTrade, InsiderTrade
from models.market_models import NewsSentiment

logger = logging.getLogger(__name__)


def get_full_graph(db: Session, job_id: str) -> Optional[dict]:
    try:
        result = db.query(GraphResult).filter(GraphResult.job_id == uuid.UUID(job_id)).first()
        return result.graph_data if result else None
    except Exception as e:
        logger.error(f"get_full_graph failed: {e}")
        return None


def get_node_detail(db: Session, job_id: str, node_id: str) -> Optional[dict]:
    """Return single node with extended trade history and related news."""
    graph = get_full_graph(db, job_id)
    if not graph:
        return None

    # Find node in graph
    node = next((n for n in graph.get("nodes", []) if n.get("id") == node_id), None)
    if not node:
        return None

    # Enrich with trade history from DB
    node["trades"] = _get_trades_for_node(db, node_id)
    node["relatedNews"] = _get_news_for_node(db, node_id, job_id)
    return node


def _get_trades_for_node(db: Session, node_id: str) -> list:
    trades = []

    # Try political trades
    rows = (
        db.query(PoliticalTrade)
        .filter(PoliticalTrade.trader_id == node_id)
        .limit(20)
        .all()
    )
    for r in rows:
        trades.append({
            "ticker": r.ticker,
            "date": str(r.trade_date),
            "type": r.direction,
            "value": float(r.trade_value) if r.trade_value else None,
            "shares": None,
        })

    # Try insider trades (node_id could be cik-XXXXX)
    if node_id.startswith("cik-"):
        cik = node_id.replace("cik-", "")
        rows_i = (
            db.query(InsiderTrade)
            .filter(InsiderTrade.cik == cik)
            .limit(20)
            .all()
        )
        for r in rows_i:
            trades.append({
                "ticker": r.ticker,
                "date": str(r.trade_date),
                "type": r.direction,
                "value": float(r.exact_value) if r.exact_value else None,
                "shares": float(r.shares) if r.shares else None,
            })
    return trades


def _get_news_for_node(db: Session, node_id: str, job_id: str) -> list:
    # Get ticker from job
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == uuid.UUID(job_id)).first()
        if not job:
            return []
        rows = (
            db.query(NewsSentiment)
            .filter(NewsSentiment.ticker == job.ticker)
            .order_by(NewsSentiment.published_at.desc())
            .limit(5)
            .all()
        )
        return [
            {
                "headline": r.headline,
                "sentiment": r.sentiment_label,
                "date": str(r.date) if r.date else None,
            }
            for r in rows
        ]
    except Exception as e:
        logger.debug(f"News enrichment failed: {e}")
        return []
