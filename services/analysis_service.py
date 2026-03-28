"""
Analysis Service
=================
Manages async analysis jobs:
  1. Create job record in DB
  2. Run pipeline in background to build clean DataFrame
  3. Call active ML model (swappable via ml/__init__.py)
  4. Store graph result in DB

ML Integration Point: change ml/__init__.py to swap the model.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, List

import pandas as pd
from sqlalchemy.orm import Session

from models.job_models import AnalysisJob, GraphResult
from database import SessionLocal

logger = logging.getLogger(__name__)


def create_job(db: Session, ticker: str, date_from: str, date_to: str, user_id: Optional[str]) -> AnalysisJob:
    """Create a queued analysis job."""
    job = AnalysisJob(
        ticker=ticker.upper(),
        date_from=date_from,
        date_to=date_to,
        status="queued",
        created_by=uuid.UUID(user_id) if user_id else None,
        estimated_time=30,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> Optional[AnalysisJob]:
    try:
        return db.query(AnalysisJob).filter(AnalysisJob.id == uuid.UUID(job_id)).first()
    except Exception:
        return None


def get_job_result(db: Session, job_id: str) -> Optional[dict]:
    try:
        result = db.query(GraphResult).filter(GraphResult.job_id == uuid.UUID(job_id)).first()
        job = get_job(db, job_id)
        if not result or not job:
            return None
        return {
            "jobId": str(job.id),
            "ticker": job.ticker,
            "completedAt": result.created_at.isoformat() if result.created_at else None,
            "overallConviction": float(result.overall_conviction) if result.overall_conviction else None,
            "riskLevel": result.risk_level,
            "graph": result.graph_data,
            "summary": result.summary,
            "modelUsed": result.model_used,
        }
    except Exception as e:
        logger.error(f"get_job_result failed: {e}")
        return None


def get_analysis_history(db: Session, ticker: str) -> List[dict]:
    jobs = (
        db.query(AnalysisJob)
        .filter(AnalysisJob.ticker == ticker.upper(), AnalysisJob.status == "complete")
        .order_by(AnalysisJob.created_at.desc())
        .limit(20)
        .all()
    )
    return [
        {
            "jobId": str(j.id),
            "ticker": j.ticker,
            "completedAt": j.completed_at.isoformat() if j.completed_at else None,
            "status": j.status,
            "modelUsed": j.model_used,
        }
        for j in jobs
    ]


def run_analysis_background(job_id: str) -> None:
    """
    Background task: runs the full pipeline + ML model + stores results.
    Called by FastAPI BackgroundTasks — has its own DB session.
    """
    db: Session = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == uuid.UUID(job_id)).first()
        if not job:
            logger.error(f"Job {job_id} not found in background task")
            return

        # Mark as processing
        job.status = "processing"
        db.commit()

        # ── Run data pipeline ──────────────────────────────────
        from pipeline.AlphaNexusPipeline import AlphaNexusPipeline
        pipeline = AlphaNexusPipeline()
        df = pipeline.run(
            ticker=job.ticker,
            date_from=str(job.date_from) if job.date_from else None,
            date_to=str(job.date_to) if job.date_to else None,
            save_to_db=True,
            export_csv=False,
        )

        # ── Run ML model ───────────────────────────────────────
        # ← This is the integration point. ML friend swaps ml/__init__.ML_MODEL
        from ml import ML_MODEL

        if df.empty:
            logger.warning(f"Empty dataframe for job {job_id}, using mock graph")
            graph = _mock_graph(job.ticker)
        else:
            graph = ML_MODEL.run_analysis(
                df=df,
                ticker=job.ticker,
                date_from=str(job.date_from) if job.date_from else None,
                date_to=str(job.date_to) if job.date_to else None,
            )

        conviction = ML_MODEL.get_overall_conviction(graph)
        risk = ML_MODEL.get_risk_level(conviction)

        # ── Store result ───────────────────────────────────────
        result = GraphResult(
            job_id=job.id,
            ticker=job.ticker,
            overall_conviction=conviction,
            risk_level=risk,
            graph_data=graph,
            summary=_generate_summary(job.ticker, conviction, risk, graph),
            model_used=ML_MODEL.get_model_name(),
        )
        db.add(result)

        job.status = "complete"
        job.completed_at = datetime.utcnow()
        job.model_used = ML_MODEL.get_model_name()
        db.commit()
        logger.info(f"Job {job_id} complete. Conviction={conviction:.2f}, Risk={risk}")

    except Exception as e:
        logger.error(f"Analysis job {job_id} failed: {e}", exc_info=True)
        try:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == uuid.UUID(job_id)).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def _generate_summary(ticker: str, conviction: float, risk: str, graph: dict) -> str:
    flagged = [n for n in graph.get("nodes", []) if n.get("flagged")]
    num_edges = len(graph.get("edges", []))
    return (
        f"{len(flagged)} flagged entity(ies) detected for {ticker}. "
        f"Overall conviction score: {conviction:.2f} ({risk} risk). "
        f"{num_edges} trade connection(s) mapped in the shadow network."
    )


def _mock_graph(ticker: str) -> dict:
    """Returns a minimal graph for when no data is available (demo safety net)."""
    return {
        "nodes": [
            {"id": "demo-trader-1", "type": "politician", "label": "Demo Trader",
             "flagged": True, "convictionScore": 0.75, "gnnScore": 0.75,
             "isolationScore": 0.75, "metadata": {}},
            {"id": ticker, "type": "ticker", "label": ticker,
             "flagged": False, "convictionScore": 0.0, "gnnScore": 0.0,
             "isolationScore": 0.0, "metadata": {}},
        ],
        "edges": [
            {"source": "demo-trader-1", "target": ticker,
             "tradeValue": 500000, "tradeDate": "2024-01-15",
             "tradeType": "buy", "weight": 0.75}
        ],
    }
