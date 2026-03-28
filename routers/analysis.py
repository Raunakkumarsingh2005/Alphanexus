"""Analysis router — 4 endpoints (trigger, poll, result, history)."""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session

from database import get_db
from routers.deps import get_current_user
from schemas.analysis import AnalysisRunRequest
import services.analysis_service as analysis_svc

router = APIRouter(prefix="/api/analysis", tags=["Analysis"])


@router.post("/run")
def trigger_analysis(
    body: AnalysisRunRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Trigger GNN analysis for a ticker. Returns a jobId immediately.
    Poll /api/analysis/{jobId}/status for progress.
    """
    job = analysis_svc.create_job(
        db,
        ticker=body.ticker,
        date_from=str(body.dateRange.from_date),
        date_to=str(body.dateRange.to_date),
        user_id=current_user.get("id"),
    )
    # Run pipeline + ML in background
    background_tasks.add_task(analysis_svc.run_analysis_background, str(job.id))

    return {
        "jobId": str(job.id),
        "status": "queued",
        "estimatedTime": job.estimated_time,
        "ticker": job.ticker,
    }


@router.get("/{job_id}/status")
def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    job = analysis_svc.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "jobId": str(job.id),
        "status": job.status,
        "ticker": job.ticker,
        "modelUsed": job.model_used,
        "createdAt": job.created_at.isoformat() if job.created_at else None,
        "completedAt": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error_message,
    }


@router.get("/history")
def get_history(
    ticker: str = Query(..., description="Ticker symbol"),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return {"history": analysis_svc.get_analysis_history(db, ticker)}


@router.get("/{job_id}/result")
def get_result(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    result = analysis_svc.get_job_result(db, job_id)
    if not result:
        job = analysis_svc.get_job(db, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "complete":
            raise HTTPException(status_code=202, detail=f"Job is {job.status}. Poll /status first.")
        raise HTTPException(status_code=404, detail="Result not found")
    return result
