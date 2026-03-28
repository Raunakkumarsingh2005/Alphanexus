"""Flags router — 3 endpoints (create, list, update)."""

import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database import get_db
from models.user_models import Flag
from routers.deps import get_current_user
from schemas.flag import FlagCreateRequest, FlagUpdateRequest

router = APIRouter(prefix="/api/flags", tags=["Flags"])


@router.post("", status_code=201)
def create_flag(
    body: FlagCreateRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    flag = Flag(
        node_id=body.nodeId,
        job_id=body.jobId,
        reason=body.reason,
        severity=body.severity,
        status="pending",
        created_by=uuid.UUID(current_user["id"]),
    )
    db.add(flag)
    db.commit()
    db.refresh(flag)
    return {
        "flagId": str(flag.id),
        "status": flag.status,
        "createdAt": flag.created_at.isoformat(),
        "createdBy": str(flag.created_by),
    }


@router.get("")
def list_flags(
    status: str = Query(None, description="Filter: pending | reviewed | dismissed"),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Flag)
    if status:
        query = query.filter(Flag.status == status)
    flags = query.order_by(Flag.created_at.desc()).limit(100).all()
    return {
        "flags": [
            {
                "flagId": str(f.id),
                "nodeId": f.node_id,
                "jobId": str(f.job_id) if f.job_id else None,
                "reason": f.reason,
                "severity": f.severity,
                "status": f.status,
                "createdAt": f.created_at.isoformat() if f.created_at else None,
            }
            for f in flags
        ],
        "total": len(flags),
    }


@router.put("/{flag_id}")
def update_flag(
    flag_id: str,
    body: FlagUpdateRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        flag = db.query(Flag).filter(Flag.id == uuid.UUID(flag_id)).first()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid flag ID")
    if not flag:
        raise HTTPException(status_code=404, detail="Flag not found")

    flag.status = body.status
    db.commit()
    return {"flagId": flag_id, "status": flag.status}
