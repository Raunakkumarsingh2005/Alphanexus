"""Watchlist router — 3 endpoints (get, add, remove)."""

import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from database import get_db
from models.user_models import Watchlist
from routers.deps import get_current_user
from schemas.watchlist import WatchlistAddRequest

router = APIRouter(prefix="/api/watchlist", tags=["Watchlist"])


@router.get("")
def get_watchlist(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Watchlist)
        .filter(Watchlist.user_id == uuid.UUID(current_user["id"]))
        .order_by(Watchlist.created_at.desc())
        .all()
    )
    return {
        "tickers": [
            {"id": str(r.id), "ticker": r.ticker, "createdAt": r.created_at.isoformat()}
            for r in rows
        ]
    }


@router.post("", status_code=201)
def add_to_watchlist(
    body: WatchlistAddRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    item = Watchlist(
        user_id=uuid.UUID(current_user["id"]),
        ticker=body.symbol.upper(),
    )
    try:
        db.add(item)
        db.commit()
        db.refresh(item)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail=f"{body.symbol.upper()} already in watchlist")

    return {"id": str(item.id), "ticker": item.ticker, "createdAt": item.created_at.isoformat()}


@router.delete("/{symbol}", status_code=204)
def remove_from_watchlist(
    symbol: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    deleted = (
        db.query(Watchlist)
        .filter(
            Watchlist.user_id == uuid.UUID(current_user["id"]),
            Watchlist.ticker == symbol.upper(),
        )
        .delete()
    )
    db.commit()
    if not deleted:
        raise HTTPException(status_code=404, detail=f"{symbol.upper()} not in watchlist")
