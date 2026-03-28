"""Graph router — 2 endpoints (full graph data, single node detail)."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from routers.deps import get_current_user
import services.graph_service as graph_svc

router = APIRouter(prefix="/api/graph", tags=["Graph"])


@router.get("/{job_id}")
def get_graph(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Full D3 graph data for a completed analysis job."""
    graph = graph_svc.get_full_graph(db, job_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Graph not found. Run /api/analysis/run first.")
    return graph


@router.get("/{job_id}/node/{node_id}")
def get_node_detail(
    job_id: str,
    node_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Single node detail with trade history and related news (for side panel)."""
    node = graph_svc.get_node_detail(db, job_id, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node
