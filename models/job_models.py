import uuid
from sqlalchemy import Column, String, Numeric, Integer, Date, DateTime, Text, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from database import Base


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(20), nullable=False)
    date_from = Column(Date)
    date_to = Column(Date)
    # queued | processing | complete | failed
    status = Column(String(20), default="queued")
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    estimated_time = Column(Integer, default=30)       # seconds
    completed_at = Column(DateTime)
    error_message = Column(Text)
    # 'isolation_forest_networkx' | 'gnn'
    model_used = Column(String(50), default="isolation_forest_networkx")
    created_at = Column(DateTime, server_default=func.now())


class GraphResult(Base):
    __tablename__ = "graph_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("analysis_jobs.id", ondelete="CASCADE"))
    ticker = Column(String(20))
    overall_conviction = Column(Numeric(6, 4))
    # 'low' | 'medium' | 'high' | 'critical'
    risk_level = Column(String(20))
    graph_data = Column(JSONB, nullable=False)          # {nodes:[...], edges:[...]}
    summary = Column(Text)
    model_used = Column(String(50))
    created_at = Column(DateTime, server_default=func.now())
