import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    display_name = Column(String(255))
    password_hash = Column(String(500), nullable=False)
    role = Column(String(50), default="trader")         # 'trader' | 'analyst'
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Flag(Base):
    __tablename__ = "flags"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_id = Column(String(255), nullable=False)
    job_id = Column(UUID(as_uuid=True), ForeignKey("analysis_jobs.id"))
    reason = Column(Text)
    severity = Column(String(20), default="medium")     # 'low' | 'medium' | 'high' | 'critical'
    status = Column(String(20), default="pending")      # 'pending' | 'reviewed' | 'dismissed'
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    reviewed_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())


class Watchlist(Base):
    __tablename__ = "watchlist"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    ticker = Column(String(20), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
