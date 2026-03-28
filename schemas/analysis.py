from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from datetime import date, datetime
import uuid


class DateRange(BaseModel):
    from_date: date
    to_date: date


class AnalysisRunRequest(BaseModel):
    ticker: str
    dateRange: DateRange


class AnalysisJobOut(BaseModel):
    jobId: uuid.UUID
    status: str
    estimatedTime: int = 30
    ticker: Optional[str] = None
    createdAt: Optional[datetime] = None
    modelUsed: Optional[str] = None

    class Config:
        from_attributes = True


class AnalysisResultOut(BaseModel):
    jobId: uuid.UUID
    ticker: str
    completedAt: Optional[datetime]
    overallConviction: Optional[float]
    riskLevel: Optional[str]
    graph: Optional[Dict[str, Any]]
    summary: Optional[str]
    modelUsed: Optional[str]


class AnalysisHistoryItem(BaseModel):
    jobId: uuid.UUID
    ticker: str
    completedAt: Optional[datetime]
    overallConviction: Optional[float]
    riskLevel: Optional[str]
    status: str
