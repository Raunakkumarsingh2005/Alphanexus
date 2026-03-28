from pydantic import BaseModel
from typing import Optional, Literal, List
from datetime import datetime
import uuid


class FlagCreateRequest(BaseModel):
    nodeId: str
    jobId: uuid.UUID
    reason: str
    severity: Literal["low", "medium", "high", "critical"] = "medium"


class FlagOut(BaseModel):
    flagId: uuid.UUID
    nodeId: str
    jobId: Optional[uuid.UUID]
    reason: Optional[str]
    severity: str
    status: str
    createdAt: datetime
    createdBy: Optional[uuid.UUID]

    class Config:
        from_attributes = True


class FlagUpdateRequest(BaseModel):
    status: Literal["pending", "reviewed", "dismissed"]


class FlagListOut(BaseModel):
    flags: List[FlagOut]
    total: int
