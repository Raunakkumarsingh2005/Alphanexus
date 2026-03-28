from pydantic import BaseModel
from typing import Optional, List, Any, Dict


class GraphNodeMetadata(BaseModel):
    party: Optional[str] = None
    state: Optional[str] = None
    committee: Optional[str] = None
    price: Optional[float] = None
    sector: Optional[str] = None


class GraphNode(BaseModel):
    id: str
    type: str                          # 'politician' | 'ticker' | 'bill'
    label: Optional[str] = None
    flagged: bool = False
    convictionScore: Optional[float] = None
    gnnScore: Optional[float] = None
    isolationScore: Optional[float] = None
    metadata: Optional[GraphNodeMetadata] = None


class GraphEdge(BaseModel):
    source: str
    target: str
    tradeValue: Optional[float] = None
    tradeDate: Optional[str] = None
    tradeType: Optional[str] = None
    weight: Optional[float] = None


class GraphOut(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class TradeInNodeDetail(BaseModel):
    ticker: str
    date: str
    type: str
    value: Optional[float]
    shares: Optional[float]


class RelatedNewsInNode(BaseModel):
    headline: str
    sentiment: Optional[str]
    date: Optional[str]


class GraphNodeDetail(GraphNode):
    """Single node with full trade history and related news (for side panel)."""
    trades: Optional[List[TradeInNodeDetail]] = None
    relatedNews: Optional[List[RelatedNewsInNode]] = None
