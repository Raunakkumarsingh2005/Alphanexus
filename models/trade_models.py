import uuid
from sqlalchemy import Column, String, Numeric, Integer, Date, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from database import Base


class PoliticalTrade(Base):
    __tablename__ = "political_trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trader_id = Column(String(100))
    trader_name = Column(String(255))
    ticker = Column(String(20), nullable=False)
    trade_date = Column(Date, nullable=False)
    trade_value_min = Column(Numeric(20, 2))
    trade_value_max = Column(Numeric(20, 2))
    trade_value = Column(Numeric(20, 2))        # midpoint
    direction = Column(String(10))              # 'buy' | 'sell'

    # Derived features
    days_before_bill_vote = Column(Integer)
    volume_zscore = Column(Numeric(10, 4))
    trade_value_zscore = Column(Numeric(10, 4))
    trader_historical_frequency = Column(Integer)

    # ML scores (written by ML pipeline after training)
    anomaly_score = Column(Numeric(10, 4))
    conviction_score = Column(Numeric(10, 4))

    created_at = Column(DateTime, server_default=func.now())


class InsiderTrade(Base):
    __tablename__ = "insider_trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cik = Column(String(20))                    # SEC canonical ID
    trader_name = Column(String(255))
    trader_title = Column(String(255))
    ticker = Column(String(20), nullable=False)
    trade_date = Column(Date, nullable=False)
    exact_value = Column(Numeric(20, 2))
    shares = Column(Numeric(20))
    price_per_share = Column(Numeric(10, 4))
    direction = Column(String(10))              # 'buy' | 'sell'
    filing_date = Column(Date)

    # ML scores
    anomaly_score = Column(Numeric(10, 4))
    conviction_score = Column(Numeric(10, 4))

    created_at = Column(DateTime, server_default=func.now())
