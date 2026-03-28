import uuid
from sqlalchemy import Column, String, Numeric, BigInteger, Date, DateTime, Text, ARRAY, func
from sqlalchemy.dialects.postgresql import UUID
from database import Base


class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(20), nullable=False)
    date = Column(Date, nullable=False)
    open_price = Column(Numeric(12, 4))
    high_price = Column(Numeric(12, 4))
    low_price = Column(Numeric(12, 4))
    close_price = Column(Numeric(12, 4))
    volume = Column(BigInteger)
    volume_zscore = Column(Numeric(10, 4))
    source = Column(String(50), default="yfinance")  # 'finnhub' | 'yfinance'


class NewsSentiment(Base):
    __tablename__ = "news_sentiment"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(20))
    date = Column(Date)
    headline = Column(Text)
    source = Column(String(255))
    url = Column(Text)
    summary = Column(Text)
    sentiment_label = Column(String(20))        # 'positive' | 'negative' | 'neutral'
    sentiment_score = Column(Numeric(6, 4))     # -1.0 to 1.0 (filled by FinBERT)
    related_tickers = Column(ARRAY(String))
    published_at = Column(DateTime)
    data_source = Column(String(50), default="newsapi")  # 'newsapi' | 'gdelt'
    created_at = Column(DateTime, server_default=func.now())
