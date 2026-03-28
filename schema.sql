-- AlphaNexus PostgreSQL Schema
-- Run: psql -U postgres -c "CREATE DATABASE alphanexus;" && psql -U postgres -d alphanexus -f schema.sql

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- USERS (local auth — JWT based)
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    password_hash VARCHAR(500) NOT NULL,
    role VARCHAR(50) DEFAULT 'trader',          -- 'trader' | 'analyst'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- SOURCE TABLE 1: political_trades (Quiver Quantitative)
-- ============================================================
CREATE TABLE IF NOT EXISTS political_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trader_id VARCHAR(100),                     -- canonical: CIK if available, else generated
    trader_name VARCHAR(255),
    ticker VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    trade_value_min NUMERIC(20, 2),
    trade_value_max NUMERIC(20, 2),
    trade_value NUMERIC(20, 2),                 -- midpoint of min/max
    direction VARCHAR(10),                      -- 'buy' | 'sell'
    -- Derived features (computed by feature_engineering.py)
    days_before_bill_vote INTEGER,
    volume_zscore NUMERIC(10, 4),
    trade_value_zscore NUMERIC(10, 4),
    trader_historical_frequency INTEGER,
    -- ML scores (filled by ML pipeline)
    anomaly_score NUMERIC(10, 4),
    conviction_score NUMERIC(10, 4),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(trader_name, ticker, trade_date, direction)
);

-- ============================================================
-- SOURCE TABLE 2: insider_trades (SEC EDGAR Form-4)
-- ============================================================
CREATE TABLE IF NOT EXISTS insider_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cik VARCHAR(20),                            -- SEC canonical ID (PRIMARY identifier)
    trader_name VARCHAR(255),
    trader_title VARCHAR(255),
    ticker VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    exact_value NUMERIC(20, 2),
    shares BIGINT,
    price_per_share NUMERIC(10, 4),
    direction VARCHAR(10),                      -- 'buy' | 'sell'
    filing_date DATE,
    -- ML scores
    anomaly_score NUMERIC(10, 4),
    conviction_score NUMERIC(10, 4),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(cik, ticker, trade_date, shares)
);

-- ============================================================
-- SOURCE TABLE 3: market_data (Finnhub + yFinance)
-- ============================================================
CREATE TABLE IF NOT EXISTS market_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open_price NUMERIC(12, 4),
    high_price NUMERIC(12, 4),
    low_price NUMERIC(12, 4),
    close_price NUMERIC(12, 4),
    volume BIGINT,
    volume_zscore NUMERIC(10, 4),               -- computed during ingest
    source VARCHAR(50) DEFAULT 'yfinance',      -- 'finnhub' | 'yfinance'
    UNIQUE(ticker, date)
);

-- ============================================================
-- SOURCE TABLE 4: news_sentiment (NewsAPI + GDELT)
-- ============================================================
CREATE TABLE IF NOT EXISTS news_sentiment (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(20),
    date DATE,
    headline TEXT,
    source VARCHAR(255),
    url TEXT,
    summary TEXT,
    sentiment_label VARCHAR(20),                -- 'positive' | 'negative' | 'neutral'
    sentiment_score NUMERIC(6, 4),              -- -1.0 to 1.0 (filled by FinBERT)
    related_tickers TEXT[],
    published_at TIMESTAMP,
    data_source VARCHAR(50) DEFAULT 'newsapi',  -- 'newsapi' | 'gdelt'
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- ANALYSIS JOBS
-- ============================================================
CREATE TABLE IF NOT EXISTS analysis_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(20) NOT NULL,
    date_from DATE,
    date_to DATE,
    status VARCHAR(20) DEFAULT 'queued',        -- 'queued' | 'processing' | 'complete' | 'failed'
    created_by UUID REFERENCES users(id),
    estimated_time INTEGER DEFAULT 30,          -- seconds
    completed_at TIMESTAMP,
    error_message TEXT,
    model_used VARCHAR(50) DEFAULT 'isolation_forest_networkx', -- 'isolation_forest_networkx' | 'gnn'
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- GRAPH RESULTS (D3 JSON contract output)
-- ============================================================
CREATE TABLE IF NOT EXISTS graph_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    ticker VARCHAR(20),
    overall_conviction NUMERIC(6, 4),
    risk_level VARCHAR(20),                     -- 'low' | 'medium' | 'high' | 'critical'
    graph_data JSONB NOT NULL,                  -- {nodes:[...], edges:[...]}
    summary TEXT,
    model_used VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- FLAGS (human review queue)
-- ============================================================
CREATE TABLE IF NOT EXISTS flags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id VARCHAR(255) NOT NULL,
    job_id UUID REFERENCES analysis_jobs(id),
    reason TEXT,
    severity VARCHAR(20) DEFAULT 'medium',      -- 'low' | 'medium' | 'high' | 'critical'
    status VARCHAR(20) DEFAULT 'pending',        -- 'pending' | 'reviewed' | 'dismissed'
    created_by UUID REFERENCES users(id),
    reviewed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- WATCHLIST
-- ============================================================
CREATE TABLE IF NOT EXISTS watchlist (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    ticker VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, ticker)
);

-- ============================================================
-- INDEXES (performance)
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_political_trades_ticker ON political_trades(ticker);
CREATE INDEX IF NOT EXISTS idx_political_trades_date ON political_trades(trade_date);
CREATE INDEX IF NOT EXISTS idx_political_trades_trader ON political_trades(trader_id);

CREATE INDEX IF NOT EXISTS idx_insider_trades_ticker ON insider_trades(ticker);
CREATE INDEX IF NOT EXISTS idx_insider_trades_date ON insider_trades(trade_date);
CREATE INDEX IF NOT EXISTS idx_insider_trades_cik ON insider_trades(cik);

CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date ON market_data(ticker, date);

CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_sentiment(ticker);
CREATE INDEX IF NOT EXISTS idx_news_date ON news_sentiment(date);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_label ON news_sentiment(sentiment_label);

CREATE INDEX IF NOT EXISTS idx_analysis_jobs_ticker ON analysis_jobs(ticker);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON analysis_jobs(status);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_created_by ON analysis_jobs(created_by);

CREATE INDEX IF NOT EXISTS idx_graph_results_job_id ON graph_results(job_id);
CREATE INDEX IF NOT EXISTS idx_flags_status ON flags(status);
CREATE INDEX IF NOT EXISTS idx_watchlist_user ON watchlist(user_id);
