"""
AlphaNexus FastAPI Application
================================
Entry point: uvicorn main:app --reload

Swagger docs: http://localhost:8000/docs
ReDoc:        http://localhost:8000/redoc
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import init_db
from routers import auth, tickers, news, analysis, graph, flags, watchlist, compliance

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("alphanexus")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB tables if they don't exist."""
    logger.info("AlphaNexus starting up...")
    try:
        init_db()
        logger.info("Database initialized ✓")
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        logger.warning("⚠ Make sure PostgreSQL is running and DATABASE_URL is correct in .env")
    yield
    logger.info("AlphaNexus shutting down.")


app = FastAPI(
    title="AlphaNexus API",
    description=(
        "AI-driven insider trading signal detection engine. "
        "Correlates SEC, congressional, and market data to surface "
        "information asymmetry before it becomes public knowledge."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────
# Permissive for hackathon — frontend dev server on any port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(tickers.router)
app.include_router(news.router)
app.include_router(analysis.router)
app.include_router(graph.router)
app.include_router(flags.router)
app.include_router(watchlist.router)
app.include_router(compliance.router)


# ── Health check ──────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "AlphaNexus API", "version": "1.0.0"}


@app.get("/", tags=["Health"])
def root():
    return {
        "message": "AlphaNexus API is running",
        "docs": "/docs",
        "health": "/health",
        "roles": {
            "retailTrader": {
                "description": "Enter ticker → ML analysis → D3 graph visualisation",
                "pipeline": [
                    "POST /api/auth/register (role=trader)",
                    "GET  /api/tickers/{symbol}",
                    "POST /api/analysis/run",
                    "GET  /api/analysis/{jobId}/status  (poll)",
                    "GET  /api/analysis/{jobId}/result  (D3 JSON + conviction)",
                    "GET  /api/graph/{jobId}            (full graph)",
                ],
            },
            "complianceAuditor": {
                "description": "Enter ticker → news + insider/political trade review",
                "pipeline": [
                    "POST /api/auth/register (role=analyst)",
                    "GET  /api/compliance/news/{ticker}    (paginated news)",
                    "GET  /api/compliance/trades/{ticker}  (insider + congressional)",
                    "GET  /api/compliance/summary/{ticker} (overview card)",
                    "POST /api/flags                       (flag suspicious nodes)",
                ],
            },
        },
    }
