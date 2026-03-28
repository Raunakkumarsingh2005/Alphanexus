# AlphaNexus — Backend API Specification

> **stack **: pocketBase (auth)  Python (FastAPI) for APIs.
> This document lists every API the frontend needs regardless of framework choice.

---

## 1. Authentication APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/register` | Register new user (email, password, displayName) |
| `POST` | `/api/auth/login` | Login → returns JWT / session token |
| `POST` | `/api/auth/logout` | Invalidate session |
| `POST` | `/api/auth/refresh` | Refresh expired access token |
| `GET`  | `/api/auth/me` | Get current user profile |
| `PUT`  | `/api/auth/me` | Update user profile (role, preferences) |
| `POST` | `/api/auth/google` | OAuth login via Google |

### Auth Models

```json
// Register Request
{
  "email": "user@example.com",
  "password": "securePassword123",
  "displayName": "John Doe",
  "role": "analyst" | "trader"
}

// Login Response
{
  "accessToken": "eyJ...",
  "refreshToken": "eyJ...",
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "displayName": "John Doe",
    "role": "analyst",
    "createdAt": "2026-03-28T00:00:00Z"
  }
}
```

---

## 2. Ticker / Stock APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/tickers/search?q={query}` | Autocomplete ticker search (e.g., "NVD" → NVDA, NVDS) |
| `GET` | `/api/tickers/{symbol}` | Get ticker summary (price, volume, market cap, 52w range) |
| `GET` | `/api/tickers/{symbol}/history?from={date}&to={date}` | Historical price data for sparklines |
| `GET` | `/api/tickers/{symbol}/insider-trades` | SEC Form-4 insider trades for this ticker |

### Ticker Models

```json
// GET /api/tickers/NVDA
{
  "symbol": "NVDA",
  "name": "NVIDIA Corp",
  "price": 142.55,
  "change": 3.21,
  "changePercent": 2.31,
  "volume": 45200000,
  "marketCap": 3500000000000,
  "high52w": 152.89,
  "low52w": 75.61
}

// GET /api/tickers/NVDA/insider-trades
{
  "trades": [
    {
      "id": "uuid",
      "traderName": "Jensen Huang",
      "traderTitle": "CEO",
      "tradeDate": "2026-03-15",
      "tradeType": "sell",
      "shares": 50000,
      "pricePerShare": 140.25,
      "totalValue": 7012500,
      "filingDate": "2026-03-17",
      "source": "SEC Form-4"
    }
  ]
}
```

---

## 3. News APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/news?ticker={symbol}&page={n}&limit={n}` | News articles for a ticker, paginated |
| `GET` | `/api/news/{articleId}` | Single article detail with full sentiment breakdown |
| `GET` | `/api/news/trending` | Top trending news across all tracked tickers |

### News Models

```json
// GET /api/news?ticker=NVDA
{
  "articles": [
    {
      "id": "uuid",
      "headline": "NVIDIA CEO sells $7M in stock ahead of earnings",
      "source": "Reuters",
      "publishedAt": "2026-03-17T10:30:00Z",
      "url": "https://reuters.com/...",
      "sentiment": "negative",
      "sentimentScore": -0.72,
      "relatedTickers": ["NVDA", "AMD"],
      "summary": "Jensen Huang sold 50,000 shares..."
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 148
  }
}
```

---

## 4. Analysis / GNN Model APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analysis/run` | Trigger GNN analysis for a ticker (async job) |
| `GET`  | `/api/analysis/{jobId}/status` | Poll job status (queued / processing / complete / failed) |
| `GET`  | `/api/analysis/{jobId}/result` | Get completed analysis results |
| `GET`  | `/api/analysis/history?ticker={symbol}` | Past analysis results for a ticker |

### Analysis Models

```json
// POST /api/analysis/run
{
  "ticker": "NVDA",
  "dateRange": {
    "from": "2025-01-01",
    "to": "2026-03-28"
  }
}

// Response
{
  "jobId": "uuid",
  "status": "queued",
  "estimatedTime": 30
}

// GET /api/analysis/{jobId}/result
{
  "jobId": "uuid",
  "ticker": "NVDA",
  "completedAt": "2026-03-28T10:35:00Z",
  "overallConviction": 0.87,
  "riskLevel": "high",
  "graph": {
    "nodes": [...],   // See Graph API below
    "edges": [...]
  },
  "summary": "High insider trading correlation detected..."
}
```

---

## 5. Graph APIs (D3 Force Graph Data)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/graph/{jobId}` | Full graph data (nodes + edges) for D3 rendering |
| `GET` | `/api/graph/{jobId}/node/{nodeId}` | Single node detail (for side panel) |

### Graph Models

```json
// GET /api/graph/{jobId}
{
  "nodes": [
    {
      "id": "node-1",
      "type": "politician",        // "politician" | "ticker" | "bill"
      "label": "Nancy Pelosi",
      "flagged": true,
      "convictionScore": 0.92,
      "gnnScore": 0.88,
      "isolationScore": 0.75,
      "metadata": {
        "party": "Democrat",
        "state": "CA",
        "committee": "Financial Services"
      }
    },
    {
      "id": "node-2",
      "type": "ticker",
      "label": "NVDA",
      "flagged": false,
      "convictionScore": 0.31,
      "metadata": {
        "price": 142.55,
        "sector": "Technology"
      }
    }
  ],
  "edges": [
    {
      "source": "node-1",
      "target": "node-2",
      "tradeValue": 1500000,
      "tradeDate": "2026-02-10",
      "tradeType": "buy",
      "weight": 0.85
    }
  ]
}

// GET /api/graph/{jobId}/node/{nodeId}
{
  "id": "node-1",
  "type": "politician",
  "label": "Nancy Pelosi",
  "flagged": true,
  "convictionScore": 0.92,
  "gnnScore": 0.88,
  "isolationScore": 0.75,
  "trades": [
    {
      "ticker": "NVDA",
      "date": "2026-02-10",
      "type": "buy",
      "value": 1500000,
      "shares": 10600
    }
  ],
  "relatedNews": [
    {
      "headline": "Pelosi discloses NVDA purchase days before CHIPS Act vote",
      "sentiment": "negative",
      "date": "2026-02-12"
    }
  ],
  "metadata": {
    "party": "Democrat",
    "state": "CA",
    "committee": "Financial Services"
  }
}
```

---

## 6. Flagging / Review APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/flags` | Flag a node for review |
| `GET`  | `/api/flags?status={status}` | List flagged items (pending / reviewed / dismissed) |
| `PUT`  | `/api/flags/{flagId}` | Update flag status |

### Flag Models

```json
// POST /api/flags
{
  "nodeId": "node-1",
  "jobId": "uuid",
  "reason": "Suspiciously timed trade before legislative action",
  "severity": "high"
}

// Response
{
  "flagId": "uuid",
  "status": "pending",
  "createdAt": "2026-03-28T10:40:00Z",
  "createdBy": "user-uuid"
}
```

---

## 7. User Preferences / Watchlist APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/watchlist` | Get user's watchlist tickers |
| `POST` | `/api/watchlist` | Add ticker to watchlist |
| `DELETE` | `/api/watchlist/{symbol}` | Remove ticker from watchlist |

---

## API Summary

| Category | Endpoints | Auth Required |
|----------|-----------|---------------|
| Auth | 7 | No (register/login), Yes (rest) |
| Tickers | 4 | Yes |
| News | 3 | Yes |
| Analysis | 4 | Yes |
| Graph | 2 | Yes |
| Flags | 3 | Yes |
| Watchlist | 3 | Yes |
| **Total** | **26** | |

---

## Tech Stack Decision (Pending)

| Option | Auth | Backend APIs | Pros | Cons |
|--------|------|-------------|------|------|
| **Firebase + Python (FastAPI)** | Firebase Auth | FastAPI | Fast dev, free tier, Python ML integration | Firebase lock-in |