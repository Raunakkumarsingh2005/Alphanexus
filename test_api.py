"""
AlphaNexus API Test Suite
=========================
Tests both roles end-to-end:
  - Retail Trader: ticker → SEC/News/Finnhub → ML → D3 JSON
  - Compliance Auditor: ticker → news articles only

Run: python test_api.py
Server must be running at http://localhost:8002
"""

import time
import requests

BASE = "http://localhost:8005"
TIMEOUT = 30  # seconds per request

session = requests.Session()

PASS = "\033[92m✔\033[0m"
FAIL = "\033[91m✘\033[0m"
WARN = "\033[93m⚠\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

results = {"pass": 0, "fail": 0}


def ok(label, resp, expected=200):
    code = resp.status_code if resp else 0
    if code == expected:
        results["pass"] += 1
        print(f"  {PASS} {label}  [{code}]")
        return True
    else:
        results["fail"] += 1
        detail = ""
        try:
            detail = f" — {resp.json()}"
        except Exception:
            detail = f" — {resp.text[:100]}"
        print(f"  {FAIL} {label}  [{code}]{detail}")
        return False


def section(title):
    print(f"\n{BOLD}{'─'*50}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*50}{RESET}")


def get(path, headers=None):
    try:
        return session.get(f"{BASE}{path}", headers=headers, timeout=TIMEOUT)
    except Exception as e:
        print(f"  {FAIL} {path} — ConnectionError: {e}")
        return type("R", (), {"status_code": 0, "json": lambda s: {}, "text": str(e)})()


def post(path, body=None, headers=None):
    try:
        return session.post(f"{BASE}{path}", json=body, headers=headers, timeout=TIMEOUT)
    except Exception as e:
        print(f"  {FAIL} {path} — ConnectionError: {e}")
        return type("R", (), {"status_code": 0, "json": lambda s: {}, "text": str(e)})()


TOKEN = None
REFRESH = None
JOB_ID = None

TS = int(time.time())
ANALYST_EMAIL = f"analyst_{TS}@alphanexus.ai"
TRADER_EMAIL  = f"trader_{TS}@alphanexus.ai"
TEST_TICKER = "AAPL"

# ─────────────────────────────────────────
# 0. Health
# ─────────────────────────────────────────
section("0 · Health")
ok("GET /health", get("/health"))
ok("GET /",       get("/"))

# ─────────────────────────────────────────
# 1. Auth
# ─────────────────────────────────────────
section("1 · Auth")

r = post("/api/auth/register", {"email": ANALYST_EMAIL, "password": "Test1234",
         "displayName": "Test Analyst", "role": "analyst"})
if ok("POST /api/auth/register (analyst)", r, 201):
    TOKEN   = r.json()["accessToken"]
    REFRESH = r.json()["refreshToken"]

r = post("/api/auth/register", {"email": TRADER_EMAIL, "password": "Test1234",
         "displayName": "Test Trader", "role": "trader"})
ok("POST /api/auth/register (trader)", r, 201)

r = post("/api/auth/login", {"email": ANALYST_EMAIL, "password": "Test1234"})
if ok("POST /api/auth/login", r):
    TOKEN = r.json()["accessToken"]

r = post("/api/auth/refresh", {"refreshToken": REFRESH})
ok("POST /api/auth/refresh", r)

AUTH = {"Authorization": f"Bearer {TOKEN}"}

ok("GET  /api/auth/me",  get("/api/auth/me", AUTH))
ok("PUT  /api/auth/me",  session.put(f"{BASE}/api/auth/me", json={"displayName": "Updated"}, headers=AUTH, timeout=TIMEOUT))

# ─────────────────────────────────────────
# 2. Tickers
# ─────────────────────────────────────────
section(f"2 · Tickers ({TEST_TICKER})")
ok("GET /api/tickers/search?q=AAPL",            get("/api/tickers/search?q=AAPL", AUTH))
ok(f"GET /api/tickers/{TEST_TICKER}",            get(f"/api/tickers/{TEST_TICKER}", AUTH))
ok(f"GET /api/tickers/{TEST_TICKER}/history",    get(f"/api/tickers/{TEST_TICKER}/history?from=2024-01-01&to=2024-06-01", AUTH))
ok(f"GET /api/tickers/{TEST_TICKER}/insider-trades", get(f"/api/tickers/{TEST_TICKER}/insider-trades", AUTH))

# ─────────────────────────────────────────
# 3. News — Compliance Auditor flow
# ─────────────────────────────────────────
section(f"3 · News / Compliance Auditor ({TEST_TICKER})")
r = get(f"/api/news?ticker={TEST_TICKER}", AUTH)
if ok(f"GET /api/news?ticker={TEST_TICKER}", r):
    arts = r.json().get("articles", [])
    print(f"       → {len(arts)} article(s) in DB")

ok("GET /api/news/trending",                       get("/api/news/trending", AUTH))
ok(f"GET /api/compliance/news/{TEST_TICKER}",      get(f"/api/compliance/news/{TEST_TICKER}", AUTH))
ok(f"GET /api/compliance/trades/{TEST_TICKER}",    get(f"/api/compliance/trades/{TEST_TICKER}", AUTH))
ok(f"GET /api/compliance/summary/{TEST_TICKER}",   get(f"/api/compliance/summary/{TEST_TICKER}", AUTH))

# ─────────────────────────────────────────
# 4. Analysis — Retail Trader ML flow
#    Fire-and-verify: don't block on ingestion
# ─────────────────────────────────────────
section(f"4 · Analysis — Retail Trader ML ({TEST_TICKER})")

r = post("/api/analysis/run", {
    "ticker": TEST_TICKER,
    "dateRange": {"from_date": "2024-01-01", "to_date": "2024-03-01"}
}, AUTH)

if ok("POST /api/analysis/run", r):
    JOB_ID = r.json().get("jobId")
    print(f"       → jobId = {JOB_ID}")
    print(f"  {WARN} Waiting 5s for job to start processing...")
    time.sleep(5)

    rs = get(f"/api/analysis/{JOB_ID}/status", AUTH)
    if ok(f"GET /api/analysis/status", rs):
        status = rs.json().get("status")
        print(f"       → status = {status}  (processing / complete — both valid here)")

    ok("GET /api/analysis/history", get(f"/api/analysis/history?ticker={TEST_TICKER}", AUTH))

    # Result might be 202 if still processing — that is correct behaviour
    rr = get(f"/api/analysis/{JOB_ID}/result", AUTH)
    if rr.status_code == 200:
        ok(f"GET /api/analysis/result", rr)
        graph = rr.json().get("graph", {})
        print(f"       → nodes={len(graph.get('nodes',[]))}, edges={len(graph.get('edges',[]))}")
        print(f"       → conviction={rr.json().get('overallConviction')}, risk={rr.json().get('riskLevel')}")
        rg = get(f"/api/graph/{JOB_ID}", AUTH)
        ok(f"GET /api/graph/{{jobId}}", rg)
        if rg.status_code == 200:
            nodes = rg.json().get("nodes", [])
            if nodes:
                nid = nodes[0]["id"]
                ok(f"GET /api/graph/node", get(f"/api/graph/{JOB_ID}/node/{nid}", AUTH))
    elif rr.status_code == 202:
        print(f"  {WARN} /api/analysis/result → 202 (still processing — pipeline is running in background ✓)")
        results["pass"] += 1
    else:
        ok(f"GET /api/analysis/result", rr)

# ─────────────────────────────────────────
# 5. Watchlist
# ─────────────────────────────────────────
section("5 · Watchlist")
ok("POST /api/watchlist",  session.post(f"{BASE}/api/watchlist", json={"symbol": TEST_TICKER}, headers=AUTH, timeout=TIMEOUT), 201)
ok("GET  /api/watchlist",  get("/api/watchlist", AUTH))
ok(f"DEL  /api/watchlist/{TEST_TICKER}", session.delete(f"{BASE}/api/watchlist/{TEST_TICKER}", headers=AUTH, timeout=TIMEOUT), 204)

# ─────────────────────────────────────────
# 6. Flags
# ─────────────────────────────────────────
section("6 · Flags")
if JOB_ID:
    r = post("/api/flags", {"nodeId": "demo-trader-1", "jobId": JOB_ID,
              "reason": "Suspicious pattern", "severity": "high"}, AUTH)
    if ok("POST /api/flags", r, 201):
        fid = r.json().get("flagId")
        ok(f"PUT  /api/flags", session.put(f"{BASE}/api/flags/{fid}", json={"status": "reviewed"}, headers=AUTH, timeout=TIMEOUT))

ok("GET  /api/flags", get("/api/flags", AUTH))

# ─────────────────────────────────────────
# 7. Logout
# ─────────────────────────────────────────
section("7 · Logout")
ok("POST /api/auth/logout", post("/api/auth/logout", headers=AUTH))

# ─────────────────────────────────────────
# Summary
# ─────────────────────────────────────────
total = results["pass"] + results["fail"]
pct   = int(100 * results["pass"] / total) if total else 0
print(f"\n{BOLD}{'='*50}{RESET}")
print(f"{BOLD}  RESULTS: {results['pass']}/{total} passed ({pct}%){RESET}")
print(f"{BOLD}  Retail Trader:    register → analysis/run → status → result → graph{RESET}")
print(f"{BOLD}  Compliance Audit: register → /compliance/news → /compliance/trades → /compliance/summary{RESET}")
print(f"{BOLD}{'='*50}{RESET}\n")

if results["fail"] > 0:
    raise SystemExit(1)
