# AlphaNexus — Project Workflow & Full Context
> Place this file at the ROOT of your repository as `ALPHANEXUS_WORKFLOW.md`.
> Every future AI session, paste its contents or reference it for full context.

---

## What AlphaNexus Is

**AlphaNexus: The Information Asymmetry Engine**

AI-driven detection of insider trading signals from public data. Built for HackCraft 3.0, Phoenix Finance track, Team Sankalp (MSIT + DCE).

The core insight: Institutional "smart money" acts on signals 24-48 hours before they become public knowledge for retail investors. AlphaNexus detects those signals retroactively and in near-real-time using public data that already exists but nobody has correlated.

**One line pitch:**
"Retail investors are always last to know when smart money moves. AlphaNexus changes that."

**Demo line:**
"When three or more connected insiders trade the same ticker within 48 hours, this node lights up. That's the signal a hedge fund pays thirty thousand dollars a year to see."

---

## Team Structure

| Member | Role | Owns |
|--------|------|------|
| Mohit | Frontend (D3.js) + ML Support | D3 force graph, red pulse animation, backtester slider, FinBERT text corpus preparation |
| Raunak | Backend | API ingestion, data cleaning, PostgreSQL, derived feature computation |
| ML Friend | Machine Learning | Isolation Forest, GNN (PyTorch Geometric), FinBERT inference |
| 4th Member | Frontend Support | PocketBase setup, auth flow, alerts feed panel, backtester UI |

**Critical rule:** ML friend touches zero APIs, zero database setup, zero cleaning. He receives one clean dataframe and builds models on it. Everything else is taken from him so he can focus entirely on GNN.

---

## The Three Act Demo Narrative

This is the story you tell judges. Every technical decision serves this narrative.

### Act 1 — Proof of Concept (2021, Pelosi/NVDA)
Paul Pelosi buys NVDA calls. Bill passes. Stock surges. CNBC covers it weeks later.
AlphaNexus would have flagged it 14 days before CNBC covered it.
Tools: Isolation Forest + NetworkX. GNN not required here.

### Act 2 — The Pattern Emerges (2022-23, Tech Committee)
Multiple politicians on the same Senate committees making correlated trades across multiple tickers over months. Individual trades look normal. The aggregate is suspicious.
Tools: Isolation Forest + NetworkX minimum. GNN makes this significantly stronger — finds connections that aren't explicit in raw data.

### Act 3 — The Live Threat (2024, AI Chip Legislation Cluster)
NVDA, AMD, INTC, TSMC. Multiple politicians. Bill progression timeline. Correlated buying patterns across people who don't explicitly appear connected.
Tools: GNN is the differentiator here. This is where propagated suspicion through network topology reveals what explicit graph mapping cannot.

**The pitch:** "This isn't a one-time anomaly. This is a pattern. Let us show you three times it happened."

---

## Data Sources

### Primary Pipeline
| Source | Data Type | Cost | Key |
|--------|-----------|------|-----|
| Quiver Quantitative | Congressional trading (who, what ticker, when, range) | Free tier | Required, no card |
| SEC EDGAR Direct | Form 4 insider trades, CIK numbers, exact amounts | Free, no key | None needed |
| Finnhub | Stock price OHLCV, volume around trade dates | Free tier | Required, no card |
| GDELT | Historical news headlines for FinBERT (2021-2024) | Free, no key | None needed |
| NewsAPI | Recent news headlines (1 month lookback) | Free tier | Required, no card |

### Backup Sources
| Primary Fails | Use Instead |
|--------------|-------------|
| Quiver | Capitol Trades API |
| SEC EDGAR | Same — EDGAR is already the primary source |
| Finnhub | yFinance (Python library, no key, no rate limits) |
| NewsAPI | GDELT (already in stack) |

### Nuclear Option (zero paid APIs)
yFinance + EDGAR direct + Capitol Trades. Complete pipeline, costs nothing, zero API keys except Capitol Trades.

---

## Data Architecture

### Four Source Tables in PostgreSQL

```sql
political_trades (trader_id, trader_name, ticker, trade_date, 
                  trade_value_min, trade_value_max, direction)

insider_trades   (cik PRIMARY KEY, trader_name, ticker, 
                  trade_date, exact_value, shares, direction)

market_data      (ticker, date, close_price, volume, volume_zscore)

news_sentiment   (ticker, date, headline, sentiment_label, 
                  sentiment_score)
```

### Clean Joined Row (what ML friend receives)

```
trader_id | trader_name | ticker | trade_date | trade_value |
direction | price_on_date | volume_zscore | avg_sentiment_score |
days_before_bill_vote | anomaly_score | conviction_score
```

### Join Strategy
- SQL JOIN for `political_trades + market_data` (exact date match)
- Pandas merge for news sentiment (7-day window around trade date)
- Fuzzy name matching BEFORE any data enters PostgreSQL

---

## Data Cleaning (Raunak owns this entirely)

### The Core Problem
"Paul Pelosi" in Quiver vs "Paul F. Pelosi" in SEC Form 4 vs "Pelosi" in Reddit. Same person. String match fails.

### The Fix

```python
from rapidfuzz import fuzz

def match_cik(quiver_name, edgar_lookup):
    # edgar_lookup = {canonical_name: cik}
    best_match = max(
        edgar_lookup.keys(),
        key=lambda x: fuzz.ratio(quiver_name.lower(), x.lower())
    )
    score = fuzz.ratio(quiver_name.lower(), best_match.lower())
    if score > 85:
        return edgar_lookup[best_match]
    return None

# Use SEC's own canonical name via CIK lookup
headers = {"User-Agent": "AlphaNexus yourname@email.com"}
requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", 
             headers=headers)
```

### Rules
1. CIK is the canonical ID wherever it exists. Never match by name when CIK is available.
2. Fuzzy match threshold: 85%. Below that, no match — don't force it.
3. Politicians without CIK (not SEC filers): fuzzy match only, accept ~85% accuracy.
4. Enrichment runs BEFORE data enters PostgreSQL. Clean data goes in, never raw.

### Derived Features Raunak Computes (not ML friend's job)

```python
# days_before_bill_vote — requires bill passage dates table
df['days_before_bill_vote'] = (bill_date - df['trade_date']).days

# volume_zscore — how anomalous was volume on trade date
df['volume_zscore'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()

# trade_value_zscore — how anomalous was this person's trade size
df['trade_value_zscore'] = df.groupby('trader_id')['trade_value'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

---

## ML Pipeline

### Step 1: Isolation Forest (the engine — detects anomalies)

```python
from sklearn.ensemble import IsolationForest

features = df[['trade_value', 'days_before_bill_vote',
               'trader_historical_frequency', 'volume_zscore']]

model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_score'] = model.fit_predict(features)
df['conviction_score'] = model.decision_function(features)
# -1 = flagged anomaly, more negative = more suspicious
```

**What it catches:** Individual trades that are statistically anomalous — wrong size, wrong timing, wrong person's history.
**What it misses:** Two politicians who each look normal individually but are correlated. That's GNN's job.

### Step 2: FinBERT (the context layer — retail trader differentiator)

```python
from transformers import pipeline

finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Mohit delivers finbert_corpus.csv
df = pd.read_csv("finbert_corpus.csv")
df["sentiment"] = df["headline"].apply(
    lambda x: finbert(x[:512])[0]["label"]
)
df["sentiment_score"] = df["headline"].apply(
    lambda x: finbert(x[:512])[0]["score"]
)
```

**What it catches:** News sentiment shifting around flagged tickers BEFORE mainstream coverage. This is the 48-hour early warning signal.
**Why it matters:** Without FinBERT you have a detection tool. With FinBERT you have a prediction-adjacent tool. Different value propositions.

**Mohit prepares the text corpus (FinBERT input):**

```python
import requests, pandas as pd

date_ranges = [
    {"label": "pelosi_nvda", "ticker": "NVDA",
     "from": "20210101000000", "to": "20210630000000"},
    {"label": "tech_committee", "ticker": "NVDA AMD INTC",
     "from": "20220101000000", "to": "20231231000000"},
    {"label": "ai_chip_2024", "ticker": "NVDA AMD TSMC",
     "from": "20240101000000", "to": "20241231000000"}
]

all_headlines = []
for r in date_ranges:
    response = requests.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={"query": f"{r['ticker']} congress trade legislation",
                "mode": "artlist",
                "startdatetime": r["from"],
                "enddatetime": r["to"],
                "format": "json"}
    )
    for article in response.json().get("articles", []):
        all_headlines.append({
            "label": r["label"], "ticker": r["ticker"],
            "headline": article["title"], "date": article["seendate"]
        })

df = pd.DataFrame(all_headlines).drop_duplicates(subset=["headline"])
df.to_csv("finbert_corpus.csv", index=False)
# Hand this file to ML friend. He never touches GDELT.
```

### Step 3: NetworkX Graph Construction (always runs — fallback if GNN fails)

```python
import networkx as nx, json

G = nx.Graph()

for _, row in df.iterrows():
    G.add_node(row['trader_id'],
               conviction_score=row['conviction_score'],
               type=row['trader_type'],
               flagged=row['anomaly_score'] == -1)
    G.add_node(row['ticker'], type='ticker')
    G.add_edge(row['trader_id'], row['ticker'],
               value=row['trade_value'], date=str(row['trade_date']))

nodes = [{"id": n, **G.nodes[n]} for n in G.nodes()]
edges = [{"source": u, "target": v, **G.edges[u,v]}
         for u,v in G.edges()]

with open("graph_output.json", "w") as f:
    json.dump({"nodes": nodes, "edges": edges}, f)
```

**NetworkX is sufficient for:** Act 1 (Pelosi/NVDA). Explicit connections, single actor, clear timing.
**NetworkX is insufficient for:** Acts 2 and 3 — correlated trades across disconnected actors where suspicion must propagate through network topology.

### Step 4: GNN — PyTorch Geometric (the differentiator for Acts 2 and 3)

**Why GNN over NetworkX:** NetworkX shows connections that EXIST in the data. GNN finds connections the data IMPLIES but doesn't state. That's the difference between a map and a prediction.

**When GNN genuinely earns its place:**
- Pattern invisible without network position (two senators, same committee, same obscure ticker, 72 hour window — neither looks suspicious alone)
- Cross-entity contamination (corporate insider + politician, no explicit link, but network neighborhood overlaps)
- Large dense graphs where human eye cannot see clusters

**GNN Implementation:**

```python
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# Graph construction from clean data
# Node types: politicians, tickers, bills, committees
# Edge types: traded, voted_for, benefited, member_of

node_features = torch.tensor(feature_matrix, dtype=torch.float)
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
data = Data(x=node_features, edge_index=edge_index)

class AlphaNexusGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # SAGEConv handles mixed node types better than GCNConv
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        return torch.sigmoid(self.conv3(x, edge_index))

model = AlphaNexusGNN(input_dim=3, hidden_dim=64, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training: GNN learns to match Isolation Forest scores
# then propagates suspicion through network topology
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.mse_loss(out[train_mask], isolation_scores[train_mask])
    loss.backward()
    optimizer.step()
```

**GNN Output to JSON (feeds D3 directly):**

```python
model.eval()
with torch.no_grad():
    gnn_scores = model(data.x, data.edge_index).numpy()

nodes = []
for i, node_id in enumerate(node_ids):
    nodes.append({
        "id": node_id,
        "type": node_types[i],
        "gnn_score": float(gnn_scores[i]),
        "isolation_score": float(isolation_scores[i]),
        "conviction": float((gnn_scores[i] + isolation_scores[i]) / 2),
        "flagged": float(gnn_scores[i]) > 0.7
    })

edges = [{"source": u, "target": v, **attrs}
         for u, v, attrs in G.edges(data=True)]

with open("graph_output.json", "w") as f:
    json.dump({"nodes": nodes, "edges": edges}, f)
```

**CRITICAL — Install PyTorch Geometric BEFORE hackathon starts:**
```bash
pip install torch torchvision torchaudio
pip install torch_geometric
# Verify it works by running the mock graph construction
# If it fails on Windows, debug tonight not at 3am
```

---

## Frontend — D3.js Force Graph (Mohit owns this)

### JSON Contract (agreed before hackathon, never changes)

```json
{
  "nodes": [
    {
      "id": "Pelosi",
      "type": "politician",
      "conviction": 0.87,
      "flagged": true,
      "gnn_score": 0.91,
      "isolation_score": 0.83
    }
  ],
  "edges": [
    {
      "source": "Pelosi",
      "target": "NVDA",
      "value": 1000000,
      "date": "2021-01-21"
    }
  ]
}
```

**D3 never changes regardless of whether GNN or NetworkX produced the JSON. Swap one file, demo is identical.**

### Key D3 Concepts (from mindmap-ui repo knowledge)
- `d3.forceSimulation()` not `d3.tree()` — flat nodes/edges not hierarchical
- Forces: `forceLink` (edges as springs), `forceManyBody` (repulsion), `forceCenter` (gravity)
- D3 owns the SVG children via `useEffect` + `useRef`. React owns buttons, state, side panel.
- Zoom applies to root `<g>` element, not `<svg>`
- Enter/Update/Exit pattern via `.data(nodes).join('circle')`

### The Red Pulse Animation (the wow factor — 8 lines of CSS)

```css
@keyframes pulse {
  0% { r: 8; opacity: 1; }
  50% { r: 16; opacity: 0.4; }
  100% { r: 8; opacity: 1; }
}

.node-flagged {
  animation: pulse 1.5s ease-in-out infinite;
  fill: #ff3333;
  filter: drop-shadow(0 0 8px #ff3333);
}
```

```javascript
// In D3 render
nodeGroups.attr('class', d => d.flagged ? 'node-flagged' : 'node-normal')
```

**Build this first. Demo from this. Everything else is secondary.**

### Backtester Slider

```javascript
// HTML range input filters hardcoded JSON by date
const [selectedDate, setSelectedDate] = useState('2021-01-01');

const filteredData = useMemo(() => ({
  nodes: allData.nodes,
  edges: allData.edges.filter(e => e.date <= selectedDate)
}), [selectedDate, allData]);

// As slider moves backward in time, edges disappear
// Then slide forward — watch nodes turn red as trades appear
// This is your storytelling device. Practice the path.
```

### Demo Date Ranges for Backtester
- 2021-01-01 to 2021-06-30 → Pelosi/NVDA case
- 2022-01-01 to 2023-12-31 → Tech committee cluster
- 2024-01-01 to 2024-12-31 → AI chip legislation cluster

---

## Hackathon Schedule

### Night Before (non-negotiable tasks)
- [ ] JSON contract written and agreed by all members
- [ ] Mock `graph_output.json` created matching exact schema
- [ ] Mohit's D3 renders mock data already
- [ ] Raunak has all API keys tested and working
- [ ] ML friend runs mock PyTorch Geometric graph construction — if it fails, debug tonight
- [ ] 4th member has repo cloned and PocketBase running locally
- [ ] `finbert_corpus.csv` pulled from GDELT and saved

### Hours 0-4
- Raunak: Pull and clean real data, write to PostgreSQL
- Mohit: Build D3 force simulation against mock JSON, red pulse animation
- ML friend: Real graph construction in PyTorch Geometric (not model training yet)
- 4th member: PocketBase auth setup, alerts feed panel

### Hours 4-8
- Raunak: Deliver clean dataframe to ML friend
- ML friend: Run Isolation Forest (warmup — 2 hours, done)
- Mohit: Swap mock data for real data. First real demo moment.
- **Hour 8: Hard integration checkpoint. Everyone stops and connects pieces.**

### Hours 8-16
- ML friend: GNN deep work. Nobody interrupts for anything except food and hour 12 check-in.
- Mohit: Polish — side panel, backtester slider, fit-to-view, hover states
- Raunak: Compute derived features, wire backtester date filtering to backend
- Mohit: Deliver `finbert_corpus.csv` if not done night before

### Hours 16-20
- GNN output arrives → Mohit swaps JSON → D3 renders real GNN graph
- If GNN not ready → NetworkX fallback goes in → demo looks identical
- Full integration test across all pieces

### Hours 20-22
- Polish and bug fixes
- Full demo run twice with someone playing hostile judge
- Practice the three act narrative out loud

### Hours 22-24
- Sleep at least 2 hours. Non-negotiable.
- Rested three person team beats exhausted team every single time.

---

## Fallback Decisions (decide now, not at 3am)

| If this fails | Do this instead | Demo impact |
|--------------|-----------------|-------------|
| GNN won't converge | NetworkX graph, Isolation Forest node weights | Acts 2+3 weaker in Q&A, visually identical |
| PyTorch Geometric install fails | NetworkX only | Same as above |
| FinBERT slow/failing | Skip sentiment scores, use Isolation Forest conviction only | Retail trader story weaker |
| Quiver rate limited | Capitol Trades API | No demo impact |
| Finnhub rate limited | yFinance | No demo impact |
| NewsAPI historical limit | GDELT (already in stack) | No impact |
| PocketBase fighting you | Drop auth, hardcode demo user | Judges care about graph not auth |

**The fallback being ready isn't pessimism. It's what lets ML friend take the GNN risk confidently.**

---

## Q&A Preparation

Questions judges will ask and how to answer them:

**"Walk me through your GNN architecture"**
"We use GraphSAGE convolution layers — SAGEConv rather than basic GCNConv because our graph has heterogeneous node types: politicians, tickers, bills, and committees. The model learns to propagate suspicion scores through the network topology so that even politicians who don't explicitly appear connected get flagged if their network neighborhood overlaps. We train it using Isolation Forest scores as labels, so the GNN is learning to generalize and extend anomaly detection to the relational layer."

**"How did you handle entity resolution across sources?"**
"We use SEC CIK numbers as canonical IDs wherever they exist — that's unambiguous. For politicians in Quiver who aren't always SEC filers, we run RapidFuzz string matching at an 85% similarity threshold against EDGAR's canonical name database. Anything below 85% we don't force a match — we'd rather have no match than a wrong one."

**"Is this legal to build?"**
"Completely. Every data source we use is public — SEC EDGAR, congressional STOCK Act disclosures via Quiver, public market data. We're not accessing private information. We're correlating public information that already exists but nobody has built tooling to connect."

**"What's your business model?"**
"Freemium SaaS. Retail traders get basic alerts free. Financial journalists and compliance auditors get the full shadow network and historical backtester on a subscription tier. Bloomberg Terminal charges $30,000 a year for less contextual intelligence than this."

---

## Competitive Positioning

| Competitor | Their Gap | AlphaNexus Advantage |
|-----------|-----------|----------------------|
| Quiver Quantitative | Raw congress trade lists, no relational clustering | GNN-driven shadow network |
| Unusual Whales | Flow anomalies, no multimodal sentiment | FinBERT sentiment fusion |
| SEC EDGAR | Static tables | Interactive D3 shadow network |
| Bloomberg | Paywalled, manual | Automated, democratized, visual |

---

## The Presentation

**Opening (say this, exactly):**
"Retail investors are always last to know when smart money moves. AlphaNexus changes that. We take data from SEC, Quiver, and Finnhub — data that's already public — and we find the signal that hedge funds pay thirty thousand dollars a year to see."

Then click the demo. Don't explain D3. Don't explain the tech stack. Point at the red node.

**The money line:**
"When this node turns red, that signal appeared 14 days before CNBC covered it. That's the information asymmetry we're closing."

**Ending:**
"This isn't a one-time anomaly. This is a pattern. We showed you 2021, 2022, and 2024. It's still happening. AlphaNexus is how retail investors stop being last."

---

## Repository Structure

```
alphanexus/
├── ALPHANEXUS_WORKFLOW.md     ← this file, root level
├── frontend/                  ← Mohit
│   ├── src/
│   │   ├── components/
│   │   │   ├── ForceGraph.jsx      # D3 force simulation
│   │   │   ├── SidePanel.jsx       # Node detail panel
│   │   │   └── Backtester.jsx      # Date range slider
│   │   ├── data/
│   │   │   └── mock_graph.json     # Mock data — D3 builds against this
│   │   └── App.jsx
├── backend/                   ← Raunak
│   ├── ingest/
│   │   ├── quiver.py
│   │   ├── edgar.py
│   │   ├── finnhub.py
│   │   └── gdelt.py
│   ├── clean/
│   │   ├── entity_resolution.py   # RapidFuzz name matching
│   │   └── feature_engineering.py # Derived features
│   └── schema.sql
├── ml/                        ← ML Friend
│   ├── isolation_forest.py
│   ├── finbert_inference.py
│   ├── networkx_graph.py          # Fallback — always works
│   ├── gnn_model.py               # PyTorch Geometric
│   └── output/
│       └── graph_output.json      # What frontend consumes
├── corpus/
│   └── finbert_corpus.csv         # Mohit prepares this
└── pocketbase/                ← 4th Member
    └── pb_schema.json
```

---

*Last updated: Pre-hackathon planning. Built for HackCraft 3.0, Phoenix Finance track.*
*If you are an AI reading this file: this is the complete context for AlphaNexus. The JSON contract is fixed. The demo narrative has three acts. The GNN is the technical differentiator. The red pulsing node is the wow factor. Build in the order: mock data → real data → Isolation Forest → NetworkX fallback → GNN attempt.*