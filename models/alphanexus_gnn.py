"""
======================================================================
   AlphaNexus GNN Pipeline
   Insider Trading & Information Asymmetry Detection

   Data Sources: SEC EDGAR (Form 4), Finnhub, News API
   
   Sections:
     1. Data Loading & Column Mapping (alphanexus_dataset.csv)
     2. IsolationForest Anomaly Preprocessing
     3. Heterogeneous Graph Construction (HeteroData)
     4. GNN Model Definition & Training Loop
     5. D3.js JSON Export
======================================================================
"""

# ============================================================================
# SECTION 1: Imports & Data Loading
# ============================================================================
# pip install torch torch-geometric scikit-learn pandas numpy networkx

import pandas as pd
import numpy as np
import json
import os
from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing torch_geometric; provide helpful error if missing
try:
    from torch_geometric.data import HeteroData
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("WARNING: torch_geometric not installed. HeteroData construction will be skipped.")
    print("Install with: pip install torch-geometric")


def load_and_map_alphanexus_data(csv_path):
    """
    Load the real AlphaNexus dataset (from SEC EDGAR / Finnhub / News API)
    and map its columns to the pipeline's expected schema.
    
    Real CSV columns:
        ticker, company, data_type, source, trade_date, file_date,
        insider_name, trade_type, shares_traded, price_per_share,
        price_on_date, price_7d_later, price_change_7d_pct,
        volume_on_date, volume_30d_avg, volume_spike_ratio,
        news_headline, news_source, news_sentiment_score,
        finnhub_headline, finnhub_sentiment,
        alpha_conviction_label, row_type
    
    Pipeline mapping:
        Insider_ID       <- insider_name
        Ticker_Symbol    <- ticker
        amount           <- shares_traded * price_per_share (trade dollar value)
        price_volatility <- abs(price_change_7d_pct) / 100  (normalized)
        sentiment_score  <- news_sentiment_score (or finnhub_sentiment fallback)
        trade_value      <- shares_traded * price_on_date   (market value at trade)
        is_suspicious    <- alpha_conviction_label mapped to binary
                            (HIGH/MEDIUM -> 1, LOW -> 0)
    
    Returns:
        pd.DataFrame with standardized columns
    """
    print(f"  Loading AlphaNexus dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"  Raw dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Tickers:  {df['ticker'].nunique()} unique")
    print(f"  Insiders: {df['insider_name'].nunique()} unique")
    print(f"  Sources:  {df['source'].unique().tolist()}")
    print(f"  Trade types: {df['trade_type'].dropna().unique().tolist()}")
    
    # --- Filter to actionable trades (Sales and Purchases) ---
    # Keep Sale, Purchase, and Option-exercise (potential insider signal)
    actionable_types = ['Sale', 'Purchase', 'Option-exercise']
    df_trades = df[df['trade_type'].isin(actionable_types)].copy()
    print(f"\n  After filtering to actionable trades ({actionable_types}):")
    print(f"    {df_trades.shape[0]:,} trades remaining")
    
    # --- Column Mapping ---
    # Insider_ID: use insider_name directly
    df_trades['Insider_ID'] = df_trades['insider_name'].fillna('UNKNOWN')
    
    # Ticker_Symbol: use ticker
    df_trades['Ticker_Symbol'] = df_trades['ticker']
    
    # amount: shares_traded * price_per_share (trade dollar value)
    df_trades['shares_traded'] = pd.to_numeric(df_trades['shares_traded'], errors='coerce').fillna(0)
    df_trades['price_per_share'] = pd.to_numeric(df_trades['price_per_share'], errors='coerce').fillna(0)
    df_trades['price_on_date'] = pd.to_numeric(df_trades['price_on_date'], errors='coerce').fillna(0)
    df_trades['amount'] = df_trades['shares_traded'] * df_trades['price_per_share']
    
    # For rows where price_per_share is 0 (e.g., option-exercise), use price_on_date
    mask_zero_price = df_trades['price_per_share'] == 0
    df_trades.loc[mask_zero_price, 'amount'] = (
        df_trades.loc[mask_zero_price, 'shares_traded'] * 
        df_trades.loc[mask_zero_price, 'price_on_date']
    )
    
    # price_volatility: derived from abs(price_change_7d_pct) normalized to 0-1
    df_trades['price_change_7d_pct'] = pd.to_numeric(df_trades['price_change_7d_pct'], errors='coerce').fillna(0)
    raw_vol = df_trades['price_change_7d_pct'].abs()
    # Clip at 30% and normalize to 0-1 (30% change = max volatility)
    df_trades['price_volatility'] = (raw_vol / 30.0).clip(0, 1)
    
    # sentiment_score: use news_sentiment_score, fallback to finnhub_sentiment
    df_trades['news_sentiment_score'] = pd.to_numeric(df_trades['news_sentiment_score'], errors='coerce')
    df_trades['finnhub_sentiment'] = pd.to_numeric(df_trades['finnhub_sentiment'], errors='coerce')
    df_trades['sentiment_score'] = df_trades['news_sentiment_score'].fillna(
        df_trades['finnhub_sentiment']
    ).fillna(0.0)  # default neutral if both missing
    # Clip to [-1, 1]
    df_trades['sentiment_score'] = df_trades['sentiment_score'].clip(-1, 1)
    
    # trade_value: shares_traded * price_on_date (market value)
    df_trades['trade_value'] = df_trades['shares_traded'] * df_trades['price_on_date']
    
    # volume_spike_ratio: already in dataset, useful as additional signal
    df_trades['volume_spike_ratio'] = pd.to_numeric(
        df_trades['volume_spike_ratio'], errors='coerce'
    ).fillna(1.0)
    
    # is_suspicious: derived from alpha_conviction_label
    # HIGH and MEDIUM conviction -> suspicious (1), LOW -> normal (0)
    conviction_map = {'HIGH': 1, 'MEDIUM': 1, 'LOW': 0}
    df_trades['is_suspicious'] = df_trades['alpha_conviction_label'].map(conviction_map).fillna(0).astype(int)
    
    # trade_date: keep as-is
    df_trades['trade_date'] = df_trades['trade_date'].fillna('')
    
    # Remove trades with 0 amount (uninformative)
    df_trades = df_trades[df_trades['amount'] > 0].copy()
    
    # Reset index
    df_trades = df_trades.reset_index(drop=True)
    
    # --- Print mapping summary ---
    print(f"\n  Column mapping applied:")
    print(f"    Insider_ID       <- insider_name  ({df_trades['Insider_ID'].nunique()} unique)")
    print(f"    Ticker_Symbol    <- ticker         ({df_trades['Ticker_Symbol'].nunique()} unique)")
    print(f"    amount           <- shares * price (${df_trades['amount'].mean():,.2f} avg)")
    print(f"    price_volatility <- |7d_pct|/30    ({df_trades['price_volatility'].mean():.4f} avg)")
    print(f"    sentiment_score  <- news/finnhub   ({df_trades['sentiment_score'].mean():.4f} avg)")
    print(f"    trade_value      <- shares * mkt   (${df_trades['trade_value'].mean():,.2f} avg)")
    print(f"    is_suspicious    <- conviction      ({df_trades['is_suspicious'].sum()} flagged)")
    print(f"\n  Conviction distribution:")
    print(f"    {df_trades['alpha_conviction_label'].value_counts().to_dict()}")
    
    return df_trades


def generate_synthetic_insider_data(n_samples=2000, random_state=42):
    """
    Fallback: generate synthetic insider trading data if no CSV is found.
    """
    np.random.seed(random_state)
    
    insiders = [
        'SEN_Pelosi', 'SEN_Burr', 'SEN_Loeffler', 'SEN_Tuberville',
        'SEN_Feinstein', 'SEN_Inhofe', 'SEN_Johnson', 'SEN_Perdue',
        'CEO_Jensen_Huang', 'CEO_Elon_Musk', 'CEO_Tim_Cook',
        'CEO_Satya_Nadella', 'CEO_Andy_Jassy', 'CEO_Sundar_Pichai',
        'CEO_Lisa_Su', 'CEO_Mark_Zuckerberg', 'CEO_Jamie_Dimon',
        'CEO_David_Solomon', 'CEO_Brian_Moynihan', 'CEO_Mary_Barra'
    ]
    tickers = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN',
               'META', 'AMD', 'JPM', 'GS', 'BAC', 'GM', 'PFE', 'MRNA', 'LMT']
    
    n_normal = int(n_samples * 0.90)
    n_suspicious = n_samples - n_normal
    
    df = pd.DataFrame({
        'Insider_ID': np.concatenate([
            np.random.choice(insiders, n_normal),
            np.random.choice(insiders[:8], n_suspicious)
        ]),
        'Ticker_Symbol': np.concatenate([
            np.random.choice(tickers, n_normal),
            np.random.choice(['NVDA', 'TSLA', 'MRNA', 'PFE', 'LMT'], n_suspicious)
        ]),
        'amount': np.concatenate([
            np.random.lognormal(9, 1.2, n_normal).clip(1000, 500_000),
            np.random.lognormal(13, 0.8, n_suspicious).clip(500_000, 5_000_000)
        ]),
        'price_volatility': np.concatenate([
            np.random.beta(2, 8, n_normal),
            np.random.beta(8, 2, n_suspicious)
        ]),
        'sentiment_score': np.concatenate([
            np.random.normal(0, 0.3, n_normal).clip(-1, 1),
            np.random.choice([-1, -0.9, 0.9, 1.0], n_suspicious)
        ]),
        'is_suspicious': np.concatenate([np.zeros(n_normal), np.ones(n_suspicious)]).astype(int),
    })
    df['trade_value'] = df['amount'] * (1 + df['price_volatility'] * 3) + np.random.normal(0, 1000, len(df))
    df['trade_value'] = df['trade_value'].clip(0)
    df['trade_date'] = '2024-01-01'
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ============================================================================
# SECTION 2: IsolationForest Anomaly Preprocessing
# ============================================================================

def filter_anomalies_isolation_forest(df, contamination=0.05, random_state=42):
    """
    Apply IsolationForest to filter the top 5% anomalies from trade data.
    Uses 'amount' and 'price_volatility' as features for anomaly detection.
    """
    print("\n" + "="*70)
    print("SECTION 2: IsolationForest Anomaly Preprocessing")
    print("="*70)
    
    features = ['amount', 'price_volatility']
    
    # Standardize features for better IsolationForest performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
        max_samples='auto'
    )
    df = df.copy()
    df['anomaly_score'] = iso_forest.fit_predict(X_scaled)
    
    # Anomalies are labeled -1 by IsolationForest
    anomalies_df = df[df['anomaly_score'] == -1].copy()
    normal_df = df[df['anomaly_score'] == 1]
    
    print(f"  Total trades:           {len(df):,}")
    print(f"  Anomalies detected:     {len(anomalies_df):,} ({len(anomalies_df)/len(df)*100:.1f}%)")
    print(f"  Normal trades:          {len(normal_df):,}")
    print(f"\n  Anomaly stats:")
    print(f"    Avg amount:           ${anomalies_df['amount'].mean():,.2f}")
    print(f"    Avg price_volatility: {anomalies_df['price_volatility'].mean():.4f}")
    print(f"    Avg sentiment_score:  {anomalies_df['sentiment_score'].mean():.4f}")
    
    # Check overlap with ground truth
    if 'is_suspicious' in anomalies_df.columns:
        true_positives = anomalies_df['is_suspicious'].sum()
        total_suspicious = df['is_suspicious'].sum()
        print(f"\n  Ground truth overlap (vs alpha_conviction):")
        print(f"    Flagged in anomalies: {int(true_positives)}/{int(total_suspicious)} total HIGH/MEDIUM")
        if len(anomalies_df) > 0:
            print(f"    Precision: {true_positives/len(anomalies_df)*100:.1f}%")
        if total_suspicious > 0:
            print(f"    Recall:    {true_positives/total_suspicious*100:.1f}%")
    
    print(f"\n  Sample anomalous trades:")
    display_cols = ['Insider_ID', 'Ticker_Symbol', 'amount', 'price_volatility',
                    'sentiment_score', 'trade_value']
    display_cols = [c for c in display_cols if c in anomalies_df.columns]
    print(anomalies_df[display_cols].head(10).to_string(index=False))
    
    return anomalies_df


# ============================================================================
# SECTION 3: Heterogeneous Graph Construction (HeteroData)
# ============================================================================

def build_hetero_graph(anomalies_df):
    """
    Build a PyTorch Geometric HeteroData object from anomalous trades.
    
    Node types:
        - 'person': Insider_ID (insiders from SEC Form 4)
        - 'ticker': Ticker_Symbol (e.g., NVDA, TSLA, AAPL)
    
    Edge type:
        - ('person', 'trades', 'ticker')
    
    Edge attributes:
        - sentiment_score (range -1 to 1)
        - trade_value (normalized float)
    """
    print("\n" + "="*70)
    print("SECTION 3: Heterogeneous Graph Construction")
    print("="*70)
    
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("  SKIPPED: torch_geometric not available.")
        return None, None, None
    
    # Encode node IDs to integer indices
    person_le = LabelEncoder()
    ticker_le = LabelEncoder()
    
    anomalies_df = anomalies_df.copy()
    anomalies_df['person_idx'] = person_le.fit_transform(anomalies_df['Insider_ID'])
    anomalies_df['ticker_idx'] = ticker_le.fit_transform(anomalies_df['Ticker_Symbol'])
    
    num_persons = len(person_le.classes_)
    num_tickers = len(ticker_le.classes_)
    
    # Create HeteroData object
    data = HeteroData()
    
    # --- Node features ---
    # Person nodes: identity features (one-hot embedding)
    data['person'].x = torch.eye(num_persons, dtype=torch.float)
    data['person'].node_id = torch.arange(num_persons)
    data['person'].label = list(person_le.classes_)
    
    # Ticker nodes: identity features
    data['ticker'].x = torch.eye(num_tickers, dtype=torch.float)
    data['ticker'].node_id = torch.arange(num_tickers)
    data['ticker'].label = list(ticker_le.classes_)
    
    # --- Edge index (COO format) ---
    src = torch.tensor(anomalies_df['person_idx'].values, dtype=torch.long)
    dst = torch.tensor(anomalies_df['ticker_idx'].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    
    data['person', 'trades', 'ticker'].edge_index = edge_index
    
    # --- Edge attributes: [sentiment_score, trade_value] ---
    # Normalize trade_value to 0-1 range
    trade_values = anomalies_df['trade_value'].values.astype(float).copy()
    tv_min, tv_max = trade_values.min(), trade_values.max()
    if tv_max > tv_min:
        trade_values_norm = (trade_values - tv_min) / (tv_max - tv_min)
    else:
        trade_values_norm = np.zeros_like(trade_values)
    
    edge_attr = torch.tensor(
        np.column_stack([
            anomalies_df['sentiment_score'].values.astype(float),
            trade_values_norm
        ]),
        dtype=torch.float
    )
    data['person', 'trades', 'ticker'].edge_attr = edge_attr
    
    # --- Edge labels (for training) ---
    if 'is_suspicious' in anomalies_df.columns:
        data['person', 'trades', 'ticker'].y = torch.tensor(
            anomalies_df['is_suspicious'].values.astype(float), dtype=torch.float
        )
    
    # Print summary
    print(f"\n  AlphaNexus Heterogeneous Graph constructed:")
    print(f"    Person nodes:  {num_persons} (Insiders from SEC Form 4)")
    print(f"    Ticker nodes:  {num_tickers} (Stock symbols)")
    print(f"    Trade edges:   {edge_index.shape[1]}")
    print(f"    Edge features: ['sentiment_score', 'trade_value'] (dim=2)")
    print(f"\n  Sample person nodes: {list(person_le.classes_[:10])}{'...' if num_persons > 10 else ''}")
    print(f"  Ticker nodes: {list(ticker_le.classes_)}")
    print(f"\n  HeteroData object:")
    print(f"    {data}")
    
    return data, person_le, ticker_le


# ============================================================================
# SECTION 4: GNN Model Definition & Training Loop
# ============================================================================

class AlphaNexusGNN(nn.Module):
    """
    Graph Neural Network for Insider Trading Detection.
    
    Architecture:
        - Input: concatenation of source node features + edge attributes
        - Hidden layer with ReLU activation + BatchNorm + Dropout
        - Output: per-edge suspicion score (single logit)
    """
    
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(AlphaNexusGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze(-1)


def train_gnn(data, num_epochs=201, lr=0.01, hidden_dim=64):
    """
    Train the AlphaNexusGNN on the heterogeneous graph data.
    
    Constructs per-edge feature vectors by concatenating:
        - Source (person) node features
        - Edge attributes (sentiment_score, trade_value)
    """
    print("\n" + "="*70)
    print("SECTION 4: GNN Model Training")
    print("="*70)
    
    if data is None:
        print("  SKIPPED: No graph data available.")
        return None, None
    
    # Extract edge data
    edge_index = data['person', 'trades', 'ticker'].edge_index
    edge_attr = data['person', 'trades', 'ticker'].edge_attr
    person_features = data['person'].x
    
    # Build per-edge input: [source_person_features | edge_attributes]
    src_indices = edge_index[0]
    src_features = person_features[src_indices]
    
    # Concatenate source features with edge attributes
    x = torch.cat([src_features, edge_attr], dim=1)
    
    input_dim = x.shape[1]
    print(f"\n  Input dimension per edge: {input_dim}")
    print(f"    (person features: {person_features.shape[1]} + edge attrs: {edge_attr.shape[1]})")
    print(f"  Number of training edges: {x.shape[0]}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {lr}")
    
    # Get targets
    if hasattr(data['person', 'trades', 'ticker'], 'y'):
        target = data['person', 'trades', 'ticker'].y
    else:
        target = torch.ones(x.shape[0], dtype=torch.float) * 0.5
    
    # Class weight for imbalanced data
    n_pos = target.sum().item()
    n_neg = len(target) - n_pos
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_pos])
        print(f"  Class balance: {int(n_neg)} normal / {int(n_pos)} suspicious (pos_weight={pos_weight.item():.2f})")
    else:
        pos_weight = torch.tensor([1.0])
    
    # Initialize model
    model = AlphaNexusGNN(input_dim, hidden_dim)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # --- Training Loop ---
    print(f"\n  Training AlphaNexus GNN...")
    print(f"  {'Epoch':>8}  {'Loss':>12}  {'Avg Pred':>12}")
    print(f"  {'-'*36}")
    
    model.train()
    for epoch in range(num_epochs):
        output = model(x)
        loss = criterion(output, target)
        
        if epoch % 20 == 0:
            avg_pred = torch.sigmoid(output).mean().item()
            print(f"  {epoch:>8}  {loss.item():>12.6f}  {avg_pred:>12.4f}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # --- Final predictions ---
    model.eval()
    with torch.no_grad():
        final_output = model(x)
        predictions = torch.sigmoid(final_output)
    
    print(f"\n  Training complete!")
    print(f"  Final loss: {loss.item():.6f}")
    print(f"  Prediction stats:")
    print(f"    Min:  {predictions.min().item():.4f}")
    print(f"    Max:  {predictions.max().item():.4f}")
    print(f"    Mean: {predictions.mean().item():.4f}")
    print(f"    Std:  {predictions.std().item():.4f}")
    
    return model, predictions


# ============================================================================
# SECTION 5: D3.js JSON Export
# ============================================================================

def export_to_d3_json(anomalies_df, predictions, person_le, ticker_le, 
                       filename="alphanexus_graph.json"):
    """
    Export the AlphaNexus graph results to a D3.js-compatible JSON file.
    
    Output format:
    {
        "nodes": [
            {"id": "HUANG JEN HSUN", "label": "person", "suspicion_score": 0.92},
            {"id": "NVDA", "label": "ticker", "suspicion_score": 0.45}
        ],
        "links": [
            {"source": "HUANG JEN HSUN", "target": "NVDA", "weight": 0.87}
        ]
    }
    """
    print("\n" + "="*70)
    print("SECTION 5: D3.js JSON Export")
    print("="*70)
    
    pred_values = predictions.numpy() if predictions is not None else np.random.rand(len(anomalies_df))
    
    # --- Compute per-node suspicion scores ---
    person_scores = {}
    for i, row in enumerate(anomalies_df.itertuples()):
        pid = row.Insider_ID
        score = float(pred_values[i])
        if pid not in person_scores:
            person_scores[pid] = []
        person_scores[pid].append(score)
    
    ticker_scores = {}
    for i, row in enumerate(anomalies_df.itertuples()):
        tid = row.Ticker_Symbol
        score = float(pred_values[i])
        if tid not in ticker_scores:
            ticker_scores[tid] = []
        ticker_scores[tid].append(score)
    
    # --- Build nodes list ---
    nodes = []
    
    for person_id in anomalies_df['Insider_ID'].unique():
        scores = person_scores.get(person_id, [0.5])
        suspicion = round(max(scores), 4)
        nodes.append({
            "id": str(person_id),
            "label": "person",
            "suspicion_score": suspicion
        })
    
    for ticker_id in anomalies_df['Ticker_Symbol'].unique():
        scores = ticker_scores.get(ticker_id, [0.5])
        suspicion = round(float(np.mean(scores)), 4)
        nodes.append({
            "id": str(ticker_id),
            "label": "ticker",
            "suspicion_score": suspicion
        })
    
    # --- Build links list ---
    links = []
    tv = anomalies_df['trade_value'].values.astype(float)
    tv_min, tv_max = tv.min(), tv.max()
    
    for i, row in enumerate(anomalies_df.itertuples()):
        weight = (row.trade_value - tv_min) / (tv_max - tv_min) if tv_max > tv_min else 0.5
        links.append({
            "source": str(row.Insider_ID),
            "target": str(row.Ticker_Symbol),
            "weight": round(float(weight), 4)
        })
    
    output = {"nodes": nodes, "links": links}
    
    # Determine output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n  AlphaNexus Graph exported to: {filepath}")
    print(f"  Nodes: {len(nodes)} ({sum(1 for n in nodes if n['label']=='person')} persons, "
          f"{sum(1 for n in nodes if n['label']=='ticker')} tickers)")
    print(f"  Links: {len(links)}")
    
    print(f"\n  Top 10 most suspicious insiders:")
    person_nodes = sorted(
        [n for n in nodes if n['label'] == 'person'],
        key=lambda x: x['suspicion_score'],
        reverse=True
    )
    for n in person_nodes[:10]:
        print(f"    {n['id']:30s}  suspicion: {n['suspicion_score']:.4f}")
    
    print(f"\n  Ticker suspicion scores:")
    ticker_nodes = sorted(
        [n for n in nodes if n['label'] == 'ticker'],
        key=lambda x: x['suspicion_score'],
        reverse=True
    )
    for n in ticker_nodes:
        print(f"    {n['id']:10s}  suspicion: {n['suspicion_score']:.4f}")
    
    print(f"\n  Sample JSON output:")
    sample = {"nodes": nodes[:3], "links": links[:3]}
    print(json.dumps(sample, indent=4))
    
    return output


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("======================================================================")
    print("   AlphaNexus GNN Pipeline")
    print("   Insider Trading & Information Asymmetry Detection")
    print("   Data: SEC EDGAR Form 4 | Finnhub | News API")
    print("=====================================================================")
    
    # -----------------------------------------------------------------------
    # SECTION 1: Load / Generate Data
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("SECTION 1: Data Loading & Column Mapping")
    print("="*70)
    
    # Load from alphanexus_dataset.csv
    real_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alphanexus_dataset.csv')
    
    if os.path.exists(real_data_path):
        df = load_and_map_alphanexus_data(real_data_path)
    else:
        print("  WARNING: alphanexus_dataset.csv not found!")
        print("  Falling back to synthetic data...")
        df = generate_synthetic_insider_data(n_samples=2000)
    
    print(f"\n  Final dataset: {df.shape[0]:,} trades")
    print(f"  Unique insiders: {df['Insider_ID'].nunique()}")
    print(f"  Unique tickers:  {df['Ticker_Symbol'].nunique()}")
    print(f"\n  Sample mapped data:")
    display_cols = ['Insider_ID', 'Ticker_Symbol', 'amount', 'price_volatility',
                    'sentiment_score', 'trade_value']
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].head(10).to_string(index=False))
    
    # -----------------------------------------------------------------------
    # SECTION 2: IsolationForest Filtering
    # -----------------------------------------------------------------------
    anomalies_df = filter_anomalies_isolation_forest(df, contamination=0.05)
    
    if anomalies_df.empty:
        print("\n  ERROR: No anomalies detected. Exiting.")
        exit(1)
    
    # -----------------------------------------------------------------------
    # SECTION 3: Build Heterogeneous Graph
    # -----------------------------------------------------------------------
    data, person_le, ticker_le = build_hetero_graph(anomalies_df)
    
    # -----------------------------------------------------------------------
    # SECTION 4: Train GNN
    # -----------------------------------------------------------------------
    model, predictions = train_gnn(data, num_epochs=201, lr=0.01, hidden_dim=64)
    
    # -----------------------------------------------------------------------
    # SECTION 5: Export to D3.js JSON
    # -----------------------------------------------------------------------
    anomalies_df = anomalies_df.reset_index(drop=True)
    graph_json = export_to_d3_json(anomalies_df, predictions, person_le, ticker_le)
    
    print("\n" + "="*70)
    print("  AlphaNexus Pipeline Complete!")
    print("="*70)
