"""
Fallback ML Model: Isolation Forest + NetworkX
================================================
This is the always-working fallback used until the GNN is ready.
It implements MLModelInterface so it slots into the pipeline identically.

When the ML friend's GNN is ready, this is replaced by swapping one
import line in services/analysis_service.py.
"""

import uuid
import logging
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest

from ml.model_interface import MLModelInterface

logger = logging.getLogger(__name__)


class IsolationForestNetworkXModel(MLModelInterface):
    """
    Act 1 / Fallback model.

    Pipeline:
      1. Isolation Forest → anomaly_score per trade row
      2. NetworkX → graph topology (trader ↔ ticker edges)
      3. Export D3-compatible JSON (same contract as GNN output)
    """

    FEATURE_COLS = [
        "trade_value",
        "volume_zscore",
        "trade_value_zscore",
        "trader_historical_frequency",
    ]

    def get_model_name(self) -> str:
        return "isolation_forest_networkx"

    def run_analysis(
        self,
        df: pd.DataFrame,
        ticker: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        if df.empty:
            logger.warning("Empty dataframe passed to IsolationForestNetworkXModel")
            return {"nodes": [], "edges": []}

        df = df.copy()
        df = self._add_missing_cols(df)

        # ── Step 1: Isolation Forest ──────────────────────────
        df = self._run_isolation_forest(df)

        # ── Step 2: NetworkX Graph ─────────────────────────────
        G = self._build_graph(df)

        # ── Step 3: Export D3 JSON ─────────────────────────────
        return self._export_d3_json(G, df)

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _add_missing_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns exist (fill with 0 if pipeline didn't compute)."""
        defaults = {
            "trade_value": 0.0,
            "volume_zscore": 0.0,
            "trade_value_zscore": 0.0,
            "trader_historical_frequency": 1,
            "avg_sentiment_score": 0.0,
            "days_before_bill_vote": 0,
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val
        return df.fillna(0)

    def _run_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        if not available:
            df["anomaly_score"] = 0
            df["conviction_score"] = 0.0
            df["flagged"] = False
            return df

        features = df[available].astype(float)
        contamination = min(0.1, max(0.01, 1 / len(df)))

        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        df["anomaly_score"] = model.fit_predict(features)      # -1 = anomaly
        raw_scores = model.decision_function(features)          # more negative = more suspicious

        # Normalise to [0, 1] where 1 = most suspicious
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s != min_s:
            df["conviction_score"] = 1 - (raw_scores - min_s) / (max_s - min_s)
        else:
            df["conviction_score"] = 0.5

        df["flagged"] = df["anomaly_score"] == -1
        return df

    def _build_graph(self, df: pd.DataFrame) -> nx.Graph:
        G = nx.Graph()
        for _, row in df.iterrows():
            trader_id = str(row.get("trader_id", str(uuid.uuid4())))
            ticker = str(row.get("ticker", "UNKNOWN"))
            conviction = float(row.get("conviction_score", 0.0))
            flagged = bool(row.get("flagged", False))
            trader_type = str(row.get("trader_type", "politician"))

            # Add trader node
            if not G.has_node(trader_id):
                G.add_node(trader_id,
                           label=str(row.get("trader_name", trader_id)),
                           type=trader_type,
                           convictionScore=round(conviction, 4),
                           isolationScore=round(conviction, 4),
                           gnnScore=round(conviction, 4),    # same as isolation until GNN runs
                           flagged=flagged)

            # Add ticker node
            if not G.has_node(ticker):
                G.add_node(ticker,
                           label=ticker,
                           type="ticker",
                           convictionScore=0.0,
                           isolationScore=0.0,
                           gnnScore=0.0,
                           flagged=False)

            # Add edge (trade)
            trade_date = str(row.get("trade_date", ""))
            trade_value = float(row.get("trade_value", 0))
            direction = str(row.get("direction", "buy"))

            G.add_edge(trader_id, ticker,
                       tradeValue=trade_value,
                       tradeDate=trade_date,
                       tradeType=direction,
                       weight=round(conviction, 4))

        return G

    def _export_d3_json(self, G: nx.Graph, df: pd.DataFrame) -> Dict[str, Any]:
        nodes: List[Dict] = []
        for node_id, attrs in G.nodes(data=True):
            node_type = attrs.get("type", "unknown")
            meta: Dict = {}
            if node_type == "politician":
                # Enrich with party/state/committee if available in df
                row = df[df["trader_id"].astype(str) == node_id]
                if not row.empty:
                    r = row.iloc[0]
                    meta = {
                        "party": r.get("party", None),
                        "state": r.get("state", None),
                        "committee": r.get("committee", None),
                    }
            elif node_type == "ticker":
                meta = {"sector": None}

            nodes.append({
                "id": node_id,
                "type": node_type,
                "label": attrs.get("label", node_id),
                "flagged": attrs.get("flagged", False),
                "convictionScore": attrs.get("convictionScore", 0.0),
                "gnnScore": attrs.get("gnnScore", 0.0),
                "isolationScore": attrs.get("isolationScore", 0.0),
                "metadata": meta,
            })

        edges: List[Dict] = []
        for u, v, attrs in G.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "tradeValue": attrs.get("tradeValue", 0),
                "tradeDate": attrs.get("tradeDate", ""),
                "tradeType": attrs.get("tradeType", "buy"),
                "weight": attrs.get("weight", 0.0),
            })

        return {"nodes": nodes, "edges": edges}
