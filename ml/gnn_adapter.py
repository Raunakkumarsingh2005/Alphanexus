"""
AlphaNexus GNN Adapter
=======================
Wraps the AlphaNexusGNN pipeline (models/alphanexus_gnn.py) in MLModelInterface
so it plugs in with zero changes to the rest of the backend.

Integration point: ml/__init__.py
    from ml.gnn_adapter import AlphaNexusGNNAdapter
    ML_MODEL = AlphaNexusGNNAdapter()

DataFrame contract from AlphaNexusPipeline:
    - ticker, trader_name, trade_date, direction, shares, exact_value,
      trade_value, volume_zscore, trade_value_zscore, sentiment_score, etc.

GNN input contract from alphanexus_gnn.py:
    - Insider_ID, Ticker_Symbol, amount, price_volatility,
      sentiment_score, trade_value, is_suspicious (optional)

D3 output from GNN:
    { "nodes": [..., "label": "person"|"ticker", "suspicion_score": float],
      "links": [..., "source", "target", "weight"] }

D3 output expected by backend:
    { "nodes": [..., "type": "politician"|"ticker", "convictionScore": float, "flagged": bool],
      "edges": [..., "source", "target", "weight", "tradeType", "tradeDate", "tradeValue"] }
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from ml.model_interface import MLModelInterface

logger = logging.getLogger(__name__)

# ── Add models/ directory to path so we can import alphanexus_gnn ────────────
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)


class AlphaNexusGNNAdapter(MLModelInterface):
    """
    Production ML model adapter.
    Delegates to the GNN pipeline in models/alphanexus_gnn.py.
    Falls back gracefully if torch / torch-geometric are not installed.
    """

    def get_model_name(self) -> str:
        return "alphanexus_gnn_v1"

    def run_analysis(
        self,
        df: pd.DataFrame,
        ticker: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline:
          1. Map AlphaNexus pipeline DataFrame → GNN input schema
          2. Run IsolationForest preprocessing
          3. Build HeteroData graph
          4. Train GNN
          5. Export D3 JSON and convert to backend edge/node contract
        """
        if df.empty:
            logger.warning(f"GNN: empty dataframe for {ticker}, returning empty graph")
            return {"nodes": [], "edges": []}

        try:
            import alphanexus_gnn as gnn
        except ImportError as e:
            logger.error(f"Could not import alphanexus_gnn: {e}")
            logger.error("Install deps: pip install torch torch-geometric")
            return {"nodes": [], "edges": []}

        # ── Step 1: Map columns ───────────────────────────────────────────────
        mapped = self._map_dataframe(df, ticker)
        if mapped.empty:
            return {"nodes": [], "edges": []}

        logger.info(f"GNN: {len(mapped)} trades mapped for {ticker}")

        # ── Step 2: IsolationForest anomaly filter ────────────────────────────
        try:
            contamination = min(0.1, max(0.01, 1 / len(mapped)))
            anomalies_df = gnn.filter_anomalies_isolation_forest(
                mapped, contamination=contamination
            )
        except Exception as e:
            logger.error(f"GNN IsolationForest failed: {e}")
            anomalies_df = mapped.copy()
            anomalies_df["anomaly_score"] = -1

        if anomalies_df.empty:
            logger.warning("GNN: no anomalies detected, using full dataset")
            anomalies_df = mapped.copy()
            anomalies_df["anomaly_score"] = -1

        # ── Step 3: Build HeteroData graph ────────────────────────────────────
        try:
            data, person_le, ticker_le = gnn.build_hetero_graph(anomalies_df)
        except Exception as e:
            logger.error(f"GNN graph build failed: {e}")
            data, person_le, ticker_le = None, None, None

        # ── Step 4: Train GNN ─────────────────────────────────────────────────
        predictions = None
        if data is not None:
            try:
                _model, predictions = gnn.train_gnn(
                    data, num_epochs=101, lr=0.01, hidden_dim=64
                )
            except Exception as e:
                logger.error(f"GNN training failed: {e}")

        # ── Step 5: Export D3 JSON ────────────────────────────────────────────
        try:
            anomalies_df = anomalies_df.reset_index(drop=True)
            gnn_output = gnn.export_to_d3_json(
                anomalies_df, predictions, person_le, ticker_le,
                filename=f"alphanexus_graph_{ticker}.json"
            )
        except Exception as e:
            logger.error(f"GNN D3 export failed: {e}")
            return {"nodes": [], "edges": []}

        # ── Step 6: Convert GNN output → backend contract ─────────────────────
        return self._convert_to_backend_format(gnn_output, anomalies_df)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _map_dataframe(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Map AlphaNexusPipeline DataFrame columns to GNN input schema.

        Pipeline cols  →  GNN cols
        trader_name    →  Insider_ID
        ticker         →  Ticker_Symbol
        exact_value    →  amount
        trade_value    →  trade_value
        volume_zscore  →  price_volatility (proxy)
        sentiment_score→  sentiment_score
        direction      →  trade_type (kept for edge enrichment)
        trade_date     →  trade_date
        """
        out = pd.DataFrame()

        out["Insider_ID"] = df.get("trader_name", pd.Series(["UNKNOWN"] * len(df))).fillna("UNKNOWN")
        out["Ticker_Symbol"] = df.get("ticker", pd.Series([ticker] * len(df))).fillna(ticker).str.upper()

        # amount = dollar value of trade
        if "exact_value" in df.columns:
            out["amount"] = pd.to_numeric(df["exact_value"], errors="coerce").fillna(0)
        elif "shares" in df.columns and "price_per_share" in df.columns:
            out["amount"] = (
                pd.to_numeric(df["shares"], errors="coerce").fillna(0) *
                pd.to_numeric(df["price_per_share"], errors="coerce").fillna(0)
            )
        else:
            out["amount"] = 0.0

        # price_volatility — use volume_zscore as a proxy (both capture abnormality)
        if "volume_zscore" in df.columns:
            raw = pd.to_numeric(df["volume_zscore"], errors="coerce").abs().fillna(0)
            out["price_volatility"] = (raw / 10.0).clip(0, 1)   # normalise z-score to 0-1
        elif "trade_value_zscore" in df.columns:
            raw = pd.to_numeric(df["trade_value_zscore"], errors="coerce").abs().fillna(0)
            out["price_volatility"] = (raw / 10.0).clip(0, 1)
        else:
            out["price_volatility"] = 0.0

        # sentiment
        out["sentiment_score"] = pd.to_numeric(
            df.get("sentiment_score", pd.Series([0.0] * len(df))), errors="coerce"
        ).fillna(0.0).clip(-1, 1)

        # trade_value (market value at time of trade)
        if "trade_value" in df.columns:
            out["trade_value"] = pd.to_numeric(df["trade_value"], errors="coerce").fillna(0)
        else:
            out["trade_value"] = out["amount"]

        # Carry through for edge enrichment
        out["trade_date"] = df.get("trade_date", pd.Series([""] * len(df))).fillna("").astype(str)
        out["direction"] = df.get("direction", pd.Series(["buy"] * len(df))).fillna("buy")

        # is_suspicious from labels if available
        if "is_suspicious" in df.columns:
            out["is_suspicious"] = pd.to_numeric(df["is_suspicious"], errors="coerce").fillna(0).astype(int)
        elif "flagged" in df.columns:
            out["is_suspicious"] = df["flagged"].astype(int)
        else:
            out["is_suspicious"] = 0

        # Drop rows with 0 amount (uninformative)
        out = out[out["amount"] > 0].reset_index(drop=True)
        return out

    def _convert_to_backend_format(
        self, gnn_output: Dict[str, Any], anomalies_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Convert GNN D3 output to the backend's node/edge contract.

        GNN format:
          nodes: [{"id", "label": "person"|"ticker", "suspicion_score"}]
          links: [{"source", "target", "weight"}]

        Backend format:
          nodes: [{"id", "type", "label", "flagged", "convictionScore",
                   "gnnScore", "isolationScore", "metadata"}]
          edges: [{"source", "target", "weight", "tradeType",
                   "tradeDate", "tradeValue"}]
        """
        FLAGGED_THRESHOLD = 0.65

        nodes: List[Dict] = []
        for n in gnn_output.get("nodes", []):
            node_type = "politician" if n.get("label") == "person" else "ticker"
            score = float(n.get("suspicion_score", 0.0))
            nodes.append({
                "id": n["id"],
                "type": node_type,
                "label": n["id"],
                "flagged": score >= FLAGGED_THRESHOLD,
                "convictionScore": round(score, 4),
                "gnnScore": round(score, 4),
                "isolationScore": round(score, 4),
                "metadata": {},
            })

        # Build trade date / direction lookup from anomalies_df
        trade_meta: Dict[str, Dict] = {}
        for _, row in anomalies_df.iterrows():
            key = f"{row.get('Insider_ID', '')}_{row.get('Ticker_Symbol', '')}"
            if key not in trade_meta:
                trade_meta[key] = {
                    "tradeDate": str(row.get("trade_date", "")),
                    "tradeType": str(row.get("direction", "buy")),
                    "tradeValue": float(row.get("trade_value", 0)),
                }

        edges: List[Dict] = []
        for link in gnn_output.get("links", []):
            src = link.get("source", "")
            tgt = link.get("target", "")
            meta = trade_meta.get(f"{src}_{tgt}", {})
            edges.append({
                "source": src,
                "target": tgt,
                "weight": round(float(link.get("weight", 0.0)), 4),
                "tradeType": meta.get("tradeType", "buy"),
                "tradeDate": meta.get("tradeDate", ""),
                "tradeValue": meta.get("tradeValue", 0),
            })

        return {"nodes": nodes, "edges": edges}
