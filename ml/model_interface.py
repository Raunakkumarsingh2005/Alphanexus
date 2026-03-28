"""
ML Model Integration Interface
================================
This file defines the contract between the backend analysis service and
the ML pipeline (owned by the ML team member).

HOW TO INTEGRATE YOUR MODEL
-----------------------------
1. Create a new file: ml/your_model.py
2. Subclass MLModelInterface
3. Implement run_analysis() and get_model_name()
4. In services/analysis_service.py, import your class and replace the
   fallback instance:
       from ml.your_model import YourGNNModel
       ML_MODEL = YourGNNModel()

The backend never changes — only this swap is needed.

INPUT CONTRACT
--------------
The dataframe passed to run_analysis has these columns:
  trader_id, trader_name, ticker, trade_date, trade_value, direction,
  price_on_date, volume_zscore, avg_sentiment_score,
  days_before_bill_vote, trader_historical_frequency,
  trade_value_zscore

OUTPUT CONTRACT (graph JSON fed to D3)
---------------------------------------
{
  "nodes": [
    {
      "id": str,                  # unique node ID
      "type": str,                # "politician" | "ticker" | "corporate_insider"
      "label": str,               # display name
      "flagged": bool,
      "convictionScore": float,   # 0.0 – 1.0
      "gnnScore": float,          # 0.0 – 1.0 (set equal to isolationScore if no GNN)
      "isolationScore": float,    # 0.0 – 1.0
      "metadata": { ... }
    }
  ],
  "edges": [
    {
      "source": str,
      "target": str,
      "tradeValue": float,
      "tradeDate": str,           # "YYYY-MM-DD"
      "tradeType": str,           # "buy" | "sell"
      "weight": float
    }
  ]
}
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class MLModelInterface(ABC):
    """
    Abstract base class for all AlphaNexus ML models.

    Implement this to plug your model (GNN, Isolation Forest, etc.)
    into the analysis pipeline without changing any backend code.
    """

    @abstractmethod
    def run_analysis(
        self,
        df: pd.DataFrame,
        ticker: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full ML pipeline on the provided dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Clean, joined dataframe from PostgreSQL (see INPUT CONTRACT above).
        ticker : str
            Primary ticker being analyzed (e.g., "NVDA").
        date_from : str, optional
            Analysis start date "YYYY-MM-DD".
        date_to : str, optional
            Analysis end date "YYYY-MM-DD".

        Returns
        -------
        dict
            Exactly matches the OUTPUT CONTRACT (D3 JSON format) above.
            Must contain keys: "nodes" and "edges".
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self) -> str:
        """Return a short identifier, e.g. 'isolation_forest_networkx' or 'gnn'."""
        raise NotImplementedError

    def get_overall_conviction(self, graph: Dict[str, Any]) -> float:
        """Utility: compute overall conviction from the graph nodes."""
        nodes = graph.get("nodes", [])
        if not nodes:
            return 0.0
        scores = [n.get("convictionScore", 0.0) for n in nodes if n.get("flagged")]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def get_risk_level(self, conviction: float) -> str:
        """Map conviction score to a risk label."""
        if conviction >= 0.85:
            return "critical"
        elif conviction >= 0.70:
            return "high"
        elif conviction >= 0.50:
            return "medium"
        return "low"
