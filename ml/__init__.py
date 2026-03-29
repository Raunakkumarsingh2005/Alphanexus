# ml/__init__.py
# ── ML Integration Point ──────────────────────────────────────────────────────
#
# Active model: AlphaNexus GNN (alphanexus_gnn.py via gnn_adapter.py)
#
# The GNN pipeline:
#   1. Maps AlphaNexusPipeline DataFrame → GNN input schema
#   2. IsolationForest anomaly preprocessing
#   3. Heterogeneous graph construction (HeteroData)
#   4. Train GNN (AlphaNexusGNN class)
#   5. Export D3-compatible JSON → backend edge/node contract
#
# To revert to the Isolation Forest fallback:
#   from ml.isolation_forest import IsolationForestNetworkXModel
#   ML_MODEL = IsolationForestNetworkXModel()
#
# Requirements: pip install torch torch-geometric
# ─────────────────────────────────────────────────────────────────────────────

import logging

logger = logging.getLogger(__name__)

# Try loading the GNN adapter; fall back to Isolation Forest if deps missing
try:
    from ml.gnn_adapter import AlphaNexusGNNAdapter
    ML_MODEL = AlphaNexusGNNAdapter()
    logger.info("✓ Active ML model: AlphaNexus GNN (alphanexus_gnn_v1)")
except Exception as e:
    logger.warning(f"GNN model unavailable ({e}), falling back to IsolationForest")
    from ml.isolation_forest import IsolationForestNetworkXModel
    ML_MODEL = IsolationForestNetworkXModel()
    logger.info("✓ Active ML model: IsolationForest + NetworkX (fallback)")

__all__ = ["ML_MODEL"]
