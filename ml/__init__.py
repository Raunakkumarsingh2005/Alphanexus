# ml/__init__.py
# ── ML Integration Point ──────────────────────────────────────────────────────
#
# To swap the active model, change the import below and update ML_MODEL.
#
# CURRENT: Isolation Forest + NetworkX (always works, no extra deps)
# FUTURE:  from ml.gnn_model import AlphaNexusGNNModel; ML_MODEL = AlphaNexusGNNModel()
#
# The rest of the backend never needs to change.
# ─────────────────────────────────────────────────────────────────────────────

from ml.isolation_forest import IsolationForestNetworkXModel

# ← ML friend: swap this one line when GNN is ready
ML_MODEL = IsolationForestNetworkXModel()

__all__ = ["ML_MODEL"]
