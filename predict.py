"""Submission interface -- this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.

Ensemble: NN + LightGBM weighted average.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from features.pipeline import FeaturePipeline
from model.architecture import ETAModel, ModelConfig

_ROOT = Path(__file__).parent
_MODEL_PATH = _ROOT / "model.pt"
_LGBM_PATH = _ROOT / "lgbm_model.txt"
_STATS_PATH = _ROOT / "data" / "zone_pair_stats" / "zone_pair_stats.pkl"

# Ensemble weight: pred = ALPHA * nn + (1 - ALPHA) * lgbm
# Set to 1.0 to use NN only (fallback if no LGBM model present)
_ENSEMBLE_ALPHA = 0.50

# --- Load NN model ---
_checkpoint = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)
_config = ModelConfig(**_checkpoint["model_config"])
_model = ETAModel(_config)
_model.load_state_dict(_checkpoint["model_state_dict"])
_model.eval()
_log_target = _checkpoint.get("log_target", False)

# --- Load feature pipeline ---
_pipeline = FeaturePipeline.from_artifacts(_STATS_PATH)
_pipeline.set_normalization_params(_checkpoint["norm_params"])

# --- Load LightGBM model (optional) ---
_lgbm_model = None
if _LGBM_PATH.exists():
    import lightgbm as lgb
    _lgbm_model = lgb.Booster(model_file=str(_LGBM_PATH))


def predict(request: dict) -> float:
    """Predict trip duration in seconds.

    Input schema:
        {
            "pickup_zone":     int,   # NYC taxi zone, 1-265
            "dropoff_zone":    int,
            "requested_at":    str,   # ISO 8601 datetime
            "passenger_count": int,
        }
    """
    cat, cont = _pipeline.transform_single(request)

    # NN prediction
    pickup = torch.tensor([cat[0]], dtype=torch.long)
    dropoff = torch.tensor([cat[1]], dtype=torch.long)
    cont_t = torch.from_numpy(cont).unsqueeze(0)

    with torch.inference_mode():
        nn_pred = _model(pickup, dropoff, cont_t)

    nn_result = float(nn_pred.item())
    if _log_target:
        nn_result = math.exp(nn_result)

    # LightGBM prediction (if available)
    if _lgbm_model is not None:
        # LGBM uses raw features (no normalization): [pickup, dropoff, ...cont_raw...]
        # transform_single already normalized cont, so we need raw
        # Re-extract raw cont from pipeline without normalization
        cat_raw, cont_raw = _pipeline.transform_single_raw(request)
        lgbm_features = np.concatenate([cat_raw.astype(np.float32), cont_raw]).reshape(1, -1)
        lgbm_result = float(_lgbm_model.predict(lgbm_features)[0])

        return _ENSEMBLE_ALPHA * nn_result + (1 - _ENSEMBLE_ALPHA) * lgbm_result

    return nn_result
