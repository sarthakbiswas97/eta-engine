"""Submission interface -- this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from features.pipeline import FeaturePipeline
from model.architecture import ETAModel, ModelConfig

_ROOT = Path(__file__).parent
_MODEL_PATH = _ROOT / "model.pt"
_STATS_PATH = _ROOT / "data" / "zone_pair_stats" / "zone_pair_stats.pkl"

# Load checkpoint
_checkpoint = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)

# Rebuild model from saved config
_config = ModelConfig(**_checkpoint["model_config"])
_model = ETAModel(_config)
_model.load_state_dict(_checkpoint["model_state_dict"])
_model.eval()

# Load feature pipeline with normalization
_pipeline = FeaturePipeline.from_artifacts(_STATS_PATH)
_pipeline.set_normalization_params(_checkpoint["norm_params"])


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

    pickup = torch.tensor([cat[0]], dtype=torch.long)
    dropoff = torch.tensor([cat[1]], dtype=torch.long)
    cont_t = torch.from_numpy(cont).unsqueeze(0)

    with torch.inference_mode():
        pred = _model(pickup, dropoff, cont_t)

    return float(pred.item())
