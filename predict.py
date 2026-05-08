"""Submission interface -- this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.

Ensemble of three models:
  1. MLP neural net (zone embeddings + residual blocks)
  2. LightGBM (gradient-boosted trees)
  3. FT-Transformer (ONNX Runtime for fast inference)

Weighted average: 0.6 * NN + 0.2 * LGBM + 0.2 * FT
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from features.pipeline import FeaturePipeline
from model.architecture import ETAModel, ModelConfig

# Reduce thread overhead for single-sample inference
torch.set_num_threads(1)

_ROOT = Path(__file__).parent
_MODEL_PATH = _ROOT / "model.pt"
_LGBM_PATH = _ROOT / "lgbm_model.txt"
_FT_ONNX_PATH = _ROOT / "ft_model.onnx"
_FT_PT_PATH = _ROOT / "ft_model.pt"
_STATS_PATH = _ROOT / "data" / "zone_pair_stats" / "zone_pair_stats.pkl"

# Ensemble weights (optimized on full dev set, 1.23M rows)
_W_NN = 0.60
_W_LGBM = 0.20
_W_FT = 0.20

# --- Load NN model ---
_nn_checkpoint = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)
_nn_config = ModelConfig(**_nn_checkpoint["model_config"])
_nn_model = ETAModel(_nn_config)
_nn_model.load_state_dict(_nn_checkpoint["model_state_dict"])
_nn_model.eval()
_nn_log_target = _nn_checkpoint.get("log_target", False)

# --- Load feature pipeline (shared) ---
_pipeline = FeaturePipeline.from_artifacts(_STATS_PATH)
_pipeline.set_normalization_params(_nn_checkpoint["norm_params"])

# --- Load FT-Transformer (prefer ONNX, fall back to PyTorch) ---
_ft_onnx_session = None
_ft_pt_model = None
_ft_norm = None

if _FT_ONNX_PATH.exists():
    import onnxruntime as ort
    _ft_onnx_session = ort.InferenceSession(
        str(_FT_ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    # Load norm params from npz
    _ft_norm_path = _ROOT / "ft_norm_params.npz"
    if _ft_norm_path.exists():
        _npz = np.load(_ft_norm_path)
        _ft_norm = {"means": _npz["means"], "stds": _npz["stds"]}
    else:
        # Fall back to checkpoint norm params
        _ft_checkpoint = torch.load(_FT_PT_PATH, map_location="cpu", weights_only=False)
        _ft_norm = _ft_checkpoint["norm_params"]

elif _FT_PT_PATH.exists():
    from model.ft_transformer import FTTransformer, FTConfig
    _ft_checkpoint = torch.load(_FT_PT_PATH, map_location="cpu", weights_only=False)
    _ft_config = FTConfig(**_ft_checkpoint["model_config"])
    _ft_pt_model = FTTransformer(_ft_config)
    _ft_pt_model.load_state_dict(_ft_checkpoint["model_state_dict"])
    _ft_pt_model.eval()
    _ft_norm = _ft_checkpoint["norm_params"]

# --- Load LightGBM model ---
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
    # Get raw features (unnormalized) for LGBM/FT, and normalized for NN
    cat, cont_normed = _pipeline.transform_single(request)
    cat_raw, cont_raw = _pipeline.transform_single_raw(request)

    # --- NN prediction ---
    pickup = torch.tensor([cat[0]], dtype=torch.long)
    dropoff = torch.tensor([cat[1]], dtype=torch.long)
    cont_t = torch.from_numpy(cont_normed).unsqueeze(0)

    with torch.inference_mode():
        nn_pred = _nn_model(pickup, dropoff, cont_t)

    nn_result = float(nn_pred.item())
    if _nn_log_target:
        nn_result = math.exp(nn_result)

    # --- Build ensemble ---
    total_weight = _W_NN
    weighted_sum = _W_NN * nn_result

    # LGBM prediction
    if _lgbm_model is not None:
        lgbm_features = np.concatenate([cat_raw.astype(np.float32), cont_raw]).reshape(1, -1)
        lgbm_result = float(_lgbm_model.predict(lgbm_features)[0])
        weighted_sum += _W_LGBM * lgbm_result
        total_weight += _W_LGBM

    # FT-Transformer prediction (ONNX or PyTorch)
    if _ft_norm is not None:
        cont_ft = ((cont_raw - _ft_norm["means"]) / _ft_norm["stds"]).astype(np.float32)

        if _ft_onnx_session is not None:
            # ONNX Runtime inference
            x_num = cont_ft.reshape(1, -1)
            x_cat = cat_raw.reshape(1, -1).astype(np.int64)
            ft_result = float(_ft_onnx_session.run(
                ["prediction"],
                {"x_numerical": x_num, "x_categorical": x_cat},
            )[0].item())

        elif _ft_pt_model is not None:
            # PyTorch fallback
            x_num = torch.from_numpy(cont_ft).unsqueeze(0)
            x_cat = torch.tensor([[cat[0], cat[1]]], dtype=torch.long)
            with torch.inference_mode():
                ft_result = float(_ft_pt_model(x_num, x_cat).item())
        else:
            ft_result = None

        if ft_result is not None:
            weighted_sum += _W_FT * ft_result
            total_weight += _W_FT

    return weighted_sum / total_weight
