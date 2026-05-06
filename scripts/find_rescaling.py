"""Find optimal prediction rescaling to fix variance collapse.

Diagnostic showed pred std = 0.872x actual. Rescaling predictions away
from the mean can recover some of that lost variance.

Optimizes: adjusted = global_mean + (pred - global_mean) * scale_factor

Usage:
    PYTHONPATH=. python scripts/find_rescaling.py
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

from features.pipeline import FeaturePipeline
from model.architecture import ETAModel, ModelConfig
from model.dataset import create_dataloader
from scripts.train_lgbm import build_lgbm_features, load_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
ENSEMBLE_ALPHA = 0.50


def get_ensemble_predictions(
    pipeline: FeaturePipeline,
    cat: np.ndarray,
    cont: np.ndarray,
) -> np.ndarray:
    """Get ensemble predictions (NN + LGBM)."""
    # NN predictions
    checkpoint = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = ETAModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log_target = checkpoint.get("log_target", False)

    norm_params = checkpoint["norm_params"]
    cont_normed = (cont - norm_params["means"]) / norm_params["stds"]

    dummy = np.zeros(len(cat), dtype=np.float32)
    loader = create_dataloader(cat, cont_normed, dummy, batch_size=16384, shuffle=False, num_workers=0, pin_memory=False)

    nn_preds = []
    with torch.no_grad():
        for pickup, dropoff, cont_batch, _ in loader:
            p = model(pickup, dropoff, cont_batch)
            if log_target:
                p = p.exp()
            nn_preds.append(p.cpu().numpy())
    nn_preds = np.concatenate(nn_preds)

    del model, checkpoint
    gc.collect()

    # LGBM predictions
    lgbm = lgb.Booster(model_file=str(ROOT / "lgbm_model.txt"))
    X = build_lgbm_features(cat, cont)
    lgbm_preds = lgbm.predict(X)

    # Ensemble
    return ENSEMBLE_ALPHA * nn_preds + (1 - ENSEMBLE_ALPHA) * lgbm_preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-sample", type=int, default=0, help="Dev sample (0=full)")
    args = parser.parse_args()

    np.random.seed(SEED)

    pipeline = FeaturePipeline.from_artifacts(ROOT / "data" / "zone_pair_stats" / "zone_pair_stats.pkl")

    dev_sample = args.dev_sample if args.dev_sample > 0 else None
    dev_cat, dev_cont, dev_targets = load_features(
        pipeline, ROOT / "data" / "dev.parquet", sample_n=dev_sample,
    )

    logger.info("Getting ensemble predictions...")
    preds = get_ensemble_predictions(pipeline, dev_cat, dev_cont)

    # Stats
    pred_mean = float(preds.mean())
    pred_std = float(preds.std())
    actual_mean = float(dev_targets.mean())
    actual_std = float(dev_targets.std())
    baseline_mae = float(np.mean(np.abs(preds - dev_targets)))
    baseline_bias = float(np.mean(preds - dev_targets))

    print(f"\n{'='*60}")
    print(f"PREDICTION RESCALING OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Dev rows: {len(dev_targets):,}")
    print(f"\nDistributions:")
    print(f"  Predictions: mean={pred_mean:.1f}, std={pred_std:.1f}")
    print(f"  Actual:      mean={actual_mean:.1f}, std={actual_std:.1f}")
    print(f"  Std ratio:   {pred_std/actual_std:.3f}")
    print(f"\nBaseline ensemble: MAE={baseline_mae:.1f}s, bias={baseline_bias:+.1f}s")

    # Try different centers for rescaling
    print(f"\n--- Rescaling around prediction mean ---")
    print(f"Formula: adjusted = center + (pred - center) * scale")
    print(f"\n{'Scale':<8} {'MAE':>8} {'Bias':>8} {'vs Base':>8}")
    print("-" * 40)

    best_scale = 1.0
    best_mae = baseline_mae
    best_center = pred_mean

    for center in [pred_mean, actual_mean, 989.0]:
        for scale in np.arange(0.90, 1.25, 0.01):
            adjusted = center + (preds - center) * scale
            mae = float(np.mean(np.abs(adjusted - dev_targets)))
            if mae < best_mae:
                best_mae = mae
                best_scale = scale
                best_center = center

    # Print results for best center
    print(f"\nBest center: {best_center:.1f}")
    for scale in np.arange(0.95, 1.20, 0.01):
        adjusted = best_center + (preds - best_center) * scale
        mae = float(np.mean(np.abs(adjusted - dev_targets)))
        bias = float(np.mean(adjusted - dev_targets))
        vs_base = mae - baseline_mae
        marker = " <-- best" if abs(scale - best_scale) < 0.005 else ""
        print(f"{scale:<8.2f} {mae:>8.1f} {bias:>+8.1f} {vs_base:>+8.1f}{marker}")

    print(f"\nBest scale: {best_scale:.2f}")
    print(f"Best MAE: {best_mae:.1f}s")
    print(f"Improvement: {best_mae - baseline_mae:+.1f}s")

    # Also try simple bias correction (just add a constant)
    print(f"\n--- Simple bias correction ---")
    print(f"Formula: adjusted = pred + offset")
    print(f"\n{'Offset':<8} {'MAE':>8} {'Bias':>8} {'vs Base':>8}")
    print("-" * 40)

    best_offset = 0.0
    best_offset_mae = baseline_mae

    for offset in np.arange(-100, 110, 5):
        adjusted = preds + offset
        mae = float(np.mean(np.abs(adjusted - dev_targets)))
        bias = float(np.mean(adjusted - dev_targets))
        vs_base = mae - baseline_mae
        if mae < best_offset_mae:
            best_offset_mae = mae
            best_offset = offset
        if abs(offset) <= 5 or abs(offset - best_offset) <= 5 or offset % 20 == 0:
            marker = " <-- best" if abs(offset - best_offset) < 2.5 else ""
            print(f"{offset:<+8.0f} {mae:>8.1f} {bias:>+8.1f} {vs_base:>+8.1f}{marker}")

    print(f"\nBest offset: {best_offset:+.0f}s")
    print(f"Best MAE: {best_offset_mae:.1f}s")
    print(f"Improvement: {best_offset_mae - baseline_mae:+.1f}s")


if __name__ == "__main__":
    main()
