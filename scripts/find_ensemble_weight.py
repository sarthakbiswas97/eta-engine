"""Find optimal ensemble weight between NN and LightGBM on dev set.

Computes: pred = alpha * nn_pred + (1-alpha) * lgbm_pred
Searches alpha in [0.0, 1.0] and reports MAE for each.

Usage:
    PYTHONPATH=. python scripts/find_ensemble_weight.py
    PYTHONPATH=. python scripts/find_ensemble_weight.py --dev-sample 0  # full dev
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


def get_nn_predictions(pipeline: FeaturePipeline, cat: np.ndarray, cont: np.ndarray) -> np.ndarray:
    """Run NN inference on data."""
    checkpoint = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = ETAModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    log_target = checkpoint.get("log_target", False)

    # Apply normalization
    norm_params = checkpoint["norm_params"]
    cont_normed = (cont - norm_params["means"]) / norm_params["stds"]

    # Dummy targets for dataloader
    dummy_targets = np.zeros(len(cat), dtype=np.float32)
    loader = create_dataloader(
        cat, cont_normed, dummy_targets,
        batch_size=16384, shuffle=False, num_workers=0, pin_memory=False,
    )

    preds_list = []
    with torch.no_grad():
        for pickup, dropoff, cont_batch, _ in loader:
            p = model(pickup, dropoff, cont_batch)
            if log_target:
                p = p.exp()
            preds_list.append(p.cpu().numpy())

    return np.concatenate(preds_list)


def get_lgbm_predictions(cat: np.ndarray, cont: np.ndarray) -> np.ndarray:
    """Run LightGBM inference on data."""
    model_path = ROOT / "lgbm_model.txt"
    model = lgb.Booster(model_file=str(model_path))
    X = build_lgbm_features(cat, cont)
    return model.predict(X)


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

    logger.info("Getting NN predictions...")
    nn_preds = get_nn_predictions(pipeline, dev_cat, dev_cont)

    logger.info("Getting LightGBM predictions...")
    lgbm_preds = get_lgbm_predictions(dev_cat, dev_cont)

    # Individual model performance
    nn_mae = float(np.mean(np.abs(nn_preds - dev_targets)))
    lgbm_mae = float(np.mean(np.abs(lgbm_preds - dev_targets)))
    nn_bias = float(np.mean(nn_preds - dev_targets))
    lgbm_bias = float(np.mean(lgbm_preds - dev_targets))

    print(f"\n{'='*60}")
    print(f"ENSEMBLE WEIGHT OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Dev rows: {len(dev_targets):,}")
    print(f"\nIndividual models:")
    print(f"  NN:      MAE={nn_mae:.1f}s, bias={nn_bias:+.1f}s")
    print(f"  LightGBM: MAE={lgbm_mae:.1f}s, bias={lgbm_bias:+.1f}s")

    # Grid search alpha
    print(f"\nEnsemble: pred = alpha * NN + (1-alpha) * LGBM")
    print(f"\n{'Alpha':<8} {'MAE':>8} {'Bias':>8} {'vs NN':>8} {'vs LGBM':>10}")
    print("-" * 50)

    best_alpha = 0.5
    best_mae = float("inf")

    for alpha in np.arange(0.0, 1.05, 0.05):
        ensemble_preds = alpha * nn_preds + (1 - alpha) * lgbm_preds
        mae = float(np.mean(np.abs(ensemble_preds - dev_targets)))
        bias = float(np.mean(ensemble_preds - dev_targets))
        vs_nn = mae - nn_mae
        vs_lgbm = mae - lgbm_mae
        marker = " <-- best" if mae < best_mae else ""
        print(f"{alpha:<8.2f} {mae:>8.1f} {bias:>+8.1f} {vs_nn:>+8.1f} {vs_lgbm:>+10.1f}{marker}")
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha:.2f}")
    print(f"Best ensemble MAE: {best_mae:.1f}s")
    print(f"Improvement over NN: {best_mae - nn_mae:+.1f}s")
    print(f"Improvement over LGBM: {best_mae - lgbm_mae:+.1f}s")

    # Error correlation
    nn_errors = nn_preds - dev_targets
    lgbm_errors = lgbm_preds - dev_targets
    correlation = float(np.corrcoef(nn_errors, lgbm_errors)[0, 1])
    print(f"\nError correlation (NN vs LGBM): {correlation:.3f}")
    if correlation > 0.95:
        print(">> HIGH correlation -- models make similar errors, ensemble gain limited")
    elif correlation > 0.85:
        print(">> MODERATE correlation -- some decorrelation, modest ensemble gain expected")
    else:
        print(">> LOW correlation -- good decorrelation, strong ensemble gain expected")


if __name__ == "__main__":
    main()
