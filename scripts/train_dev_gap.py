"""Compute train MAE vs dev MAE to measure generalization gap.

Uses batch inference (not single-request predict()) for speed.
Samples train set to keep runtime reasonable on CPU.

Usage:
    python scripts/train_dev_gap.py
    python scripts/train_dev_gap.py --train-sample 500000 --dev-sample 0
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from features.pipeline import FeaturePipeline
from model.architecture import ETAModel, ModelConfig
from model.dataset import create_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def load_model() -> tuple[ETAModel, FeaturePipeline]:
    """Load trained model and pipeline."""
    checkpoint = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = ETAModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pipeline = FeaturePipeline.from_artifacts(ROOT / "data" / "zone_pair_stats" / "zone_pair_stats.pkl")
    pipeline.set_normalization_params(checkpoint["norm_params"])
    return model, pipeline


def load_and_transform(
    pipeline: FeaturePipeline,
    path: Path,
    sample_n: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load parquet, transform to arrays."""
    import gc
    import pandas as pd

    logger.info("Loading %s...", path.name)
    df = pd.read_parquet(path)
    if sample_n is not None and sample_n > 0 and len(df) > sample_n:
        logger.info("Sampling %s from %s rows", f"{sample_n:,}", f"{len(df):,}")
        df = df.sample(n=sample_n, random_state=SEED).reset_index(drop=True)
    logger.info("Transforming %s rows...", f"{len(df):,}")

    chunk_size = 2_000_000
    cat_chunks, cont_chunks, target_chunks = [], [], []

    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        chunk = df.iloc[start:end].copy()
        cat, cont, targets = pipeline.transform_dataframe(chunk)
        cont = pipeline.normalize(cont)
        cat_chunks.append(cat)
        cont_chunks.append(cont)
        if targets is not None:
            target_chunks.append(targets)
        del chunk
        gc.collect()

    del df
    gc.collect()
    return np.concatenate(cat_chunks), np.concatenate(cont_chunks), np.concatenate(target_chunks)


@torch.no_grad()
def compute_mae(model: ETAModel, cat: np.ndarray, cont: np.ndarray, targets: np.ndarray, batch_size: int = 16384) -> dict:
    """Compute MAE and error statistics."""
    loader = create_dataloader(cat, cont, targets, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    all_preds, all_targets = [], []
    for pickup, dropoff, cont_batch, target in loader:
        preds = model(pickup, dropoff, cont_batch)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(target.numpy())

    preds = np.concatenate(all_preds)
    actuals = np.concatenate(all_targets)
    errors = preds - actuals
    abs_errors = np.abs(errors)

    return {
        "n": len(preds),
        "mae": float(abs_errors.mean()),
        "bias": float(errors.mean()),
        "median_ae": float(np.median(abs_errors)),
        "std_error": float(errors.std()),
        "p90_ae": float(np.percentile(abs_errors, 90)),
        "p95_ae": float(np.percentile(abs_errors, 95)),
        "p99_ae": float(np.percentile(abs_errors, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sample", type=int, default=500_000, help="Train sample size (0 = full, default 500k)")
    parser.add_argument("--dev-sample", type=int, default=0, help="Dev sample size (0 = full)")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model, pipeline = load_model()
    logger.info("Model loaded: %s params", f"{model.count_parameters():,}")

    # Train MAE
    train_sample = args.train_sample if args.train_sample > 0 else None
    t0 = time.time()
    train_cat, train_cont, train_targets = load_and_transform(
        pipeline, ROOT / "data" / "train.parquet", sample_n=train_sample,
    )
    train_stats = compute_mae(model, train_cat, train_cont, train_targets)
    train_time = time.time() - t0
    del train_cat, train_cont, train_targets

    # Dev MAE
    dev_sample = args.dev_sample if args.dev_sample > 0 else None
    t0 = time.time()
    dev_cat, dev_cont, dev_targets = load_and_transform(
        pipeline, ROOT / "data" / "dev.parquet", sample_n=dev_sample,
    )
    dev_stats = compute_mae(model, dev_cat, dev_cont, dev_targets)
    dev_time = time.time() - t0

    # Report
    print("\n" + "=" * 60)
    print("TRAIN vs DEV GAP ANALYSIS (v3 model)")
    print("=" * 60)

    print(f"\n{'Metric':<20} {'Train':>12} {'Dev':>12} {'Gap':>12}")
    print("-" * 56)
    print(f"{'N rows':<20} {train_stats['n']:>12,} {dev_stats['n']:>12,}")
    print(f"{'MAE':<20} {train_stats['mae']:>12.1f} {dev_stats['mae']:>12.1f} {dev_stats['mae'] - train_stats['mae']:>+12.1f}")
    print(f"{'Bias (mean error)':<20} {train_stats['bias']:>12.1f} {dev_stats['bias']:>12.1f} {dev_stats['bias'] - train_stats['bias']:>+12.1f}")
    print(f"{'Median AE':<20} {train_stats['median_ae']:>12.1f} {dev_stats['median_ae']:>12.1f} {dev_stats['median_ae'] - train_stats['median_ae']:>+12.1f}")
    print(f"{'Std of error':<20} {train_stats['std_error']:>12.1f} {dev_stats['std_error']:>12.1f} {dev_stats['std_error'] - train_stats['std_error']:>+12.1f}")
    print(f"{'P90 AE':<20} {train_stats['p90_ae']:>12.1f} {dev_stats['p90_ae']:>12.1f} {dev_stats['p90_ae'] - train_stats['p90_ae']:>+12.1f}")
    print(f"{'P95 AE':<20} {train_stats['p95_ae']:>12.1f} {dev_stats['p95_ae']:>12.1f} {dev_stats['p95_ae'] - train_stats['p95_ae']:>+12.1f}")
    print(f"{'P99 AE':<20} {train_stats['p99_ae']:>12.1f} {dev_stats['p99_ae']:>12.1f} {dev_stats['p99_ae'] - train_stats['p99_ae']:>+12.1f}")
    print(f"\nTrain eval time: {train_time:.1f}s  |  Dev eval time: {dev_time:.1f}s")

    gap = dev_stats['mae'] - train_stats['mae']
    print(f"\n--- Summary ---")
    print(f"Train MAE:  {train_stats['mae']:.1f}s")
    print(f"Dev MAE:    {dev_stats['mae']:.1f}s")
    print(f"Gap:        {gap:+.1f}s")
    if gap > 30:
        print(">> SIGNIFICANT overfitting. Model memorizes training patterns that don't generalize.")
    elif gap > 15:
        print(">> MODERATE overfitting. Some room for regularization.")
    elif gap > 5:
        print(">> MILD overfitting. Model generalizes reasonably well.")
    else:
        print(">> MINIMAL gap. Model is well-regularized (or underfitting).")

    # Eval projection
    print(f"\n--- Eval Projection ---")
    print(f"CHALLENGE.md notes ~15s dev-to-eval gap (baseline: 351 dev -> 367 eval)")
    print(f"Projected eval MAE: ~{dev_stats['mae'] + 15:.0f}s (if same gap applies)")


if __name__ == "__main__":
    main()
