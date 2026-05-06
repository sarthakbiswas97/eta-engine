"""Train a LightGBM model on the same features as the neural net.

LightGBM captures different patterns than the NN:
  - Handles zone IDs as native categoricals (no embedding needed)
  - Partition-based splits work well for rare/sparse data
  - Inherently handles feature interactions via tree splits

Usage:
    python scripts/train_lgbm.py
    python scripts/train_lgbm.py --sample 10000000 --run-name lgbm-10M
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.pipeline import FeaturePipeline, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATS_PATH = DATA_DIR / "zone_pair_stats" / "zone_pair_stats.pkl"
SEED = 42


def load_features(
    pipeline: FeaturePipeline,
    path: Path,
    sample_n: int | None = None,
    chunk_size: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load parquet and transform. Returns (cat, cont, targets)."""
    logger.info("Loading %s...", path.name)
    df = pd.read_parquet(path)
    if sample_n is not None and sample_n > 0 and len(df) > sample_n:
        logger.info("Sampling %s from %s rows", f"{sample_n:,}", f"{len(df):,}")
        df = df.sample(n=sample_n, random_state=SEED).reset_index(drop=True)

    total_rows = len(df)
    logger.info("Transforming %s rows...", f"{total_rows:,}")

    cat_chunks, cont_chunks, target_chunks = [], [], []
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk = df.iloc[start:end].copy()
        cat, cont, targets = pipeline.transform_dataframe(chunk)
        cat_chunks.append(cat)
        cont_chunks.append(cont)
        if targets is not None:
            target_chunks.append(targets)
        del chunk
        gc.collect()

    del df
    gc.collect()
    return np.concatenate(cat_chunks), np.concatenate(cont_chunks), np.concatenate(target_chunks)


def build_lgbm_features(cat: np.ndarray, cont: np.ndarray) -> np.ndarray:
    """Combine categorical + continuous into a single feature matrix for LGBM.

    LGBM handles categoricals natively -- we pass zone IDs as-is (not one-hot).
    Feature order: [pickup_zone, dropoff_zone, ...24 continuous features...]
    """
    return np.hstack([cat.astype(np.float32), cont])


def get_feature_names() -> list[str]:
    """Return feature names in same order as build_lgbm_features."""
    return CATEGORICAL_FEATURES + CONTINUOUS_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM for ETA prediction")
    parser.add_argument("--sample", type=int, default=10_000_000, help="Train sample size (0=full, default 10M)")
    parser.add_argument("--dev-sample", type=int, default=100_000, help="Dev sample for early stopping")
    parser.add_argument("--num-leaves", type=int, default=255, help="Max leaves per tree")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--n-estimators", type=int, default=1000, help="Max boosting rounds")
    parser.add_argument("--early-stopping", type=int, default=50, help="Early stopping rounds")
    parser.add_argument("--min-child-samples", type=int, default=100, help="Min samples per leaf")
    parser.add_argument("--run-name", type=str, default="lgbm-v1", help="Run name for logging")
    args = parser.parse_args()

    np.random.seed(SEED)

    # Load pipeline (no normalization needed for trees)
    pipeline = FeaturePipeline.from_artifacts(STATS_PATH)

    # Load data
    train_sample = args.sample if args.sample > 0 else None
    train_cat, train_cont, train_targets = load_features(
        pipeline, DATA_DIR / "train.parquet", sample_n=train_sample,
    )
    dev_cat, dev_cont, dev_targets = load_features(
        pipeline, DATA_DIR / "dev.parquet", sample_n=args.dev_sample,
    )

    # Build feature matrices
    X_train = build_lgbm_features(train_cat, train_cont)
    X_dev = build_lgbm_features(dev_cat, dev_cont)
    feature_names = get_feature_names()

    logger.info("Train: %s rows, %d features", f"{X_train.shape[0]:,}", X_train.shape[1])
    logger.info("Dev: %s rows", f"{X_dev.shape[0]:,}")

    del train_cat, train_cont, dev_cat, dev_cont
    gc.collect()

    # Create LightGBM datasets
    # Mark zone columns as categorical
    cat_indices = [0, 1]  # pickup_zone, dropoff_zone

    train_data = lgb.Dataset(
        X_train, label=train_targets,
        feature_name=feature_names,
        categorical_feature=[feature_names[i] for i in cat_indices],
        free_raw_data=True,
    )
    dev_data = lgb.Dataset(
        X_dev, label=dev_targets,
        feature_name=feature_names,
        categorical_feature=[feature_names[i] for i in cat_indices],
        reference=train_data,
        free_raw_data=True,
    )

    del X_train, train_targets
    gc.collect()

    # Train
    # Auto-detect GPU
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    params = {
        "objective": "mae",
        "metric": "mae",
        "num_leaves": args.num_leaves,
        "learning_rate": args.lr,
        "min_child_samples": args.min_child_samples,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": SEED,
        "num_threads": -1,
    }

    if has_gpu:
        params["device"] = "gpu"
        logger.info("Using GPU for LightGBM training")

    logger.info("Training LightGBM: %s", params)
    t0 = time.time()

    callbacks = [
        lgb.early_stopping(args.early_stopping),
        lgb.log_evaluation(50),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=args.n_estimators,
        valid_sets=[dev_data],
        valid_names=["dev"],
        callbacks=callbacks,
    )

    train_time = time.time() - t0
    logger.info("Training complete in %.1fs, %d trees", train_time, model.num_trees())

    # Evaluate
    dev_preds = model.predict(X_dev)
    dev_mae = float(np.mean(np.abs(dev_preds - dev_targets)))
    dev_bias = float(np.mean(dev_preds - dev_targets))
    logger.info("Dev MAE: %.1f s", dev_mae)
    logger.info("Dev bias: %+.1f s", dev_bias)

    # Save model
    model_path = ROOT / "lgbm_model.txt"
    model.save_model(str(model_path))
    model_size = model_path.stat().st_size / 1e6
    logger.info("Model saved to %s (%.1f MB)", model_path, model_size)

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])
    logger.info("Top 10 features by gain:")
    for name, imp in feat_imp[:10]:
        logger.info("  %s: %.0f", name, imp)

    # Summary
    print(f"\n{'='*50}")
    print(f"LightGBM Training Summary")
    print(f"{'='*50}")
    print(f"Run: {args.run_name}")
    print(f"Train rows: {train_sample or 'full':,}" if train_sample else f"Train rows: full")
    print(f"Trees: {model.num_trees()}")
    print(f"Dev MAE: {dev_mae:.1f} s")
    print(f"Dev bias: {dev_bias:+.1f} s")
    print(f"Training time: {train_time:.1f}s")
    print(f"Model size: {model_size:.1f} MB")
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    main()
