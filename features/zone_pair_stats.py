"""Zone-pair statistical features for ETA prediction.

Precomputes duration statistics at three granularity levels from training data:
  1. (pickup_zone, dropoff_zone) pair level
  2. pickup_zone level (fallback)
  3. dropoff_zone level (fallback)
  4. global level (last resort)

Uses Bayesian shrinkage to smooth sparse pair estimates toward the
pickup-zone prior, so rare pairs don't produce noisy features.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_PATH = Path(__file__).parent.parent / "data" / "zone_pair_stats" / "zone_pair_stats.pkl"
SHRINKAGE_PRIOR_COUNT = 20


@dataclass(frozen=True)
class ZonePairFeatures:
    """Features returned for a single (pickup, dropoff) lookup."""

    pair_mean_smoothed: float
    pair_median: float
    pair_std: float
    pair_count: int
    pair_p25: float
    pair_p75: float
    pu_mean: float
    do_mean: float


def _compute_pair_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duration stats per (pickup_zone, dropoff_zone) pair."""
    return (
        train_df
        .groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"]
        .agg(
            pair_mean="mean",
            pair_median="median",
            pair_std="std",
            pair_count="count",
            pair_p25=lambda x: x.quantile(0.25),
            pair_p75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )


def _compute_zone_stats(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate duration stats per pickup zone and per dropoff zone."""
    pu_stats = (
        train_df
        .groupby("pickup_zone")["duration_seconds"]
        .agg(pu_mean="mean", pu_median="median", pu_count="count")
        .reset_index()
    )
    do_stats = (
        train_df
        .groupby("dropoff_zone")["duration_seconds"]
        .agg(do_mean="mean", do_median="median", do_count="count")
        .reset_index()
    )
    return pu_stats, do_stats


def _compute_global_stats(train_df: pd.DataFrame) -> dict[str, float]:
    """Compute global duration statistics."""
    dur = train_df["duration_seconds"]
    return {
        "global_mean": float(dur.mean()),
        "global_median": float(dur.median()),
        "global_std": float(dur.std()),
        "global_p25": float(dur.quantile(0.25)),
        "global_p75": float(dur.quantile(0.75)),
    }


def _apply_shrinkage(
    pair_stats: pd.DataFrame,
    pu_stats: pd.DataFrame,
    global_stats: dict[str, float],
    prior_count: int = SHRINKAGE_PRIOR_COUNT,
) -> pd.DataFrame:
    """Apply Bayesian shrinkage to smooth pair means toward pickup-zone prior.

    smoothed = (n * pair_mean + prior_count * prior_mean) / (n + prior_count)

    Where prior_mean = pickup-zone mean (falls back to global if pickup zone
    is also sparse).
    """
    merged = pair_stats.merge(
        pu_stats[["pickup_zone", "pu_mean"]],
        on="pickup_zone",
        how="left",
    )
    prior_mean = merged["pu_mean"].fillna(global_stats["global_mean"])

    n = merged["pair_count"]
    merged["pair_mean_smoothed"] = (
        (n * merged["pair_mean"] + prior_count * prior_mean) / (n + prior_count)
    )
    return merged


def compute_stats(train_path: Path) -> dict[str, Any]:
    """Compute all zone-pair statistics from training data.

    Returns a dict containing:
      - "pair": dict keyed by (pickup_zone, dropoff_zone) -> stat dict
      - "pickup": dict keyed by pickup_zone -> stat dict
      - "dropoff": dict keyed by dropoff_zone -> stat dict
      - "global": global stat dict
      - "prior_count": shrinkage prior used
    """
    logger.info("Loading training data from %s", train_path)
    train_df = pd.read_parquet(train_path, columns=["pickup_zone", "dropoff_zone", "duration_seconds"])
    logger.info("Loaded %s rows", f"{len(train_df):,}")

    logger.info("Computing pair-level stats...")
    pair_stats = _compute_pair_stats(train_df)
    logger.info("Computed stats for %s zone pairs", f"{len(pair_stats):,}")

    logger.info("Computing zone-level stats...")
    pu_stats, do_stats = _compute_zone_stats(train_df)

    logger.info("Computing global stats...")
    global_stats = _compute_global_stats(train_df)

    logger.info("Applying Bayesian shrinkage (prior_count=%d)...", SHRINKAGE_PRIOR_COUNT)
    pair_stats = _apply_shrinkage(pair_stats, pu_stats, global_stats)

    # Fill NaN std (pairs with count=1 have NaN std)
    pair_stats["pair_std"] = pair_stats["pair_std"].fillna(global_stats["global_std"])

    # Build lookup dicts for fast access
    pair_dict: dict[tuple[int, int], dict[str, float]] = {}
    for row in pair_stats.itertuples(index=False):
        pair_dict[(row.pickup_zone, row.dropoff_zone)] = {
            "pair_mean_smoothed": float(row.pair_mean_smoothed),
            "pair_median": float(row.pair_median),
            "pair_std": float(row.pair_std),
            "pair_count": int(row.pair_count),
            "pair_p25": float(row.pair_p25),
            "pair_p75": float(row.pair_p75),
            "pu_mean": float(row.pu_mean),
        }

    pu_dict: dict[int, dict[str, float]] = {}
    for row in pu_stats.itertuples(index=False):
        pu_dict[row.pickup_zone] = {
            "pu_mean": float(row.pu_mean),
            "pu_median": float(row.pu_median),
            "pu_count": int(row.pu_count),
        }

    do_dict: dict[int, dict[str, float]] = {}
    for row in do_stats.itertuples(index=False):
        do_dict[row.dropoff_zone] = {
            "do_mean": float(row.do_mean),
            "do_median": float(row.do_median),
            "do_count": int(row.do_count),
        }

    return {
        "pair": pair_dict,
        "pickup": pu_dict,
        "dropoff": do_dict,
        "global": global_stats,
        "prior_count": SHRINKAGE_PRIOR_COUNT,
    }


def lookup(stats: dict[str, Any], pickup_zone: int, dropoff_zone: int) -> ZonePairFeatures:
    """Look up zone-pair features for a single request.

    Applies fallback hierarchy:
      pair-level -> pickup-zone -> dropoff-zone -> global
    """
    global_stats = stats["global"]
    pu_data = stats["pickup"].get(pickup_zone)
    do_data = stats["dropoff"].get(dropoff_zone)

    pu_mean = pu_data["pu_mean"] if pu_data else global_stats["global_mean"]
    do_mean = do_data["do_mean"] if do_data else global_stats["global_mean"]

    pair_data = stats["pair"].get((pickup_zone, dropoff_zone))
    if pair_data is not None:
        return ZonePairFeatures(
            pair_mean_smoothed=pair_data["pair_mean_smoothed"],
            pair_median=pair_data["pair_median"],
            pair_std=pair_data["pair_std"],
            pair_count=pair_data["pair_count"],
            pair_p25=pair_data["pair_p25"],
            pair_p75=pair_data["pair_p75"],
            pu_mean=pu_mean,
            do_mean=do_mean,
        )

    # Unseen pair: construct from zone-level stats with global fallback
    fallback_mean = (pu_mean + do_mean) / 2.0
    return ZonePairFeatures(
        pair_mean_smoothed=fallback_mean,
        pair_median=fallback_mean,
        pair_std=global_stats["global_std"],
        pair_count=0,
        pair_p25=global_stats["global_p25"],
        pair_p75=global_stats["global_p75"],
        pu_mean=pu_mean,
        do_mean=do_mean,
    )


def enrich_dataframe(df: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    """Add zone-pair stat columns to a DataFrame (vectorized).

    Adds columns: pair_mean_smoothed, pair_median, pair_std, pair_count,
    pair_p25, pair_p75, pu_mean, do_mean.
    """
    pair_dict = stats["pair"]
    pu_dict = stats["pickup"]
    do_dict = stats["dropoff"]
    global_stats = stats["global"]

    # Build pair keys
    pair_keys = list(zip(df["pickup_zone"], df["dropoff_zone"]))

    # Vectorized lookup via Series.map
    pair_series = pd.Series(pair_keys)

    feature_names = ["pair_mean_smoothed", "pair_median", "pair_std", "pair_count", "pair_p25", "pair_p75"]
    for feat in feature_names:
        mapping = {k: v[feat] for k, v in pair_dict.items()}
        col = pair_series.map(mapping)

        if feat == "pair_count":
            col = col.fillna(0).astype(np.int32)
        elif feat == "pair_std":
            col = col.fillna(global_stats["global_std"])
        elif feat == "pair_median":
            # For unseen pairs, fall back to avg of pu_mean + do_mean
            col = col.fillna(np.nan)
        elif feat == "pair_p25":
            col = col.fillna(global_stats["global_p25"])
        elif feat == "pair_p75":
            col = col.fillna(global_stats["global_p75"])

        df = df.assign(**{feat: col})

    # Pickup-zone and dropoff-zone means
    pu_mean_map = {z: d["pu_mean"] for z, d in pu_dict.items()}
    do_mean_map = {z: d["do_mean"] for z, d in do_dict.items()}

    pu_mean_col = df["pickup_zone"].map(pu_mean_map).fillna(global_stats["global_mean"])
    do_mean_col = df["dropoff_zone"].map(do_mean_map).fillna(global_stats["global_mean"])

    df = df.assign(pu_mean=pu_mean_col, do_mean=do_mean_col)

    # Fill unseen pair_mean_smoothed with avg of zone means
    fallback = (pu_mean_col + do_mean_col) / 2.0
    df["pair_mean_smoothed"] = df["pair_mean_smoothed"].fillna(fallback)
    df["pair_median"] = df["pair_median"].fillna(fallback)

    return df


def save_stats(stats: dict[str, Any], path: Path = DEFAULT_ARTIFACT_PATH) -> None:
    """Serialize stats dict to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved zone-pair stats to %s", path)


def load_stats(path: Path = DEFAULT_ARTIFACT_PATH) -> dict[str, Any]:
    """Load stats dict from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_path = Path(__file__).parent.parent / "data" / "train.parquet"
    stats = compute_stats(train_path)
    save_stats(stats)

    # Quick summary
    pair_count = len(stats["pair"])
    logger.info("Pair-level entries: %s", f"{pair_count:,}")
    logger.info("Pickup zones: %d", len(stats["pickup"]))
    logger.info("Dropoff zones: %d", len(stats["dropoff"]))
    logger.info("Global mean: %.1f s", stats["global"]["global_mean"])
