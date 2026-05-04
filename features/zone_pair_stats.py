"""Zone-pair statistical features for ETA prediction.

Precomputes duration statistics at multiple granularity levels:
  1. (pickup_zone, dropoff_zone) pair level
  2. (pickup_zone, dropoff_zone, time_bucket) pair-temporal level
  3. pickup_zone level (fallback)
  4. dropoff_zone level (fallback)
  5. global level (last resort)

Uses Bayesian shrinkage to smooth sparse estimates toward priors.
"""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_PATH = (
    Path(__file__).parent.parent / "data" / "zone_pair_stats" / "zone_pair_stats.pkl"
)
SHRINKAGE_PRIOR_COUNT = 20

# 6 time buckets capturing distinct traffic regimes
TIME_BUCKETS = {
    0: 0,  # Late Night (12-5AM)
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 1,  # Early Morning (5-8AM)
    6: 1,
    7: 1,
    8: 2,  # AM Rush (8-11AM)
    9: 2,
    10: 2,
    11: 3,  # Midday (11AM-4PM)
    12: 3,
    13: 3,
    14: 3,
    15: 3,
    16: 4,  # PM Rush (4-8PM)
    17: 4,
    18: 4,
    19: 4,
    20: 5,  # Evening (8PM-12AM)
    21: 5,
    22: 5,
    23: 5,
}
NUM_TIME_BUCKETS = 6


def hour_to_bucket(hour: int) -> int:
    """Map hour (0-23) to time bucket (0-5)."""
    return TIME_BUCKETS[hour]


@dataclass(frozen=True)
class ZonePairFeatures:
    """Features returned for a single (pickup, dropoff) lookup."""

    pair_mean_smoothed: float
    pair_median: float
    pair_std: float
    pair_count: int
    pair_p25: float
    pair_p75: float
    pair_iqr: float
    log_pair_count: float
    same_zone: int
    pu_mean: float
    do_mean: float
    pair_tb_mean: float
    pair_tb_median: float
    pair_rarity: float  # 1/(1+log1p(count)) -- high for rare pairs, low for common


def _compute_pair_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duration stats per (pickup_zone, dropoff_zone) pair."""
    return (
        train_df.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"]
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


def _compute_pair_temporal_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duration stats per (pickup_zone, dropoff_zone, time_bucket)."""
    ts = pd.to_datetime(train_df["requested_at"])
    train_df = train_df.assign(time_bucket=ts.dt.hour.map(TIME_BUCKETS).astype(np.int8))
    return (
        train_df.groupby(["pickup_zone", "dropoff_zone", "time_bucket"])[
            "duration_seconds"
        ]
        .agg(tb_mean="mean", tb_median="median", tb_count="count")
        .reset_index()
    )


def _compute_zone_stats(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate duration stats per pickup zone and per dropoff zone."""
    pu_stats = (
        train_df.groupby("pickup_zone")["duration_seconds"]
        .agg(pu_mean="mean", pu_median="median", pu_count="count")
        .reset_index()
    )
    do_stats = (
        train_df.groupby("dropoff_zone")["duration_seconds"]
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
    """Apply Bayesian shrinkage to smooth pair means toward pickup-zone prior."""
    merged = pair_stats.merge(
        pu_stats[["pickup_zone", "pu_mean"]],
        on="pickup_zone",
        how="left",
    )
    prior_mean = merged["pu_mean"].fillna(global_stats["global_mean"])

    n = merged["pair_count"]
    merged["pair_mean_smoothed"] = (
        n * merged["pair_mean"] + prior_count * prior_mean
    ) / (n + prior_count)
    return merged


def _apply_temporal_shrinkage(
    tb_stats: pd.DataFrame,
    pair_dict: dict[tuple[int, int], dict[str, float]],
    global_stats: dict[str, float],
    prior_count: int = 10,
) -> pd.DataFrame:
    """Shrink temporal bucket means toward overall pair mean."""
    pair_means = []
    for _, row in tb_stats.iterrows():
        key = (int(row["pickup_zone"]), int(row["dropoff_zone"]))
        pair_data = pair_dict.get(key)
        pair_means.append(
            pair_data["pair_mean_smoothed"]
            if pair_data
            else global_stats["global_mean"]
        )
    tb_stats = tb_stats.assign(pair_overall_mean=pair_means)

    n = tb_stats["tb_count"]
    tb_stats = tb_stats.assign(
        tb_mean_smoothed=(n * tb_stats["tb_mean"] + prior_count * tb_stats["pair_overall_mean"])
        / (n + prior_count)
    )
    return tb_stats


def compute_stats(train_path: Path) -> dict[str, Any]:
    """Compute all zone-pair statistics from training data.

    Returns a dict containing:
      - "pair": dict keyed by (pickup_zone, dropoff_zone) -> stat dict
      - "pair_temporal": dict keyed by (pickup_zone, dropoff_zone, time_bucket)
      - "pickup": dict keyed by pickup_zone -> stat dict
      - "dropoff": dict keyed by dropoff_zone -> stat dict
      - "global": global stat dict
      - "prior_count": shrinkage prior used
    """
    logger.info("Loading training data from %s", train_path)
    train_df = pd.read_parquet(
        train_path,
        columns=["pickup_zone", "dropoff_zone", "requested_at", "duration_seconds"],
    )
    logger.info("Loaded %s rows", f"{len(train_df):,}")

    logger.info("Computing pair-level stats...")
    pair_stats = _compute_pair_stats(train_df)
    logger.info("Computed stats for %s zone pairs", f"{len(pair_stats):,}")

    logger.info("Computing zone-level stats...")
    pu_stats, do_stats = _compute_zone_stats(train_df)

    logger.info("Computing global stats...")
    global_stats = _compute_global_stats(train_df)

    logger.info(
        "Applying Bayesian shrinkage (prior_count=%d)...", SHRINKAGE_PRIOR_COUNT
    )
    pair_stats = _apply_shrinkage(pair_stats, pu_stats, global_stats)

    # Fill NaN std (pairs with count=1 have NaN std)
    pair_stats["pair_std"] = pair_stats["pair_std"].fillna(global_stats["global_std"])

    # Build pair lookup dict
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

    # Temporal zone-pair stats
    logger.info("Computing temporal zone-pair stats (6 time buckets)...")
    tb_stats = _compute_pair_temporal_stats(train_df)
    logger.info(
        "Computed %s pair-temporal entries", f"{len(tb_stats):,}"
    )
    tb_stats = _apply_temporal_shrinkage(tb_stats, pair_dict, global_stats)

    tb_dict: dict[tuple[int, int, int], dict[str, float]] = {}
    for row in tb_stats.itertuples(index=False):
        tb_dict[
            (row.pickup_zone, row.dropoff_zone, row.time_bucket)
        ] = {
            "tb_mean": float(row.tb_mean_smoothed),
            "tb_median": float(row.tb_median),
            "tb_count": int(row.tb_count),
        }

    # Zone-level dicts
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
        "pair_temporal": tb_dict,
        "pickup": pu_dict,
        "dropoff": do_dict,
        "global": global_stats,
        "prior_count": SHRINKAGE_PRIOR_COUNT,
    }


def lookup(
    stats: dict[str, Any],
    pickup_zone: int,
    dropoff_zone: int,
    hour: int | None = None,
) -> ZonePairFeatures:
    """Look up zone-pair features for a single request.

    Applies fallback hierarchy:
      pair-level -> pickup-zone -> dropoff-zone -> global
      pair-temporal -> pair-level (for time-bucketed stats)
    """
    global_stats = stats["global"]
    pu_data = stats["pickup"].get(pickup_zone)
    do_data = stats["dropoff"].get(dropoff_zone)

    pu_mean = pu_data["pu_mean"] if pu_data else global_stats["global_mean"]
    do_mean = do_data["do_mean"] if do_data else global_stats["global_mean"]

    same_zone = 1 if pickup_zone == dropoff_zone else 0

    pair_data = stats["pair"].get((pickup_zone, dropoff_zone))

    if pair_data is not None:
        p25 = pair_data["pair_p25"]
        p75 = pair_data["pair_p75"]
        count = pair_data["pair_count"]
        pair_mean_sm = pair_data["pair_mean_smoothed"]
        pair_median = pair_data["pair_median"]
        pair_std = pair_data["pair_std"]
    else:
        fallback_mean = (pu_mean + do_mean) / 2.0
        p25 = global_stats["global_p25"]
        p75 = global_stats["global_p75"]
        count = 0
        pair_mean_sm = fallback_mean
        pair_median = fallback_mean
        pair_std = global_stats["global_std"]

    # Temporal lookup
    tb_mean = pair_mean_sm
    tb_median = pair_median
    if hour is not None:
        tb = hour_to_bucket(hour)
        tb_data = stats.get("pair_temporal", {}).get(
            (pickup_zone, dropoff_zone, tb)
        )
        if tb_data is not None:
            tb_mean = tb_data["tb_mean"]
            tb_median = tb_data["tb_median"]

    log_count = math.log1p(count)
    pair_rarity = 1.0 / (1.0 + log_count)

    return ZonePairFeatures(
        pair_mean_smoothed=pair_mean_sm,
        pair_median=pair_median,
        pair_std=pair_std,
        pair_count=count,
        pair_p25=p25,
        pair_p75=p75,
        pair_iqr=p75 - p25,
        log_pair_count=log_count,
        same_zone=same_zone,
        pu_mean=pu_mean,
        do_mean=do_mean,
        pair_tb_mean=tb_mean,
        pair_tb_median=tb_median,
        pair_rarity=pair_rarity,
    )


def _pair_key(pickup: int, dropoff: int) -> int:
    """Encode (pickup, dropoff) as a single integer."""
    return pickup * 1000 + dropoff


def _pair_tb_key(pickup: int, dropoff: int, tb: int) -> int:
    """Encode (pickup, dropoff, time_bucket) as a single integer."""
    return pickup * 10000 + dropoff * 10 + tb


def enrich_dataframe(df: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    """Add zone-pair stat columns to a DataFrame (vectorized, memory-efficient).

    Adds columns: pair_mean_smoothed, pair_median, pair_std, pair_count,
    pair_p25, pair_p75, pair_iqr, log_pair_count, same_zone,
    pu_mean, do_mean, pair_tb_mean, pair_tb_median.
    """
    pair_dict = stats["pair"]
    tb_dict = stats.get("pair_temporal", {})
    pu_dict = stats["pickup"]
    do_dict = stats["dropoff"]
    global_stats = stats["global"]

    # Integer-keyed pair lookup
    pu_vals = df["pickup_zone"].values
    do_vals = df["dropoff_zone"].values
    pair_int_keys = pu_vals * 1000 + do_vals

    # Build integer-keyed mappings once
    feature_names = [
        "pair_mean_smoothed",
        "pair_median",
        "pair_std",
        "pair_count",
        "pair_p25",
        "pair_p75",
    ]
    int_mappings: dict[str, dict[int, float]] = {}
    for feat in feature_names:
        int_mappings[feat] = {
            _pair_key(k[0], k[1]): v[feat] for k, v in pair_dict.items()
        }

    pair_key_series = pd.Series(pair_int_keys)

    for feat in feature_names:
        col = pair_key_series.map(int_mappings[feat])

        if feat == "pair_count":
            col = col.fillna(0).astype(np.int32)
        elif feat == "pair_std":
            col = col.fillna(global_stats["global_std"])
        elif feat == "pair_median":
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

    # Fill unseen pair stats with zone-level fallback
    fallback = (pu_mean_col + do_mean_col) / 2.0
    df["pair_mean_smoothed"] = df["pair_mean_smoothed"].fillna(fallback)
    df["pair_median"] = df["pair_median"].fillna(fallback)

    # Derived features
    log_pair_count = np.log1p(df["pair_count"])
    df = df.assign(
        pair_iqr=df["pair_p75"] - df["pair_p25"],
        log_pair_count=log_pair_count,
        same_zone=(df["pickup_zone"] == df["dropoff_zone"]).astype(np.int8),
        pair_rarity=1.0 / (1.0 + log_pair_count),
    )

    # Temporal zone-pair stats
    if tb_dict:
        ts = pd.to_datetime(df["requested_at"])
        hours = ts.dt.hour
        time_buckets = hours.map(TIME_BUCKETS).astype(np.int8)
        tb_int_keys = pu_vals * 10000 + do_vals * 10 + time_buckets.values

        tb_mean_map = {
            _pair_tb_key(k[0], k[1], k[2]): v["tb_mean"]
            for k, v in tb_dict.items()
        }
        tb_median_map = {
            _pair_tb_key(k[0], k[1], k[2]): v["tb_median"]
            for k, v in tb_dict.items()
        }

        tb_key_series = pd.Series(tb_int_keys)
        tb_mean_col = tb_key_series.map(tb_mean_map).fillna(df["pair_mean_smoothed"])
        tb_median_col = tb_key_series.map(tb_median_map).fillna(df["pair_median"])

        df = df.assign(pair_tb_mean=tb_mean_col, pair_tb_median=tb_median_col)
    else:
        df = df.assign(
            pair_tb_mean=df["pair_mean_smoothed"],
            pair_tb_median=df["pair_median"],
        )

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

    logger.info("Pair-level entries: %s", f"{len(stats['pair']):,}")
    logger.info("Pair-temporal entries: %s", f"{len(stats['pair_temporal']):,}")
    logger.info("Pickup zones: %d", len(stats["pickup"]))
    logger.info("Dropoff zones: %d", len(stats["dropoff"]))
    logger.info("Global mean: %.1f s", stats["global"]["global_mean"])
