"""Feature pipeline for ETA prediction.

Combines zone-pair statistics and temporal features into a unified
feature representation. Separates categorical inputs (zone IDs for
embeddings) from continuous inputs for the neural net.

Usage:
    # Batch (training):
    pipeline = FeaturePipeline.from_artifacts(stats_path)
    cat, cont, targets = pipeline.transform_dataframe(train_df)

    # Single request (inference):
    cat, cont = pipeline.transform_single(request_dict)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.temporal import (
    FEATURE_COLUMNS as TEMPORAL_COLUMNS,
    enrich_dataframe as temporal_enrich,
    extract_single as temporal_extract,
)
from features.zone_pair_stats import (
    load_stats,
    lookup as zone_pair_lookup,
    enrich_dataframe as zone_pair_enrich,
)

logger = logging.getLogger(__name__)

ZONE_PAIR_CONTINUOUS = [
    "pair_mean_smoothed",
    "pair_median",
    "pair_std",
    "pair_count",
    "pair_p25",
    "pair_p75",
    "pu_mean",
    "do_mean",
]

CONTINUOUS_FEATURES = ZONE_PAIR_CONTINUOUS + TEMPORAL_COLUMNS

CATEGORICAL_FEATURES = ["pickup_zone", "dropoff_zone"]

NUM_ZONES = 266  # zone IDs 1-265, index 0 reserved for padding/unknown


@dataclass(frozen=True)
class FeatureSchema:
    """Describes the feature layout for the model."""

    categorical: list[str]
    continuous: list[str]
    num_zones: int
    n_continuous: int


SCHEMA = FeatureSchema(
    categorical=CATEGORICAL_FEATURES,
    continuous=CONTINUOUS_FEATURES,
    num_zones=NUM_ZONES,
    n_continuous=len(CONTINUOUS_FEATURES),
)


class FeaturePipeline:
    """Transforms raw request data into model-ready features."""

    def __init__(self, zone_pair_stats: dict[str, Any]) -> None:
        self._stats = zone_pair_stats
        self._cont_means: np.ndarray | None = None
        self._cont_stds: np.ndarray | None = None

    @classmethod
    def from_artifacts(cls, stats_path: Path) -> FeaturePipeline:
        """Load pipeline from saved artifacts."""
        stats = load_stats(stats_path)
        return cls(zone_pair_stats=stats)

    def transform_single(self, request: dict) -> tuple[np.ndarray, np.ndarray]:
        """Transform a single request dict into (categorical, continuous) arrays.

        Returns:
            cat: shape (2,) int32 -- [pickup_zone, dropoff_zone]
            cont: shape (n_continuous,) float32 -- continuous features
        """
        pickup_zone = int(request["pickup_zone"])
        dropoff_zone = int(request["dropoff_zone"])

        # Categorical
        cat = np.array([pickup_zone, dropoff_zone], dtype=np.int32)

        # Zone-pair stats
        zp = zone_pair_lookup(self._stats, pickup_zone, dropoff_zone)
        zp_values = [
            zp.pair_mean_smoothed,
            zp.pair_median,
            zp.pair_std,
            float(zp.pair_count),
            zp.pair_p25,
            zp.pair_p75,
            zp.pu_mean,
            zp.do_mean,
        ]

        # Temporal
        tf = temporal_extract(request["requested_at"])
        tf_values = [getattr(tf, col) for col in TEMPORAL_COLUMNS]

        cont = np.array(zp_values + tf_values, dtype=np.float32)

        # Normalize if stats are fitted
        if self._cont_means is not None:
            cont = (cont - self._cont_means) / self._cont_stds

        return cat, cont

    def transform_dataframe(
        self, df: pd.DataFrame, target_col: str | None = "duration_seconds",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Transform a DataFrame into (categorical, continuous, targets) arrays.

        Returns:
            cat: shape (n, 2) int32
            cont: shape (n, n_continuous) float32
            targets: shape (n,) float32 or None if target_col not in df
        """
        logger.info("Enriching %s rows with zone-pair stats...", f"{len(df):,}")
        enriched = zone_pair_enrich(df.copy(), self._stats)

        logger.info("Enriching with temporal features...")
        enriched = temporal_enrich(enriched)

        # Categorical array
        cat = enriched[CATEGORICAL_FEATURES].to_numpy(dtype=np.int32)

        # Continuous array
        cont = enriched[CONTINUOUS_FEATURES].to_numpy(dtype=np.float32)

        # Replace any remaining NaN with 0 (rare unseen zone pairs)
        cont = np.nan_to_num(cont, nan=0.0)

        # Targets
        targets = None
        if target_col and target_col in enriched.columns:
            targets = enriched[target_col].to_numpy(dtype=np.float32)

        return cat, cont, targets

    def fit_normalization(self, cont: np.ndarray) -> None:
        """Compute mean/std from training continuous features for normalization.

        Call this once on training data, then transform_single will apply it.
        """
        self._cont_means = cont.mean(axis=0).astype(np.float32)
        self._cont_stds = cont.std(axis=0).astype(np.float32)
        # Prevent division by zero for constant features
        self._cont_stds[self._cont_stds < 1e-8] = 1.0
        logger.info("Fitted normalization on %d features", len(self._cont_means))

    def normalize(self, cont: np.ndarray) -> np.ndarray:
        """Apply normalization to a continuous feature array."""
        if self._cont_means is None:
            raise RuntimeError("Call fit_normalization first")
        return (cont - self._cont_means) / self._cont_stds

    @property
    def normalization_params(self) -> dict[str, np.ndarray] | None:
        """Return normalization params for serialization."""
        if self._cont_means is None:
            return None
        return {"means": self._cont_means, "stds": self._cont_stds}

    def set_normalization_params(self, params: dict[str, np.ndarray]) -> None:
        """Restore normalization params from serialized form."""
        self._cont_means = params["means"]
        self._cont_stds = params["stds"]

    @property
    def schema(self) -> FeatureSchema:
        return SCHEMA
