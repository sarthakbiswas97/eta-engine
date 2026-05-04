"""Temporal feature extraction for ETA prediction.

Extracts time-based features from ISO 8601 timestamps. All features are
generic -- no domain-specific geography or location knowledge.

Features:
  - Cyclical encoding (sin/cos) for hour, day-of-week
  - Normalized minute-of-day (0-1)
  - Binary flags: is_weekend, is_rush_hour, is_night

Removed (constant in dev/eval date ranges):
  - month_sin, month_cos (dev = all Dec, eval = winter holidays)
  - day_of_month (2-week slices have near-zero variance)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class TemporalFeatures:
    """Features returned for a single timestamp lookup."""

    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float
    minute_of_day: float
    is_weekend: int
    is_rush_hour: int
    is_night: int


def extract_single(requested_at: str) -> TemporalFeatures:
    """Extract temporal features from a single ISO 8601 timestamp string."""
    ts = datetime.fromisoformat(requested_at)

    hour_frac = (ts.hour + ts.minute / 60.0) / 24.0
    dow_frac = ts.weekday() / 7.0

    is_wknd = 1 if ts.weekday() >= 5 else 0
    is_rush = 1 if (not is_wknd and (7 <= ts.hour <= 9 or 16 <= ts.hour <= 19)) else 0
    is_ngt = 1 if (ts.hour >= 23 or ts.hour < 5) else 0

    return TemporalFeatures(
        hour_sin=math.sin(_TWO_PI * hour_frac),
        hour_cos=math.cos(_TWO_PI * hour_frac),
        dow_sin=math.sin(_TWO_PI * dow_frac),
        dow_cos=math.cos(_TWO_PI * dow_frac),
        minute_of_day=(ts.hour * 60 + ts.minute) / 1439.0,
        is_weekend=is_wknd,
        is_rush_hour=is_rush,
        is_night=is_ngt,
    )


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal feature columns to a DataFrame (vectorized).

    Expects a 'requested_at' column with ISO 8601 timestamp strings.
    Returns a new DataFrame with temporal columns added.
    """
    ts = pd.to_datetime(df["requested_at"])

    hour_frac = (ts.dt.hour + ts.dt.minute / 60.0) / 24.0
    dow_frac = ts.dt.dayofweek / 7.0

    is_weekend = (ts.dt.dayofweek >= 5).astype(np.int8)

    return df.assign(
        hour_sin=np.sin(_TWO_PI * hour_frac),
        hour_cos=np.cos(_TWO_PI * hour_frac),
        dow_sin=np.sin(_TWO_PI * dow_frac),
        dow_cos=np.cos(_TWO_PI * dow_frac),
        minute_of_day=(ts.dt.hour * 60 + ts.dt.minute) / 1439.0,
        is_weekend=is_weekend,
        is_rush_hour=((~is_weekend.astype(bool)) & ((ts.dt.hour >= 7) & (ts.dt.hour <= 9) | (ts.dt.hour >= 16) & (ts.dt.hour <= 19))).astype(np.int8),
        is_night=((ts.dt.hour >= 23) | (ts.dt.hour < 5)).astype(np.int8),
    )


FEATURE_COLUMNS = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "minute_of_day",
    "is_weekend", "is_rush_hour", "is_night",
]
