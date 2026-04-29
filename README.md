# NYC ETA Engine

Predicting NYC taxi trip duration from pickup/dropoff zones, request
timestamp, and passenger count. Built for the Gobblecube ETA Challenge.

See [CHALLENGE.md](CHALLENGE.md) for the original problem statement and rules.

---

## Approach

### Problem

Given a ride request (pickup zone, dropoff zone, timestamp, passenger count),
predict trip duration in seconds. Scored on MAE against a held-out 2024
winter-holiday eval set.

### Architecture

Deep learning model with learned zone embeddings and rich engineered features,
trained on ~37M NYC yellow taxi trips from 2023.

**Feature groups:**

1. **Zone-pair statistics** -- precomputed mean, median, std, p25, p75, and
   trip count per (pickup, dropoff) pair with Bayesian shrinkage for sparse
   pairs. Fallback hierarchy: pair -> pickup-zone -> dropoff-zone -> global.
2. **Spatial features** -- zone centroid coordinates, haversine/Manhattan
   distance, bearing, borough-level categoricals (from NYC TLC shapefile).
3. **Temporal features** -- cyclical hour/day/month encoding, rush hour,
   weekend, and holiday flags.

**Model:** Tabular neural net with entity embeddings for zone IDs, combined
with continuous features through an MLP. Trained with MAE (L1) loss on GPU,
inference on CPU.

### Baseline Comparison

| Method | Dev MAE |
|--------|---------|
| Predict global mean | ~580 s |
| XGBoost baseline (6 features) | 351 s |
| Zone-pair average (10 lines) | ~300 s |
| Zone-pair smoothed mean (this repo) | 302.7 s |
| Zone-pair median (this repo) | 296.7 s |
| **Full model (target)** | **< 280 s** |

---

## Project Structure

```
.
├── CHALLENGE.md              # Original challenge README (reference)
├── SUBMISSION_TEMPLATE.md    # Writeup template for final submission
├── baseline.py               # Original XGBoost baseline (reference)
├── predict.py                # Submission interface (grader imports this)
├── grade.py                  # Local scoring harness
├── Dockerfile                # Submission packaging
├── requirements.txt          # Python dependencies
├── features/                 # Feature engineering modules
│   └── zone_pair_stats.py    # Zone-pair statistical features
├── data/
│   ├── download_data.py      # Fetches NYC TLC data, builds train/dev splits
│   ├── schema.md             # Data schema documentation
│   └── zone_pair_stats/      # Generated artifacts (gitignored)
└── tests/
    └── test_submission.py    # Smoke tests for submission contract
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download data (~500 MB, one-time)
python data/download_data.py

# Compute zone-pair stats
python -m features.zone_pair_stats

# Run baseline (reference)
python baseline.py

# Score on dev set
python grade.py
```

---

## What Worked

- Zone-pair median as a standalone predictor (296.7s) beats the XGBoost
  baseline (351s) with zero ML -- confirms zone-pair is the dominant signal.
- Bayesian shrinkage for sparse pairs prevents overfitting on low-count
  zone pairs while preserving signal from high-count ones.

## What We Tried That Didn't Work

*(Updated as we go)*

## Next Experiments

*(Updated as we go)*

---

## Constraints

- Inference: <= 200 ms per request on CPU
- Docker image: <= 2.5 GB
- No external API calls at inference time
- No 2024 data in training
