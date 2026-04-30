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

### Design Philosophy

The model is **generic** -- it learns all spatial relationships from the trip
data itself via embeddings, not from external geography (no shapefiles, no
hardcoded coordinates). If the zone IDs mapped to a different city, the model
would learn equally well given the same trip patterns.

### Architecture

Tabular neural net with learned zone embeddings and engineered features,
trained on ~37M NYC yellow taxi trips from 2023.

**Feature groups (21 total):**

1. **Zone-pair statistics (8 features)** -- precomputed mean, median, std,
   p25, p75, trip count per (pickup, dropoff) pair with Bayesian shrinkage
   for sparse pairs. Fallback hierarchy: pair -> pickup-zone -> dropoff-zone
   -> global.
2. **Temporal features (11 features)** -- cyclical sin/cos encoding for hour,
   day-of-week, month. Binary flags for weekend, rush hour, night. Normalized
   minute-of-day and day-of-month.
3. **Zone embeddings (2 categorical)** -- separate learned embeddings for
   pickup and dropoff zones (dim=50 each), plus a hash-based zone-pair
   embedding (16k buckets, dim=16) for direct pair-level learning.

**Model:** Combined MLP (372k parameters). Zone embeddings and continuous
features are processed through separate branches, then merged. Trained with
MAE (L1) loss on GPU, inference on CPU (<1ms per request).

### Results

| Method | Dev MAE | Improvement |
|--------|---------|-------------|
| Predict global mean | ~580 s | -- |
| XGBoost baseline (6 features) | 351 s | -- |
| Zone-pair average (10 lines) | ~300 s | 14% vs XGB |
| Zone-pair smoothed mean | 302.7 s | 14% vs XGB |
| Zone-pair median | 296.7 s | 15% vs XGB |
| **Neural net v1** | **272.1 s** | **22% vs XGB** |

---

## Project Structure

```
.
├── CHALLENGE.md              # Original challenge README (reference)
├── SUBMISSION_TEMPLATE.md    # Writeup template for final submission
├── baseline.py               # Original XGBoost baseline (reference)
├── predict.py                # Submission interface (grader imports this)
├── grade.py                  # Local scoring harness
├── train.py                  # Training script (GPU)
├── Dockerfile                # Submission packaging
├── requirements.txt          # Python dependencies
├── features/                 # Feature engineering modules
│   ├── zone_pair_stats.py    # Zone-pair statistical features
│   ├── temporal.py           # Temporal feature extraction
│   └── pipeline.py           # Unified feature pipeline
├── model/                    # Neural network
│   ├── architecture.py       # ETAModel definition
│   └── dataset.py            # PyTorch Dataset/DataLoader
├── scripts/
│   └── upload_data_hf.py     # Push data to HF Hub
├── notebooks/
│   └── train_gpu.ipynb       # Colab/Kaggle training notebook
├── data/
│   ├── download_data.py      # Fetches NYC TLC data
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

# Train (GPU recommended, or use notebooks/train_gpu.ipynb on Colab)
python train.py --epochs 10 --batch-size 8192 --lr 5e-4

# Score on dev set
python grade.py
```

---

## What Worked

- **Zone-pair statistics as features:** Zone-pair median alone (296.7s) beats
  the XGBoost baseline (351s) with zero ML. Bayesian shrinkage handles sparse
  pairs without overfitting.
- **Learned zone embeddings:** The neural net learns spatial relationships
  purely from trip patterns. No external geography needed. Combined with
  zone-pair stats, achieves 272.1s.
- **OneCycleLR with warmup:** Stable training from step 1, avoids the NaN
  gradients we saw with CosineAnnealingLR.
- **Chunked data processing:** Processing 37M rows in 2M chunks keeps memory
  under 6GB, enabling training on Colab's 12GB RAM.

## What Didn't Work

- **Training on small samples (500k rows):** Model converged to ~945s MAE,
  far worse than the zone-pair baseline. Embeddings need the full dataset to
  learn meaningful zone relationships.
- **CosineAnnealingLR without warmup:** NaN loss in first epoch due to
  unstable gradients on randomly initialized embeddings.

## Next Experiments

- Interaction features (zone-pair stats x temporal)
- Temporal zone-pair statistics (pair mean by time bucket)
- Architecture improvements (residual connections, deeper MLP)
- Hyperparameter sweep (embedding dim, dropout, LR)
- LightGBM ensemble
- Additional derived features (same_zone flag, pair IQR)

---

## Constraints

- Inference: <= 200 ms per request on CPU (actual: <1 ms)
- Docker image: <= 2.5 GB (estimated: ~500 MB)
- No external API calls at inference time
- No 2024 data in training
