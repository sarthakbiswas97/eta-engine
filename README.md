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

### Architecture (v3)

Tabular neural net with learned zone embeddings, engineered features, and
residual connections. Trained on ~37M NYC yellow taxi trips from 2023.

```
Zone Branch:
  pickup_zone  -> Embedding(266, 50)  --\
  dropoff_zone -> Embedding(266, 50)  ---+-- [pu, do, pu*do, pu-do, pair_hash]
  (pu, do)     -> HashEmbed(16384, 16) -/        |
                                           concat(216) -> MLP(128) x2 -> 128-dim

Continuous Branch:
  24 features -> BatchNorm(24) -> MLP(128) x2 -> 128-dim

Combined:
  concat(256) -> ResidualBlock(256)
             -> project(128) -> ResidualBlock(128)
             -> Linear(64) -> SiLU -> Linear(1)
```

**559,897 trainable parameters.**

**Feature groups (26 total):**

1. **Zone-pair statistics (13 features)** -- precomputed mean, median, std,
   p25, p75, IQR, trip count per (pickup, dropoff) pair with Bayesian
   shrinkage for sparse pairs. Time-bucketed mean/median (6 time-of-day
   buckets). Fallback hierarchy: pair -> pickup-zone -> dropoff-zone -> global.
2. **Temporal features (11 features)** -- cyclical sin/cos encoding for hour,
   day-of-week, month. Binary flags for weekend, rush hour, night. Normalized
   minute-of-day and day-of-month.
3. **Zone embeddings (2 categorical)** -- separate learned embeddings for
   pickup and dropoff zones (dim=50 each), plus element-wise product
   (similarity) and difference (directionality), plus a hash-based zone-pair
   embedding (16k buckets, dim=16).

---

## Results

| Method | Dev MAE | Params | vs XGBoost |
|--------|---------|--------|------------|
| Predict global mean | ~580 s | -- | -- |
| XGBoost baseline (6 features) | 351.0 s | -- | -- |
| Zone-pair smoothed mean | 302.7 s | -- | -14% |
| Zone-pair median | 296.7 s | -- | -15% |
| Zone-pair time-bucketed mean | 277.9 s | -- | -21% |
| Neural net v1 (L1, 19 features) | 272.1 s | 372k | -22% |
| Neural net v2 (Huber, 24 features) | 266.2 s | 373k | -24% |
| **Neural net v3 (residual, Huber)** | **264.5 s** | **560k** | **-25%** |

---

## Experiments

### v1: Baseline Neural Net

**What:** Tabular neural net with separate zone embedding and continuous
feature branches, combined MLP, L1 loss. 19 continuous features (8 zone-pair
stats + 11 temporal). 372k parameters.

**Why:** Zone-pair median alone (296.7s) beats XGBoost (351s), so the dominant
signal is the pickup-dropoff pair. A neural net can learn nonlinear
interactions between zone embeddings and temporal features that simple
statistics miss.

| Epoch | Train Loss | Dev MAE |
|-------|-----------|---------|
| 1 | 968.5 | 858.7 s |
| 2 | 669.0 | 414.5 s |
| 3 | 281.8 | 275.2 s |
| 4 | 245.3 | **272.1 s** |
| 5 | 241.9 | 273.9 s |
| 6 | 239.9 | 272.4 s |
| 7 | 238.9 | 274.6 s |

**Result:** 272.1s (epoch 4, early stopped at 7). Train-dev gap of ~33s
indicates moderate overfitting.

**Takeaway:** Most learning happens in epochs 2-3 as embeddings lock onto
zone-pair patterns. After that, diminishing returns. Error analysis revealed
systematic underprediction on long trips (-970s bias for 2400s+ trips).

---

### v2: Temporal Features + Huber Loss

**What changed:**
- 5 new features (19 -> 24): time-bucketed zone-pair mean/median (6 time-of-day
  buckets with Bayesian shrinkage), pair IQR, log trip count, same-zone flag
- Huber loss (delta=300) instead of L1: L2 penalty for errors < 300s, L1 for
  larger errors

**Why:** Error analysis of v1 showed the same zone pair varies 2.2x by time of
day (e.g., zone 237->236: 251s at 5AM vs 552s at 2PM). Temporal zone-pair
stats capture this. Huber loss addresses the long-trip underprediction bias by
penalizing large errors less aggressively than L2 but more than L1.

| Epoch | Train Loss | Dev MAE |
|-------|-----------|---------|
| 1 | 246048.6 | 923.5 s |
| 2 | 156510.2 | 394.5 s |
| 3 | 50746.7 | 270.8 s |
| 4 | 41775.0 | 270.3 s |
| 5 | 40995.2 | **266.2 s** |
| 6 | 40563.6 | 269.0 s |
| 7 | 40314.5 | 270.9 s |
| 8 | 40164.9 | 270.4 s |

**Result:** 266.2s (epoch 5, early stopped at 8). 6s improvement over v1.

**Takeaway:** Temporal features helped but less than expected -- the model may
already learn temporal-zone interactions via embeddings. The plateau at ~266s
suggested architecture was the bottleneck, not features.

---

### v3: Residual Architecture + Embedding Interactions

**What changed:**
- Element-wise product (`pu * do`) captures zone similarity (zones that
  co-occur in similar trip patterns get similar embeddings, so their product
  is large)
- Element-wise difference (`pu - do`) captures directionality (A->B vs B->A
  have opposite signs)
- Deeper zone interaction MLP (2 layers instead of 1)
- Wider continuous branch (128-dim instead of 64-dim)
- Residual blocks in combined MLP for better gradient flow
- Higher dropout (0.3) and embed dropout (0.15)
- Parameters: 372k -> 560k

**Why:** v2 plateaued at 266s despite strong features, suggesting the
architecture couldn't fully exploit the inputs. Residual connections help
deeper networks train stably. Embedding interactions provide explicit
similarity/direction signals without the model needing to learn them from
scratch.

| Epoch | Train Loss | Dev MAE |
|-------|-----------|---------|
| 1 | 94684.9 | 300.8 s |
| 2 | 41304.7 | 279.7 s |
| 3 | 39168.2 | 272.3 s |
| 4 | 38266.6 | 268.7 s |
| 5 | 37758.8 | **264.5 s** |
| 6 | 37414.8 | 268.2 s |
| 7 | 37193.0 | 269.3 s |
| 8 | 37054.8 | 271.2 s |

**Result:** 264.5s (epoch 5, early stopped at 8). 1.7s improvement over v2.

**Takeaway:** Architecture changes gave modest gains. The residual blocks
helped stabilize deeper training, but the model still plateaus after epoch 5.
Train loss continues dropping while dev MAE rises -- classic overfitting
signal. Next steps: L1 vs Huber A/B test, reduced hash buckets.

---

### Learning Curves

```mermaid
xychart-beta
    title "Dev MAE Across Training (lower is better)"
    x-axis "Epoch" [1, 2, 3, 4, 5, 6, 7, 8]
    y-axis "Dev MAE (seconds)" 250 --> 950
    line [858.7, 414.5, 275.2, 272.1, 273.9, 272.4, 274.6, 274.6]
    line [923.5, 394.5, 270.8, 270.3, 266.2, 269.0, 270.9, 270.4]
    line [300.8, 279.7, 272.3, 268.7, 264.5, 268.2, 269.3, 271.2]
```

```mermaid
xychart-beta
    title "Dev MAE (Zoomed: Epochs 3-8)"
    x-axis "Epoch" [3, 4, 5, 6, 7, 8]
    y-axis "Dev MAE (seconds)" 260 --> 280
    line [275.2, 272.1, 273.9, 272.4, 274.6, 274.6]
    line [270.8, 270.3, 266.2, 269.0, 270.9, 270.4]
    line [272.3, 268.7, 264.5, 268.2, 269.3, 271.2]
```

**Key observations from the curves:**
- All versions converge rapidly (epochs 1-3), then plateau
- v3 starts lower (300.8 vs 858/923) due to better architecture initialization
- The convergence gap narrows with each version: diminishing returns on architecture alone
- All versions show dev MAE rising after epoch 5 -- overfitting window is consistent

---

## Project Structure

```
.
├── CHALLENGE.md              # Original challenge README (reference)
├── SUBMISSION_TEMPLATE.md    # Writeup template for final submission
├── baseline.py               # Original XGBoost baseline (reference)
├── predict.py                # Submission interface (grader imports this)
├── grade.py                  # Local scoring harness
├── train.py                  # Training script (GPU, MLflow tracked)
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

# Train (GPU recommended, or use notebooks/train_gpu.ipynb on Colab/Kaggle)
python train.py --epochs 10 --batch-size 8192 --lr 5e-4 --loss huber --run-name v3

# Score on dev set
python grade.py
```

---

## What Worked

- **Zone-pair statistics as features:** Zone-pair median alone (296.7s) beats
  XGBoost (351s) with zero ML. Bayesian shrinkage handles sparse pairs.
- **Learned zone embeddings:** The neural net learns spatial relationships
  purely from trip patterns. No external geography needed.
- **Element-wise embedding interactions:** Product captures zone similarity,
  difference captures trip directionality (A->B vs B->A).
- **Residual blocks:** Better gradient flow through deeper combined MLP,
  enabling the model to learn more complex feature interactions.
- **Huber loss (delta=300):** Reduced underprediction bias on long trips
  compared to L1, without the outlier sensitivity of L2.
- **OneCycleLR with warmup:** Stable training from step 1. Avoids NaN
  gradients on randomly initialized embeddings.
- **Chunked data processing:** 37M rows in 2M chunks keeps memory under 6GB,
  enabling training on free-tier Colab/Kaggle (12-16GB RAM).
- **MLflow experiment tracking:** All runs logged with hyperparameters,
  metrics, and artifacts. Archived to HF Hub for persistence.

## What Didn't Work

- **Training on small samples (500k rows):** Converged to ~945s MAE.
  Embeddings need the full 37M rows to learn meaningful zone relationships.
- **CosineAnnealingLR without warmup:** NaN loss in epoch 1 due to unstable
  gradients on random embeddings.
- **Temporal zone-pair features (diminishing returns):** Standalone tb_mean
  achieves 277.9s, but only added ~6s to the neural net -- the model already
  learns temporal-zone interactions implicitly via embeddings.
- **Wider architecture alone:** Going from 372k to 560k parameters gave only
  1.7s improvement. The model overfits after epoch 5 regardless of size.

## Next Steps

- L1 vs Huber A/B test (same v3 architecture)
- Reduced hash buckets (16k -> 8k) to cut parameter count
- LightGBM ensemble
- Hyperparameter sweep (embedding dim, learning rate, batch size)
- Final submission packaging (Dockerfile, writeup)

---

## Constraints

- Inference: <= 200 ms per request on CPU (actual: <1 ms)
- Docker image: <= 2.5 GB (estimated: ~500 MB)
- No external API calls at inference time
- No 2024 data in training
