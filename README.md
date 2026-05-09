# NYC ETA Engine

**Dev MAE: 253s** | 3-model ensemble | 28% below baseline

Predicting NYC taxi trip duration from pickup/dropoff zones, request
timestamp, and passenger count. Built for the Gobblecube ETA Challenge.

---

## TL;DR

Three models, one feature pipeline, diagnostic-driven iteration.

Started with a zone-pair statistical baseline (297s), built a neural net
with learned zone embeddings (264s), hit a ceiling, ran deep diagnostics
to find the bottleneck (rare-pair underprediction bias), solved it with a
LightGBM ensemble (254s), then added an FT-Transformer built from scratch
for further diversity (253s). Each step was motivated by analyzing what the
previous model got wrong.

---

## Approach

### Problem

Given a ride request (pickup zone, dropoff zone, timestamp, passenger count),
predict trip duration in seconds. Scored on MAE against a held-out 2024
winter-holiday eval set.

### Design Philosophy

**Generic, not geographic.** The model learns all spatial relationships from
trip data via embeddings -- no shapefiles, no hardcoded coordinates, no
NYC-specific knowledge. If the zone IDs mapped to a different city, the model
would learn equally well given the same trip patterns.

### Architecture: 3-Model Ensemble

```
Request -> Feature Pipeline (24 features) -> [NN, LightGBM, FT-Transformer]
                                                        |
                                          0.5 * NN + 0.3 * LGBM + 0.2 * FT
                                                        |
                                                  predicted duration
```

**Why three models?** Each has a different inductive bias. The NN interpolates
through learned embeddings -- great for common routes, but underpredicts rare
ones (bias: -106s). LightGBM partitions the feature space with hard splits --
near-zero bias (-6s) on rare pairs. The FT-Transformer discovers cross-feature
interactions via self-attention -- adds diversity the other two miss.

| Model | Type | Params | Dev MAE | Bias | Strength |
|-------|------|--------|---------|------|----------|
| NN (MLP) | Neural net | 560k | 261s | -106s | Precision on common routes |
| LightGBM | Gradient-boosted trees | 81 trees | 262s | -6s | Rare pairs, low bias |
| FT-Transformer | Tabular transformer | 169k | 267s | -1s | Cross-feature attention |
| **Ensemble** | **Weighted average** | -- | **253s** | **-43s** | **Best of all three** |

### Model 1: Tabular Neural Net

```
Zone Branch:
  pickup_zone  -> Embedding(266, 50)  --\
  dropoff_zone -> Embedding(266, 50)  ---+-- [pu, do, pu*do, pu-do, pair_hash]
  (pu, do)     -> HashEmbed(16384, 16) -/        |
                                           concat(216) -> MLP(128) x2 -> 128-dim

Continuous Branch:
  24 features -> BatchNorm -> MLP(128) x2 -> 128-dim

Combined:
  concat(256) -> ResidualBlock(256) -> ResidualBlock(128) -> Linear(1)
```

560k params. Trained on full 37M rows, Huber loss, OneCycleLR with 10% warmup,
Kaggle T4 GPU. The element-wise product (`pu * do`) captures zone similarity;
the difference (`pu - do`) captures trip directionality (A->B vs B->A).

### Model 2: LightGBM

81 gradient-boosted trees (2.4 MB). Trained on 10M rows with MAE objective.
Zone IDs as native categoricals -- trees split directly on zone values, no
embedding needed. Key finding: 10M subsample outperformed full 37M (cleaner
signal, fewer outlier trips diluting splits).

### Model 3: FT-Transformer (built from scratch)

Feature Tokenizer Transformer following Gorishniy et al. (NeurIPS 2021).
Each feature is projected into a 96-dim token; a [CLS] token aggregates
via 2-layer self-attention with 4 heads. 169k params.

```
24 numerical features -> 24 independent Linear(1, 96) projections
2 zone IDs            -> 2 independent Embedding(266, 96) lookups
[CLS]                 -> learnable parameter
                              |
                     27 tokens x 96-dim
                              |
                    TransformerEncoder (2 layers, 4 heads, pre-norm, GELU)
                              |
                    CLS output -> LayerNorm -> Linear(96, 1)
```

Exported to ONNX for optimized CPU inference. Self-attention discovers which
features interact without hand-designed branches.

### Feature Pipeline (shared across all models)

**24 continuous + 2 categorical = 26 features total.**

1. **Zone-pair statistics (14 features)** -- Bayesian-shrunk mean, median,
   std, p25, p75, IQR, count, time-bucketed mean/median (6 traffic regimes),
   pair rarity signal, zone-level means. Fallback hierarchy handles unseen
   pairs: pair -> pickup-zone -> dropoff-zone -> global.

2. **Temporal (10 features)** -- cyclical sin/cos for hour, day-of-week,
   month. Binary flags for weekend, rush hour, night. Normalized minute-of-day.

3. **Zone embeddings (2 categorical)** -- pickup and dropoff zone IDs, fed
   directly to NN/FT as learned embeddings, and to LightGBM as native
   categoricals.

---

## Results

| Method | Dev MAE | Improvement |
|--------|---------|-------------|
| Global mean prediction | ~580 s | -- |
| XGBoost baseline (6 features) | 351 s | -- |
| Zone-pair median (statistics only) | 297 s | -15% |
| Zone-pair time-bucketed mean | 278 s | -21% |
| Neural net v1 (L1 loss, 19 features) | 272 s | -22% |
| Neural net v2 (Huber, 24 features) | 266 s | -24% |
| Neural net v3 (residual blocks) | 265 s | -25% |
| Neural net v4b (tuned) | 264 s | -25% |
| LightGBM standalone | 262 s | -25% |
| FT-Transformer standalone | 267 s | -24% |
| NN + LightGBM (2-model) | 254 s | -28% |
| **NN + LGBM + FT (3-model)** | **253 s** | **-28%** |

---

## The Journey: What Actually Happened

### Phase 1: Feature engineering beats ML

Zone-pair median (297s) beat the XGBoost baseline (351s) with zero ML.
Lesson: understand the data before reaching for models. The dominant signal
is the pickup-dropoff pair -- everything else is refinement.

### Phase 2: Neural net adds 30s over statistics

Built a tabular NN with learned zone embeddings, temporal features, and
Bayesian-shrunk zone-pair statistics. Iterated through 4 versions (v1-v4b),
gaining 272s -> 264s. Each version was motivated by error analysis of the
previous one -- not random tuning.

### Phase 3: Diagnostics reveal the ceiling

Ran deep diagnostics: parameter health, rare-pair MAE by frequency bucket,
dropout variance, prediction distribution. Found that the NN's -106s bias
was entirely driven by rare zone pairs (MAE 926s for unseen pairs vs 251s
for common ones). No amount of NN tuning could fix this -- embeddings
interpolate, and rare pairs have nothing to interpolate from.

### Phase 4: Ensemble breaks through

LightGBM solved the rare-pair problem (bias: -6s vs -106s). Trees partition
rather than interpolate. The FT-Transformer added further diversity via
self-attention. Ensemble brought 264s -> 253s.

### Phase 5: Inference optimization

FT-Transformer inference was 8.5ms (80% of total). Exported to ONNX Runtime
for fused attention kernels. Also experimented with smaller architecture
(d=96, 2 layers vs d=128, 3 layers) -- 58% fewer params with only marginal
ensemble impact.

---

## What Worked

- **3-model ensemble** -- Different inductive biases (embeddings, tree splits,
  attention) make different errors. Blending reduces MAE by 8s over best
  single model.
- **Zone-pair statistics with Bayesian shrinkage** -- The single most
  impactful feature. Median alone beats XGBoost by 54s.
- **Diagnostic-driven iteration** -- Analyzing rare-pair bias, dropout
  variance, and parameter health prevented wasted experiments and directly
  motivated the LightGBM addition.
- **FT-Transformer from scratch** -- Self-attention discovers cross-feature
  interactions without hand-designed branches. Built following the NeurIPS
  2021 paper, exported to ONNX for production.
- **10M subsample for LightGBM** -- More data hurt (267s on 37M vs 263s on
  10M). Fewer outlier trips = cleaner tree splits.

## What Didn't Work

- **Reducing hash buckets (16k -> 8k):** 13s regression. Hash embeddings are
  47% of NN params but critical for pair-level learning.
- **Removing month features:** 13s regression. Training needs seasonal signal
  even when eval is a single month.
- **Log-target + Huber loss:** Huber(delta=300) in log-space becomes pure MSE
  (errors never exceed 6). Loss/metric mismatch.
- **LightGBM on full 37M rows:** Worse than 10M. More data = more outliers
  diluting tree splits.
- **Prediction rescaling/bias correction:** Only 0.8s gain. Not worth the
  overfitting risk on unseen eval data.

## If I Kept Going

1. **Stacking meta-learner** -- Train a small model on the three models'
   predictions. Adaptively weight based on input (e.g., give LGBM more weight
   for rare pairs).
2. **Larger FT-Transformer** -- Current small config (169k params) achieves
   267s. Original config (406k params, 287s) provides more ensemble diversity
   despite worse standalone MAE. A middle-ground architecture could help.
3. **XGBoost as 4th ensemble member** -- Different tree implementation,
   potentially captures different split patterns.
4. **Holiday/event features** -- The eval set is winter holidays. A
   `is_holiday` feature (generic, not NYC-specific) could capture the traffic
   regime shift.

---

## Project Structure

```
.
├── predict.py                # Submission interface (3-model ensemble)
├── grade.py                  # Local scoring harness
├── train.py                  # MLP training (GPU, MLflow)
├── Dockerfile                # Submission packaging (1.4 GB image)
├── requirements.txt
├── model.pt                  # Trained MLP (560k params, 2.3 MB)
├── lgbm_model.txt            # Trained LightGBM (81 trees, 2.4 MB)
├── ft_model.onnx             # FT-Transformer ONNX (optimized inference)
├── ft_model.pt               # FT-Transformer PyTorch (fallback)
├── ft_norm_params.npz        # FT normalization params
├── features/
│   ├── zone_pair_stats.py    # Bayesian-shrunk zone-pair statistics
│   ├── temporal.py           # Temporal feature extraction
│   └── pipeline.py           # Unified feature pipeline
├── model/
│   ├── architecture.py       # ETAModel (MLP with embeddings)
│   ├── ft_transformer.py     # FT-Transformer (from scratch)
│   └── dataset.py            # PyTorch Dataset/DataLoader
├── scripts/
│   ├── train_lgbm.py         # LightGBM training
│   ├── train_ft.py           # FT-Transformer training
│   ├── export_ft_onnx.py     # ONNX export + verification
│   ├── find_ensemble_weight.py
│   ├── diagnose.py           # Model diagnostics
│   └── train_dev_gap.py      # Generalization analysis
├── notebooks/
│   └── train_gpu.ipynb       # Colab/Kaggle training notebook
└── tests/
    └── test_submission.py    # Submission contract smoke tests
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download data (~500 MB)
python data/download_data.py
python -m features.zone_pair_stats

# Train all three models (or download pre-trained from HF)
python train.py --epochs 10 --batch-size 8192 --lr 5e-4 --loss huber
python scripts/train_lgbm.py --sample 10000000
python scripts/train_ft.py --small --sample 10000000 --epochs 15 --batch-size 2048 --lr 3e-4 --loss l1
python scripts/export_ft_onnx.py

# Score locally
python grade.py

# Docker
docker build -t my-eta .
docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv
```

Pre-trained models: [huggingface.co/sarthakbiswas/eta-engine](https://huggingface.co/sarthakbiswas/eta-engine)

---

## Constraints

- Inference: <= 200 ms/request on CPU (actual: ~4ms with ONNX)
- Docker image: <= 2.5 GB (actual: 1.4 GB)
- Total model weights: 6.1 MB
- No external API calls at inference time
- No 2024 data in training
