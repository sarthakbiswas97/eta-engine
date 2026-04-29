# ETA Prediction Model -- Implementation Plan

## Goal
Beat the baseline XGBoost (351s MAE) and zone-pair average (300s MAE) significantly using deep learning with learned zone embeddings + rich features.

**Constraints:** CPU inference < 200ms, Docker < 2.5GB, no 2024 data, no external API calls at inference.

---

## Phase 1: Feature Engineering & Data Pipeline

### 1. Zone-pair precomputed statistics -- Complexity: Medium
- Compute per (pickup, dropoff) pair: mean, median, std, count
- Smoothed fallback: pair -> pickup-zone -> global
- Serve as input features to the neural net AND fallback for rare pairs

### 2. Spatial features from zone centroids -- Complexity: Medium
- Download taxi zone shapefile, compute centroid lat/lon per zone (265 zones)
- Haversine distance, Manhattan distance, bearing between pickup/dropoff
- Same-zone flag
- Borough-level features (from zone lookup CSV)

### 3. Temporal features -- Complexity: Low
- Hour (sin/cos encoded), minute-of-day, day-of-week (sin/cos), month (sin/cos)
- is_weekend, is_rush_hour, is_holiday (2023 US holidays + Christmas/NYE week)
- Day-of-month for holiday proximity

### 4. Build a PyTorch Dataset / DataLoader pipeline -- Complexity: Medium
- Efficient parquet reading with chunked loading (37M rows)
- Separate categorical inputs (zone IDs for embeddings) from continuous features
- Train/dev split already done

---

## Phase 2: Deep Learning Model

### 5. Architecture: Tabular Neural Net with Entity Embeddings -- Complexity: High
- Zone embeddings: Learned embeddings for pickup_zone (dim ~32-64) and dropoff_zone (dim ~32-64)
- Interaction layer: Concat pickup + dropoff embeddings -> MLP for zone-pair interactions
- Continuous features: Temporal + spatial + precomputed stats, batch normed
- Combined MLP: Concat embedding output + continuous -> 512 -> 256 -> 128 -> 1
- Output: Linear, predicting duration in seconds (or log-seconds)
- Activation: ReLU or SiLU, dropout for regularization

### 6. Training setup -- Complexity: Medium
- Loss: MAE (L1Loss) -- directly optimizes evaluation metric
- Optimizer: AdamW with cosine annealing LR schedule
- Batch size: 4096-8192
- Train on full 37M rows, validate on dev
- Early stopping on dev MAE
- Train on Kaggle/Colab GPU

### 7. Training script -- Complexity: Medium
- train.py producing model weights (model.pt) + artifacts (zone stats, centroids, scaler)
- Log metrics per epoch, save best checkpoint by dev MAE

---

## Phase 3: Ensemble (if time permits)

### 8. LightGBM as complementary model -- Complexity: Medium
- Train on same feature set (minus embeddings, plus zone IDs as categoricals)
- Weighted average: alpha * nn_pred + (1-alpha) * lgbm_pred
- Tune alpha on dev set

---

## Phase 4: Advanced Features (diminishing returns)

### 9. OSRM precomputed distances -- Complexity: High
- Precompute 265x265 driving distance/time matrix from zone centroids
- Ship as numpy array

### 10. Weather (NOAA) -- Complexity: High
- Hourly weather for 2023 joined on timestamp
- Risk: eval is 2024, may skip

---

## Phase 5: Ship

### 11. Update predict.py -- Complexity: Low
- Load PyTorch model + lookup tables at import time
- torch.inference_mode() for speed
- Verify < 200ms on CPU

### 12. Update Dockerfile + requirements.txt -- Complexity: Low
- torch CPU-only wheel, bundle weights + lookup tables
- Verify < 2.5 GB

### 13. Validate -- Complexity: Low
- grade.py on dev, pytest tests/, Docker build + run

### 14. README writeup -- Complexity: Low

---

## Execution Order

```
1 -> 2 -> 3  (features, can partially parallelize)
    |
4 -> 5 -> 6 -> 7  (model architecture + training)
    |
8  (LightGBM ensemble, optional)
    |
11 -> 12 -> 13 -> 14  (ship)
```

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU training time exceeds free tier | High | Use 1M sample for arch search, full data for final |
| PyTorch CPU inference > 200ms | Medium | Keep model small (< 5M params), precompute features |
| Docker image too large with PyTorch | High | Use torch CPU-only wheel (~200MB vs ~2GB) |
| Zone embeddings overfit rare zones | Medium | Embedding dropout, min-frequency threshold |
| Eval holiday shift | Medium | Holiday flags, dev is already holiday period |
