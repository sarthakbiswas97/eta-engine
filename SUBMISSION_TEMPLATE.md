# Submission Writeup

---

## Your final score

Dev MAE: **252.7 s** (3-model ensemble on full 1.23M dev rows)

---

## Your approach, in one paragraph

A 3-model ensemble of a tabular MLP (560k params, Huber loss, learned zone
embeddings with hash-based pair encoding), a LightGBM (81 trees, MAE
objective, zone IDs as native categoricals), and an FT-Transformer (406k
params, self-attention over per-feature tokens, built from scratch following
Gorishniy et al. 2021). All three models share the same 24-feature pipeline:
14 zone-pair statistics with Bayesian shrinkage (including time-bucketed
means, IQR, and a pair-rarity signal) and 10 temporal features (cyclical
encodings, binary flags). The key insight was that the NN has high precision
on common routes but a -106s underprediction bias on rare/long trips, while
LightGBM has near-zero bias (-6s) because trees partition rather than
interpolate. Adding the FT-Transformer with its +65s positive bias further
improved diversity. Ensemble weights (0.6/0.2/0.2) were optimized via grid
search on the full dev set. Trained on Kaggle T4 GPUs (NN on 37M rows, LGBM
and FT on 10M subsamples).

## What you tried that didn't work

1. **Reducing hash embedding buckets (16k -> 8k):** Expected to cut
   memorization by halving parameters. Instead caused a 13s regression -- hash
   embeddings were the most important component, not the most wasteful. Same
   session, removing month features (constant in dev/eval) also regressed 13s
   because training still needed seasonal signal to learn zone representations.
   Lesson: changes that look logical from eval-set analysis can backfire when
   they affect the training-time learning dynamics.

2. **Log-target training with Huber loss:** Hypothesized that log-space would
   naturally handle the right-skewed target distribution. But Huber(delta=300)
   in log-space (where errors range 0-6) becomes pure MSE -- the delta never
   triggers. Result: 265s vs 264s for standard Huber. The loss/metric mismatch
   was subtle and only became obvious after analyzing the gradient behavior.

3. **LightGBM on full 37M rows:** Expected more data to help trees learn rare
   pairs. Instead, MAE went from 263s (10M) to 267-274s (37M). More data
   included more outlier trips that diluted tree splits. Similarly, prediction
   rescaling and bias correction gave only -0.8s and -0.4s respectively --
   the systematic bias is too interleaved with real prediction errors to
   correct post-hoc.

## Where AI tooling sped you up most

Claude Code (via CLI) was used throughout. The highest-leverage moments were:

- **Diagnostic-driven iteration:** Claude ran full error analysis (train-dev
  gap, rare-pair MAE by frequency bucket, parameter health, dropout variance)
  directly from the training artifacts. This revealed that the NN's -106s bias
  was concentrated in rare zone pairs -- a finding that directly motivated
  adding LightGBM (which solved it). Without this analysis, I would have kept
  tuning the NN architecture, which was already at its ceiling.

- **FT-Transformer from scratch:** Claude translated the paper's architecture
  into a clean PyTorch implementation (~150 lines) with per-feature
  tokenization, [CLS] token, and pre-norm transformer encoder. This would
  have taken significantly longer to implement and debug manually.

- **Feature engineering pipeline:** The zone-pair statistics module (Bayesian
  shrinkage, temporal bucketing, fallback hierarchy) and the chunked data
  processing (2M rows at a time to stay within Kaggle's 13GB RAM) were
  designed and implemented interactively, with Claude handling the
  memory-optimization concerns.

Where it fell short: Claude initially batched multiple architecture changes
(dropout + hash buckets + feature removal) into one step, making it
impossible to isolate which caused the 13s regression. After feedback, it
switched to one-change-at-a-time with separate commits -- a workflow
correction that saved significant debugging time.

## Next experiments

If I kept going, the next experiments in priority order:

1. **Larger FT-Transformer (d_token=192, 4 layers):** Current FT gets 287s
   standalone. More capacity could bring it closer to the NN's 264s, and a
   stronger FT in the ensemble could push below 250s.

2. **Feature interaction engineering:** The diagnostic showed specific
   feature pairs that matter (e.g., pair_rarity x pair_mean, hour x zone_pair).
   Explicit interaction features could help all three models.

3. **XGBoost as a fourth ensemble member:** Different tree implementation
   from LightGBM, potentially captures different split patterns. Low effort,
   moderate expected gain.

4. **Stacking (meta-learner):** Instead of fixed weights, train a small model
   on the three models' predictions. Could adaptively weight models based on
   input characteristics (e.g., give LGBM more weight for rare pairs).

## How to reproduce

```bash
git clone https://github.com/sarthakbiswas97/eta-engine.git
cd eta-engine

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download data (~500 MB, one-time)
python data/download_data.py

# Compute zone-pair stats (~2 min)
python -m features.zone_pair_stats

# Train MLP (GPU, ~90 min on 37M rows)
python train.py --epochs 10 --batch-size 8192 --lr 5e-4 --patience 3 \
    --loss huber --run-name v4b-huber

# Train LightGBM (CPU, ~5 min on 10M rows)
python scripts/train_lgbm.py --sample 10000000 --dev-sample 100000 \
    --run-name lgbm-v1

# Train FT-Transformer (GPU, ~90 min on 10M rows)
python scripts/train_ft.py --sample 10000000 --epochs 15 --batch-size 2048 \
    --lr 3e-4 --patience 4 --loss l1 --run-name ft-v3

# Score on dev set
python grade.py

# Docker build and test
docker build -t my-eta .
docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv
```

Pre-trained models are also available on HuggingFace:
[sarthakbiswas/eta-engine](https://huggingface.co/sarthakbiswas/eta-engine)

---

_Total time spent on this challenge: ~30 hours across 7 days._
