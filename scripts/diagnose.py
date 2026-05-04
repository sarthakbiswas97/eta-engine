"""Deep diagnostic: parameter health, rare-data behavior, feature sparsity, regularization.

Usage:
    PYTHONPATH=. python scripts/diagnose.py
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features.pipeline import FeaturePipeline, CONTINUOUS_FEATURES
from features.zone_pair_stats import load_stats
from model.architecture import ETAModel, ModelConfig
from model.dataset import create_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def load_model() -> tuple[ETAModel, dict, FeaturePipeline]:
    checkpoint = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = ETAModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    pipeline = FeaturePipeline.from_artifacts(ROOT / "data" / "zone_pair_stats" / "zone_pair_stats.pkl")
    pipeline.set_normalization_params(checkpoint["norm_params"])
    return model, checkpoint, pipeline


def diagnose_parameters(model: ETAModel) -> None:
    """Check weight distributions for dead neurons, saturation, collapse."""
    print("\n" + "=" * 70)
    print("1. PARAMETER DISTRIBUTION CHECK")
    print("=" * 70)

    print(f"\n{'Layer':<45} {'Shape':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'|Dead|':>7}")
    print("-" * 120)

    total_params = 0
    total_near_zero = 0
    total_saturated = 0

    for name, param in model.named_parameters():
        data = param.data
        n = data.numel()
        total_params += n

        mean = data.mean().item()
        std = data.std().item()
        mn = data.min().item()
        mx = data.max().item()

        # Dead params: |w| < 1e-6
        near_zero = (data.abs() < 1e-6).sum().item()
        total_near_zero += near_zero
        dead_pct = near_zero / n * 100

        # Saturated: |w| > 5 (unusually large)
        saturated = (data.abs() > 5.0).sum().item()
        total_saturated += saturated

        shape_str = str(tuple(data.shape))
        print(f"{name:<45} {shape_str:<20} {mean:>+8.4f} {std:>8.4f} {mn:>+8.3f} {mx:>+8.3f} {dead_pct:>6.1f}%")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Near-zero (|w| < 1e-6): {total_near_zero:,} ({total_near_zero/total_params*100:.2f}%)")
    print(f"Saturated (|w| > 5): {total_saturated:,} ({total_saturated/total_params*100:.2f}%)")

    # Embedding analysis
    print("\n--- Embedding Health ---")
    for name, emb_layer in [("pickup_embed", model.pickup_embed), ("dropoff_embed", model.dropoff_embed)]:
        w = emb_layer.weight.data
        norms = w.norm(dim=1)  # L2 norm per zone
        # Exclude padding idx 0
        norms_active = norms[1:]
        print(f"\n{name} (266 zones, dim=50):")
        print(f"  Norm: mean={norms_active.mean():.3f}, std={norms_active.std():.3f}, min={norms_active.min():.3f}, max={norms_active.max():.3f}")
        # Check if padding is truly zero
        print(f"  Padding (idx 0) norm: {norms[0]:.6f}")
        # Cosine similarity between random pairs of zone embeddings
        cos_sims = []
        for _ in range(1000):
            i, j = np.random.randint(1, 266, size=2)
            cos = torch.nn.functional.cosine_similarity(w[i].unsqueeze(0), w[j].unsqueeze(0)).item()
            cos_sims.append(cos)
        cos_sims = np.array(cos_sims)
        print(f"  Random pair cosine sim: mean={cos_sims.mean():.3f}, std={cos_sims.std():.3f}")
        if cos_sims.mean() > 0.8:
            print("  >> WARNING: Embeddings are collapsing (high avg similarity)")
        elif cos_sims.std() < 0.1:
            print("  >> WARNING: Low variance in similarities -- embeddings may not be differentiating zones")

    # Hash embedding
    hash_w = model.pair_hasher.embedding.weight.data
    hash_norms = hash_w.norm(dim=1)
    near_zero_buckets = (hash_norms < 0.01).sum().item()
    print(f"\nHash embedding (16384 buckets, dim=16):")
    print(f"  Norm: mean={hash_norms.mean():.3f}, std={hash_norms.std():.3f}, min={hash_norms.min():.3f}, max={hash_norms.max():.3f}")
    print(f"  Near-zero buckets: {near_zero_buckets} ({near_zero_buckets/16384*100:.1f}%)")
    print(f"  Parameters: {hash_w.numel():,} ({hash_w.numel()/sum(p.numel() for p in model.parameters())*100:.1f}% of model)")

    # BatchNorm stats
    print("\n--- BatchNorm Running Stats ---")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            rm = module.running_mean
            rv = module.running_var
            if rm is not None:
                dead_bn = (rv < 1e-6).sum().item()
                print(f"  {name}: features={module.num_features}, var_min={rv.min():.4f}, var_max={rv.max():.4f}, dead_features(var<1e-6)={dead_bn}")


def diagnose_unseen_rare(pipeline: FeaturePipeline, model: ETAModel, stats: dict) -> None:
    """Check model behavior on unseen and rare zone pairs."""
    print("\n" + "=" * 70)
    print("2. MODEL BEHAVIOR ON UNSEEN / RARE DATA")
    print("=" * 70)

    dev_df = pd.read_parquet(ROOT / "data" / "dev.parquet")
    train_pairs = set(stats["pair"].keys())

    # Classify dev rows by pair frequency
    dev_df = dev_df.assign(
        pair_key=list(zip(dev_df["pickup_zone"], dev_df["dropoff_zone"])),
    )
    dev_df = dev_df.assign(
        pair_in_train=dev_df["pair_key"].isin(train_pairs),
    )

    # Get pair counts from training
    pair_counts = {k: v["pair_count"] for k, v in stats["pair"].items()}
    dev_df = dev_df.assign(
        train_count=dev_df["pair_key"].map(pair_counts).fillna(0).astype(int),
    )

    # Categorize
    dev_df = dev_df.assign(
        freq_category=pd.cut(
            dev_df["train_count"],
            bins=[-1, 0, 10, 100, 1000, 10000, float("inf")],
            labels=["unseen", "1-10", "11-100", "101-1k", "1k-10k", "10k+"],
        ),
    )

    # Batch predict
    cat_all, cont_all, targets_all = [], [], []
    chunk_size = 500_000
    for start in range(0, len(dev_df), chunk_size):
        chunk = dev_df.iloc[start:start + chunk_size].copy()
        cat, cont, targets = pipeline.transform_dataframe(chunk)
        cont = pipeline.normalize(cont)
        cat_all.append(cat)
        cont_all.append(cont)
        if targets is not None:
            target_all_chunk = targets
            targets_all.append(target_all_chunk)

    cat_np = np.concatenate(cat_all)
    cont_np = np.concatenate(cont_all)
    targets_np = np.concatenate(targets_all)

    # Run model
    loader = create_dataloader(cat_np, cont_np, targets_np, batch_size=16384, shuffle=False, num_workers=0, pin_memory=False)
    preds_list = []
    with torch.no_grad():
        for pickup, dropoff, cont_batch, _ in loader:
            p = model(pickup, dropoff, cont_batch)
            preds_list.append(p.cpu().numpy())
    preds = np.concatenate(preds_list)

    dev_df = dev_df.assign(pred=preds, actual=targets_np)
    dev_df = dev_df.assign(
        error=dev_df["pred"] - dev_df["actual"],
        abs_error=np.abs(dev_df["pred"] - dev_df["actual"]),
    )

    # Report by frequency category
    print(f"\n{'Category':<12} {'N rows':>10} {'MAE':>8} {'Bias':>8} {'Med AE':>8} {'P95 AE':>8} {'Avg Dur':>8}")
    print("-" * 75)
    for cat in ["unseen", "1-10", "11-100", "101-1k", "1k-10k", "10k+"]:
        subset = dev_df[dev_df["freq_category"] == cat]
        if len(subset) == 0:
            continue
        mae = subset["abs_error"].mean()
        bias = subset["error"].mean()
        med = subset["abs_error"].median()
        p95 = subset["abs_error"].quantile(0.95)
        avg_dur = subset["actual"].mean()
        print(f"{cat:<12} {len(subset):>10,} {mae:>8.1f} {bias:>+8.1f} {med:>8.1f} {p95:>8.1f} {avg_dur:>8.1f}")

    total_mae = dev_df["abs_error"].mean()
    print(f"{'ALL':<12} {len(dev_df):>10,} {total_mae:>8.1f} {dev_df['error'].mean():>+8.1f} {dev_df['abs_error'].median():>8.1f} {dev_df['abs_error'].quantile(0.95):>8.1f} {dev_df['actual'].mean():>8.1f}")

    # Unseen pair analysis
    unseen = dev_df[~dev_df["pair_in_train"]]
    seen = dev_df[dev_df["pair_in_train"]]
    print(f"\nUnseen pairs: {unseen['pair_key'].nunique()} unique pairs, {len(unseen):,} rows ({len(unseen)/len(dev_df)*100:.1f}%)")
    print(f"Seen pairs:   {seen['pair_key'].nunique()} unique pairs, {len(seen):,} rows ({len(seen)/len(dev_df)*100:.1f}%)")

    # Hash collision analysis
    print("\n--- Hash Collision Analysis ---")
    num_buckets = 16384
    pair_to_bucket = {}
    bucket_to_pairs: dict[int, list] = {}
    for pu, do in train_pairs:
        bucket = (pu * 7919 + do * 104729) % num_buckets
        pair_to_bucket[(pu, do)] = bucket
        bucket_to_pairs.setdefault(bucket, []).append((pu, do))

    collision_counts = [len(v) for v in bucket_to_pairs.values()]
    used_buckets = len(bucket_to_pairs)
    empty_buckets = num_buckets - used_buckets
    max_collision = max(collision_counts) if collision_counts else 0
    avg_collision = np.mean(collision_counts) if collision_counts else 0

    print(f"Total train pairs: {len(train_pairs):,}")
    print(f"Used buckets: {used_buckets:,} / {num_buckets} ({used_buckets/num_buckets*100:.1f}%)")
    print(f"Empty buckets: {empty_buckets:,} ({empty_buckets/num_buckets*100:.1f}%)")
    print(f"Avg pairs per used bucket: {avg_collision:.1f}")
    print(f"Max pairs in one bucket: {max_collision}")
    collision_dist = Counter(collision_counts)
    print(f"Buckets with 1 pair: {collision_dist.get(1, 0)}")
    print(f"Buckets with 2-3 pairs: {sum(collision_dist.get(i, 0) for i in range(2, 4))}")
    print(f"Buckets with 4+ pairs: {sum(v for k, v in collision_dist.items() if k >= 4)}")

    # Check if high-collision buckets hurt MAE
    dev_df = dev_df.assign(
        hash_bucket=((dev_df["pickup_zone"] * 7919 + dev_df["dropoff_zone"] * 104729) % num_buckets),
    )
    bucket_pair_count = dev_df["hash_bucket"].map(
        {b: len(ps) for b, ps in bucket_to_pairs.items()}
    ).fillna(0).astype(int)
    dev_df = dev_df.assign(bucket_collision_count=bucket_pair_count)

    print(f"\n{'Collisions':<15} {'N rows':>10} {'MAE':>8} {'Bias':>8}")
    print("-" * 45)
    for label, lo, hi in [("0 (unseen)", -1, 0), ("1 (unique)", 1, 1), ("2-3", 2, 3), ("4+", 4, 999)]:
        subset = dev_df[(dev_df["bucket_collision_count"] >= lo) & (dev_df["bucket_collision_count"] <= hi)]
        if len(subset) == 0:
            continue
        print(f"{label:<15} {len(subset):>10,} {subset['abs_error'].mean():>8.1f} {subset['error'].mean():>+8.1f}")


def diagnose_features(pipeline: FeaturePipeline) -> None:
    """Check feature sparsity and cardinality."""
    print("\n" + "=" * 70)
    print("3. FEATURE SPARSITY + CARDINALITY")
    print("=" * 70)

    dev_df = pd.read_parquet(ROOT / "data" / "dev.parquet")
    logger.info("Transforming dev data for feature analysis...")

    # Transform without normalization to see raw values
    pipeline_raw = FeaturePipeline.from_artifacts(ROOT / "data" / "zone_pair_stats" / "zone_pair_stats.pkl")
    cat, cont_raw, targets = pipeline_raw.transform_dataframe(dev_df.iloc[:500_000].copy())

    feature_names = CONTINUOUS_FEATURES
    print(f"\n{'Feature':<22} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Zero%':>7} {'NaN%':>6} {'Uniq':>8}")
    print("-" * 90)

    for i, name in enumerate(feature_names):
        col = cont_raw[:, i]
        mean = np.nanmean(col)
        std = np.nanstd(col)
        mn = np.nanmin(col)
        mx = np.nanmax(col)
        zero_pct = (col == 0).sum() / len(col) * 100
        nan_pct = np.isnan(col).sum() / len(col) * 100
        unique = len(np.unique(col[~np.isnan(col)]))
        flag = ""
        if zero_pct > 50:
            flag = " << SPARSE"
        if std < 1e-6:
            flag = " << CONSTANT"
        if nan_pct > 1:
            flag += " << NaN"
        print(f"{name:<22} {mean:>10.2f} {std:>10.2f} {mn:>10.2f} {mx:>10.2f} {zero_pct:>6.1f}% {nan_pct:>5.1f}% {unique:>8,}{flag}")

    # Zone cardinality
    print(f"\n--- Zone Cardinality ---")
    pu_zones = dev_df["pickup_zone"].nunique()
    do_zones = dev_df["dropoff_zone"].nunique()
    pair_count = dev_df.groupby(["pickup_zone", "dropoff_zone"]).size()
    print(f"Unique pickup zones: {pu_zones}")
    print(f"Unique dropoff zones: {do_zones}")
    print(f"Unique zone pairs: {len(pair_count):,}")
    print(f"Pair frequency: min={pair_count.min()}, median={pair_count.median():.0f}, max={pair_count.max()}, mean={pair_count.mean():.1f}")

    # How many pairs have < 10 trips in dev
    sparse_pairs = (pair_count < 10).sum()
    print(f"Pairs with < 10 dev trips: {sparse_pairs:,} ({sparse_pairs/len(pair_count)*100:.1f}%)")


def diagnose_regularization(model: ETAModel, pipeline: FeaturePipeline) -> None:
    """Check if regularization is actually working."""
    print("\n" + "=" * 70)
    print("4. REGULARIZATION ANALYSIS")
    print("=" * 70)

    # Weight magnitude analysis
    print("\n--- Weight Magnitude by Layer Group ---")
    groups = {
        "pickup_embed": [],
        "dropoff_embed": [],
        "pair_hash": [],
        "zone_mlp": [],
        "cont_mlp": [],
        "combined_mlp": [],
    }
    for name, param in model.named_parameters():
        if "pickup_embed" in name:
            groups["pickup_embed"].append(param.data)
        elif "dropoff_embed" in name:
            groups["dropoff_embed"].append(param.data)
        elif "pair_hasher" in name:
            groups["pair_hash"].append(param.data)
        elif "zone_mlp" in name:
            groups["zone_mlp"].append(param.data)
        elif "cont_mlp" in name or "cont_bn" in name:
            groups["cont_mlp"].append(param.data)
        elif "combined_mlp" in name:
            groups["combined_mlp"].append(param.data)

    print(f"\n{'Group':<20} {'Params':>10} {'L2 Norm':>10} {'Mean |w|':>10} {'Max |w|':>10}")
    print("-" * 65)
    for group_name, tensors in groups.items():
        if not tensors:
            continue
        all_w = torch.cat([t.flatten() for t in tensors])
        n = all_w.numel()
        l2 = all_w.norm(2).item()
        mean_abs = all_w.abs().mean().item()
        max_abs = all_w.abs().max().item()
        print(f"{group_name:<20} {n:>10,} {l2:>10.2f} {mean_abs:>10.4f} {max_abs:>10.4f}")

    # Effective dropout check: run same input twice in train mode
    print("\n--- Dropout Effectiveness ---")
    dev_df = pd.read_parquet(ROOT / "data" / "dev.parquet")
    sample = dev_df.sample(n=1000, random_state=SEED)
    cat, cont, targets = pipeline.transform_dataframe(sample.copy())
    cont = pipeline.normalize(cont)

    pickup_t = torch.tensor(cat[:, 0], dtype=torch.long)
    dropoff_t = torch.tensor(cat[:, 1], dtype=torch.long)
    cont_t = torch.tensor(cont, dtype=torch.float32)

    # Eval mode: deterministic
    model.eval()
    with torch.no_grad():
        eval_pred = model(pickup_t, dropoff_t, cont_t).numpy()

    # Train mode: stochastic (dropout active)
    model.train()
    train_preds = []
    for _ in range(10):
        with torch.no_grad():
            p = model(pickup_t, dropoff_t, cont_t).numpy()
            train_preds.append(p)
    model.eval()

    train_preds = np.stack(train_preds)  # (10, 1000)
    pred_std = train_preds.std(axis=0)  # per-sample std across dropout runs
    pred_mean_std = pred_std.mean()
    pred_max_std = pred_std.max()

    # Compare eval vs train-mode mean
    train_mean = train_preds.mean(axis=0)
    eval_train_diff = np.abs(eval_pred - train_mean).mean()

    print(f"Dropout variance (10 forward passes, train mode):")
    print(f"  Mean std per sample: {pred_mean_std:.2f}s")
    print(f"  Max std per sample:  {pred_max_std:.2f}s")
    print(f"  Eval vs train-mean diff: {eval_train_diff:.2f}s")

    if pred_mean_std < 5:
        print("  >> LOW dropout effect -- dropout may not be regularizing meaningfully")
    elif pred_mean_std < 20:
        print("  >> MODERATE dropout effect -- reasonable")
    else:
        print("  >> HIGH dropout effect -- predictions are very sensitive to dropout")

    # Check output range
    print("\n--- Output Range Check ---")
    print(f"Eval predictions: min={eval_pred.min():.1f}, max={eval_pred.max():.1f}, mean={eval_pred.mean():.1f}, std={eval_pred.std():.1f}")
    actual = targets
    print(f"Actual durations: min={actual.min():.1f}, max={actual.max():.1f}, mean={actual.mean():.1f}, std={actual.std():.1f}")
    print(f"Pred/Actual std ratio: {eval_pred.std()/actual.std():.3f}")
    if eval_pred.std() / actual.std() < 0.7:
        print("  >> VARIANCE COLLAPSE: model predicts a narrower range than reality")
    negative_preds = (eval_pred < 0).sum()
    print(f"Negative predictions: {negative_preds} ({negative_preds/len(eval_pred)*100:.1f}%)")


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model, checkpoint, pipeline = load_model()
    stats = load_stats(ROOT / "data" / "zone_pair_stats" / "zone_pair_stats.pkl")

    diagnose_parameters(model)
    diagnose_unseen_rare(pipeline, model, stats)
    diagnose_features(pipeline)
    diagnose_regularization(model, pipeline)

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
