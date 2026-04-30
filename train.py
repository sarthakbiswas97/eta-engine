"""Training script for ETA prediction model.

Trains the neural net on precomputed features, validates on dev set,
saves best checkpoint. Auto-detects cuda/mps/cpu.

Usage:
    # Quick iteration on small sample (local CPU/MPS)
    python train.py --sample 100000

    # Full training (GPU)
    python train.py

    # Custom hyperparameters
    python train.py --epochs 10 --batch-size 8192 --lr 5e-4
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from features.pipeline import FeaturePipeline
from model.architecture import ETAModel, ModelConfig
from model.dataset import create_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
STATS_PATH = DATA_DIR / "zone_pair_stats" / "zone_pair_stats.pkl"
MODEL_DIR = Path(__file__).parent
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
SEED = 42


def detect_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_transform(
    pipeline: FeaturePipeline,
    path: Path,
    sample_n: int | None = None,
    chunk_size: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load parquet, optionally sample, and transform in chunks.

    Processes data in chunks of chunk_size rows to stay within memory
    limits on constrained environments (e.g., Colab T4 with 12GB RAM).
    """
    import gc

    import pandas as pd

    logger.info("Loading %s...", path.name)
    df = pd.read_parquet(path)
    if sample_n is not None and len(df) > sample_n:
        logger.info("Sampling %s rows from %s", f"{sample_n:,}", f"{len(df):,}")
        df = df.sample(n=sample_n, random_state=SEED).reset_index(drop=True)

    total_rows = len(df)
    logger.info("Transforming %s rows in chunks of %s...", f"{total_rows:,}", f"{chunk_size:,}")

    cat_chunks: list[np.ndarray] = []
    cont_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []

    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk_df = df.iloc[start:end].copy()
        cat, cont, targets = pipeline.transform_dataframe(chunk_df)
        cat_chunks.append(cat)
        cont_chunks.append(cont)
        if targets is not None:
            target_chunks.append(targets)
        del chunk_df
        gc.collect()
        logger.info("  Processed rows %s-%s / %s", f"{start:,}", f"{end:,}", f"{total_rows:,}")

    # Free the original DataFrame
    del df
    gc.collect()

    all_cat = np.concatenate(cat_chunks)
    all_cont = np.concatenate(cont_chunks)
    all_targets = np.concatenate(target_chunks) if target_chunks else None

    del cat_chunks, cont_chunks, target_chunks
    gc.collect()

    return all_cat, all_cont, all_targets


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    log_every: int = 500,
) -> float:
    """Train for one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (pickup, dropoff, cont, target) in enumerate(loader):
        pickup = pickup.to(device, non_blocking=True)
        dropoff = dropoff.to(device, non_blocking=True)
        cont = cont.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(pickup, dropoff, cont)
        loss = criterion(preds, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx > 0 and batch_idx % log_every == 0:
            avg_loss = total_loss / n_batches
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "  Epoch %02d step %d/%d  loss=%.1f  lr=%.2e",
                epoch, batch_idx, len(loader), avg_loss, lr,
            )

    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Evaluate on a dataset, return MAE in seconds."""
    model.eval()
    all_preds = []
    all_targets = []

    for pickup, dropoff, cont, target in loader:
        pickup = pickup.to(device, non_blocking=True)
        dropoff = dropoff.to(device, non_blocking=True)
        cont = cont.to(device, non_blocking=True)

        preds = model(pickup, dropoff, cont)
        all_preds.append(preds.cpu())
        all_targets.append(target)

    preds_t = torch.cat(all_preds)
    targets_t = torch.cat(all_targets)
    mae = torch.mean(torch.abs(preds_t - targets_t)).item()
    return mae


def save_checkpoint(
    model: nn.Module,
    config: ModelConfig,
    norm_params: dict[str, np.ndarray] | None,
    dev_mae: float,
    epoch: int,
    path: Path,
) -> None:
    """Save model checkpoint with metadata."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "norm_params": norm_params,
        "dev_mae": dev_mae,
        "epoch": epoch,
    }
    torch.save(checkpoint, path)
    logger.info("Saved checkpoint to %s (dev MAE: %.1f s)", path, dev_mae)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ETA prediction model")
    parser.add_argument("--sample", type=int, default=None, help="Sample N rows from training data for fast iteration")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Max learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--dev-sample", type=int, default=50_000, help="Dev set sample size for evaluation")
    parser.add_argument("--save-every", type=int, default=2, help="Save checkpoint every N epochs")
    parser.add_argument("--loss", type=str, default="huber", choices=["l1", "huber"], help="Loss function")
    args = parser.parse_args()

    set_seed(SEED)
    device = detect_device()
    logger.info("Device: %s", device)

    # Load pipeline
    if not STATS_PATH.exists():
        raise SystemExit(f"Missing {STATS_PATH}. Run `python -m features.zone_pair_stats` first.")
    pipeline = FeaturePipeline.from_artifacts(STATS_PATH)

    # Load and transform data
    train_cat, train_cont, train_targets = load_and_transform(
        pipeline, DATA_DIR / "train.parquet", sample_n=args.sample,
    )
    dev_cat, dev_cont, dev_targets = load_and_transform(
        pipeline, DATA_DIR / "dev.parquet", sample_n=args.dev_sample,
    )

    # Fit normalization on training data
    pipeline.fit_normalization(train_cont)
    train_cont = pipeline.normalize(train_cont)
    dev_cont = pipeline.normalize(dev_cont)

    # Save normalization params for inference
    norm_params = pipeline.normalization_params

    # Create data loaders
    pin_memory = device.type == "cuda"
    train_loader = create_dataloader(
        train_cat, train_cont, train_targets,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )
    dev_loader = create_dataloader(
        dev_cat, dev_cont, dev_targets,
        batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    logger.info("Train: %s rows, %d batches/epoch", f"{len(train_loader.dataset):,}", len(train_loader))
    logger.info("Dev: %s rows, %d batches", f"{len(dev_loader.dataset):,}", len(dev_loader))

    # Model
    config = ModelConfig(n_continuous=train_cont.shape[1])
    model = ETAModel(config).to(device)
    logger.info("Model parameters: %s", f"{model.count_parameters():,}")

    # Training setup
    if args.loss == "huber":
        criterion = nn.HuberLoss(delta=300.0)
        logger.info("Loss: HuberLoss(delta=300)")
    else:
        criterion = nn.L1Loss()
        logger.info("Loss: L1Loss (MAE)")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # OneCycleLR: warmup from lr/25 -> lr, then cosine decay to lr/1000
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
        div_factor=25.0,       # initial_lr = max_lr / 25
        final_div_factor=1000, # final_lr = max_lr / 1000
    )

    # Checkpoint directory
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Training loop
    best_mae = float("inf")
    patience_counter = 0
    best_path = MODEL_DIR / "model.pt"

    logger.info("Starting training: %d epochs, %d total steps", args.epochs, total_steps)
    logger.info("LR schedule: warmup %.2e -> %.2e -> decay to %.2e",
                args.lr / 25, args.lr, args.lr / 1000)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch,
        )
        dev_mae = evaluate(model, dev_loader, device)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %02d/%02d  train_loss=%.1f  dev_mae=%.1f s  lr=%.2e  time=%.0fs",
            epoch, args.epochs, train_loss, dev_mae, lr, elapsed,
        )

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            epoch_path = CHECKPOINT_DIR / f"epoch_{epoch:02d}.pt"
            save_checkpoint(model, config, norm_params, dev_mae, epoch, epoch_path)

        # Save best checkpoint
        if dev_mae < best_mae:
            best_mae = dev_mae
            patience_counter = 0
            save_checkpoint(model, config, norm_params, dev_mae, epoch, best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
                break

    logger.info("Best dev MAE: %.1f s", best_mae)
    logger.info("Best checkpoint: %s", best_path)
    logger.info("Periodic checkpoints: %s", CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
