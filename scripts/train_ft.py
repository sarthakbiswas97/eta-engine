"""Training script for FT-Transformer model.

Trains the Feature Tokenizer Transformer on the same feature pipeline
as the MLP. Saves checkpoint compatible with ensemble predict.py.

Usage:
    # Quick local test
    PYTHONPATH=. python scripts/train_ft.py --sample 100000 --run-name ft-test

    # Full training on Kaggle
    PYTHONPATH=. python scripts/train_ft.py --sample 10000000 --epochs 10 --batch-size 1024 --lr 1e-4 --run-name ft-v1
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from features.pipeline import FeaturePipeline
from model.ft_transformer import FTTransformer, FTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATS_PATH = DATA_DIR / "zone_pair_stats" / "zone_pair_stats.pkl"
MLRUNS_DIR = ROOT / "mlruns"
SEED = 42


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
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
    """Load parquet and transform in chunks. Returns (cat, cont, targets)."""
    import pandas as pd

    logger.info("Loading %s...", path.name)
    df = pd.read_parquet(path)
    if sample_n is not None and sample_n > 0 and len(df) > sample_n:
        logger.info("Sampling %s from %s rows", f"{sample_n:,}", f"{len(df):,}")
        df = df.sample(n=sample_n, random_state=SEED).reset_index(drop=True)

    total = len(df)
    logger.info("Transforming %s rows...", f"{total:,}")

    cat_chunks, cont_chunks, target_chunks = [], [], []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = df.iloc[start:end].copy()
        cat, cont, targets = pipeline.transform_dataframe(chunk)
        cat_chunks.append(cat)
        cont_chunks.append(cont)
        if targets is not None:
            target_chunks.append(targets)
        del chunk
        gc.collect()
        logger.info("  Processed rows %s-%s / %s", f"{start:,}", f"{end:,}", f"{total:,}")

    del df
    gc.collect()
    return np.concatenate(cat_chunks), np.concatenate(cont_chunks), np.concatenate(target_chunks)


class FTDataset(torch.utils.data.Dataset):
    """Dataset for FT-Transformer: returns (x_numerical, x_categorical, target)."""

    def __init__(self, cat: np.ndarray, cont: np.ndarray, targets: np.ndarray) -> None:
        self.x_cat = torch.from_numpy(cat).long()
        self.x_num = torch.from_numpy(cont).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_num[idx], self.x_cat[idx], self.targets[idx]


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
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (x_num, x_cat, target) in enumerate(loader):
        x_num = x_num.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(x_num, x_cat)
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
    model.eval()
    all_preds, all_targets = [], []

    for x_num, x_cat, target in loader:
        x_num = x_num.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)

        preds = model(x_num, x_cat)
        all_preds.append(preds.cpu())
        all_targets.append(target)

    preds_t = torch.cat(all_preds)
    targets_t = torch.cat(all_targets)
    return torch.mean(torch.abs(preds_t - targets_t)).item()


def save_checkpoint(
    model: nn.Module,
    config: FTConfig,
    norm_params: dict[str, np.ndarray] | None,
    dev_mae: float,
    epoch: int,
    path: Path,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "norm_params": norm_params,
        "dev_mae": dev_mae,
        "epoch": epoch,
        "model_type": "ft_transformer",
    }
    torch.save(checkpoint, path)
    logger.info("Saved checkpoint to %s (dev MAE: %.1f s)", path, dev_mae)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FT-Transformer for ETA prediction")
    parser.add_argument("--sample", type=int, default=10_000_000, help="Train sample (0=full, default 10M)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Max learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--dev-sample", type=int, default=50_000, help="Dev sample size")
    parser.add_argument("--save-every", type=int, default=2, help="Checkpoint every N epochs")
    parser.add_argument("--loss", type=str, default="l1", choices=["l1", "huber", "mse"], help="Loss function")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    args = parser.parse_args()

    set_seed(SEED)
    device = detect_device()
    logger.info("Device: %s", device)

    # MLflow
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment("eta-engine")

    run_name = args.run_name
    if run_name is None:
        sample_tag = f"-{args.sample // 1_000_000}M" if args.sample else "-full"
        run_name = f"ft-{args.loss}{sample_tag}"

    # Load pipeline
    if not STATS_PATH.exists():
        raise SystemExit(f"Missing {STATS_PATH}. Run `python -m features.zone_pair_stats` first.")
    pipeline = FeaturePipeline.from_artifacts(STATS_PATH)

    # Load data
    train_sample = args.sample if args.sample > 0 else None
    train_cat, train_cont, train_targets = load_and_transform(
        pipeline, DATA_DIR / "train.parquet", sample_n=train_sample,
    )
    dev_cat, dev_cont, dev_targets = load_and_transform(
        pipeline, DATA_DIR / "dev.parquet", sample_n=args.dev_sample,
    )

    # Normalization (FT-Transformer benefits from normalized inputs)
    pipeline.fit_normalization(train_cont)
    train_cont = pipeline.normalize(train_cont)
    dev_cont = pipeline.normalize(dev_cont)
    norm_params = pipeline.normalization_params

    # DataLoaders
    pin_memory = device.type == "cuda"
    train_ds = FTDataset(train_cat, train_cont, train_targets)
    dev_ds = FTDataset(dev_cat, dev_cont, dev_targets)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    logger.info("Train: %s rows, %d batches/epoch", f"{len(train_ds):,}", len(train_loader))
    logger.info("Dev: %s rows, %d batches", f"{len(dev_ds):,}", len(dev_loader))

    # Model
    config = FTConfig(n_numerical=train_cont.shape[1])
    model = FTTransformer(config).to(device)
    logger.info("FT-Transformer parameters: %s", f"{model.count_parameters():,}")

    # Loss
    if args.loss == "l1":
        criterion = nn.L1Loss()
        logger.info("Loss: L1 (MAE)")
    elif args.loss == "huber":
        criterion = nn.HuberLoss(delta=300.0)
        logger.info("Loss: Huber(delta=300)")
    else:
        criterion = nn.MSELoss()
        logger.info("Loss: MSE")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000,
    )

    # Checkpoint dir
    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    best_path = ROOT / "ft_model.pt"

    # MLflow run
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_type": "ft_transformer",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "loss": args.loss,
            "patience": args.patience,
            "sample": args.sample or "full",
            "dev_sample": args.dev_sample,
            "train_rows": len(train_ds),
            "dev_rows": len(dev_ds),
            "n_numerical": config.n_numerical,
            "total_params": model.count_parameters(),
            "device": str(device),
            "seed": SEED,
        })
        mlflow.log_params({f"ft_{k}": v for k, v in asdict(config).items()})

        if device.type == "cuda":
            mlflow.log_param("gpu", torch.cuda.get_device_name(0))

        # Training loop
        best_mae = float("inf")
        patience_counter = 0
        training_start = time.time()

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

            mlflow.log_metrics({
                "train_loss": train_loss,
                "dev_mae": dev_mae,
                "lr": lr,
                "epoch_time": elapsed,
            }, step=epoch)

            if epoch % args.save_every == 0:
                epoch_path = checkpoint_dir / f"ft_epoch_{epoch:02d}.pt"
                save_checkpoint(model, config, norm_params, dev_mae, epoch, epoch_path)

            if dev_mae < best_mae:
                best_mae = dev_mae
                patience_counter = 0
                save_checkpoint(model, config, norm_params, dev_mae, epoch, best_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
                    break

        total_time = time.time() - training_start

        mlflow.log_metrics({
            "best_dev_mae": best_mae,
            "best_epoch": epoch - patience_counter,
            "total_training_minutes": total_time / 60,
            "epochs_completed": epoch,
        })
        mlflow.log_artifact(str(best_path))

        logger.info("Best dev MAE: %.1f s", best_mae)
        logger.info("Best checkpoint: %s", best_path)
        logger.info("MLflow run: %s", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
