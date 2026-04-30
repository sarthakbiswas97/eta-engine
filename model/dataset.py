"""PyTorch Dataset and DataLoader factory for ETA prediction.

Wraps precomputed numpy arrays (from FeaturePipeline) into a PyTorch
Dataset. DataLoader handles batching, shuffling, and parallel loading.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ETADataset(Dataset):
    """Dataset holding precomputed categorical, continuous, and target arrays.

    Args:
        categorical: (n, 2) int32 array -- [pickup_zone, dropoff_zone]
        continuous: (n, n_continuous) float32 array
        targets: (n,) float32 array or None (inference mode)
    """

    def __init__(
        self,
        categorical: np.ndarray,
        continuous: np.ndarray,
        targets: np.ndarray | None = None,
    ) -> None:
        self._cat = torch.from_numpy(categorical).long()
        self._cont = torch.from_numpy(continuous).float()
        self._targets = torch.from_numpy(targets).float() if targets is not None else None

    def __len__(self) -> int:
        return len(self._cat)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        pickup = self._cat[idx, 0]
        dropoff = self._cat[idx, 1]
        cont = self._cont[idx]
        target = self._targets[idx] if self._targets is not None else torch.tensor(0.0)
        return pickup, dropoff, cont, target


def create_dataloader(
    categorical: np.ndarray,
    continuous: np.ndarray,
    targets: np.ndarray | None = None,
    batch_size: int = 4096,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader from precomputed feature arrays."""
    dataset = ETADataset(categorical, continuous, targets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
