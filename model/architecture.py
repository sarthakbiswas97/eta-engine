"""Neural network architecture for ETA prediction.

Tabular neural net with:
  - Separate learned embeddings for pickup and dropoff zones
  - Hash-based zone-pair embedding for direct pair-level learning
  - Continuous feature branch with batch normalization
  - Combined MLP predicting trip duration in seconds
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters."""

    num_zones: int = 266           # zone IDs 1-265, index 0 = padding/unknown
    zone_embed_dim: int = 50       # embedding dim per zone (pickup and dropoff separate)
    pair_hash_buckets: int = 16384 # hash buckets for zone-pair embedding
    pair_embed_dim: int = 16       # embedding dim for hashed zone pair
    n_continuous: int = 24         # number of continuous input features
    zone_mlp_dim: int = 64         # output dim of zone interaction MLP
    cont_mlp_dim: int = 64         # output dim of continuous branch
    hidden_dims: tuple[int, ...] = (256, 128, 64)  # combined MLP hidden layers
    dropout: float = 0.2
    embed_dropout: float = 0.1


class ZonePairHasher(nn.Module):
    """Hash-based embedding for (pickup, dropoff) zone pairs.

    Maps arbitrary zone pairs to a fixed-size embedding table using
    a deterministic hash. Handles unseen pairs naturally through
    hash collisions -- no enumeration needed.
    """

    def __init__(self, num_buckets: int, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _hash_pair(self, pickup: torch.Tensor, dropoff: torch.Tensor) -> torch.Tensor:
        """Deterministic hash of (pickup, dropoff) to bucket index."""
        # Large primes for mixing, mod by num_buckets
        return ((pickup.long() * 7919 + dropoff.long() * 104729) % self.num_buckets)

    def forward(self, pickup: torch.Tensor, dropoff: torch.Tensor) -> torch.Tensor:
        bucket_idx = self._hash_pair(pickup, dropoff)
        return self.dropout(self.embedding(bucket_idx))


class ETAModel(nn.Module):
    """Tabular neural net for trip duration prediction.

    Architecture:
        Zone branch:
            pickup_zone -> Embedding(266, 50)  --|
            dropoff_zone -> Embedding(266, 50) --|-- concat --> MLP --> 64-dim
            (pu, do) -> HashEmbed(16384, 16)   --|

        Continuous branch:
            19 features -> BatchNorm -> Linear -> ReLU -> 64-dim

        Combined:
            concat(zone_64, cont_64) = 128 -> MLP(256, 128, 64) -> Linear(1)
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config

        # Zone embeddings (separate tables for pickup and dropoff)
        self.pickup_embed = nn.Embedding(c.num_zones, c.zone_embed_dim, padding_idx=0)
        self.dropoff_embed = nn.Embedding(c.num_zones, c.zone_embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(c.embed_dropout)

        # Zone-pair hash embedding
        self.pair_hasher = ZonePairHasher(c.pair_hash_buckets, c.pair_embed_dim, c.embed_dropout)

        # Zone interaction MLP:
        # [pu_emb, do_emb, pu*do, pu-do, pair_hash] -> zone_mlp_dim
        zone_input_dim = c.zone_embed_dim * 4 + c.pair_embed_dim
        self.zone_mlp = nn.Sequential(
            nn.Linear(zone_input_dim, c.zone_mlp_dim),
            nn.SiLU(),
            nn.Dropout(c.dropout),
        )

        # Continuous branch: n_continuous -> cont_mlp_dim
        self.cont_bn = nn.BatchNorm1d(c.n_continuous)
        self.cont_mlp = nn.Sequential(
            nn.Linear(c.n_continuous, c.cont_mlp_dim),
            nn.SiLU(),
            nn.Dropout(c.dropout),
        )

        # Combined MLP
        combined_dim = c.zone_mlp_dim + c.cont_mlp_dim
        layers: list[nn.Module] = []
        prev_dim = combined_dim
        for hidden_dim in c.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(c.dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.combined_mlp = nn.Sequential(*layers)

    def forward(
        self,
        pickup_zone: torch.Tensor,
        dropoff_zone: torch.Tensor,
        continuous: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pickup_zone: (batch,) int tensor, zone IDs 1-265
            dropoff_zone: (batch,) int tensor, zone IDs 1-265
            continuous: (batch, n_continuous) float tensor

        Returns:
            (batch,) float tensor -- predicted duration in seconds
        """
        # Zone branch
        pu_emb = self.embed_drop(self.pickup_embed(pickup_zone))   # (batch, 50)
        do_emb = self.embed_drop(self.dropoff_embed(dropoff_zone)) # (batch, 50)
        pair_emb = self.pair_hasher(pickup_zone, dropoff_zone)     # (batch, 16)

        # Element-wise interactions: similarity (product) and direction (difference)
        emb_product = pu_emb * do_emb    # (batch, 50)
        emb_diff = pu_emb - do_emb       # (batch, 50)

        zone_features = torch.cat(
            [pu_emb, do_emb, emb_product, emb_diff, pair_emb], dim=1,
        )  # (batch, 216)
        zone_out = self.zone_mlp(zone_features)  # (batch, zone_mlp_dim)

        # Continuous branch
        cont_normed = self.cont_bn(continuous)    # (batch, 19)
        cont_out = self.cont_mlp(cont_normed)     # (batch, 64)

        # Combined
        combined = torch.cat([zone_out, cont_out], dim=1)  # (batch, 128)
        output = self.combined_mlp(combined)                # (batch, 1)

        return output.squeeze(1)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
