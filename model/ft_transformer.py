"""Feature Tokenizer Transformer for tabular regression.

Implements the FT-Transformer architecture from:
  "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., NeurIPS 2021)

Core idea: each input feature (numerical or categorical) is projected into a
d-dimensional token. A [CLS] token is prepended, and the full sequence is
processed by a standard Transformer encoder. The [CLS] output is used for
regression.

This captures cross-feature interactions via self-attention -- every feature
can attend to every other feature, discovering interaction patterns that
hand-designed branches (like our MLP's zone vs continuous split) might miss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FTConfig:
    """FT-Transformer hyperparameters."""

    n_numerical: int = 24          # number of numerical input features
    n_categorical: int = 2         # number of categorical features (pickup, dropoff)
    cat_cardinalities: tuple[int, ...] = (266, 266)  # vocab size per categorical
    d_token: int = 128             # token / embedding dimension
    n_blocks: int = 3              # number of transformer layers
    n_heads: int = 8               # attention heads
    ffn_multiplier: float = 4 / 3  # FFN hidden dim = d_token * ffn_multiplier
    attention_dropout: float = 0.2
    ffn_dropout: float = 0.1
    residual_dropout: float = 0.0


class NumericalTokenizer(nn.Module):
    """Project each numerical feature into a d_token-dimensional token.

    Each feature gets its own independent linear projection:
      token_j = W_j * x_j + b_j

    This is NOT a shared projection -- each feature learns its own
    mapping into the token space, so the model can learn feature-specific
    scales and representations.
    """

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        self._init_weights()

    def _init_weights(self) -> None:
        # Xavier-uniform per feature, following the paper's approach
        d = self.weight.shape[1]
        std = 1.0 / math.sqrt(d)
        nn.init.uniform_(self.weight, -std, std)
        nn.init.uniform_(self.bias, -std, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, n_features) float tensor

        Returns:
            (batch, n_features, d_token) -- one token per feature
        """
        # x: (batch, n_features) -> (batch, n_features, 1)
        # weight: (n_features, d_token)
        # result: (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class CategoricalTokenizer(nn.Module):
    """Embed each categorical feature into a d_token-dimensional token.

    Each categorical feature gets its own embedding table.
    """

    def __init__(self, cardinalities: tuple[int, ...], d_token: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, d_token, padding_idx=0)
            for card in cardinalities
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, n_categorical) long tensor

        Returns:
            (batch, n_categorical, d_token) -- one token per categorical feature
        """
        tokens = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(tokens, dim=1)


class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for tabular regression.

    Architecture:
        1. Tokenize: each feature -> d_token-dim vector
        2. Prepend learnable [CLS] token
        3. Process through Transformer encoder (pre-norm)
        4. Take [CLS] output -> LayerNorm -> Linear(1) -> prediction
    """

    def __init__(self, config: FTConfig | None = None) -> None:
        super().__init__()
        self.config = config or FTConfig()
        c = self.config

        # Feature tokenizers
        self.num_tokenizer = NumericalTokenizer(c.n_numerical, c.d_token)
        self.cat_tokenizer = CategoricalTokenizer(c.cat_cardinalities, c.d_token)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, c.d_token))

        # Transformer encoder (pre-norm via norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_token,
            nhead=c.n_heads,
            dim_feedforward=int(c.d_token * c.ffn_multiplier),
            dropout=c.ffn_dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=c.n_blocks,
        )

        # Attention and residual dropout applied manually
        self.attn_dropout = nn.Dropout(c.attention_dropout)
        self.resid_dropout = nn.Dropout(c.residual_dropout)

        # Output head: [CLS] token -> prediction
        self.head = nn.Sequential(
            nn.LayerNorm(c.d_token),
            nn.Linear(c.d_token, 1),
        )

        self._init_head()

    def _init_head(self) -> None:
        """Initialize output head with small weights for stable start."""
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x_numerical: torch.Tensor,
        x_categorical: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_numerical: (batch, n_numerical) float tensor
            x_categorical: (batch, n_categorical) long tensor

        Returns:
            (batch,) float tensor -- predicted duration in seconds
        """
        batch_size = x_numerical.shape[0]

        # Tokenize features
        num_tokens = self.num_tokenizer(x_numerical)    # (batch, 24, d)
        cat_tokens = self.cat_tokenizer(x_categorical)  # (batch, 2, d)

        # Prepend [CLS] token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d)

        # Full token sequence: [CLS, num_0, ..., num_23, cat_0, cat_1]
        tokens = torch.cat([cls, num_tokens, cat_tokens], dim=1)  # (batch, 27, d)

        # Transformer encoder
        output = self.transformer(tokens)  # (batch, 27, d)

        # Take [CLS] token output and predict
        cls_output = output[:, 0]  # (batch, d)
        prediction = self.head(cls_output)  # (batch, 1)

        return prediction.squeeze(1)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
