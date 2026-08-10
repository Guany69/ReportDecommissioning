"""The PyTorch duplicate classifier.

A compact feed-forward binary classifier is the right shape here: the input is 18
low-dimensional, already-normalized structured features, not text or images. A
larger network would only overfit the small hand-reviewed label sets this system
can realistically collect.

The forward pass returns a **raw logit**, never a probability — training pairs it
with `BCEWithLogitsLoss` (numerically stabler than sigmoid + BCE) and inference
applies the sigmoid explicitly at the one place probabilities are wanted.
"""
from __future__ import annotations

import torch
from torch import nn

from .features import FEATURE_COUNT

# Recorded in the artifact so a future architecture change can be detected on load.
ARCHITECTURE = "duplicate_mlp_v1"
DEFAULT_HIDDEN_SIZES: tuple[int, ...] = (32, 16)


class DuplicateMLP(nn.Module):
    """input -> Linear -> ReLU -> Linear -> ReLU -> Linear(1), emitting a logit."""

    def __init__(
        self,
        input_size: int = FEATURE_COUNT,
        hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer.")

        self.input_size = input_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = float(dropout)

        layers: list[nn.Module] = []
        prev = input_size
        for width in self.hidden_sizes:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, input_size) -> (batch,) raw logits."""
        return self.net(x).squeeze(-1)


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Logits -> probabilities in 0..1. The only place sigmoid is applied."""
    return torch.sigmoid(logits)


def build_model(
    input_size: int = FEATURE_COUNT,
    hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES,
    dropout: float = 0.0,
) -> DuplicateMLP:
    return DuplicateMLP(input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout)
