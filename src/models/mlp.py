import torch
from torch import nn


class MLP(nn.Module):
    """784-200-200-10 MLP; useful for the earliest correctness checks."""

    def __init__(self, num_classes: int = 10, hidden: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
