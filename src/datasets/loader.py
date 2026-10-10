from dataclasses import dataclass

import torch
from torchvision import datasets

# Standard normalisation constants for each dataset.
_DATASETS = {
    "mnist": (datasets.MNIST, 0.1307, 0.3081),
    "fashion_mnist": (datasets.FashionMNIST, 0.2860, 0.3530),
}


@dataclass
class TensorDataset:
    """Whole dataset held as tensors: x is (N, 1, 28, 28) float, y is (N,) long."""
    x: torch.Tensor
    y: torch.Tensor

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, indices) -> "TensorDataset":
        idx = torch.as_tensor(indices, dtype=torch.long)
        return TensorDataset(self.x[idx], self.y[idx])

    def to(self, device) -> "TensorDataset":
        return TensorDataset(self.x.to(device), self.y.to(device))


def load_dataset(name: str, root: str) -> tuple[TensorDataset, TensorDataset]:
    """Download (if needed) and return normalised train and test sets.

    MNIST-sized data fits easily in memory, so we skip DataLoader and batch tensors
    directly; this is much faster on both CPU and a Colab GPU.
    """
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(_DATASETS)}")
    cls, mean, std = _DATASETS[name]
    out = []
    for train in (True, False):
        raw = cls(root, train=train, download=True)
        x = raw.data.float().div(255.0).sub(mean).div(std).unsqueeze(1)
        out.append(TensorDataset(x, raw.targets.long()))
    return out[0], out[1]
