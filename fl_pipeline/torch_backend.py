"""PyTorch model backend for FederatedRLEnv (--model torch_mlp / torch_cnn).

Imported lazily by rl_env/experiment so `import fl_pipeline` works without torch.
"""
from __future__ import annotations

import time
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch

from .torch_fl import MNISTCNN, MNISTMLP, eval_torch, fedavg_torch, train_client_torch


class TorchBackend:
    """Global PyTorch model with FedAvg rounds over per-client (x, y) tensor partitions."""

    def __init__(
        self,
        model_name: str,
        partitions: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        test_data: Tuple[torch.Tensor, torch.Tensor],
        seed: int = 42,
        lr_batch_size: int = 64,
    ) -> None:
        if model_name not in ("torch_mlp", "torch_cnn"):
            raise ValueError(f"Unsupported torch model: {model_name!r}.")
        self.model_name = model_name
        self.partitions = list(partitions)
        self.val_data = val_data
        self.test_data = test_data
        self.seed = int(seed)
        self.batch_size = int(lr_batch_size)
        self.model: Any = None
        self.reset()

    def reset(self) -> None:
        """Build a fresh global model; seeding makes every episode start from the same weights."""
        torch.manual_seed(self.seed)
        self.model = MNISTMLP() if self.model_name == "torch_mlp" else MNISTCNN()

    def train_round(
        self, trained_ids: Sequence[int], epochs: int, lr: float
    ) -> Tuple[List[float], np.ndarray]:
        """Train each trained client from the global model, FedAvg-aggregate, load into global."""
        states, counts, secs = [], [], []
        for i in trained_ids:
            x, y = self.partitions[i]
            t0 = time.perf_counter()
            # inplace=False: train_client_torch deep-copies the global model per client.
            states.append(
                train_client_torch(
                    self.model, x, y, epochs=epochs, lr=lr, batch_size=self.batch_size
                )
            )
            secs.append(time.perf_counter() - t0)
            counts.append(len(y))
        agg_state, norm_weights = fedavg_torch(states, counts)
        self.model.load_state_dict(agg_state)
        return secs, np.asarray(norm_weights, dtype=float)

    def evaluate(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, float]:
        """Return (accuracy, cross-entropy loss) of the global model on (x, y) tensors."""
        acc, loss = eval_torch(self.model, data[0], data[1], return_loss=True)
        return float(acc), float(loss)
