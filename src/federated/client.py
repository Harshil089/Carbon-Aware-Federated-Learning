import copy
import time

import torch
from torch import nn

from src.datasets.loader import TensorDataset
from src.federated.training import train_local
from src.simulation.client_profiles import ClientProfile


class FederatedClient:
    """A simulated client: holds its private data shard and a hardware profile.

    Raw data never leaves the client; only the trained model state is returned.
    """

    def __init__(self, profile: ClientProfile, data: TensorDataset, seed: int):
        self.profile = profile
        self.data = data
        self.generator = torch.Generator().manual_seed(seed + profile.client_id)

    @property
    def client_id(self) -> int:
        return self.profile.client_id

    @property
    def num_samples(self) -> int:
        return len(self.data)

    def fit(self, global_model: nn.Module, cfg: dict) -> dict:
        model = copy.deepcopy(global_model)
        start = time.perf_counter()
        loss = train_local(
            model, self.data,
            epochs=cfg["local_epochs"], batch_size=cfg["batch_size"],
            lr=cfg["learning_rate"], momentum=cfg["momentum"], generator=self.generator,
            hflip=cfg.get("augment_hflip", False),
        )
        if self.data.x.is_cuda:
            torch.cuda.synchronize()
        self.profile.participation_count += 1
        return {
            "state": model.state_dict(),
            "num_samples": self.num_samples,
            "local_loss": loss,
            "train_wall_time_s": time.perf_counter() - start,
        }
