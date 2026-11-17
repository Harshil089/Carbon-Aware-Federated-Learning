"""PyTorch Federated Learning (MNIST MLP/CNN, FedAvg) and Deep Q-Network (DQN) Scheduler."""
from __future__ import annotations

from collections import deque
import copy
import socket
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clients import FederatedClient
from .data import partition_dirichlet, partition_iid
from .fedavg import compute_aggregation_weights
from .rl_env import FederatedRLEnv
from .schedulers import BaseScheduler

SYNTHETIC_MNIST_PROVENANCE = "SYNTHETIC_28x28_FALLBACK"
TORCHVISION_MNIST_PROVENANCE = "TORCHVISION_MNIST"

TorchPartition = Tuple[torch.Tensor, torch.Tensor]


class MNISTPartitionBundle(NamedTuple):
    """Container for partitioned MNIST client tensors, held-out test set, and provenance tag."""

    partitions: List[TorchPartition]
    test_data: TorchPartition
    provenance: str


def _generate_synthetic_mnist_arrays(
    n_samples: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate deterministic 28x28 10-class synthetic image arrays with learnable spatial templates."""
    rng = np.random.default_rng(seed)
    n_classes = 10
    base_counts = np.full(n_classes, n_samples // n_classes, dtype=int)
    base_counts[: n_samples % n_classes] += 1
    y = np.repeat(np.arange(n_classes, dtype=np.int64), base_counts)
    rng.shuffle(y)

    grid_y, grid_x = np.meshgrid(
        np.linspace(-1.0, 1.0, 28, dtype=np.float32),
        np.linspace(-1.0, 1.0, 28, dtype=np.float32),
        indexing="ij",
    )
    templates = np.zeros((n_classes, 28, 28), dtype=np.float32)
    for c in range(n_classes):
        angle = (c / n_classes) * np.pi
        freq = 1.5 + 0.35 * (c % 5)
        cx = 0.55 * np.cos(2.0 * np.pi * c / n_classes)
        cy = 0.55 * np.sin(2.0 * np.pi * c / n_classes)
        gaussian_blob = np.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / 0.14)
        oriented_wave = 0.5 * (
            1.0 + np.cos(freq * np.pi * (grid_x * np.cos(angle) + grid_y * np.sin(angle)))
        )
        ring = np.exp(-((np.sqrt(grid_x**2 + grid_y**2) - (0.20 + 0.06 * c)) ** 2) / 0.03)
        templates[c] = np.clip(0.55 * gaussian_blob + 0.25 * oriented_wave + 0.20 * ring, 0.0, 1.0)

    x = templates[y] + rng.normal(0.0, 0.08, size=(n_samples, 28, 28)).astype(np.float32)
    x = np.clip(x, 0.0, 1.0)[:, None, :, :]  # (N, 1, 28, 28)
    return x.astype(np.float32), y.astype(np.int64)


_OFFLINE_FALLBACK_CACHED = False


def _has_local_mnist_files(root: Path) -> bool:
    """Check whether torchvision MNIST raw or processed files already exist on disk."""
    raw_dir = root / "MNIST" / "raw"
    proc_dir = root / "MNIST" / "processed"
    if raw_dir.exists() and any(raw_dir.iterdir()):
        return True
    if proc_dir.exists() and any(proc_dir.iterdir()):
        return True
    return False


def _load_raw_mnist_or_fallback(
    max_train_samples: int = 1600,
    max_test_samples: int = 400,
    seed: int = 42,
    data_dir: Union[str, Path] = "data/mnist",
    allow_download: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load MNIST via torchvision.datasets.MNIST or fall back to deterministic 28x28 synthetic tensors."""
    global _OFFLINE_FALLBACK_CACHED
    root = Path(data_dir)
    try:
        from torchvision import datasets

        has_local = _has_local_mnist_files(root)
        if not has_local and not allow_download:
            # Call torchvision.datasets.MNIST(..., download=False), which raises RuntimeError if absent
            train_ds = datasets.MNIST(root=str(root), train=True, download=False)
            test_ds = datasets.MNIST(root=str(root), train=False, download=False)
        elif not has_local and _OFFLINE_FALLBACK_CACHED:
            raise RuntimeError("Offline MNIST fallback cached.")
        elif not has_local:
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(0.4)
            try:
                train_ds = datasets.MNIST(root=str(root), train=True, download=True)
                test_ds = datasets.MNIST(root=str(root), train=False, download=True)
            except Exception:
                _OFFLINE_FALLBACK_CACHED = True
                raise
            finally:
                socket.setdefaulttimeout(old_timeout)
        else:
            train_ds = datasets.MNIST(root=str(root), train=True, download=False)
            test_ds = datasets.MNIST(root=str(root), train=False, download=False)

        rng = np.random.default_rng(seed)
        n_tr = min(int(max_train_samples), len(train_ds.data))
        n_te = min(int(max_test_samples), len(test_ds.data))
        tr_idx = rng.choice(len(train_ds.data), size=n_tr, replace=False)
        te_idx = rng.choice(len(test_ds.data), size=n_te, replace=False)

        x_train = (train_ds.data[tr_idx].numpy().astype(np.float32) / 255.0)[:, None, :, :]
        y_train = train_ds.targets[tr_idx].numpy().astype(np.int64)
        x_test = (test_ds.data[te_idx].numpy().astype(np.float32) / 255.0)[:, None, :, :]
        y_test = test_ds.targets[te_idx].numpy().astype(np.int64)
        return x_train, y_train, x_test, y_test, TORCHVISION_MNIST_PROVENANCE
    except Exception:
        x_train, y_train = _generate_synthetic_mnist_arrays(max_train_samples, seed=seed)
        x_test, y_test = _generate_synthetic_mnist_arrays(max_test_samples, seed=seed + 10_000)
        return x_train, y_train, x_test, y_test, SYNTHETIC_MNIST_PROVENANCE


def load_mnist_partitions(
    n_clients: int = 4,
    mode: str = "dirichlet",
    alpha: float = 0.5,
    seed: int = 42,
    max_train_samples: int = 1600,
    max_test_samples: int = 400,
    data_dir: Union[str, Path] = "data/mnist",
) -> MNISTPartitionBundle:
    """Load and partition MNIST (or deterministic 28x28 fallback) across federated clients.

    Guarantees:
    - Zero empty partitions (`len(y_i) > 0` for every client).
    - Deterministic given `seed`.
    - Returns `MNISTPartitionBundle(partitions, test_data, provenance)` which can be
      unpacked as `partitions, test_data, provenance` or accessed via attributes.
    """
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1.")
    if max_train_samples < n_clients:
        raise ValueError("max_train_samples must be >= n_clients to ensure non-empty partitions.")

    x_train, y_train, x_test, y_test, provenance = _load_raw_mnist_or_fallback(
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        seed=seed,
        data_dir=data_dir,
    )

    mode_norm = mode.strip().lower()
    min_per_client = max(1, min(10, len(y_train) // n_clients))
    if mode_norm == "iid":
        np_parts = partition_iid(x_train, y_train, n_clients=n_clients, seed=seed + 1)
    elif mode_norm in ("dirichlet", "non-iid", "non_iid"):
        np_parts = partition_dirichlet(
            x_train,
            y_train,
            n_clients=n_clients,
            alpha=alpha,
            min_samples_per_client=min_per_client,
            seed=seed + 1,
        )
    else:
        raise ValueError(f"Unsupported partition mode: {mode}")

    torch_parts: List[TorchPartition] = []
    for xi, yi in np_parts:
        if len(yi) == 0:
            raise RuntimeError("Empty client partition encountered.")
        torch_parts.append(
            (
                torch.as_tensor(xi, dtype=torch.float32),
                torch.as_tensor(yi, dtype=torch.long),
            )
        )

    test_pair: TorchPartition = (
        torch.as_tensor(x_test, dtype=torch.float32),
        torch.as_tensor(y_test, dtype=torch.long),
    )
    return MNISTPartitionBundle(
        partitions=torch_parts,
        test_data=test_pair,
        provenance=provenance,
    )


class MNISTMLP(nn.Module):
    """2-Layer Multi-Layer Perceptron for 28x28 MNIST classification (784 -> 128 -> 10)."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class MNISTCNN(nn.Module):
    """Lightweight CNN for 28x28 MNIST: Conv2d(1,8,3) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear -> 10."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 13 * 13, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv(x)))
        x = self.flatten(x)
        return self.fc(x)


def _to_tensor_pair(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
) -> TorchPartition:
    x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
    y_t = y if isinstance(y, torch.Tensor) else torch.as_tensor(y, dtype=torch.long)
    return x_t.float(), y_t.long()


def train_client_torch(
    model: nn.Module,
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    epochs: int = 1,
    lr: float = 0.05,
    batch_size: int = 64,
    inplace: bool = False,
) -> Dict[str, torch.Tensor]:
    """Train a client PyTorch model on local data (x, y) and return its updated state_dict."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if lr <= 0:
        raise ValueError("lr must be > 0.")
    x_t, y_t = _to_tensor_pair(x, y)
    if len(y_t) == 0:
        raise ValueError("Cannot train on an empty client partition.")

    local_model = model if inplace else copy.deepcopy(model)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = len(y_t)
    step = max(1, int(batch_size))
    for _ in range(epochs):
        for start in range(0, n, step):
            end = min(start + step, n)
            optimizer.zero_grad()
            logits = local_model(x_t[start:end])
            loss = criterion(logits, y_t[start:end])
            loss.backward()
            optimizer.step()

    return {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}


def fedavg_torch(
    client_states: Sequence[Union[Mapping[str, torch.Tensor], nn.Module]],
    sample_counts: Sequence[int],
) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    """Sample-weighted FedAvg aggregation over PyTorch state_dicts.

    Returns:
        (aggregated_state_dict, normalized_weights) where normalized_weights sum to 1.0.
    """
    if len(client_states) == 0:
        raise ValueError("client_states cannot be empty.")
    if len(client_states) != len(sample_counts):
        raise ValueError("client_states and sample_counts must have matching lengths.")

    norm_weights = compute_aggregation_weights(sample_counts)
    if not np.isclose( float(np.sum(norm_weights)), 1.0, atol=1e-6 ):
        raise ValueError("Normalized FedAvg aggregation weights failed sum-to-1 verification.")

    state_dicts: List[Mapping[str, torch.Tensor]] = [
        s.state_dict() if isinstance(s, nn.Module) else s for s in client_states
    ]
    keys = list(state_dicts[0].keys())
    aggregated: Dict[str, torch.Tensor] = {}

    for k in keys:
        ref = state_dicts[0][k].detach().cpu()
        if ref.is_floating_point():
            acc_tensor = torch.zeros_like(ref, dtype=torch.float32)
            for w_i, sd in zip(norm_weights, state_dicts):
                acc_tensor += float(w_i) * sd[k].detach().cpu().float()
            aggregated[k] = acc_tensor.to(dtype=ref.dtype)
        else:
            aggregated[k] = ref.clone()

    return aggregated, norm_weights


class TorchEvalMetric(float):
    """Float accuracy value in [0, 1] that also exposes `.accuracy` and `.loss` attributes."""

    accuracy: float
    loss: float

    def __new__(cls, accuracy: float, loss: float = 0.0) -> "TorchEvalMetric":
        obj = super().__new__(cls, float(accuracy))
        obj.accuracy = float(accuracy)
        obj.loss = float(loss)
        return obj


def eval_torch(
    model: nn.Module,
    x_test: Union[torch.Tensor, np.ndarray],
    y_test: Union[torch.Tensor, np.ndarray],
    return_loss: bool = False,
) -> Union[TorchEvalMetric, Tuple[float, float]]:
    """Evaluate a PyTorch model on held-out test tensors.

    Returns accuracy as a float (with `.accuracy` and `.loss` attributes),
    or `(accuracy, loss)` if `return_loss=True`.
    """
    x_t, y_t = _to_tensor_pair(x_test, y_test)
    if len(y_t) == 0:
        raise ValueError("Test dataset cannot be empty.")

    model.eval()
    with torch.no_grad():
        logits = model(x_t)
        loss = float(F.cross_entropy(logits, y_t).item())
        preds = torch.argmax(logits, dim=1)
        acc = float((preds == y_t).float().mean().item())

    if return_loss:
        return acc, loss
    return TorchEvalMetric(acc, loss)


class QNetwork(nn.Module):
    """2-Layer MLP Q-Network mapping continuous environment observations to action Q-values."""

    def __init__(self, obs_dim: int = 15, num_actions: int = 11, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.fc2(self.relu(self.fc1(x)))


class ReplayBuffer:
    """Fixed-capacity FIFO experience replay buffer for DQN training."""

    def __init__(self, capacity: int = 2000) -> None:
        self.capacity = max(1, int(capacity))
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.capacity
        )

    def push(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action_idx),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = min(len(self.buffer), max(1, int(batch_size)))
        indices = rng.choice(len(self.buffer), size=n, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in indices))
        return (
            torch.as_tensor(np.stack(states), dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.long),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.stack(next_states), dtype=torch.float32),
            torch.as_tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNScheduler(BaseScheduler):
    """PyTorch Deep Q-Network (DQN) scheduler over combinatorial federated client subsets."""

    name = "DQN"

    def __init__(
        self,
        obs_dim: Union[int, Sequence[Tuple[int, ...]]] = 15,
        actions: Optional[Sequence[Tuple[int, ...]]] = None,
        seed: int = 42,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.86,
        epsilon_start: float = 0.80,
        epsilon_min: float = 0.05,
        buffer_capacity: int = 2000,
        batch_size: int = 16,
    ) -> None:
        # Support both DQNScheduler(obs_dim=15, actions=...) and DQNScheduler(actions, seed=42)
        if not isinstance(obs_dim, (int, np.integer)):
            actions = obs_dim
            obs_dim = 15
        if actions is None:
            actions = [
                (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
                (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3),
                (0, 1, 2, 3),
            ]
        self.obs_dim = int(obs_dim)
        self.actions: List[Tuple[int, ...]] = [tuple(int(i) for i in a) for a in actions]
        if not self.actions:
            raise ValueError("DQNScheduler requires a non-empty action space.")

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon = float(epsilon_start)
        self.batch_size = int(batch_size)

        self.q_net = QNetwork(
            obs_dim=self.obs_dim, num_actions=len(self.actions), hidden_dim=self.hidden_dim
        )
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.train_steps = 0

    def _fit_obs_dim(self, obs_vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs_vec, dtype=np.float32).reshape(-1)
        if len(arr) == self.obs_dim:
            return arr
        if len(arr) > self.obs_dim:
            return arr[: self.obs_dim]
        padded = np.zeros(self.obs_dim, dtype=np.float32)
        padded[: len(arr)] = arr
        return padded

    def _clients_to_obs(
        self,
        clients: Sequence[FederatedClient],
        round_idx: int = 0,
        participation_counts: Optional[Sequence[int]] = None,
        acc: float = 0.0,
        max_rounds: int = 20,
    ) -> np.ndarray:
        n = len(clients)
        progress = float(round_idx) / max(1, int(max_rounds))
        est_loss = float(max(0.05, -np.log(max(0.05, min(0.99, float(acc) if acc > 0 else 0.33)))))
        if participation_counts is not None:
            counts = np.asarray(participation_counts, dtype=np.float32)
            total_part = max(1.0, float(np.sum(counts)))
            part_rates = counts / total_part
        else:
            part_rates = np.zeros(n, dtype=np.float32)
        speeds = np.array([c.relative_speed for c in clients], dtype=np.float32)
        cis = np.array([c.grid_kgco2_per_kwh for c in clients], dtype=np.float32)
        raw = np.concatenate(
            [
                np.array([progress, float(acc), est_loss], dtype=np.float32),
                part_rates,
                speeds,
                cis,
            ]
        )
        return self._fit_obs_dim(raw)

    def _resolve_action_index(self, action: Union[int, Sequence[int]]) -> int:
        if isinstance(action, (int, np.integer)):
            idx = int(action)
            if 0 <= idx < len(self.actions):
                return idx
            raise ValueError(f"Action index {idx} out of bounds.")
        tup = tuple(sorted(int(i) for i in action))
        if tup in self.actions:
            return self.actions.index(tup)
        raise ValueError(f"Action tuple {tup} not found in DQNScheduler actions.")

    def push(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: Union[int, Sequence[int]],
        reward: float,
        next_state: Union[np.ndarray, torch.Tensor],
        done: bool = False,
    ) -> None:
        """Store a transition in the replay buffer."""
        s_np = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
        sp_np = (
            next_state.detach().cpu().numpy()
            if isinstance(next_state, torch.Tensor)
            else np.asarray(next_state)
        )
        a_idx = self._resolve_action_index(action)
        self.replay_buffer.push(
            self._fit_obs_dim(s_np),
            a_idx,
            float(reward),
            self._fit_obs_dim(sp_np),
            bool(done),
        )

    remember = push

    def select(
        self,
        clients_or_obs: Union[Sequence[FederatedClient], np.ndarray, torch.Tensor, Sequence[float]],
        round_idx: int = 0,
        k: Optional[int] = None,
        participation_counts: Optional[Sequence[int]] = None,
        acc: float = 0.0,
        max_rounds: int = 20,
        epsilon: Optional[float] = None,
    ) -> Tuple[int, ...]:
        """Select a client subset action using epsilon-greedy Q-values.

        Supports both:
        - `select(obs, epsilon=...)` with a 15-D observation vector
        - `select(clients, round_idx=..., participation_counts=..., acc=..., max_rounds=...)`
        """
        if (
            isinstance(clients_or_obs, (np.ndarray, torch.Tensor))
            or (
                isinstance(clients_or_obs, Sequence)
                and len(clients_or_obs) > 0
                and isinstance(clients_or_obs[0], (int, float, np.number))
            )
        ):
            obs_arr = (
                clients_or_obs.detach().cpu().numpy()
                if isinstance(clients_or_obs, torch.Tensor)
                else np.asarray(clients_or_obs, dtype=np.float32)
            )
            obs = self._fit_obs_dim(obs_arr)
            eps = self.epsilon if epsilon is None else float(epsilon)
        else:
            clients = list(clients_or_obs)  # type: ignore[arg-type]
            obs = self._clients_to_obs(
                clients,
                round_idx=round_idx,
                participation_counts=participation_counts,
                acc=acc,
                max_rounds=max_rounds,
            )
            eps = 0.0 if epsilon is None else float(epsilon)

        if eps > 0.0 and self.rng.random() < eps:
            a_idx = int(self.rng.integers(len(self.actions)))
            return self.actions[a_idx]

        self.q_net.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_vals = self.q_net(obs_t).squeeze(0)
            a_idx = int(torch.argmax(q_vals).item())
        return self.actions[a_idx]

    def train_step(
        self,
        state: Optional[Union[np.ndarray, torch.Tensor]] = None,
        action: Optional[Union[int, Sequence[int]]] = None,
        reward: Optional[float] = None,
        next_state: Optional[Union[np.ndarray, torch.Tensor]] = None,
        done: bool = False,
        batch_size: Optional[int] = None,
    ) -> float:
        """Perform a single DQN replay-buffer optimization step and return the scalar TD loss."""
        if state is not None and action is not None and reward is not None and next_state is not None:
            self.push(state, action, float(reward), next_state, bool(done))

        if len(self.replay_buffer) == 0:
            return 0.0

        b_size = self.batch_size if batch_size is None else int(batch_size)
        states_t, actions_t, rewards_t, next_states_t, dones_t = self.replay_buffer.sample(
            b_size, self.rng
        )

        self.q_net.train()
        q_pred = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            q_target = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = F.smooth_l1_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        # Soft-update target network parameters
        tau = 0.15
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)

        return float(loss.item())

    def train(
        self,
        env: FederatedRLEnv,
        episodes: int = 15,
    ) -> List[float]:
        """Train the DQN scheduler on FederatedRLEnv for the given number of episodes."""
        episode_losses: List[float] = []
        for ep in range(max(1, int(episodes))):
            obs = env.reset(seed=self.seed + ep)
            eps = max(
                self.epsilon_min,
                self.epsilon_start * (1.0 - ep / max(1, int(episodes))),
            )
            self.epsilon = eps
            step_losses: List[float] = []
            for _ in range(env.max_rounds):
                chosen = self.select(obs, epsilon=eps)
                next_obs, reward, done, _ = env.step(chosen)
                loss = self.train_step(obs, chosen, reward, next_obs, done)
                step_losses.append(loss)
                obs = next_obs
                if done:
                    break
            episode_losses.append(float(np.mean(step_losses)) if step_losses else 0.0)

        self.epsilon = 0.0
        return episode_losses
