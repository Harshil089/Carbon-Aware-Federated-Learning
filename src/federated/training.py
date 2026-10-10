import torch
from torch import nn

from src.datasets.loader import TensorDataset


def train_local(
    model: nn.Module,
    data: TensorDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    generator: torch.Generator,
) -> float:
    """Run local SGD on ``data`` in place and return the mean training loss of the last epoch."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()
    n = len(data)
    last_epoch_loss = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=generator).to(data.x.device)
        total, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = data.x[idx], data.y[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            seen += len(idx)
        last_epoch_loss = total / seen
    return last_epoch_loss


@torch.no_grad()
def evaluate(model: nn.Module, data: TensorDataset, batch_size: int = 1000) -> tuple[float, float]:
    """Return (accuracy, mean cross-entropy loss) on ``data``."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    correct, loss = 0, 0.0
    for start in range(0, len(data), batch_size):
        xb, yb = data.x[start:start + batch_size], data.y[start:start + batch_size]
        logits = model(xb)
        loss += loss_fn(logits, yb).item()
        correct += (logits.argmax(1) == yb).sum().item()
    return correct / len(data), loss / len(data)
