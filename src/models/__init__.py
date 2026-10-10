from torch import nn

from .cnn import SmallCNN
from .mlp import MLP

MODELS = {"cnn": SmallCNN, "mlp": MLP}


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {list(MODELS)}")
    return MODELS[name](num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
