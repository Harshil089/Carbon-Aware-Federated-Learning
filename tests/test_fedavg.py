import torch

from src.federated.fedavg import fedavg
from src.models import build_model


def test_weighted_average_matches_formula():
    a = {"w": torch.tensor([0.0, 2.0])}
    b = {"w": torch.tensor([4.0, 6.0])}
    out = fedavg([a, b], [1, 3])
    assert torch.allclose(out["w"], torch.tensor([3.0, 5.0]))


def test_identical_clients_return_same_model():
    model = build_model("cnn")
    state = model.state_dict()
    out = fedavg([state, state, state], [10, 20, 30])
    for k in state:
        assert torch.allclose(out[k], state[k], atol=1e-6)


def test_output_loads_into_model():
    states = [build_model("mlp").state_dict() for _ in range(3)]
    model = build_model("mlp")
    model.load_state_dict(fedavg(states, [5, 5, 5]))
