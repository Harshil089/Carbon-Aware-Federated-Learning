from collections import OrderedDict

import torch

StateDict = "OrderedDict[str, torch.Tensor]"


def fedavg(states: list[StateDict], num_samples: list[int]) -> StateDict:
    """w_{t+1} = sum_i (n_i / sum_j n_j) * w_i^{t+1} over the participating clients."""
    if not states or len(states) != len(num_samples):
        raise ValueError("Need one sample count per client state.")
    total = float(sum(num_samples))
    if total <= 0:
        raise ValueError("Total sample count must be positive.")
    weights = [n / total for n in num_samples]
    averaged = OrderedDict()
    for key, ref in states[0].items():
        if not torch.is_floating_point(ref):
            # Integer buffers (e.g. BatchNorm's num_batches_tracked) are not averaged.
            averaged[key] = ref.clone()
            continue
        averaged[key] = sum(w * s[key].to(torch.float64) for w, s in zip(weights, states)).to(ref.dtype)
    return averaged
