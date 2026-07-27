from collections.abc import Callable
from contextlib import contextmanager

import torch.nn as nn

# Modules that change behavior between train/eval and could skew distributions
_FROZEN_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.LazyBatchNorm1d,
    nn.LazyBatchNorm2d,
    nn.LazyBatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)
# Handle common stochastic layers from PyTorch and timm by class name
_FROZEN_NAMES = {"StochasticDepth", "DropPath", "DropPath1d", "DropPath2d", "DropPath3d"}


def _is_freezable(m: nn.Module) -> bool:
    return isinstance(m, _FROZEN_TYPES) or type(m).__name__ in _FROZEN_NAMES


def freeze_stochastic(model: nn.Module) -> list[tuple[nn.Module, bool]]:
    """Freeze layers that introduce randomness or running-stat updates. Returns their previous states."""
    states = []
    for m in model.modules():
        if _is_freezable(m):
            states.append((m, m.training))
            m.eval()
    return states


def restore_stochastic(states: list[tuple[nn.Module, bool]]) -> None:
    """Restore modules to their original training/eval state."""
    for m, was_training in states:
        if isinstance(m, nn.Module):
            m.train(was_training)


@contextmanager
def frozen_stochastic(model: nn.Module):
    """Context manager for temporarily freezing stochastic/stateful layers."""
    states = freeze_stochastic(model)
    try:
        yield model
    finally:
        restore_stochastic(states)


def init_norm(norm_type: str, dim: int, num_groups: int = 4) -> Callable:
    if norm_type == "batch":
        return nn.BatchNorm1d(dim)
    elif norm_type == "group":
        return nn.GroupNorm(num_groups, dim)
    elif norm_type == "layer":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unsupported normalization: {norm_type}")


def init_activation(act_type: str) -> Callable:
    if act_type == "gelu":
        return nn.GELU()
    elif act_type == "relu":
        return nn.ReLU(inplace=True)
    elif act_type == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif act_type == "swish":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation type: {act_type}")
