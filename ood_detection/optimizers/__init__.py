import math

from omegaconf import DictConfig
from schedulefree import SGDScheduleFree
import torch
import torch.nn as nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader

IMPLEMENTED_OPTIMIZERS = {"SGD": SGD, "SGDScheduleFree": SGDScheduleFree}
IMPLEMENTED_SCHEDULERS = {
    "OneCycleLR": OneCycleLR,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
}


class StaircaseCosineAnnealingWarmRestarts:
    """
    Cosine restarts with a descending staircase envelope.

    Unlike PyTorch's default warm restarts, each new cycle starts from a lower
    peak. Intermediate cycles decay to half of their own peak, and only the
    final cycle decays all the way to eta_min.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        total_epochs: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        if T_0 <= 0:
            raise ValueError("T_0 must be > 0.")
        if T_mult < 1:
            raise ValueError("T_mult must be >= 1.")
        if total_epochs <= 0:
            raise ValueError("total_epochs must be > 0.")

        self.optimizer = optimizer
        self.T_0 = int(T_0)
        self.T_mult = int(T_mult)
        self.eta_min = float(eta_min)
        self.total_epochs = float(total_epochs)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = float(last_epoch)
        self._last_lr = list(self.base_lrs)

        self._cycle_starts, self._cycle_lengths = self._build_cycles()
        self.step(0.0 if last_epoch < 0 else float(last_epoch))

    def _build_cycles(self) -> tuple[list[float], list[float]]:
        starts: list[float] = []
        lengths: list[float] = []
        current_start = 0.0
        current_length = float(self.T_0)

        while current_start < self.total_epochs:
            starts.append(current_start)
            lengths.append(current_length)
            current_start += current_length
            current_length *= self.T_mult

        return starts, lengths

    def _get_cycle_index(self, epoch_progress: float) -> int:
        for idx, start in enumerate(self._cycle_starts):
            end = start + self._cycle_lengths[idx]
            if epoch_progress < end:
                return idx
        return len(self._cycle_starts) - 1

    def _get_cycle_bounds(self, cycle_idx: int) -> tuple[list[float], list[float]]:
        is_last_cycle = cycle_idx == len(self._cycle_starts) - 1

        cycle_max_lrs = [base_lr / (2**cycle_idx) for base_lr in self.base_lrs]
        if is_last_cycle:
            cycle_min_lrs = [self.eta_min for _ in self.base_lrs]
        else:
            cycle_min_lrs = [max(self.eta_min, base_lr / (2 ** (cycle_idx + 1))) for base_lr in self.base_lrs]

        return cycle_max_lrs, cycle_min_lrs

    def get_last_lr(self) -> list[float]:
        return self._last_lr

    def state_dict(self) -> dict:
        return {
            "T_0": self.T_0,
            "T_mult": self.T_mult,
            "eta_min": self.eta_min,
            "total_epochs": self.total_epochs,
            "base_lrs": self.base_lrs,
            "last_epoch": self.last_epoch,
            "_last_lr": self._last_lr,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.last_epoch = float(state_dict["last_epoch"])
        self._last_lr = list(state_dict["_last_lr"])
        self.step(self.last_epoch)

    def step(self, epoch: float | None = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1.0

        epoch = max(0.0, float(epoch))
        cycle_idx = self._get_cycle_index(epoch)
        cycle_start = self._cycle_starts[cycle_idx]
        cycle_length = self._cycle_lengths[cycle_idx]
        cycle_progress = min(max((epoch - cycle_start) / max(cycle_length, 1e-12), 0.0), 1.0)
        cycle_max_lrs, cycle_min_lrs = self._get_cycle_bounds(cycle_idx)

        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
        lrs = []
        for group, cycle_max_lr, cycle_min_lr in zip(self.optimizer.param_groups, cycle_max_lrs, cycle_min_lrs):
            lr = cycle_min_lr + (cycle_max_lr - cycle_min_lr) * cosine_factor
            group["lr"] = lr
            lrs.append(lr)

        self.last_epoch = epoch
        self._last_lr = lrs


class MultiOptimizerWrapper(Optimizer):
    """
    Wrapper that combines multiple optimizers into a single optimizer interface.
    Useful for handling different parameter groups with different optimization strategies.
    """

    def __init__(self, optimizers: list[Optimizer]):
        self.optimizers = optimizers
        self.param_groups = []

        # Combine param_groups from all optimizers
        for optim in self.optimizers:
            self.param_groups.extend(optim.param_groups)

    def zero_grad(self):
        """Zero gradients for all optimizers"""
        for optim in self.optimizers:
            optim.zero_grad()

    def step(self, closure=None):
        """Step all optimizers"""
        for optim in self.optimizers:
            optim.step(closure)

    def state_dict(self):
        """Return combined state dict from all optimizers"""
        state = {}
        for i, optim in enumerate(self.optimizers):
            state[f"optimizer_{i}"] = optim.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load state dict into all optimizers"""
        for i, optim in enumerate(self.optimizers):
            if f"optimizer_{i}" in state_dict:
                optim.load_state_dict(state_dict[f"optimizer_{i}"])

    def add_param_group(self, param_group):
        """Add parameter group to the first optimizer (or implement custom logic)"""
        self.optimizers[0].add_param_group(param_group)
        self.param_groups.append(param_group)

    # Optional: Implement scheduler compatibility methods
    def get_lr(self):
        """Get learning rates from all optimizers"""
        lrs = []
        for optim in self.optimizers:
            for group in optim.param_groups:
                lrs.append(group["lr"])
        return lrs

    def set_lr(self, lr):
        """Set learning rates for all optimizers"""
        for optim in self.optimizers:
            for group in optim.param_groups:
                group["lr"] = lr


def init_scheduler(
    scheduler_name: str,
    optimizer: Optimizer,
    train_loader: DataLoader,
    config: DictConfig,
    pct_start: float = 0.01,
    max_lrs: list[float] | None = None,
):
    """Initialize the scheduler."""
    if scheduler_name == "OneCycleLR":
        batches_per_epoch = len(train_loader)
        total_steps = config.trainer.args.epochs * batches_per_epoch

        warmup_steps = int(pct_start * total_steps)
        if warmup_steps <= 1:
            pct_start = 0.0
            if config.verbose:
                print(f"Adjusted pct_start to {pct_start} to avoid <1 warmup step.")

        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lrs if max_lrs is not None else config.optimizer.args.lr,
            pct_start=pct_start,
            total_steps=total_steps,
            cycle_momentum=False,
        )
        if config.verbose:
            print("lr_scheduler initialized with total_steps:", total_steps)
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler_args = config.lr_scheduler.get("args", {})
        t_0 = int(scheduler_args.get("T_0", 10))
        t_mult = int(scheduler_args.get("T_mult", 1))
        eta_min = float(scheduler_args.get("eta_min", 0.0))
        last_epoch = int(scheduler_args.get("last_epoch", -1))
        staircase = bool(scheduler_args.get("staircase", False))

        if staircase:
            scheduler = StaircaseCosineAnnealingWarmRestarts(
                optimizer,
                T_0=t_0,
                T_mult=t_mult,
                eta_min=eta_min,
                total_epochs=config.trainer.args.epochs,
                last_epoch=last_epoch,
            )
        else:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t_0,
                T_mult=t_mult,
                eta_min=eta_min,
                last_epoch=last_epoch,
            )
        if config.verbose:
            print(
                "lr_scheduler initialized with "
                f"T_0={t_0}, T_mult={t_mult}, eta_min={eta_min}, "
                f"last_epoch={last_epoch}, staircase={staircase}"
            )
    else:
        raise NotImplementedError(f"{scheduler_name} is not implemented.")

    return scheduler


def init_optimizer(
    optimizer_name: str,
    scheduler_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    config: DictConfig,
) -> tuple[torch.optim.Optimizer, object]:
    """
    Returns (optimizer, scheduler) based on config.
    Uses a single SGD with param groups:
        - decay (weight decay)
        - no_decay (biases & norm layers)
        - proto (classifier/prototypes; no wd, no momentum)
    """
    if optimizer_name != "SGD":
        raise NotImplementedError(f"Optimizer '{optimizer_name}' is not implemented. Available optimizers: ['SGD']")

    optim_args = config.optimizer.args
    lr = float(optim_args.lr)
    wd = float(optim_args.weight_decay)
    momentum = float(getattr(optim_args, "momentum", 0.9))
    nesterov = bool(getattr(optim_args, "nesterov", False))

    # Map of module names -> modules, so we can resolve the parent module for each param
    modules = dict(model.named_modules())

    decay_params, no_decay_params, proto_params = [], [], []

    for full_name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # parent module to check for norms
        parent_name = full_name.rsplit(".", 1)[0] if "." in full_name else ""
        parent_module = modules.get(parent_name, model)
        param_name = full_name.split(".")[-1]

        # 1) Prototypes / classifier params
        if "classifier" in full_name:
            proto_params.append(param)
            continue

        # 2) No-decay: biases and norm layers
        if param_name == "bias" or isinstance(
            parent_module, (nn.LayerNorm | nn.BatchNorm1d | nn.BatchNorm2d | nn.GroupNorm)
        ):
            no_decay_params.append(param)
            continue

        # 3) Everything else: decay
        decay_params.append(param)

    # Build param groups (skip empties so scheduler aligns with real groups)
    param_groups = []
    max_lrs = []

    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": wd, "momentum": momentum})
        max_lrs.append(lr)

    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0, "momentum": momentum})
        max_lrs.append(lr)

    if proto_params:
        # prototypes: no wd, no momentum (as in your intent)
        param_groups.append({"params": proto_params, "weight_decay": 0.0, "momentum": 0.0})
        max_lrs.append(lr)

    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.SGD(param_groups, lr=lr, nesterov=nesterov)

    scheduler = init_scheduler(
        scheduler_name=scheduler_name,
        optimizer=optimizer,
        train_loader=train_loader,
        config=config,
        pct_start=0.01,
        max_lrs=max_lrs,  # must match number/order of groups above
    )

    return optimizer, scheduler
