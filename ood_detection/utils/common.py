from datetime import datetime
import glob
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.val, self.avg, self.sum, self.count = None, None, None, None
        self.reset()

    def reset(self):
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TermColors:
    """
    Border Color values for pretty printing in terminal
    Sample Use:
        print(f"{TermColors.WARN}Warning: Information.{TermColors.ENDC}"
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARN = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_env_var(name: str, required: bool = True, default: str = None) -> str:
    value = os.getenv(name)
    if required and default:
        raise OSError(f"Required environment variable '{name}' cannot have a default value.")
    if required and value is None:
        raise OSError(f"Required environment variable '{name}' not set.")
    if value is None:
        return default
    return value


def to_numpy(t):
    if isinstance(t, torch.Tensor):
        a = t.detach().cpu()
        if a.dtype == torch.bfloat16 or a.dtype == torch.float64:
            a = a.to(torch.float32)
        return a.numpy()
    return t


def name_with_datetime(prefix="default"):
    now = datetime.now()
    return prefix + "_" + now.strftime("%Y%m%d_%H%M%S")


def init_environment(gpu: int, seed: int | None = None, device: str = "cuda", reproducible: bool = True):
    """
    Initialize the computing environment for PyTorch.

    Parameters:
    - gpu (int): GPU number to use. Ignored if no CUDA device is available.
    - seed (int): Seed value for reproducibility. If None, no seed is set.
    - device (str): Device to use ("cuda" or "cpu").
    - reproducible (bool): If True, sets deterministic behavior and disables benchmarking.

    Returns:
    - torch.device: The configured device.
    - torch.Generator: A PyTorch random number generator initialized with the seed.
    """
    use_cuda = torch.cuda.is_available() and device == "cuda"
    device = torch.device(f"cuda:{gpu}" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.set_device(device)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        if use_cuda:
            torch.cuda.manual_seed(seed + 3)

    if reproducible:
        print(
            f"{TermColors.WARN}Warning: Setting torch.backends.cudnn.deterministic = True "
            f"and cudnn.benchmark = False. This may slow down GPU training.{TermColors.ENDC}"
        )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    return device, generator


def get_git_revision_hash() -> str:
    """Get the git hash of the current commit. Returns None if run from a non-git init repo"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except subprocess.CalledProcessError as excep:
        print(excep, "Couldn't get git hash of the current repo. Returning None")
    return None


def get_latest_file_in_dir(directory: str, extension: str) -> str | None:
    """
    Get the latest (most recently modified) file in a directory with a given extension.

    :param directory: Path to the directory to search.
    :param extension: File extension to filter by (e.g., 'txt', 'csv').
    :return: Path to the latest file, or None if no matching files are found.
    """
    files = glob.glob(os.path.join(directory, f"*.{extension}"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def get_latest_folder(directory: Path):
    dirs = [p for p in directory.iterdir() if p.is_dir()]
    return max(dirs, key=lambda d: d.stat().st_mtime, default=None)


############################ conversion utils ############################


def flatten_dict(d, parent_key="", sep="."):
    """
    Flattens a nested dictionary into a single level dict.
    E.g., {'a': {'b': 1}} becomes {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_structure(x, path=""):
    """
    Recursively flatten a nested structure (dict or list/tuple) into a list of tuples.
    Each tuple contains the path to the leaf node and the value of the leaf node.

    Args:
        x (dict or list/tuple): The nested structure to flatten.
        path (str): The path to the current node in the structure. Defaults to "".

    Returns:
        list[tuple]: A list of tuples containing the path to each leaf node and its value.
    """
    leaves = []
    if isinstance(x, dict):
        for k in sorted(x.keys()):
            leaves.extend(flatten_structure(x[k], f"{path}/{k}" if path else str(k)))
    elif isinstance(x, list | tuple):
        for i, v in enumerate(x):
            leaves.extend(flatten_structure(v, f"{path}/{i}" if path else str(i)))
    else:
        leaves.append((path if path else "output", x))
    return leaves
