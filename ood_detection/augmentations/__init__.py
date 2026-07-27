from collections.abc import Callable

import numpy as np
from omegaconf import DictConfig

from .time_series_cls import (
    HorizontalFlip,
    Jitter,
    Mask,
    RandomCropIfLonger,
    RandomCropResize,
    Resize,
    ResizeDownIfLonger,
    Scaling,
    Standardize,
    TimeWarp,
    TsCompose,
    TsToTensor,
    VerticalFlip,
)

IMPLEMENTED_AUGMENTATIONS = {
    "Standardize": Standardize,
    "TsToTensor": TsToTensor,
    "RandomCropIfLonger": RandomCropIfLonger,
    "RandomCropResize": RandomCropResize,
    "Mask": Mask,
    "Jitter": Jitter,
    "Scaling": Scaling,
    "Resize": Resize,
    "ResizeDownIfLonger": ResizeDownIfLonger,
    "HorizontalFlip": HorizontalFlip,
    "VerticalFlip": VerticalFlip,
    "TimeWarp": TimeWarp,
}


def build_augmentations(
    aug_list: list[dict], train_data: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> list[Callable]:
    """
    Builds a list of callable agus based on the provided list of dict configurations.
    Args:
        aug_list (List[dict]): List of dict configs for each aug. Each dict should contain 'type' and optionally 'args'.
        train_data (np.ndarray): Training data. (Must be of shape [N, Feat, Time])
        mean (np.ndarray): Mean values for standardization.
        std (np.ndarray): Standard deviation values for standardization.

    Returns:
        List[Callable]: List of callable augmentation functions.
    """
    transforms = []
    for aug_cfg in aug_list:
        aug_type = aug_cfg["type"]
        aug_args = aug_cfg.get("args", {})
        aug_class = IMPLEMENTED_AUGMENTATIONS.get(aug_type)
        if aug_class is None:
            raise ValueError(f"Unknown augmentation type: {aug_type}")

        ######## Add any custom runtime params to augs here ########
        # Inject mean/std for Standardize
        if aug_type == "Standardize":
            aug_args = {**aug_args, "mean": mean, "std": std}
        ############################################################
        transforms.append(aug_class(**aug_args))
    return transforms


def get_transform(config: DictConfig, train_data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict[str, Callable]:
    """
    Get a dict of callable transformations for training, testing, synthetic outlier generation or ood dataset loading.
    """
    transforms = {"train": None, "test": None, "synthetic_outlier": None, "ood": None}
    if config.dataset.augmentations.get("train"):
        transforms["train"] = TsCompose(build_augmentations(config.dataset.augmentations.train, train_data, mean, std))
    if config.dataset.augmentations.get("test"):
        transforms["test"] = TsCompose(build_augmentations(config.dataset.augmentations.test, train_data, mean, std))
    if config.dataset.augmentations.get("synthetic_outlier"):
        transforms["synthetic_outlier"] = TsCompose(
            build_augmentations(config.dataset.augmentations.synthetic_outlier, train_data, mean, std)
        )
    if config.ood_eval.enabled:
        if config.dataset.augmentations.get("ood"):
            transforms["ood"] = TsCompose(build_augmentations(config.dataset.augmentations.ood, train_data, mean, std))
        else:
            print("No augmentations specified for ood, using test augmentations instead.")
            transforms["ood"] = TsCompose(build_augmentations(config.dataset.augmentations.test, train_data, mean, std))

    return transforms
