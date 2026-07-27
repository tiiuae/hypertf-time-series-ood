from collections.abc import Callable
from functools import partial
import multiprocessing
import os

import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from ood_detection.augmentations import get_transform
from ood_detection.datasets import init_dataset
from ood_detection.utils.common import TermColors as Tc
from ood_detection.utils.data.common import (
    filter_arr_by_class,
    remove_invalid_instances,
    resize_time_series,
)
from ood_detection.utils.data.loader import load_ucr, load_uea
from ood_detection.utils.logger import EXP_LOGGER_NAME, LoggerSingleton


def get_loaders(config: DictConfig, generator: Callable | None = None):
    """
    Creates DataLoaders for both training and testing data for a given dataset type and name.

    Args:
        config (DictConfig): Configuration object
        generator (torch.Generator, optional): Generator for controlling randomness in the DataLoaders.

    Returns:
        tuple: (train_loader, test_loader), both are PyTorch DataLoaders.
    """
    dataset_type = config.dataset.type  # Dataset class type (e.g., 'CustomTimeSeriesDataset')
    dataset_name = config.dataset.args.name  # Dataset name (e.g., 'ECG5000', 'Epilepsy', etc)
    loader_type = config.dataset.args.loader  # Dataset Loader type ('UCR', 'UEA')
    id_classes = config.dataset.args.get("id_classes", None)  # List of classes to load, if None load all

    # Load datasets -> functions return data in shape [n_instances, n_timestamps, n_features]
    if loader_type == "UCR":
        in_dist_filter = partial(filter_arr_by_class, class_names=id_classes, include_classes=True)
        load_func = partial(load_ucr, config=config, data_filter_func=in_dist_filter)
    elif loader_type == "UEA":
        in_dist_filter = partial(filter_arr_by_class, class_names=id_classes, include_classes=True)
        load_func = partial(load_uea, config=config, data_filter_func=in_dist_filter)
    else:
        raise ValueError(f"Unsupported loader type: {loader_type}")
    train_data, train_labels, test_data, test_labels, mean, std, label_remap = load_func()

    # logging data load info
    logger = LoggerSingleton.get_logger(EXP_LOGGER_NAME)
    if np.isnan(train_data).any():
        logger.warning(f"{Tc.WARN}train_data contains NaN values.{Tc.ENDC}")
    if np.isnan(test_data).any():
        logger.warning(f"{Tc.WARN}test_data contains NaN values.{Tc.ENDC}")
    if np.isnan(mean).any():
        msg = "mean contains NaN values."
        logger.error(msg)
        raise ValueError(msg)
    if np.isnan(std).any():
        msg = "std contains NaN values."
        logger.error(msg)
        raise ValueError(msg)

    # add synthetic outlier class, which is assigned to a new class == len(label_remap)
    if config.dataset.augmentations.get("synthetic_outlier", None) is not None:
        syn_class_count, syn_class_label = len(train_labels), len(label_remap)
        logger.info(
            "\tIncreased train classes by 1 to include synthetic outliers class for binary in-vs-out separation"
        )
        logger.info(f"\tSynthetic outlier assigned to class {syn_class_label} with count {syn_class_count}")
    logger.info(f"\tLabel remap: {label_remap}")
    logger.info(f"\tTrain classes counts: {dict(zip(*np.unique(train_labels, return_counts=True), strict=False))}")
    logger.info(f"\tTest classes counts: {dict(zip(*np.unique(test_labels, return_counts=True), strict=False))}")

    # Assuming train_labels is your array of training labels
    unique_classes, class_counts = np.unique(train_labels, return_counts=True)
    total_samples = len(train_labels)
    class_weights = total_samples / (len(unique_classes) * class_counts)
    logger.info(f"\tClass Weights: {dict(zip(unique_classes, class_weights, strict=False))}")

    # Ensure the input data has the correct shape
    if train_data.ndim != 3 or test_data.ndim != 3:
        raise ValueError("Input data must have shape (n_instances, n_timestamps, n_features).")

    # Resize train and test data seq length to max_seq_len if provided in dataset.args
    if config.dataset.args.get("max_seq_len", None) is not None:
        orig_seq_len = train_data.shape[1]
        max_seq_len = config.dataset.args.max_seq_len
        # Resize the data to the maximum sequence length if it exceeds the original length
        if orig_seq_len > max_seq_len:
            train_data = resize_time_series(train_data, max_seq_len)
            test_data = resize_time_series(test_data, max_seq_len)
            logger.info(f"\tResized train & test seq len from {orig_seq_len} to max seq len {max_seq_len}")

    logger.info(
        f"Loading data... Loader: {loader_type}, Dataset: {dataset_name}, "
        f"Train Shape (bef. transpose): train={train_data.shape}, test={test_data.shape}"
    )
    ts_length, num_features = train_data.shape[1], train_data.shape[2]  # before any potential transposition

    # Introduce random NaNs into train and test data
    # nan_ratio = 0.25  # 25% of the data will be replaced with NaN values
    # from ood_detection.utils.data.common import introduce_random_nans
    # train_data = introduce_random_nans(train_data, nan_ratio=nan_ratio)
    # test_data = introduce_random_nans(test_data, nan_ratio=nan_ratio)

    # We assume encoder require features second dimension
    train_data = train_data.transpose(0, 2, 1)
    test_data = test_data.transpose(0, 2, 1)
    logger.info(f"Shape (after transpose to set feature second axis): train={train_data.shape}, test={test_data.shape}")

    # Remove invalid instances with NaN values
    train_data = remove_invalid_instances(train_data)
    test_data = remove_invalid_instances(test_data)

    # save mean and std dev vectors to the models dir
    np.save(os.path.join(config.experiment_models_dir, "mean.npy"), mean)
    np.save(os.path.join(config.experiment_models_dir, "std.npy"), std)

    # Get transforms
    transforms = get_transform(config, train_data, mean, std)

    # Create custom dataset instances for training and testing
    train_dataset = init_dataset(
        dataset_type,
        config=config,
        ts_length=ts_length,
        num_features=num_features,
        data=train_data,
        labels=train_labels,
        label_remap=label_remap,
        transforms_dict={
            "transform": transforms["train"],
            "secondary_transform": transforms["test"],
            "synthetic_outlier_gen_transform": transforms["synthetic_outlier"],
        },
    )
    # Note: Test dataset is always CustomTimeSeriesDataset
    test_dataset = init_dataset(
        "CustomTimeSeriesDataset",
        config=config,
        ts_length=ts_length,
        num_features=num_features,
        data=test_data,
        labels=test_labels,
        label_remap=label_remap,
        transforms_dict={"transform": transforms["test"]},
    )

    # Create DataLoaders for training and testing
    cpus_per_proc = max(2, (multiprocessing.cpu_count() - 8) // 8)
    train_workers = min(4, cpus_per_proc - 1)
    test_workers = max(2, train_workers // 2)

    logger.info(
        f"Dataloader workers: Train workers={train_workers}, Test workers={test_workers} with {cpus_per_proc} cpus per training process"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataloader.args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=train_workers,
        prefetch_factor=2,
        persistent_workers=(train_workers > 0),
        pin_memory=config.device == "cuda" and torch.cuda.is_available(),
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataloader.args.batch_size * 4,
        shuffle=False,
        drop_last=False,
        prefetch_factor=2,
        num_workers=test_workers,
        persistent_workers=(test_workers > 0),
        pin_memory=config.device == "cuda" and torch.cuda.is_available(),
    )

    logger.info(f"Loaders length: Train: {len(train_loader)}, Test: {len(test_loader)}")

    return train_loader, test_loader, transforms, class_weights
