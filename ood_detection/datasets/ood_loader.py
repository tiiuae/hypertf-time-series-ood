from collections.abc import Callable
from copy import deepcopy
from functools import partial

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ood_detection.datasets import init_dataset
from ood_detection.datasets.loader import load_ucr, load_uea
from ood_detection.utils.common import TermColors, TermColors as Tc
from ood_detection.utils.data.common import (
    filter_arr_by_class,
    filter_non_id_datasets_by_type,
)
from ood_detection.utils.data.info import UCR_DATASETS_TYPE, UEA_DATASETS_TYPE
from ood_detection.utils.data.loader import load_and_preprocess_timeseries_data
from ood_detection.utils.logger import EXP_LOGGER_NAME, LoggerSingleton


def get_ood_dataloader(
    config: DictConfig,
    ood_dataset: str,
    load_function: Callable,
    id_seq_len: int | None = None,
    id_feat_len: int | None = None,
    transform: Callable | None = None,
    verbose: bool = False,
) -> DataLoader:
    """
    Args:
        config: Config object.
        ood_dataset: Name of the OOD dataset.
        load_function: Function to load the OOD dataset.
        id_seq_len: ID seq length for time series. If None, use ood_data seq length.
        id_feat_len: ID feature length. If None, use ood_data feature length.
        transform: Transform function to apply to the data.
        verbose: If True, print additional information.
    """
    ood_data, ood_labels, label_remap, ref_seq_len, ref_feat_len = load_and_preprocess_timeseries_data(
        config=config,
        dataset_name=ood_dataset,
        load_function=load_function,
        id_seq_len=id_seq_len,
        id_feat_len=id_feat_len,
        verbose=verbose,
        oe_feature_strategy="pre_match_features_to_id",
    )

    # Create a single dataset instance
    ood_dataset_instance = init_dataset(
        "CustomTimeSeriesDataset",
        config=config,
        ts_length=ref_seq_len,
        num_features=ref_feat_len,
        data=ood_data,
        labels=ood_labels,
        label_remap=label_remap,
        transforms_dict={"transform": transform},
    )

    # Create a DataLoader
    ood_loader = DataLoader(ood_dataset_instance, batch_size=1024, shuffle=False, drop_last=False, num_workers=0)
    return ood_loader


def get_near_ood_loaders(config: DictConfig, transform: Callable | None = None) -> dict[str, DataLoader]:
    """
    Loads and prepares near OOD dataset by taking specific classes from the in-distribution dataset,
    instead of using a separate OOD dataset.
    Merges the OOD class train/test splits.

    Args:
        config (DictConfig): Configuration object.
        transform (Optional[Callable]): Transform function to apply to the data.

    Returns:
        dict: A dictionary of OOD DataLoaders {dataset_name: DataLoader}.
    """
    logger = LoggerSingleton.get_logger(EXP_LOGGER_NAME)
    dataset_name = config.dataset.args.name  # e.g., 'ECG5000', 'Epilepsy', etc
    loader_type = config.dataset.args.loader  # 'UCR', 'UEA'
    id_classes_to_drop = config.dataset.args.get("id_classes", None)  # List of ID classes to drop from the ID dataset

    logger.info("")
    logger.info(f"Loading near OOD dataset for loader {loader_type} and dataset {dataset_name}...")
    ood_loaders = {}
    if loader_type == "UCR":
        # filter out the id classes from the same dataset
        near_ood_filter = partial(filter_arr_by_class, class_names=id_classes_to_drop, include_classes=False)
        load_function = partial(load_ucr, data_filter_func=near_ood_filter)
    elif loader_type == "UEA":
        # filter out the id classes from the same dataset
        near_ood_filter = partial(filter_arr_by_class, class_names=id_classes_to_drop, include_classes=False)
        load_function = partial(load_uea, data_filter_func=near_ood_filter)
    else:
        message = f"Unsupported near OOD loader type: {loader_type}"
        logger.error(message)
        raise ValueError(message)

    ood_loaders[dataset_name] = get_ood_dataloader(config, dataset_name, load_function, None, None, transform=transform)
    # near ood data info logging
    ood_data = ood_loaders[dataset_name].dataset.data
    ood_labels = ood_loaders[dataset_name].dataset.labels
    if np.isnan(ood_data).any():
        logger.warning(f"{Tc.WARN}train_data contains NaN values.{Tc.ENDC}")
    logger.info(f"\tOOD data shape: {ood_data.shape}")
    logger.info(f"\tOOD classes counts: {dict(zip(*np.unique(ood_labels, return_counts=True), strict=False))}")

    return ood_loaders


def get_far_ood_loaders(
    config: DictConfig, id_seq_len: int, id_feat_len: int, transform: Callable | None = None
) -> dict[str, DataLoader]:
    """
    Loads and prepares separate datasets as far OOD datasets for evaluation.
    Merges the OOD train/test splits.
    Ensures consistency in time series length and number of features.

    Args:
        config (DictConfig): Configuration object.
        id_seq_len (int): Reference sequence length of ID dataset for resizing OOD datasets.
        id_feat_len (int): Reference feature length of ID dataset for resizing OOD datasets.
        transform (Optional[Callable]): Transform function to apply to the data.

    Returns:
        dict: A dictionary of OOD DataLoaders {dataset_name: DataLoader}.
    """
    logger = LoggerSingleton.get_logger(EXP_LOGGER_NAME)
    id_dataset = config.dataset.args.name  # e.g., 'ECG5000', 'Epilepsy', etc
    loader_type = config.dataset.args.loader  # 'UCR', 'UEA'

    # Get list of all datasets in the selected archive
    if loader_type == "UCR":
        load_function = partial(load_ucr)
        all_datasets = list(UCR_DATASETS_TYPE.keys())
    elif loader_type == "UEA":
        load_function = partial(load_uea)
        all_datasets = list(UEA_DATASETS_TYPE.keys())
    else:
        message = f"{TermColors.FAIL}Unsupported far OOD loader type: {loader_type}{TermColors.ENDC}"
        logger.error(message)
        raise ValueError(message)

    logger.info("")
    far_ood_subtype = config.ood_eval.data.far_ood_subtype
    assert far_ood_subtype in {"same", "diff", "all"}, f"Invalid far_ood_subtype: {far_ood_subtype}"

    # Remove the in-distribution dataset from the OOD candidates
    ood_datasets = [d for d in all_datasets if d != id_dataset]
    same_type_ood_datasets = filter_non_id_datasets_by_type(
        ood_datasets, id_dataset, loader_type, use_most_similar_if_no_match=True, logger=logger
    )

    if far_ood_subtype == "same":
        logger.info("Using OOD datasets of the same subtype as ID dataset")
        ood_datasets = same_type_ood_datasets
    elif far_ood_subtype == "diff":
        logger.info("Using OOD datasets of different subtype from ID dataset")
        ood_datasets = [d for d in ood_datasets if d not in same_type_ood_datasets]
    elif far_ood_subtype == "all":
        logger.info("Using all far OOD datasets (both same and different subtypes)")
        # ood_datasets already includes all minus the ID dataset

    ood_loaders = {}
    logger.info(
        f"For {loader_type} ID dataset {id_dataset}, Loading {len(ood_datasets)} far OOD datasets:'{ood_datasets}'"
    )
    logger.info("")
    for ood_dataset in ood_datasets:
        try:
            far_ood_config = deepcopy(config)
            far_ood_config.dataset.args.name = ood_dataset
            # Set id_classes to None to load all classes for OOD datasets
            far_ood_config.dataset.args.id_classes = None
            far_ood_config.dataset.args.id_classes_remap = None

            ood_loader = get_ood_dataloader(
                far_ood_config, ood_dataset, load_function, id_seq_len, id_feat_len, transform
            )
            # Store the OOD loader
            ood_loaders[ood_dataset] = ood_loader

        except Exception as e:
            logger.error(f"{TermColors.FAIL}Error loading far ood dataset '{ood_dataset}': {e}{TermColors.ENDC}")
    if not ood_datasets:
        msg = f"{TermColors.FAIL}No OOD datasets found for ID dataset '{id_dataset}'{TermColors.ENDC}"
        logger.error(msg)
        raise ValueError(msg)

    return ood_loaders
