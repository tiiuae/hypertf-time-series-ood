from collections.abc import Callable
import hashlib
import inspect
import json
import os

from aeon.datasets import load_classification
import numpy as np
from omegaconf import DictConfig

from ood_detection.utils.common import TermColors as Tc
from ood_detection.utils.data.common import (
    interpolate_nans_fast,
    limit_samples_per_class,
    match_features,
    pad_with_nans,
    remap_labels_to_sequential_ids,
    resize_time_series,
)
from ood_detection.utils.logger import EXP_LOGGER_NAME, LoggerSingleton


def load_ucr_cached_raw(
    cache_dir: str, cache_key: str, config: DictConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Loads raw UCR train and test data from .tsv files and caches parsed arrays.
    Metadata is stored separately as JSON due to dict type incompatibility with np.savez.

    Args:
        cache_dir: Directory to save cache.
        cache_key: Key to use for caching.
        config: Hydra config containing dataset details.

    Returns:
        x_train, y_train, x_test, y_test, metadata
    """
    logger = LoggerSingleton.get_logger(EXP_LOGGER_NAME)
    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, f"{cache_key}.npz")
    json_path = os.path.join(cache_dir, f"{cache_key}_metadata.json")

    if os.path.exists(npz_path) and os.path.exists(json_path):
        data = np.load(npz_path, allow_pickle=True)
        with open(json_path, encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(f"{Tc.OKBLUE}Loading cached data from {npz_path} and metadata from {json_path}.{Tc.ENDC}")
        return data["x_train"], data["y_train"], data["x_test"], data["y_test"], metadata

    logger.info(f"No cached data found. Parsing fresh data and caching to {npz_path} and {json_path}.")

    # Load train and test using aeon
    dataset = config.dataset.args.name
    data_root_dir = config.dataset.args.get("root", "data")
    ucr_dir = os.path.join(data_root_dir, "UCR")

    x_train, y_train, metadata = load_classification(
        dataset,
        split="train",
        extract_path=ucr_dir,
        load_equal_length=False,
        load_no_missing=False,
        return_metadata=True,
    )
    x_test, y_test, _ = load_classification(
        dataset,
        split="test",
        extract_path=ucr_dir,
        load_equal_length=False,
        load_no_missing=False,
        return_metadata=True,
    )

    if not metadata.get("equallength", True):
        max_len = max(max(x.shape[1] for x in x_train), max(x.shape[1] for x in x_test))
        x_train = [pad_with_nans(x.astype(np.float32), max_len).T for x in x_train]
        x_test = [pad_with_nans(x.astype(np.float32), max_len).T for x in x_test]
    else:
        x_train = [x.astype(np.float32).T for x in x_train]
        x_test = [x.astype(np.float32).T for x in x_test]

    # Convert list to numpy array
    x_train = np.stack(x_train).astype(np.float32)
    x_test = np.stack(x_test).astype(np.float32)

    # Save data arrays
    np.savez(npz_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    # Save metadata as JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    return x_train, y_train, x_test, y_test, metadata


def load_uea_cached_raw(
    cache_dir: str, cache_key: str, config: DictConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Loads raw UEA train and test data using aeon and caches parsed arrays and metadata.

    Args:
        cache_dir: Directory to save cache.
        cache_key: Key to use for caching.
        config: Config object containing dataset info.

    Returns:
        x_train, y_train, x_test, y_test, metadata
    """
    logger = LoggerSingleton.get_logger(EXP_LOGGER_NAME)
    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, f"{cache_key}.npz")
    json_path = os.path.join(cache_dir, f"{cache_key}_metadata.json")

    if os.path.exists(npz_path) and os.path.exists(json_path):
        data = np.load(npz_path, allow_pickle=True)
        with open(json_path, encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(f"{Tc.OKBLUE}Loading cached data from {npz_path} and metadata from {json_path}.{Tc.ENDC}")
        return data["x_train"], data["y_train"], data["x_test"], data["y_test"], metadata

    logger.info(f"No cached data found. Parsing fresh data and caching to {npz_path} and {json_path}.")

    dataset = config.dataset.args.name
    data_root_dir = config.dataset.args.get("root", "data")
    uea_dir = os.path.join(data_root_dir, "UEA")

    # aeon handles download and extraction
    x_train, y_train, metadata = load_classification(
        dataset,
        split="train",
        extract_path=uea_dir,
        load_equal_length=False,
        load_no_missing=False,
        return_metadata=True,
    )
    x_test, y_test, _ = load_classification(
        dataset,
        split="test",
        extract_path=uea_dir,
        load_equal_length=False,
        load_no_missing=False,
        return_metadata=True,
    )

    # If sequences are not all equal length, pad with NaNs
    if not metadata.get("equallength", True):
        max_len = max(max(x.shape[1] for x in x_train), max(x.shape[1] for x in x_test))
        x_train = [pad_with_nans(x.astype(np.float32), max_len).T for x in x_train]
        x_test = [pad_with_nans(x.astype(np.float32), max_len).T for x in x_test]
    else:
        x_train = [x.astype(np.float32).T for x in x_train]
        x_test = [x.astype(np.float32).T for x in x_test]

    # Convert list to numpy array
    x_train = np.stack(x_train).astype(np.float32)
    x_test = np.stack(x_test).astype(np.float32)

    # Save arrays
    np.savez(npz_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    # Save metadata
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    return x_train, y_train, x_test, y_test, metadata


def load_ucr(
    config: DictConfig, data_filter_func: Callable | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Returns x_train, y_train, x_test, y_test of a UCR dataset in shape [num_samples, seq_len, feats_len].
    Downloads the entire UCR set using aeon if it doesn't exist.

    Args:
        config: Config object. config.dataset.arg.name must be a UCR dataset. e.g. ECG200, GunPoint etc.
        data_filter_func: Function to filter the train & test numpy arrays.
    """
    assert config.dataset.args.loader == "UCR", "Loader must be UCR"
    dataset = config.dataset.args.name
    logger = LoggerSingleton.get_logger(EXP_LOGGER_NAME)

    data_hash = hashlib.md5((inspect.getsource(load_ucr_cached_raw)).encode()).hexdigest()[
        :8
    ]  # use load_ucr_cached_raw src code as cache key
    x_train, y_train, x_test, y_test, metadata = load_ucr_cached_raw(".cache/UCR", f"{dataset}_{data_hash}", config)

    # Ensure shape [samples, seq_len, feat_dim]
    if x_train.ndim == 2:  # univariate series
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

    # Move the labels from {'class1', 'class2', ...} to {0, ..., L-1}
    y_train, y_test, label_remap = remap_labels_to_sequential_ids(y_train, y_test)

    # Filter the data (to keep/remove classes) if a filter function is provided
    if data_filter_func is not None:
        logger.info(f"\tBefore data filtering, label remap: {label_remap}")
        x_train, y_train, x_test, y_test = data_filter_func(x_train, y_train, x_test, y_test)

        # Preserve original label names after filtering while keeping IDs contiguous
        value_to_label = {idx: label for label, idx in label_remap.copy().items()}
        y_train, y_test, label_remap = remap_labels_to_sequential_ids(
            y_train,
            y_test,
            label_value_to_key=value_to_label,
        )

    # set max_sample_per_class to limit dataset sizes for tests only
    max_sample_per_class = config.dataset.args.get("max_sample_per_class", None)
    if max_sample_per_class is not None:
        logger.warning(
            f"{Tc.WARN}Limiting per class dataset size to {max_sample_per_class}. "
            f"Only enable this for pytests.{Tc.ENDC}"
        )
        x_train, y_train, x_test, y_test = limit_samples_per_class(
            x_train, y_train, x_test, y_test, max_sample_per_class
        )

    # Compute mean and std for normalization
    mean = np.nanmean(x_train, keepdims=True).astype(np.float32)
    std = np.nanstd(x_train, keepdims=True).astype(np.float32)
    std[std == 0] = 1  # avoid divide-by-zero

    return x_train, y_train, x_test, y_test, mean, std, label_remap


def load_uea(
    config: DictConfig, data_filter_func: Callable | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Returns x_train, y_train, x_test, y_test of a UEA dataset in shape [num_samples seq_len, feats_len].
    Downloads the entire UEA set using aeon if it doesn't exist.
    Args:
        config: Config object. config.dataset.arg.name must be a UCR dataset. e.g. Epilepsy, RacketSports etc.
        data_filter_func: Function to filter the train & test numpy arrays.
    """
    assert config.dataset.args.loader == "UEA", "Loader must be UEA"
    dataset = config.dataset.args.name
    logger = LoggerSingleton.get_logger(EXP_LOGGER_NAME)

    # X data has shape [num_samples, seq_len, feats_len]
    data_hash = hashlib.md5((inspect.getsource(load_uea_cached_raw)).encode()).hexdigest()[
        :8
    ]  # use load_uea_cached_raw src code as cache key
    x_train, y_train, x_test, y_test, metadata = load_uea_cached_raw(".cache/UEA", f"{dataset}_{data_hash}", config)

    # Compute mean and std deviation without applying StandardScaler
    # Compute mean across instances and time
    mean = np.nanmean(x_train, axis=(0, 1)).astype(np.float32)
    std = np.nanstd(x_train, axis=(0, 1)).astype(np.float32)

    # Change shape to match final post-transposed x_train shape as [bsize, feats_len, seq_len]
    mean = mean[np.newaxis, ..., np.newaxis]  # [1, n_feat, 1]
    std = std[np.newaxis, ..., np.newaxis]  # [1, n_feat, 1]

    # Move the labels from {'class1', 'class2', ...} to {0, ..., L-1}
    y_train, y_test, label_remap = remap_labels_to_sequential_ids(y_train, y_test)

    # Filter the data (to keep/remove classes) if a filter function is provided
    if data_filter_func is not None:
        logger.info(f"\tBefore data filtering, label remap: {label_remap}")
        x_train, y_train, x_test, y_test = data_filter_func(x_train, y_train, x_test, y_test)

        # Preserve original label names after filtering while keeping IDs contiguous
        value_to_label = {idx: label for label, idx in label_remap.copy().items()}
        y_train, y_test, label_remap = remap_labels_to_sequential_ids(
            y_train,
            y_test,
            label_value_to_key=value_to_label,
        )

    # Ensure data is float32
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # set max_sample_per_class to limit dataset sizes for tests only
    max_sample_per_class = config.dataset.args.get("max_sample_per_class", None)
    if max_sample_per_class is not None:
        logger.warning(
            f"{Tc.WARN}Limiting per class dataset size to {max_sample_per_class}. "
            f"Only enable this for pytests.{Tc.ENDC}"
        )
        x_train, y_train, x_test, y_test = limit_samples_per_class(
            x_train, y_train, x_test, y_test, max_sample_per_class
        )

    return x_train, y_train, x_test, y_test, mean, std, label_remap


def load_and_preprocess_timeseries_data(
    config: DictConfig,
    dataset_name: str,
    load_function: Callable,
    id_seq_len: int | None = None,
    id_feat_len: int | None = None,
    verbose: bool = False,
    oe_feature_strategy: str = "pre_match_features_to_id",
) -> tuple[np.ndarray, np.ndarray, dict, int, int]:
    """
    Loads and preprocesses time series data: merges splits, interpolates NaNs, resizes, matches features, transposes.
    Returns preprocessed data, labels, label remap, final sequence length, final feature dim.

    Args:
        config: OmegaConf config.
        dataset_name: Name of the dataset for logging.
        load_function: Callable that returns train/test splits.
        id_seq_len: Reference ID sequence length to resize to. If None, keep dataset length.
        id_feat_len: Reference ID feature count for matching. If None, keep dataset feature count.
        verbose: If True, log intermediate stats.
    """
    if oe_feature_strategy not in {"pre_match_features_to_id", "hybrid_feature_mix", "id_oe_mixup"}:
        raise ValueError(
            f"Invalid ood_eval.oe_feature_strategy: {oe_feature_strategy}. "
            "Must be 'pre_match_features_to_id' or 'hybrid_feature_mix' or 'id_oe_mixup'."
        )
    data_train, labels_train, data_test, labels_test, _, _, label_remap = load_function(config)

    # Merge train and test into a single set
    data = np.concatenate([data_train, data_test], axis=0)
    labels = np.concatenate([labels_train, labels_test], axis=0)

    if verbose:
        logger = LoggerSingleton.get_logger(EXP_LOGGER_NAME)
        logger.info(f"\t{dataset_name} label remap: {label_remap}")
        logger.info(
            f"\t{dataset_name} combined classes counts: {dict(zip(*np.unique(labels, return_counts=True), strict=False))}"
        )
        logger.info(f"\t{dataset_name} data shape before resize & feat. match (N, L, F): {data.shape}")

    # Interpolate NaNs only if NaNs present and input projection in model is disabled
    if np.isnan(data).any() and not config.model.args.input_projection.enabled:
        data = interpolate_nans_fast(data)

    _, seq_len, feat_len = data.shape
    ref_seq_len = seq_len if id_seq_len is None else id_seq_len
    ref_feat_len = feat_len if id_feat_len is None else id_feat_len

    if data.ndim != 3:
        raise ValueError(f"Skipping {dataset_name}: Invalid shape {data.shape}")

    if seq_len != ref_seq_len:
        data = resize_time_series(data, target_length=ref_seq_len)
    # Disable when creating hybrid OE mixes since feat matching is done in the __getitem__() later per-sample
    if oe_feature_strategy == "pre_match_features_to_id" and feat_len != ref_feat_len:
        data = match_features(data, target_features=ref_feat_len)

    if verbose:
        logger.info(f"\t{dataset_name} data shape after all processing (N, L, F): {data.shape}")
    final_seq_len, final_feat_len = data.shape[1], data.shape[2]

    data = data.transpose(0, 2, 1)  # [N, F, L]
    return data, labels, label_remap, final_seq_len, final_feat_len
