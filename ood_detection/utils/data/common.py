import logging
from typing import Any

from numba import njit, prange
import numpy as np
from scipy.interpolate import interp1d

from ood_detection.utils.common import TermColors as Tc
from ood_detection.utils.data.info import (
    UCR_DATASETS_TYPE,
    UCR_MODALITY_SIMILARITY,
    UEA_DATASETS_TYPE,
    UEA_MODALITY_SIMILARITY,
)


def remap_labels_to_sequential_ids(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    label_value_to_key: dict[Any, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[Any, int]]:
    """Remap labels to a contiguous ``[0, N-1]`` range.

    Args:
        train_labels: Training labels to remap.
        test_labels: Testing labels to remap using the same mapping as ``train_labels``.
        label_value_to_key: Optional mapping from the label values in ``train_labels`` to
            human-readable keys (e.g. original class names). When provided, the returned
            ``label_remap`` dict uses these keys instead of the raw label values.

    Returns:
        Tuple containing the remapped ``train_labels``, ``test_labels`` and the
        ``label_remap`` dictionary.
    """

    unique_labels = np.unique(train_labels)
    value_to_new_idx = {label: idx for idx, label in enumerate(unique_labels)}

    remap_fn = np.vectorize(value_to_new_idx.get, otypes=[np.int64])
    remapped_train = remap_fn(train_labels).astype(np.int64)
    remapped_test = remap_fn(test_labels).astype(np.int64)

    if label_value_to_key is not None:
        label_remap = {label_value_to_key[label]: new_idx for label, new_idx in value_to_new_idx.items()}
    else:
        label_remap = value_to_new_idx.copy()

    return remapped_train, remapped_test, label_remap


def filter_arr_by_class(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    class_names: list = None,
    include_classes: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters the train and test data so that only the specified classes are included/excluded.
    Args:
        train_x: Training data array.
        train_y: Training labels array.
        test_x: Testing data array.
        test_y: Testing labels array.
        class_names: List of class names (str) or labels (int). if None, return as is
        include_classes: If True, includes rows with class names matching the list.
    """
    if class_names is None:
        return train_x, train_y, test_x, test_y
    train_mask = np.isin(train_y, class_names)
    test_mask = np.isin(test_y, class_names)
    if include_classes:
        return train_x[train_mask], train_y[train_mask], test_x[test_mask], test_y[test_mask]
    return train_x[~train_mask], train_y[~train_mask], test_x[~test_mask], test_y[~test_mask]


def match_features(data: np.ndarray, target_features: int) -> np.ndarray:
    """
    Adjusts the number of features in the dataset by either truncating or replicating features.

    Args:
        data (np.ndarray): Input time series of shape (N, L, F).
        target_features (int): Desired number of features.

    Returns:
        np.ndarray: Adjusted time series of shape (N, L, target_features).
    """
    N, L, F = data.shape

    if target_features < F:
        # Drop extra features to match target size
        return data[:, :, :target_features]

    elif target_features > F:
        # Replicate features cyclically until we reach target features
        num_repeats = (target_features // F) + 1  # Ensure enough repetitions
        # Tile along feature axis
        expanded_data = np.tile(data, (1, 1, num_repeats))
        return expanded_data[:, :, :target_features]  # Trim excess features

    return data


def resize_time_series_fast(data: np.ndarray, target_length: int) -> np.ndarray:
    """
    Vectorized resizing of time series to target length using linear interpolation.

    Args:
        data (np.ndarray): Input shape (N, L, F)
        target_length (int): Target time series length

    Returns:
        np.ndarray: Output shape (N, target_length, F)
    """
    N, L, F = data.shape

    # Original and target linspace
    orig_x = np.linspace(0, 1, L)
    target_x = np.linspace(0, 1, target_length)

    # Reshape for broadcasting: (N*F, L)
    data_reshaped = data.transpose(0, 2, 1).reshape(-1, L)

    # Apply interpolation per row (N*F rows total)
    interp_func = interp1d(orig_x, data_reshaped, kind="linear", axis=1, fill_value="extrapolate", assume_sorted=True)
    resized = interp_func(target_x)  # Shape: (N*F, target_length)

    # Reshape back to (N, target_length, F)
    resized = resized.reshape(N, F, target_length).transpose(0, 2, 1).astype(np.float32)

    return resized


def resize_time_series_less_memory(data: np.ndarray, target_length: int) -> np.ndarray:
    """
    Resizes the time series to the target length using interpolation.

    Args:
        data (np.ndarray): Input time series of shape (N, L, F) where L is the original length.
        target_length (int): Desired time series length.

    Returns:
        np.ndarray: Resized time series of shape (N, target_length, F).
    """
    N, L, F = data.shape
    new_data = np.zeros((N, target_length, F), dtype=np.float32)

    for i in range(N):
        for j in range(F):
            interp_func = interp1d(np.linspace(0, 1, L), data[i, :, j], kind="linear", fill_value="extrapolate")
            new_data[i, :, j] = interp_func(np.linspace(0, 1, target_length))

    return new_data


@njit(parallel=True, fastmath=True, cache=True)
def resize_time_series_minimal_memory(data: np.ndarray, target_length: int) -> np.ndarray:
    N, L, F = data.shape
    output = np.empty((N, target_length, F), dtype=np.float32)

    # Compute scale factor once
    scale = (L - 1) / max(1, (target_length - 1))

    for n in prange(N):  # Parallel across samples
        for t in range(target_length):
            # Direct interpolation without intermediate arrays
            src_pos = t * scale
            src_idx = int(src_pos)

            if src_idx >= L - 1:
                # Copy last value
                for f in range(F):
                    output[n, t, f] = data[n, L - 1, f]
            else:
                # Linear interpolation
                weight = src_pos - src_idx
                for f in range(F):
                    output[n, t, f] = data[n, src_idx, f] * (1 - weight) + data[n, src_idx + 1, f] * weight

    return output


def resize_time_series(data: np.ndarray, target_length: int) -> np.ndarray:
    """
    Chooses the best resizing strategy based on estimated memory size.

    Args:
        data (np.ndarray): Input array of shape (N, L, F)
        target_length (int): Desired time series length

    Returns:
        np.ndarray: Resized array of shape (N, target_length, F)
    """
    return resize_time_series_minimal_memory(data.copy(), target_length)


def interpolate_nans(data: np.ndarray) -> np.ndarray:
    """
    Performs linear interpolation for NaN values in time series data.
    Ensures that start and end values are filled with nearest valid values.

    Args:
        data (np.ndarray): Input time series of shape (N, L, F).

    Returns:
        np.ndarray: Interpolated time series of the same shape.
    """
    N, L, F = data.shape
    interpolated_data = np.copy(data)

    for i in range(N):  # Iterate over instances
        for j in range(F):  # Iterate over features
            ts = data[i, :, j]  # Extract 1D time series for interpolation
            nan_mask = np.isnan(ts)  # Find NaN indices

            if np.any(nan_mask):  # Only process if there are NaNs
                valid_idx = np.where(~nan_mask)[0]  # Indices of non-NaN values
                valid_values = ts[valid_idx]  # Corresponding valid values

                if len(valid_idx) == 0:  # Entire series is NaN (should not happen in well-formed data)
                    raise ValueError(f"Sample {i}, feature {j} is entirely NaN!")

                # Linear interpolation function
                interp_func = interp1d(valid_idx, valid_values, kind="linear", fill_value="extrapolate")

                # Replace NaNs with interpolated values
                interpolated_data[i, :, j] = interp_func(np.arange(L))

    return interpolated_data


def interp1d_like_np(ts, idx, xp, fp):
    """
    Fast fill & linear-extrapolate like scipy.interp1d(..., extrapolate).
    ts: original 1D array (with NaNs)
    idx: np.arange(len(ts))
    xp, fp: valid indices & values
    """
    # 1) interpolate and clamp
    y = np.interp(idx, xp, fp)

    # 2) compute left slope and apply to idx < xp[0]
    if xp.size > 1:
        slope_left = (fp[1] - fp[0]) / (xp[1] - xp[0])
        left_mask = idx < xp[0]
        y[left_mask] = fp[0] + slope_left * (idx[left_mask] - xp[0])

        # 3) compute right slope and apply to idx > xp[-1]
        slope_right = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        right_mask = idx > xp[-1]
        y[right_mask] = fp[-1] + slope_right * (idx[right_mask] - xp[-1])

    return y


def interpolate_nans_fast(data: np.ndarray):
    """
    Performs fast inear interpolation for NaN values in time series data.
    Extrapolates NaN values at the start and end of the series.
    Ensures that start and end values are filled with nearest valid values.

    Args:
        data (np.ndarray): Input time series of shape (N, L, F).

    Returns:
        np.ndarray: Interpolated time series of the same shape.
    """
    N, L, F = data.shape
    flat = data.transpose(0, 2, 1).reshape(-1, L)
    idx = np.arange(L)

    for k, ts in enumerate(flat):
        mask = np.isnan(ts)
        if not mask.any():
            continue
        valid = ~mask
        xp = idx[valid]
        fp = ts[valid]
        flat[k] = interp1d_like_np(ts, idx, xp, fp)

    return flat.reshape(N, F, L).transpose(0, 2, 1)


def pad_with_nans(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pads a 2D time series with NaNs to match the specified target length.

    Args:
        x (np.ndarray): Input time series of shape (n_features, seq_len).
        target_len (int): Desired sequence length after padding.

    Returns:
        np.ndarray: Padded time series of shape (n_features, target_len).
    """
    seq_len = x.shape[1]
    if seq_len == target_len:
        return x
    pad_width = target_len - seq_len
    padding = np.full((x.shape[0], pad_width), np.nan, dtype=x.dtype)
    return np.concatenate([x, padding], axis=1)


def introduce_random_nans(data: np.ndarray, nan_ratio=0.1):
    """Function to introduce random NaN values"""
    # Ensure the data is a NumPy array
    data = data.copy()  # Avoid modifying the original dataset
    total_elements = data.size
    num_nans = int(total_elements * nan_ratio)  # Number of NaNs to introduce

    # Randomly select indices to replace with NaN
    nan_indices = np.random.choice(total_elements, size=num_nans, replace=False)
    flat_data = data.flatten()  # Flatten the array for easy indexing
    flat_data[nan_indices] = np.nan  # Assign NaN to selected indices

    return flat_data.reshape(data.shape)  # Reshape back to original dimensions


def remove_invalid_instances(data: np.ndarray):
    """
    Remove multivariate time series where any feature (channel) is fully NaN across all time steps.

    Parameters:
        data (numpy.ndarray): Input data, expected to be a 3D array with shape [num_samples, feat, time].

    Returns:
        numpy.ndarray: Filtered data without instances containing any fully NaN channels.
    """
    # Identify samples with at least one fully NaN channel
    # Check if all values along time are NaN per feature
    fully_nan_channel_mask = np.isnan(data).all(axis=2)
    # Check if any channel is fully NaN for each sample
    any_fully_nan_channel = fully_nan_channel_mask.any(axis=1)

    # Count the number of samples to remove
    num_invalid = np.sum(any_fully_nan_channel)

    # Print message if invalid samples are removed
    if num_invalid > 0:
        print(f"Removed {num_invalid} multivariate time series containing at least one fully NaN channel.")

    # Keep only valid samples
    valid_mask = ~any_fully_nan_channel
    return data[valid_mask]


def filter_non_id_datasets_by_type(
    non_id_datasets: list[str],
    in_dist_dataset: str,
    loader_type: str,
    use_most_similar_if_no_match: bool = False,
    logger: logging.Logger | None = None,
) -> list[str]:
    """
    Filters a list of Non-ID datasets to only include those of the same type as the ID dataset.

    Args:
        non_id_datasets: List of all available OOD dataset names.
        in_dist_dataset: ID dataset name.
        loader_type: Loader type ('UCR', 'UEA').
        use_most_similar_if_no_match: Whether to use the most similar dataset type if no match for dsets of same type as ID dset.
        logger: Optional logger for logging.

    Returns:
        List of OOD dataset names belonging to the same type as the in-distribution dataset or
        closest modality match if `use_most_similar_if_no_match` is True.
    """
    # Identify the type of the in-distribution dataset
    datasets_type = {"UCR": UCR_DATASETS_TYPE, "UEA": UEA_DATASETS_TYPE}[loader_type]
    in_dist_type = datasets_type.get(in_dist_dataset)

    if in_dist_type is None:
        raise ValueError(f"Dataset '{in_dist_dataset}' not found in dataset_table.")

    # Filter Non-ID datasets based on type
    filtered_non_id_datasets = [d for d in non_id_datasets if datasets_type.get(d) == in_dist_type]

    if use_most_similar_if_no_match and not filtered_non_id_datasets:
        # If no match for dsets of same type as ID dset, use the most similar dataset type
        if loader_type == "UCR":
            most_similar_type = UCR_MODALITY_SIMILARITY[in_dist_type]
        elif loader_type == "UEA":
            most_similar_type = UEA_MODALITY_SIMILARITY[in_dist_type]
        else:
            raise NotImplementedError(f"Most similar modality not found for loader type: {loader_type}")

        warn_msg = (
            f"{Tc.WARN}No other datasets of the ID dataset type ({in_dist_type}) found for ID dset {in_dist_dataset}. "
            + f"Using subtype of the most similar modality ({most_similar_type}) instead.{Tc.ENDC}"
        )
        if logger is not None:
            logger.warning(warn_msg)
        else:
            print(warn_msg)
        filtered_non_id_datasets = [d for d in non_id_datasets if datasets_type.get(d) == most_similar_type]

    return filtered_non_id_datasets


def limit_samples_per_class(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    per_class_limit: int | dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep at most `per_class_limit` examples per class in both splits."""
    if per_class_limit is None:
        return x_train, y_train, x_test, y_test

    def _limit_split(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            limit = per_class_limit if isinstance(per_class_limit, int) else per_class_limit.get(cls)
            if limit is None or cls_idx.size <= limit:
                indices.append(cls_idx)
                continue
            # take the first occurances of the classes
            chosen = np.sort(cls_idx[:limit])
            indices.append(chosen)

        keep = np.concatenate(indices)
        keep.sort()
        return x[keep], y[keep]

    x_train_limited, y_train_limited = _limit_split(x_train, y_train)
    x_test_limited, y_test_limited = _limit_split(x_test, y_test)
    return x_train_limited, y_train_limited, x_test_limited, y_test_limited
