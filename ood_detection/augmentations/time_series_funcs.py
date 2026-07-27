import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import convolve1d, zoom


def jitter(x: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """
    Apply random Gaussian noise to time series data.

    Args:
        x (np.ndarray): Input array of shape [channels, time_steps].
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Jittered array of the same shape as input.
    """
    return x + np.random.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)


def scaling(x: np.ndarray, sigma: float = 1.1) -> np.ndarray:
    """
    Apply random scaling to time series data.

    Args:
        x (np.ndarray): Input array of shape [channels, time_steps].
        sigma (float): Standard deviation of the scaling factor.

    Returns:
        np.ndarray: Scaled array of the same shape as input.
    """
    time_steps = x.shape[-1]
    factor = np.random.normal(loc=2.0, scale=sigma, size=(1, time_steps))
    return (x * factor).astype(np.float32)


def permutation(x: np.ndarray, max_seg: int = 5, seg_mode: str = "random") -> np.ndarray:
    """
    Permute segments of a time series along the time_steps axis.

    Args:
        x (np.ndarray): Input array of shape [channels, time_steps].
        max_seg (int): Maximum number of segments to divide the sequence into.
        seg_mode (str): "random" for uneven splits, "equal" for fixed-length segments.

    Returns:
        np.ndarray: Array with permuted time steps (same permutation applied to all channels).
    """
    tm_steps = x.shape[-1]
    orig_steps = np.arange(tm_steps)
    num_segs = np.random.randint(1, max_seg + 1)
    num_segs = min(num_segs, tm_steps)

    if num_segs > 1:
        if seg_mode == "random":
            split_points = np.sort(np.random.choice(np.arange(1, tm_steps), num_segs - 1, replace=False))
            splits = np.split(orig_steps, split_points)
        else:  # seg_mode == "equal"
            splits = np.array_split(orig_steps, num_segs)

        permuted_split_idxs = np.random.permutation(len(splits))
        warp = np.concatenate([splits[i] for i in permuted_split_idxs])
        # Apply permutation to all channels
        return x[:, warp]
    # No permutation if only 1 segment
    return x


def random_crop_if_longer(x: np.ndarray, target_seq_len: int) -> np.ndarray:
    """
    Randomly crop the input sequence to a fixed target length.

    Args:
        x (np.ndarray): Input array of shape [channels, time_steps].
        target_seq_len (int): Desired sequence length after cropping.

    Returns:
        np.ndarray: Cropped array of shape [channels, target_seq_len],
                    or original if target_seq_len >= time_steps.
    """
    _, seq_len = x.shape

    if target_seq_len >= seq_len:
        return x

    start_idx = np.random.randint(0, seq_len - target_seq_len + 1)
    return x[:, start_idx : start_idx + target_seq_len].copy()


def random_crop_resize_views(x: np.ndarray, scale=(0.5, 1.0)):
    """
    Randomly crop and resize to original length. No overlap guarantee.
    Each view has independent random crop size and position.

    Args:
        x (np.ndarray): Input array of shape [channels, time_steps]
        scale (tuple): Range of scales for cropping (e.g., (0.5, 1.0))

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two independently cropped and resized views
    """
    _, length = x.shape
    if length == 0:
        raise ValueError("Input time series has zero length.")

    # Get random crop sizes for each view (independent)
    crop_scale1 = np.random.uniform(*scale)
    crop_scale2 = np.random.uniform(*scale)

    crop_size1 = max(1, int(round(length * crop_scale1)))
    crop_size2 = max(1, int(round(length * crop_scale2)))

    # Get random start positions for each view
    start1 = np.random.randint(0, length - crop_size1 + 1)
    start2 = np.random.randint(0, length - crop_size2 + 1)

    # Crop the samples
    crop1 = x[:, start1 : start1 + crop_size1]
    crop2 = x[:, start2 : start2 + crop_size2]

    # Resize each crop back to original length
    zf1 = [1, length / crop_size1]
    zf2 = [1, length / crop_size2]

    return zoom(crop1, zf1, order=1), zoom(crop2, zf2, order=1)


def random_crop_resize(x: np.ndarray, scale=(0.5, 1.0)) -> np.ndarray:
    """
    Randomly crop and resize to the original length.

    Args:
        x (np.ndarray): Input array of shape [channels, time_steps].
        scale (tuple): Range of scales for cropping (e.g., (0.5, 1.0)).
        min_overlap (float): Unused, retained for compatibility.

    Returns:
        np.ndarray: A single cropped and resized view.
    """
    _, length = x.shape

    # Ensure length is valid
    if length == 0:
        raise ValueError("Input time series has zero length.")

    # Determine crop size based on random scale
    crop_scale = np.random.uniform(*scale)
    crop_size = max(1, int(length * crop_scale))  # Ensure crop_size is at least 1

    # Random start index for crop
    start_idx = np.random.randint(0, length - crop_size + 1)

    # Crop the sample
    cropped_sample = x[:, start_idx : start_idx + crop_size]

    # Resize crop back to original length
    zoom_factors = [1, length / crop_size] if crop_size > 0 else [1, 1]
    resized_x = zoom(cropped_sample, zoom_factors, order=1)

    return resized_x


def random_crop_resize_with_overlap(x: np.ndarray, scale=(0.5, 1.0), min_overlap=0.2, ensure_different=True):
    _, length = x.shape
    if length == 0:
        raise ValueError("Input time series has zero length.")

    crop_scale = np.random.uniform(*scale)
    crop_size = max(1, int(round(length * crop_scale)))

    start1 = np.random.randint(0, length - crop_size + 1)

    # required overlap in samples (use ceil to avoid 0)
    min_ov = int(math.ceil(crop_size * float(min_overlap)))
    max_delta = crop_size - min_ov  # <= THIS is the key change

    low = max(0, start1 - max_delta)
    high = min(length - crop_size, start1 + max_delta)

    # pick start2 (try to avoid identical crops if possible)
    if ensure_different and high > low:
        start2 = start1
        while start2 == start1:
            start2 = np.random.randint(low, high + 1)
    else:
        start2 = np.random.randint(low, high + 1)

    crop1 = x[:, start1 : start1 + crop_size]
    crop2 = x[:, start2 : start2 + crop_size]

    zf = [1, length / crop_size]
    return zoom(crop1, zf, order=1), zoom(crop2, zf, order=1)


def resize(x: np.ndarray, size: int) -> np.ndarray:
    """
    Resizes the given 2D NumPy array to a new size.

    If the new size is equal to the original size of the second dimension of the array,
    a copy of the original array is returned.

    Parameters:
        x(numpy.ndarray): Input array of shape [num_features, seq_len].
        size(int): An integer specifying the new size of the second dimension of the array.

    Returns:
        A 2D NumPy array of the specified size that is a resized version of the original array.
    """
    num_features, seq_len = x.shape
    if size == seq_len:
        return x.copy()

    resized = np.empty((num_features, size))
    for i in range(num_features):
        ind = np.linspace(0, seq_len - 1, size)
        resized[i] = np.interp(ind, np.arange(seq_len), x[i])

    return resized


def resize_down_if_longer(x: np.ndarray, max_length: int) -> np.ndarray:
    """
    Resize down the input sequence only if its length exceeds max_length.
    x must be of shape [num_features, seq_len].

    Args:
        x (np.ndarray): Input array of shape [num_features, seq_len].
        max_length (int): Maximum length of the output sequence.

    Returns:
        np.ndarray: Resized array if seq_len > max_length, otherwise original array.
    """
    _, seq_len = x.shape
    if seq_len > max_length:
        return resize(x, max_length)
    return x.copy()  # No resizing needed


def horizontal_flip(x: np.ndarray) -> np.ndarray:
    """
    Flips the given NumPy array horizontally.
    Ensures the returned array is contiguous in memory.
    """
    return x[..., ::-1].copy()


def vertical_flip(x: np.ndarray) -> np.ndarray:
    """
    Flips the given NumPy array vertically
    x must be of shape [num_features, seq_len].
    """
    midpoint = (x.max(axis=1, keepdims=True) + x.min(axis=1, keepdims=True)) / 2
    return 2 * midpoint - x


def time_warp(x: np.ndarray, warp_factor: float = 0.5):
    """
    Apply time warp augmentation to a time series.
    Parameters:
        x (numpy.ndarray): A 2D numpy array [num_features, sequence_length].
        warp_factor (float): A factor to determine the amount of warping. Default is 0.5.
    Returns:
        numpy.ndarray: A time-warped 2D numpy array of the same shape.
    """
    num_features, seq_len = x.shape

    # Generate a smooth warping path using a sine wave
    warp_path = np.sin(np.linspace(0, np.pi, seq_len)) * warp_factor + 1

    # Normalize the warp path so that it fits the original sequence length
    warp_path_normalized = np.cumsum(warp_path)
    warp_path_normalized = (warp_path_normalized / warp_path_normalized[-1]) * (seq_len - 1)

    # Apply the warp to each feature
    warped_x = np.zeros_like(x)
    for i in range(num_features):
        # Ensure that the interp points fit within the orig seq len
        # by clipping the warp path
        clipped_warp_path = np.clip(warp_path_normalized, 0, seq_len - 1)
        warped_x[i] = np.interp(np.arange(seq_len), clipped_warp_path, x[i])

    return warped_x


def _gaussian_kernel(sigma=1, size=5) -> np.ndarray:
    """
    Generate a 1D Gaussian kernel.

    Parameters:
        sigma (float): Standard deviation of the Gaussian kernel.
        size (int): Size of the kernel (default: 5).

    Returns:
        numpy.ndarray: 1D Gaussian kernel.
    """
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize to ensure the sum of weights is 1
    return kernel


def smoothen(x: np.ndarray, sigma=1, size=5, dtype=np.float32) -> np.ndarray:
    """
    Smooth a NumPy array along each feature axis using a Gaussian kernel,
    Parameters:
        x (numpy.ndarray): Input array of shape [num_features, seq_len].
        sigma (float): Standard deviation of the Gaussian kernel (default: 1).
        size (int): Size of the Gaussian kernel (default: 5).
        dtype (type): Data type of the resulting array (default: np.float32).

    Returns:
        numpy.ndarray: Smoothed array of the same shape as x.
    """
    num_features, _ = x.shape
    X_smooth = np.zeros_like(x)

    for feature_index in range(num_features):
        feature_data = x[feature_index, :]
        smoothed_feature = convolve1d(feature_data, weights=_gaussian_kernel(sigma, size), mode="constant")
        X_smooth[feature_index, :] = smoothed_feature

    return X_smooth.astype(dtype)


def random_scaling(x: np.ndarray, scale_range: tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """
    Applies random scaling to a numpy array
    Parameters:
        x (numpy.ndarray): Input array of shape [num_features, seq_len].
        scale_range (tuple): Range of scaling factors to apply to the input array.
            Default is (0.9, 1.1).

    Returns:
            numpy.ndarray: Scaled array of the same shape as x.
    """
    factors = np.random.uniform(scale_range[0], scale_range[1], x.shape[0])
    return x * factors[:, np.newaxis]


def magnitude_warping(x: np.ndarray, sigma: float = 0.1, knot: int = 4) -> np.ndarray:
    """
    Applies magnitude warping to a time series.
    Parameters:
        x (numpy.ndarray): A 2D numpy array of shape [num_features, sequence_length].
        sigma (float): Standard deviation of the Gaussian kernel (default: 0.1).
        knot (int): Number of knots for the spline (default: 4).
    Returns:
        numpy.ndarray: A time-warped 2D numpy array of the same shape.
    """
    time_steps = np.linspace(0, 1, x.shape[1])
    warp = np.random.normal(1, sigma, knot + 2)
    spline = CubicSpline(np.linspace(0, 1, knot + 2), warp)
    warping_factor = spline(time_steps)
    return x * warping_factor


def skip_n_and_resize(x: np.ndarray, skip_n: int, size: int = None, dtype=np.float32) -> np.ndarray:
    """
    Drops every other n points and resizes the remaining part to size.
    If size is None, the size of the original array is used.
    Parameters:
        x (numpy.ndarray): Input array of shape [num_features, seq_len].
        skip_n (int): Drops every other n point.
        size (int): Final array size (default: None). If None, the orig array size is used.
        dtype (type): Data type of the final array (default: np.float32).
    Returns:
        numpy.ndarray: Dropped & resized array of the same shape as x.
    """
    _, seq_len = x.shape
    size = size if size is not None else seq_len
    return resize(x[:, ::skip_n], size).astype(dtype)


def drop_rand_frac_and_resize(x: np.ndarray, drop_frac: float, size: int = None, dtype=np.float32) -> np.ndarray:
    """
    Randomly drops drop_frac of points and resizes the remaining part to size.
    If size is None, the size of the original array is used.
    Parameters:
        x (numpy.ndarray): Input array of shape [num_features, seq_len].
        drop_perc (float): Randomly drops drop_frac of points.
        size (int): Final array size (default: None). If None, the orig array size is used.
        dtype (type): Data type of the final array (default: np.float32).
    Returns:
        numpy.ndarray: Dropped & resized array of the same shape as x.
    """
    if not 0 < drop_frac < 1:
        raise ValueError(f"Drop fraction {drop_frac} must be between 0 and 1")
    _, seq_len = x.shape
    size = size if size is not None else seq_len
    drop_n = int(drop_frac * size)
    seq_idxs = sorted(np.random.choice(size, size - drop_n, replace=False))
    return resize(x[:, seq_idxs], size).astype(dtype)
