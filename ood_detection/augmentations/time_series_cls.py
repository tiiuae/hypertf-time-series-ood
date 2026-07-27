"""
Possible agumentations.
Note, have to be careful with adding some augmentation as they could simulate anomalous data instead
Note all functions should return new arrays

• Jittering: A way of simulating additive sensor noise.
• Scaling: Changes the magnitude of the data in a window by multiplying by a random scalar.
• Rotations: Upside-down placement of the sensor which changes the position of the sensor you are wearing.
• Permutation: Slice the data into N same length segments, & randomly permute the segments to create a new window.
• Time-warping: Smoothly distorting the time intervals between samples to change the temporal locations of the samples.
• Magnitude-warping: changes the magnitude of each sample by convolving the data window with a smooth curve varying around one.
• Cropping: Cut and delete collected data after a certain time.
• Scaling: Randomly scaling data.
• Magnitude Warping: Applies smooth multiplicative noise(random scaling factors applied smoothly over time).

Note for sensor data augmentations like Gaussian Noise, Time Warping, Rotation Transformations, Scaling, Permutation, Magnitude Warping are appropriate, but horizontal_flip and especially vertical_flip should be avoided.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import torch

from ood_detection.augmentations.base_augmentation import BaseAugmentation
from ood_detection.augmentations.time_series_funcs import (
    drop_rand_frac_and_resize,
    horizontal_flip,
    jitter,
    permutation,
    random_crop_if_longer,
    random_crop_resize,
    random_scaling,
    resize,
    resize_down_if_longer,
    scaling,
    skip_n_and_resize,
    smoothen,
    time_warp,
    vertical_flip,
)

################ Base Transformations ################


class TsCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def __repr__(self):
        return f"TsCompose({self.transforms})"


class TsToTensor:
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class Standardize:
    def __init__(self, mean, std):
        assert mean.dtype == np.float32
        assert std.dtype == np.float32
        self.mean = mean.reshape(-1, 1)  # Reshape to (n_features, 1)
        self.std = std.reshape(-1, 1)  # Reshape to (n_features, 1)

    def __call__(self, x):
        return (x - self.mean) / self.std

    def __repr__(self):
        return f"Standardize(mean={self.mean}, std={self.std})"


class MinMaxScaler:
    def __init__(self, min_val=0, max_val=1):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x):
        min_sample = np.min(x, axis=(1, 2), keepdims=True)
        max_sample = np.max(x, axis=(1, 2), keepdims=True)
        return self.min_val + (x - min_sample) * (self.max_val - self.min_val) / (max_sample - min_sample + 1e-8)

    def __repr__(self):
        return f"MinMaxScaler(min_val={self.min_val}, max_val={self.max_val})"


#################### Augmentations ####################


class Jitter(BaseAugmentation):
    """
    Apply random Gaussian noise to time series data.
    Considered a strong aug in combination with permutationm i.e. jitter(permutation(x))
    https://arxiv.org/pdf/1706.00527.pdf
    x (np.ndarray): Input array of shape [channels, time_steps].

    Args:
        sigma (float): Standard deviation of the Gaussian noise.
        p (float): probability of applying jitter (Set 1 to always apply)

    Returns:
        np.ndarray: Jittered array of the same shape as input.
    """

    def __init__(self, sigma: float = 0.8, p: float = 1.0):
        super().__init__(p)
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return jitter(x, self.sigma)
        return x.copy()

    def __repr__(self) -> str:
        return f"Jitter(sigma={self.sigma})"


class Scaling(BaseAugmentation):
    """
    Apply random scaling to time series data as a weak augmentation.
    Scaled by same factor across channels
    Considered a weak augmentation: https://arxiv.org/pdf/1706.00527.pdf
    x (np.ndarray): Input array of shape [channels, time_steps].

    Args:
        sigma (float): Standard deviation of the scaling factor.
        p (float): probability of applying scaling (Set 1 to always apply)

    Returns:
        np.ndarray: Scaled array of the same shape as input.
    """

    def __init__(self, sigma: float = 1.1, p: float = 1.0):
        super().__init__(p)
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return scaling(x, self.sigma)
        return x.copy()

    def __repr__(self) -> str:
        return f"Scaling(sigma={self.sigma})"


class Permutation(BaseAugmentation):
    """
    Permute segments of a time series along the time_steps axis.
    x (np.ndarray): Input array of shape [batch_size, channels, time_steps].

    Args:
        max_seg (int): Maximum number of segments to divide each time series.
        seg_mode (str): Mode of segment division.
                        "random" for random split points, "equal" for equal-sized segments.
        p (float): probability of applying permutation (Set 1 to always apply)

    Returns:
        np.ndarray: Array with permuted time_steps for each sample.
    """

    def __init__(self, max_seg: int = 5, seg_mode: str = "random", p: float = 1.0):
        super().__init__(p)
        assert seg_mode in ["random", "equal"], "seg_mode must be either 'random' or 'equal'"
        self.max_seg = max_seg
        self.seg_mode = seg_mode

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        if np.random.rand() < self.p:
            return permutation(x, self.max_seg, self.seg_mode)
        return x

    def __repr__(self) -> str:
        return f"Permutation(max_seg={self.max_seg}, seg_mode={self.seg_mode})"


class RandomCropIfLonger(BaseAugmentation):
    """
    Randomly crop the input sequence to a fixed target length,
    if the input sequence is longer than the target length.
    x (np.ndarray): Input array of shape [channels, time_steps].

    Args:
        target_seq_len (int): Desired sequence length after cropping.
        p (float): Probability of applying the transform (default: 1.0).

    Returns:
        np.ndarray: Cropped array of shape [channels, target_seq_len],
                    or original if target_seq_len >= input length.
    """

    def __init__(self, target_seq_len: int, p: float = 1.0):
        super().__init__(p)
        self.target_seq_len = target_seq_len

    def __call__(self, x: np.ndarray) -> np.ndarray:
        _, seq_len = x.shape
        if np.random.rand() < self.p and self.target_seq_len < seq_len:
            return random_crop_if_longer(x, self.target_seq_len)
        return x.copy()

    def __repr__(self) -> str:
        return f"RandomCrop(target_seq_len={self.target_seq_len}, p={self.p})"


class RandomCropResize(BaseAugmentation):
    """
    Randomly crop and resize to the original length.
    x (np.ndarray): Input array of shape [channels, time_steps].

    Args:
        scale (tuple): Range of scales for cropping (e.g., (0.5, 1.0)).
        p (float): Probability of applying the transform (default: 1.0).

    Returns:
        np.ndarray: Array cropped and resized to the original shape.
    """

    def __init__(self, scale: tuple[float, float] = (0.5, 1.0), p: float = 1.0):
        super().__init__(p)
        self.scale = scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.p == 1.0:
            # Skip the random check entirely when p=1.0
            return random_crop_resize(x, self.scale)
        elif np.random.rand() < self.p:
            return random_crop_resize(x, self.scale)
        return x.copy()

    def __repr__(self) -> str:
        return f"RandomCropResize(scale={self.scale})"


class Mask(BaseAugmentation):
    """
    Mask a time series sequence with zeros
    x is an np.ndarray of shape [channels, time_steps].

    Args:
        mask_ratio (float): Proportion of time sequence sample to mask
        p (float): probability of applying mask (Set 1 to always apply)
    """

    def __init__(self, mask_ratio=0.1, p: float = 1.0):
        super().__init__(p)
        assert 0 <= mask_ratio <= 1, "mask_ratio must be between 0 and 1."
        self.mask_ratio = mask_ratio

    def __call__(self, x: np.ndarray):
        if np.random.rand() >= self.p:
            return x.copy()
        n_features, length = x.shape

        # Ensure at least one time step is masked
        num_masked_steps = max(1, int(length * self.mask_ratio))

        # Randomly choose time steps to mask
        masked_indices = np.random.choice(length, num_masked_steps, replace=False)

        # Create a mask array and set selected indices to 0
        mask = np.ones((n_features, length), dtype=x.dtype)
        mask[:, masked_indices] = 0

        # Apply the mask to the sample
        return x * mask

    def __repr__(self):
        return f"Mask(mask_ratio={self.mask_ratio})"


######################## Additional Augmentations ########################


class Resize(BaseAugmentation):
    """
    Resizes the NumPy array
    x must be of shape [num_features, seq_len].
    Args:
        size (int): The size to which the array should be resized.
        dtype (numpy.dtype): The data type of the resized array. Defaults to np.float32.
        p (float): probability of applying mask (Set 1 to always apply)
    """

    def __init__(self, size: int, dtype=np.float32, p: float = 1.0):
        super().__init__(p)
        self.size = size
        self.dtype = dtype

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return resize(x, self.size).astype(self.dtype)
        return x.copy()

    def __repr__(self):
        return f"Resize(size={self.size})"


class ResizeDownIfLonger(BaseAugmentation):
    """
    Downsizes the input NumPy array only if its sequence length exceeds `max_length`.
    Expects x of shape [num_features, seq_len].

    Args:
        max_length (int): Maximum allowed sequence length. Input is resized down if it exceeds this.
        dtype (np.dtype): Output array dtype. Defaults to np.float32.
        p (float): Probability of applying the transformation. Defaults to 1.0.
    """

    def __init__(self, max_length: int, dtype=np.float32, p: float = 1.0):
        super().__init__(p)
        self.max_length = max_length
        self.dtype = dtype

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return resize_down_if_longer(x, self.max_length).astype(self.dtype)
        return x.copy()

    def __repr__(self):
        return f"ResizeDownIfLonger(max_length={self.max_length})"


class HorizontalFlip(BaseAugmentation):
    """
    Randomly flips a given NumPy array horizontally with a specified probability `p` (0 ≤ p ≤ 1).
    The input array `x` must have a shape of [num_features, seq_len].

    Args:
        p (float): Probability of applying the horizontal flip. Default is 1.0.
    """

    def __init__(self, p: float = 1.0):
        super().__init__(p)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return horizontal_flip(x)
        return x.copy()

    def __repr__(self):
        return f"HorizontalFlip(p={self.p})"


class VerticalFlip(BaseAugmentation):
    """
    Randomly flips a given NumPy array vertically with a specified probability `p` (0 ≤ p ≤ 1).
    The input array `x` must have a shape of [num_features, seq_len].

    Args:
        p (float): Probability of applying the vertical flip. Default is 1.0.
    """

    def __init__(self, p: float = 1.0):
        super().__init__(p)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return vertical_flip(x)
        return x.copy()

    def __repr__(self):
        return f"VerticalFlip(p={self.p})"


class TimeWarp(BaseAugmentation):
    """
    TApplies a time-warping transformation to a given 2D NumPy array,
    randomly with a specified probability `p` (0 ≤ p ≤ 1).
    This transformation alters the temporal alignment of the sequence.
    The input array `x` must have a shape of [num_features, seq_len].

    Args:
        warp_factor (float): The degree of warping to apply. Default is 1.0.
        p (float): Probability of applying the time warp. Default is 1.0.
    """

    def __init__(self, warp_factor: float = 0.5, p: float = 1.0):
        super().__init__(p)
        self.warp_factor = warp_factor

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return time_warp(x, self.warp_factor)
        return x.copy()

    def __repr__(self):
        return f"TimeWarp(warp_factor={self.warp_factor}, p={self.p})"


class Smoothen(BaseAugmentation):
    """
    Smooths the input NumPy array along each feature axis using a Gaussian kernel,
    randomly with a specified probability `p` (0 ≤ p ≤ 1).
    The smoothing is controlled by the Gaussian kernel's `sigma` (std dev) & `size` (window size).
    The input array `x` must have a shape of [num_features, seq_len].

    Args:
        sigma (float): Standard deviation of the Gaussian kernel. Default is 1.
        size (int): Size of the Gaussian kernel window. Default is 5.
        p (float): Probability of applying smoothing. Default is 1.0.
    """

    def __init__(self, sigma=1, size=5, p=1.0):
        super().__init__(p)
        self.sigma = sigma
        self.size = size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return smoothen(x, self.sigma, self.size)
        return x.copy()

    def __repr__(self):
        return f"Smoothen(sigma={self.sigma}, size={self.size}, p={self.p})"


class RandomScaling(BaseAugmentation):
    """
    Randomly scales the values of a NumPy array along the feature dimension within a specified `scale_range`,
    with a given probability `p` (0 ≤ p ≤ 1).
    The input array `x` must have a shape of [num_features, seq_len].

    Args:
        scale_range (Tuple[float, float]): Range for random scaling factors (min, max). Default is (0.9, 1.1).
        dtype (np.dtype): Data type for the output array. Default is np.float32.
        p (float): Probability of applying random scaling. Default is 1.0.

    """

    def __init__(self, scale_range: tuple[float, float] = (0.9, 1.1), dtype=np.float32, p: float = 1.0):
        super().__init__(p)
        if scale_range[0] >= scale_range[1]:
            raise ValueError("Scale range must have the first value less than the second value")
        self.scale_range = scale_range
        self.dtype = dtype

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return random_scaling(x, self.scale_range).astype(self.dtype)
        return x.copy()

    def __repr__(self):
        return f"RandomScaling(scale_range={self.scale_range}, p={self.p})"


class MagnitudeWarping(BaseAugmentation):
    """
    Applies a smooth multiplicative noise to the input NumPy array along the sequence dimension,
    randomly with a specified probability `p` (0 ≤ p ≤ 1).
    The noise is generated using a cubic spline interpolation with `knot` points and a specified `sigma` (standard deviation).
    If `seq_len` is provided, the warping factor is precomputed; otherwise, it is generated dynamically.
    The input array `x` must have a shape of [num_features, seq_len].

    Args:
        sigma (float): Standard deviation for the noise generation. Must be positive. Default is 0.1.
        knot (int): Number of knots used for cubic spline interpolation. Must be at least 1. Default is 4.
        seq_len (int, optional): Length of the sequence for precomputing the warping factor. Default is None.
        dtype (np.dtype): Data type for the output array. Default is np.float32.
        p (float): Probability of applying magnitude warping. Default is 1.0.
    """

    def __init__(self, sigma: float = 0.1, knot: int = 4, seq_len: int = None, dtype=np.float32, p: float = 1.0):
        super().__init__(p)
        if sigma <= 0:
            raise ValueError("Sigma must be positive")
        if knot < 1:
            raise ValueError("Knots must be at least 1")
        self.sigma = sigma
        self.knot = knot
        self.warping_factor = self._generate_warp(seq_len) if seq_len is not None else None
        self.dtype = dtype

    def _generate_warp(self, seq_len: int) -> np.ndarray:
        """Generate a warping factor for magnitude warping"""
        time_steps = np.linspace(0, 1, seq_len)
        warp = np.random.normal(1, self.sigma, self.knot + 2)
        spline = CubicSpline(np.linspace(0, 1, self.knot + 2), warp)
        return spline(time_steps)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            if self.warping_factor is None or len(self.warping_factor) != x.shape[1]:
                # Generate warping factor dynamically if needed
                self.warping_factor = self._generate_warp(x.shape[1])
            return (x * self.warping_factor).astype(self.dtype)
        return x.copy()

    def __repr__(self):
        return f"MagnitudeWarping(sigma={self.sigma}, knot={self.knot}, p={self.p})"


class SkipNAndResize(BaseAugmentation):
    """
    Drops every `n`-th point in the input NumPy array and resizes the remaining data to its original size,
    randomly with a specified probability `p` (0 ≤ p ≤ 1).
    The input array `x` must have a shape of [num_features, seq_len].

    Args:
        skip_n (int): Number of points to skip between retained points.
        size (int, optional): Desired size of the output array. If None, it matches the original size. Default is None.
        p (float): Probability of applying the transformation. Default is 1.0.
        dtype (np.dtype): Data type for the output array. Default is np.float32.
    """

    def __init__(self, skip_n: int, size: int = None, p: float = 1.0, dtype=np.float32):
        super().__init__(p)
        self.skip_n = skip_n
        self.size = size
        self.dtype = dtype

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return skip_n_and_resize(x, self.skip_n, self.size, self.dtype)
        return x.copy()

    def __repr__(self):
        return f"SkipNAndResize(skip_n={self.skip_n},size={self.size})"


class DropRandPercAndResize(BaseAugmentation):
    """
    Randomly drops a specified fraction of points from the input NumPy array and resizes the remaining data to its original size,
    with a given probability `p` (0 ≤ p ≤ 1).
    The fraction of points to drop is specified by `drop_frac` (0 < drop_frac < 1).
    The input array `x` must have a shape of [num_features, seq_len].

    Args:
        drop_frac (float): Fraction of points to randomly drop. Must be between 0 and 1.
        size (int, optional): Desired size of the output array. If None, it matches the original size. Default is None.
        p (float): Probability of applying the transformation. Default is 1.0.
        dtype (np.dtype): Data type for the output array. Default is np.float32.
    """

    def __init__(self, drop_frac: float, size: int = None, p: float = 1.0, dtype=np.float32):
        super().__init__(p)
        if not 0 < drop_frac < 1:
            raise ValueError(f"Drop fraction {drop_frac} must be between 0 and 1")
        self.drop_frac = drop_frac
        self.size = size
        self.dtype = dtype

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return drop_rand_frac_and_resize(x, self.drop_frac, self.size, self.dtype)
        return x.copy()

    def __repr__(self):
        return f"DropRandPercAndResize(drop_frac={self.drop_frac},size={self.size})"


if __name__ == "__main__":
    # Example usage
    from ood_detection.utils.visualization import plot_two_views_per_aug

    # Example input data [batch_size, channels, time_steps/seq_length]
    # Generate sine wave data of shape [3, 2, 500]
    tsteps = np.linspace(0, 2 * np.pi, 500)  # 500 time steps
    sample = np.array(
        [
            [
                np.sin(tsteps),  # Channel 1: Sin wave
                np.sin(tsteps + np.pi / 2),  # Channel 2: Cos wave (phase shift)
            ]
            for _ in range(3)  # Batch size: 3
        ]
    )

    plot_two_views_per_aug(
        sample[0],
        [
            Jitter(),
            Scaling(),
            Permutation(max_seg=10),
            RandomCropResize(scale=(0.5, 1.0)),
        ],
        show=False,
        savepath="sample_augs1.jpg",
    )

    plot_two_views_per_aug(
        sample[0],
        [HorizontalFlip(), VerticalFlip(), TimeWarp(warp_factor=1.5), Smoothen(sigma=5), Mask(mask_ratio=0.4)],
        show=False,
        savepath="sample_augs2.jpg",
    )

    plot_two_views_per_aug(
        sample[0],
        [
            RandomScaling(scale_range=(0.8, 1.2)),
            MagnitudeWarping(sigma=0.5, knot=4),
            SkipNAndResize(skip_n=3),
            DropRandPercAndResize(drop_frac=0.3),
        ],
        show=False,
        savepath="sample_augs3.jpg",
    )

    plot_two_views_per_aug(
        sample[0],
        [
            Resize(size=30),
            ResizeDownIfLonger(max_length=100),
            ResizeDownIfLonger(max_length=501),
            RandomCropIfLonger(target_seq_len=20),
            RandomCropIfLonger(target_seq_len=501),
        ],
        show=False,
        savepath="sample_augs4.jpg",
    )
