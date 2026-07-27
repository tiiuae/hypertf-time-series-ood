from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np


def plot_multivariate_time_series(
    data, mission_name, feature_names, frequency_hz=100, show: bool = True, savepath: str = None
):
    time_steps = np.arange(len(data)) / frequency_hz

    fig, ax = plt.subplots(len(feature_names), 1, figsize=(10, 2 * len(feature_names)), sharex=True)
    fig.suptitle(f"Time Series for {mission_name}", fontsize=16)

    if len(feature_names) == 1:
        ax = [ax]  # Ensure ax is always iterable

    for idx, feature in enumerate(feature_names):
        ax[idx].plot(time_steps, data[feature], label=feature)
        ax[idx].set_ylabel(feature)
        ax[idx].legend(loc="upper right")

    ax[-1].set_xlabel("Time (seconds)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close(fig)


def plot_random_windows(x, y, feature_names=None, num_windows=3):
    # Check if the data has at least the number of requested windows
    if x.shape[0] < num_windows:
        raise ValueError(
            "The total number of windows in x is less than the requested number of random windows to plot."
        )

    # Randomly select indices for the windows to plot
    indices = np.random.choice(x.shape[0], size=num_windows, replace=False)

    # Create a figure with subplots
    fig, axes = plt.subplots(num_windows, 1, figsize=(10, num_windows * 3))

    # If there's only one window, axes might not be an array
    if num_windows == 1:
        axes = [axes]  # Make it iterable

    # Plot each selected window
    for i, ax in enumerate(axes):
        window_index = indices[i]
        # Transpose to plot features as separate lines
        ax.plot(x[window_index].T)
        ax.set_title(f"Window {window_index}: {y[window_index]}")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Feature Value")
        ax.legend(feature_names)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_two_views_per_aug(x: np.ndarray, augmentations: list[Callable], show: bool = True, savepath: str = None):
    """
    Plot two augmented views of the time-series data `x` applying the augmentations sequentially.

    Parameters:
        x (np.ndarray): Input time-series data of shape (batch_size, channels, time_steps).
        augmentations (List[Callable]): List of augmentation functions to apply.
        savepath (str): plot savepath
    """
    fig, axes = plt.subplots(len(augmentations), 2, figsize=(12, 5 * len(augmentations)))

    for i, aug in enumerate(augmentations):
        # Generate two augmented views
        aug_view_1 = aug(x)
        aug_view_2 = aug(x)

        # Select one example from the batch for visualization
        sample_feat = 0
        # First channel of the first example
        orig_example = x[sample_feat, :]
        aug_example_1 = aug_view_1[sample_feat, :]
        aug_example_2 = aug_view_2[sample_feat, :]

        # Plot original and augmented views side by side
        axes[i, 0].plot(orig_example, label="Original", color="blue", linestyle="--", alpha=0.5)
        axes[i, 0].plot(aug_example_1, label="Augmented View 1", color="orange", alpha=0.9)
        axes[i, 0].set_title(f"Aug {i + 1} - View 1 ({aug})")
        axes[i, 0].legend()

        axes[i, 1].plot(orig_example, label="Original", color="blue", linestyle="--", alpha=0.5)
        axes[i, 1].plot(aug_example_2, label="Augmented View 2", color="green", alpha=0.9)
        axes[i, 1].set_title(f"Aug {i + 1} - View 2 ({aug})")
        axes[i, 1].legend()

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close(fig)


def plot_samples_side_by_side(
    x: np.ndarray, idxs_col1: list, idxs_col2: list, title: str, save_path: str = "samples_plot.png"
):
    """
    Plots and saves side-by-side comparisons of samples from x given index cols idxs_col1 and idxs_col2

    Args:
        x (np.ndarray): Numpy array of shape (N, Channel, Time Sequence)
        idxs_col1 (list): List of indices for col 1 in `data`.
        idxs_col2 (list): List of indices for col 2 in `data`.
        save_path (str): File path to save the resulting plot.
    """
    assert x.ndim == 3, "Data should be 3D (batch, channels, seq_len)"
    assert len(idxs_col1) == len(idxs_col2), "idxs_col1 and idxs_col2 must be the same length"

    num_samples = len(idxs_col1)
    seq_len = x.shape[-1]

    fig, axs = plt.subplots(num_samples, 2, figsize=(12, 2.5 * num_samples))
    fig.suptitle(title, fontsize=16)

    for i, (idx1, idx2) in enumerate(zip(idxs_col1, idxs_col2, strict=False)):
        sample1, sample2 = x[idx1], x[idx2]

        for ch in range(sample1.shape[0]):
            axs[i, 0].plot(sample1[ch], alpha=0.7)
            axs[i, 1].plot(sample2[ch], alpha=0.7)

        axs[i, 0].set_title(f"Sample index {idx1}")
        axs[i, 1].set_title(f"Sample index {idx2}")
        axs[i, 0].set_xlim(0, seq_len)
        axs[i, 1].set_xlim(0, seq_len)
        axs[i, 0].set_ylabel("Channels")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()


def plot_ood_stacked(ood_scores, bins=60, figsize=(12, 8), savepath: str = None):
    """
    Plot stacked histograms with density normalization.
    ood_score_results (dict): OOD scores for each evaluation method.
                Format:
                {
                    'MethodName': {
                        'test_id': [scores],
                        'ood_dataset1': [scores],
                        'ood_dataset2': [scores],
                        ...
                    }
                }
    """
    methods = list(ood_scores.keys())
    fig, axes = plt.subplots(len(methods), 1, figsize=figsize)
    axes = [axes] if len(methods) == 1 else axes

    red_shades = ["#FF4444", "#CC2222", "#AA0000", "#880000", "#FF6666", "#FF8888"]

    for ax, method in zip(axes, methods, strict=True):
        scores = ood_scores[method]

        # Percentile-based binning
        all_scores = np.concatenate(list(scores.values()))
        bins_edges = np.percentile(all_scores, np.linspace(0, 100, bins + 1))

        # Plot with density normalization
        for i, (name, data) in enumerate(scores.items()):
            if name == "test_id":
                ax.hist(
                    data,
                    bins=bins_edges,
                    alpha=0.7,
                    color="blue",
                    density=True,
                    label=f"ID (n={len(data)})",
                    edgecolor="darkblue",
                )
                ax.axvline(data.mean(), color="blue", linestyle="--", linewidth=2, alpha=0.8)
            else:
                color = red_shades[i % len(red_shades)]
                ax.hist(
                    data,
                    bins=bins_edges,
                    alpha=0.6,
                    color=color,
                    density=True,
                    label=f"OOD: {name[:20]} (n={len(data)})",
                    edgecolor="darkred",
                )
                ax.axvline(data.mean(), color=color, linestyle="--", linewidth=1.5, alpha=0.7)

        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.set_title(method)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, linestyle=":")

    plt.suptitle("OOD Score Distributions (Normalized)")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0.1, dpi=350)
    else:
        plt.show()
    plt.close(fig)


def plot_ood_histograms(
    ood_scores: dict, bins: int = 50, alpha: float = 0.6, figsize: tuple[int, int] = (15, 5), savepath: str = None
):
    """
    Plot histograms of OOD scores for each scoring method.
    ood_score_results (dict): OOD scores for each evaluation method.
                Format:
                {
                    'MethodName': {
                        'test_id': [scores],
                        'ood_dataset1': [scores],
                        'ood_dataset2': [scores],
                        ...
                    }
                }
    """
    methods = list(ood_scores.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=figsize)
    axes = [axes] if len(methods) == 1 else axes

    red_shades = ["#FF4444", "#CC2222", "#AA0000", "#880000", "#FF6666", "#FF8888"]

    for ax, method in zip(axes, methods, strict=True):
        scores = ood_scores[method]

        # Get score range for consistent binning
        all_scores = np.concatenate(list(scores.values()))
        bins_edges = np.linspace(all_scores.min(), all_scores.max(), bins + 1)

        # Plot ID and OOD scores
        for i, (name, data) in enumerate(scores.items()):
            if name == "test_id":
                ax.hist(data, bins=bins_edges, alpha=alpha, color="blue", label="ID", edgecolor="black", linewidth=0.5)
            else:
                ax.hist(
                    data,
                    bins=bins_edges,
                    alpha=alpha,
                    color=red_shades[i % len(red_shades)],
                    label=f"OOD: {name[:20]}",
                    edgecolor="black",
                    linewidth=0.5,
                )

        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.set_title(method)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("OOD Score Distributions")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0.1, dpi=350)
    else:
        plt.show()
    plt.close(fig)
