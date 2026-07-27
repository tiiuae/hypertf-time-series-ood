import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.manifold import TSNE
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix
import umap


def plot_metrics(
    data_dict: dict[str, pd.DataFrame],
    x_col: str,
    y_col: str,
    title: str = "Metric Over Epochs",
    xlabel: str = "Epochs",
    ylabel: str = "Metric",
    savepath: str = "metric.png",
) -> None:
    """
    General function to plot any metric over a specified x-axis column.

    Args:
        data_dict (dict): A dictionary where keys are method names and values are pandas DataFrames.
        x_col (str): Column name to use for the x-axis (e.g., 'epoch').
        y_col (str): Column name to use for the y-axis (e.g., 'auroc').
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        savepath (str): Path to save the plot image.
    """
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(data_dict))

    fig, ax = plt.subplots(figsize=(12, 8))
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "+", "x", "d", "|", "_"]

    for idx, (method_name, df) in enumerate(data_dict.items()):
        if x_col in df.columns and y_col in df.columns:
            # Plot individual data points
            ax.scatter(df[x_col], df[y_col], label=None, color=palette[idx], alpha=0.4, s=25)

            # Compute group mean and std
            grouped = df.groupby(x_col)[y_col].agg(["mean", "std"]).reset_index()

            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]

            # Plot mean line
            ax.plot(
                grouped[x_col],
                grouped["mean"],
                label=method_name,
                color=palette[idx],
                linestyle=line_style,
                marker=marker,
                markersize=5,
            )

            # Fill ±1 std band
            ax.fill_between(
                grouped[x_col],
                grouped["mean"] - grouped["std"],
                grouped["mean"] + grouped["std"],
                color=palette[idx],
                alpha=0.2,
            )

            # Annotate final mean value
            last_x = grouped[x_col].iloc[-1]
            last_y = grouped["mean"].iloc[-1]
            ax.annotate(
                f"{last_y:.2f}",
                xy=(last_x, last_y),
                textcoords="offset points",
                xytext=(5, 0),
                ha="left",
                va="center",
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
            )
        else:
            print(f"Warning: {method_name} does not contain {x_col} or {y_col}. Skipping.")

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="best", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close(fig)


def plot_roc_curve(y_true: ArrayLike, y_pred: ArrayLike, savepath: str = "roc_curve.png", **kwargs) -> None:
    """Plot the ROC curve for a binary classification problem."""
    fig = plt.figure(figsize=(15, 15))
    RocCurveDisplay.from_predictions(y_true, y_pred, **kwargs)

    plt.savefig(savepath, bbox_inches="tight", pad_inches=0.1, dpi=350)
    plt.close(fig)


def plot_pr_curve(y_true: ArrayLike, y_pred: ArrayLike, savepath: str = "precision_recall_curve.png", **kwargs) -> None:
    """Plot the Precision-Recall curve for a binary classification problem."""
    fig = plt.figure(figsize=(15, 15))
    PrecisionRecallDisplay.from_predictions(y_true, y_pred, **kwargs)

    plt.savefig(savepath, bbox_inches="tight", pad_inches=0.1, dpi=350)
    plt.close(fig)


def plot_calibration_curve(
    y_true: ArrayLike, y_score: ArrayLike, savepath: str = "calibration_curve.png", **kwargs
) -> None:
    """Plot the Calibration curve for a binary classification problem."""
    fig = plt.figure(figsize=(15, 15))
    CalibrationDisplay.from_predictions(y_true, y_prob=y_score, n_bins=10, **kwargs)

    plt.savefig(savepath, bbox_inches="tight", pad_inches=0.1, dpi=350)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_preds: ArrayLike,
    label_remap: dict[int | str, int] | None = None,
    title: str = "Confusion Matrix",
    save_path: str = "confusion_matrix.png",
) -> None:
    """
    Plots a confusion matrix with optional custom labels mapped using label_remap.

    Parameters:
        y_true (ArrayLike): Ground truth labels.
        y_preds (ArrayLike): Predicted labels.
        label_remap (dict, optional): Dict mapping orig class names to class ids e.g. {"Class1": 0, "Class2": 1}.
        title (str): Title of the plot.
        save_path (str): Path to save the confusion matrix image.
    """
    if len(y_true) == 0 or len(y_preds) == 0:
        print("No validation predictions available; skipping confusion matrix.")
        return

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_preds)
    if label_remap:
        # reverse label_remap
        id_label_map = {v: k for k, v in label_remap.items()}
        label_names = [id_label_map[cid] for cid in sorted(id_label_map.keys())]
    else:
        unique_labels = sorted(set(y_true) | set(y_preds))
        label_names = [str(label) for label in unique_labels]  # Convert to strings for plotting

    # Plot confusion matrix
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names, ax=ax_cm)
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_title(title)

    # Adjust margins to prevent text cutoff
    plt.xticks(rotation=45, ha="right")  # Rotate xticks for better readability
    plt.yticks(rotation=0)  # Keep yticks horizontal
    plt.tight_layout()  # Auto-adjust layout

    # Save and close the figure
    plt.savefig(save_path)
    plt.close(fig_cm)


def plot_train_val_metrics(
    train_df: pd.DataFrame | ArrayLike,
    val_df: pd.DataFrame | ArrayLike,
    metrics: list[str] | None = None,
    titles: dict[str, str] | None = None,
    y_labels: dict[str, str] | None = None,
    save_path: str = "train_vs_val_metrics.png",
    max_cols: int = 3,  # Max columns per row
) -> None:
    """
    Plots training vs. validation metrics if available.
    Arranges plots dynamically, moving to a new row after `max_cols` plots.

    Parameters:
    - train_df (DataFrame or ArrayLike): Training metrics with optional columns like ["epoch", "loss", "accuracy", "f1"].
    - val_df (DataFrame or ArrayLike): Validation metrics with optional columns.
    - save_path (str): Path to save the generated plot.
    - metrics (list, optional): List of metrics to plot. Defaults to ["loss", "accuracy", "f1"].
    - titles (dict, optional): Custom titles for each metric.
    - y_labels (dict, optional): Custom y-axis labels for each metric.
    - max_cols (int, optional): Max number of plots per row. Defaults to 3.
    """
    if metrics or titles or y_labels:
        assert len(metrics) == len(titles) == len(y_labels), "Number of metrics, titles, and y_labels must match."
    # Default values for metrics, titles, and y_labels is the train_df + val_df columns
    default_metrics = train_df.columns.tolist() + val_df.columns.tolist()
    default_metrics = [metric for metric in default_metrics if metric.lower() not in ["epoch"]]
    default_y_labels = {metric: metric for metric in default_metrics}
    default_titles = {metric: f"{metric}_vs_epochs" for metric in default_metrics}

    metrics = metrics or default_metrics
    titles = titles or default_titles
    y_labels = y_labels or default_y_labels

    # Determine which metrics are available in either train_df or val_df
    available_metrics = {metric: metric in train_df.columns or metric in val_df.columns for metric in metrics}
    available_metrics = {k: v for k, v in available_metrics.items() if v}  # Filter out missing metrics

    if not available_metrics:
        print("No valid metrics found to plot.")
        return

    num_plots = len(available_metrics)
    # Compute rows and columns for subplots
    num_rows = int(np.ceil(num_plots / max_cols))
    num_cols = min(num_plots, max_cols)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(6 * num_cols, 5 * num_rows))

    # If there's only one row, ensure `axes` is iterable
    axes = np.array(axes).reshape(-1)  # Flatten in case of single row

    for ax, metric in zip(axes, available_metrics.keys(), strict=False):
        if metric in train_df.columns:
            ax.plot(train_df["epoch"], train_df[metric], label="Train")
        if metric in val_df.columns:
            ax.plot(val_df["epoch"], val_df[metric], "--", label="Validation")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(y_labels.get(metric, metric))  # Default to metric name if y_label is missing
        ax.set_title(
            titles.get(metric, metric), fontsize=14, fontweight="bold"
        )  # Default to metric name if title is missing
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    # Remove any unused subplots
    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_embedding(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    label_remap: dict[int | str, int] | None = None,
    title: str = "Embedding Plot",
    metric: str = "euclidean",
    palette: dict[int | str, str] | None = None,
    savepath: str = "embedding_plot.png",
) -> None:
    """
    Plots a 2D visualization of embeddings using t-SNE or UMAP.

    Args:
        embeddings: NumPy array of shape (N, D) where N is the number of samples, and D is the feature dimension.
        labels: NumPy array of shape (N,) representing class labels (0,1,2,... for ID and -1,-2,-3,... for OOD).
        method: Dimensionality reduction method to use ('tsne' or 'umap').
        label_remap: Optional dict with remapped labels, e.g. {"Class1": 0, "Class2": 1, "OOD_Class1": -1,...}.
        title: Title for the plot.
        metric: Metric to use for the chosen method (e.g., 'euclidean', 'cosine').
        palette: Optional dictionary mapping class labels to specific colors.
        savepath: Path to save the plot.
    """
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, metric=metric, random_state=42)
    elif method.lower() == "umap":
        reducer = umap.UMAP(n_components=2, metric=metric, random_state=42)
    else:
        raise ValueError("Invalid method. Choose either 'tsne' or 'umap'.")

    reduced_emb = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    # map labels from [0,1,2,...-1, -2, -3,...] to ["Class1", "Class2", ..., "OOD_Class-1", "OOD_Class-2", "OOD_Class-3", ...]
    if label_remap:
        reversed_label_remap = {
            cls_id: cls_name if cls_id >= 0 else "OOD_" + cls_name for cls_name, cls_id in label_remap.items()
        }
        labels = np.vectorize(reversed_label_remap.get)(labels)  # Map labels

    labels_str = labels.astype(str)
    unique_labels = np.unique(labels_str)
    id_labels = [lb for lb in unique_labels if not lb.startswith("OOD_")]
    ood_labels = [lb for lb in unique_labels if lb.startswith("OOD_")]

    if palette is None:
        base_palette = sns.color_palette("tab10", len(id_labels) + 1)
        base_palette = [color for i, color in enumerate(base_palette) if i != 3]  # avoid red
        id_palette = dict(zip(id_labels, base_palette, strict=False))

        ood_colors = sns.cubehelix_palette(
            len(ood_labels), start=0.05, rot=0.9, dark=0.1, light=0.95, gamma=0.8, reverse=True
        )
        ood_palette = dict(zip(ood_labels, ood_colors, strict=False))

        palette = {**id_palette, **ood_palette}

    # Plot each class separately to control marker type
    for label in unique_labels:
        mask = labels_str == label
        color = palette.get(label, "gray")
        scatter_size = 18
        marker = "X" if label.startswith("OOD_") else "o"
        alpha = 0.85 if label.startswith("OOD_") else 0.7
        sns.scatterplot(
            x=reduced_emb[mask, 0],
            y=reduced_emb[mask, 1],
            label=label,
            color=color,
            alpha=alpha,
            edgecolor="k",
            s=scatter_size,
            marker=marker,
        )

    plt.title(title)
    # Retrieve and sort legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    sorted_labels_handles = sorted(zip(labels, handles, strict=False), key=lambda x: x[0])
    sorted_labels, sorted_handles = zip(*sorted_labels_handles, strict=False)
    ax.legend(sorted_handles, sorted_labels, title="Classes", prop={"size": 6})

    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
