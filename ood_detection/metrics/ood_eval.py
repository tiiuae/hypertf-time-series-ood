from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def init_metric_function(metric_name: str) -> Callable:
    """Return the corresponding function for an evaluation metric using sklearn."""
    metric_functions = {
        "auroc": compute_auroc,
        "fpr95": partial(compute_fpr_at_tpr, tpr_level=0.95),
        "aupr": compute_aupr,
    }
    return metric_functions.get(metric_name.lower())


def compute_auroc(id_scores, ood_scores) -> np.float32:
    """Compute Area Under the Receiver Operating Characteristic Curve (AUROC)."""
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(labels, scores)


def compute_aupr(id_scores, ood_scores) -> np.float32:
    """Compute Area Under the Precision-Recall Curve (AUPR)."""
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return average_precision_score(labels, scores)


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """
    Computes cumulative sum with higher precision and checks numerical stability.

    Args:
        arr (array-like): Input array to cumulatively sum.
        rtol (float): Relative tolerance for numerical check.
        atol (float): Absolute tolerance for numerical check.

    Returns:
        np.ndarray: Cumulative sum of `arr` with float64 precision.
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError("cumsum was found to be unstable: its last element does not correspond to sum")
    return out


def compute_fpr_at_tpr(id_scores, ood_scores, tpr_level=0.95, pos_label=None):
    """
    Compute the False Positive Rate (FPR) at a given tpr/recall level (e.g., 95%).
    https://github.com/deeplearning-wisc/cider/blob/dce8ad36b035ec3043cb3936bb70508e86c3af19/utils/detection_util.py
    """
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_score = np.concatenate([id_scores, ood_scores])

    classes = np.unique(y_true)
    if pos_label is None and not (
        np.array_equal(classes, [0, 1])
        or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0])
        or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.0

    # make y_true a boolean vector
    y_true = y_true == pos_label

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = (
        np.r_[recall[sl], 1],
        np.r_[fps[sl], 0],
        np.r_[tps[sl], 0],
        thresholds[sl],
    )

    cutoff = np.argmin(np.abs(recall - tpr_level))
    fpr_at_tpr_level = fps[cutoff] / (np.sum(np.logical_not(y_true)))
    fdr_at_tpr_level = fps[cutoff] / (fps[cutoff] + tps[cutoff])
    return fpr_at_tpr_level

