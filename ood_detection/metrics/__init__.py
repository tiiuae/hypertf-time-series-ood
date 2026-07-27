from .plots import plot_calibration_curve, plot_pr_curve, plot_roc_curve

IMPLEMENTED_METRICS_PLOTS = {
    "roc_curve": plot_roc_curve,
    "pr_curve": plot_pr_curve,
    "calibration_curve": plot_calibration_curve,
}

DEFAULT_VAL_METRIC_PRECISIONS = {
    "epoch": 0,
    "loss": 4,
    "accuracy": 2,
    "f1": 4,
    "accuracy_nn": 2,
    "f1_nn": 4,
    "accuracy_prot": 2,
    "f1_prot": 4,
    "pre_proj_dispersion": 4,
    "post_proj_dispersion": 4,
    "pre_proj_compactness": 4,
    "post_proj_compactness": 4,
}
DEFAULT_FALLBACK_PRECISION = 4


def filter_kwargs(kwargs: dict, exclude_keys: set) -> dict:
    """Filter arguments based on the required inputs for a metric."""
    return {k: v for k, v in kwargs.items() if k not in exclude_keys}


def plot_metric(metric_name: str, **kwargs) -> None:
    """Plot the metric."""
    try:
        if metric_name in {"roc_curve", "pr_curve"}:
            kwargs = filter_kwargs(kwargs, "y_score")
        elif metric_name in {"calibration_curve"}:
            kwargs = filter_kwargs(kwargs, "y_pred")
        IMPLEMENTED_METRICS_PLOTS[metric_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{metric_name} is not implemented. "
            + f"Available metrics for plotting are {IMPLEMENTED_METRICS_PLOTS.keys()}"
        ) from exc
