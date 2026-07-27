from .ts_dataset import CustomTimeSeriesDataset
from .ts_auxiliary_oe_dataset import AuxiliaryOutlierExposureTimeSeriesDataset
from .ts_contrastive_dataset import ContrastiveTimeSeriesDataset

IMPLEMENTED_DATASETS = {
    "CustomTimeSeriesDataset": CustomTimeSeriesDataset,
    "ContrastiveTimeSeriesDataset": ContrastiveTimeSeriesDataset,
    "AuxiliaryOutlierExposureTimeSeriesDataset": AuxiliaryOutlierExposureTimeSeriesDataset,
}


def init_dataset(dataset_type: str, **kwargs):
    """Initialize the dataset."""
    try:
        dataset = IMPLEMENTED_DATASETS[dataset_type](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{dataset_type} is not implemented. " + f"Available Datasets: {IMPLEMENTED_DATASETS.keys()}"
        ) from exc
    return dataset
