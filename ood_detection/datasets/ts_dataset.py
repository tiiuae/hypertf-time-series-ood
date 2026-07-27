from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset


class CustomTimeSeriesDataset(Dataset):
    def __init__(
        self,
        config: DictConfig,
        ts_length: int,
        num_features: int,
        data: ArrayLike,
        labels: ArrayLike,
        label_remap: dict[Any, int],
        transforms_dict: dict[str, Callable],
    ):
        """
        Custom dataset for time-series data with optional transformations.

        Args:
            ts_length (int): The length of the time-series.
            num_features (int): The number of features in each time-step.
            data (numpy.ndarray): The time-series data.
            labels (numpy.ndarray): Corresponding labels, shape (n_instances,).
            label_remap (dict): Mapping from original labels to numeric labels.
            transforms_dict (dict): Holds the main transform to apply to each sample and secondary tfms
        """
        expected_shape = (num_features, ts_length)
        if data.shape[1:] != expected_shape:
            raise ValueError(f"Data dimensions {data.shape[1:]} do not match expected shape {expected_shape}.")
        transform = transforms_dict.get("transform")
        secondary_transform = transforms_dict.get("secondary_transform")

        self.config = config
        self.ts_length = ts_length
        self.num_features = num_features
        self.data = data  # keep as np as it is a requirement for transforms
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.label_remap = label_remap
        self.num_classes = len(np.unique(labels))
        self.transform = transform  # transform being used in get_item
        self.main_transform = transform
        self.secondary_transform = secondary_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the time-series sample and corresponding label at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Time-series sample and label.
        """
        sample = self.data[idx]
        label = self.labels[idx]

        # Apply transformation if provided
        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def set_main_transform(self):
        self.transform = self.main_transform

    def set_secondary_transform(self):
        if self.secondary_transform is None:
            raise ValueError("Secondary transform is None.")
        self.transform = self.secondary_transform
