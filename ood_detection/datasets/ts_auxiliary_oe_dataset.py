from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from omegaconf import DictConfig
import torch

from ood_detection.augmentations.time_series_funcs import random_crop_resize_with_overlap
from ood_detection.datasets import CustomTimeSeriesDataset
from ood_detection.utils.common import TermColors
from ood_detection.utils.data.common import filter_non_id_datasets_by_type, match_features
from ood_detection.utils.data.info import UCR_DATASETS_TYPE, UEA_DATASETS_TYPE
import ood_detection.utils.data.loader as loader_module
from ood_detection.utils.logger import EXP_LOGGER_NAME, LoggerSingleton


class AuxiliaryOutlierExposureTimeSeriesDataset(CustomTimeSeriesDataset):
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
        Custom dataset for time-series data with outlier exposure with optional train/test transformations.

        Args:
            ts_length (int): The length of the time-series.
            num_features (int): The number of features in each time-step.
            data (numpy.ndarray): The time-series data of shape (n_instances, num_features, ts_length).
            labels (numpy.ndarray): Corresponding labels, shape (n_instances,).
            label_remap (dict): Mapping from original labels to numeric labels.
            same_outlier_exposure_sub_type (bool): Whether to use the same dataset subtype when training with OE.
            transforms_dict (dict): Holds the main transform to apply to each sample, synthetic outlier gen and secondary tfms
        """
        super().__init__(
            config=config,
            ts_length=ts_length,
            num_features=num_features,
            data=data,  # keep as np as it is a requirement for transforms
            labels=torch.tensor(labels, dtype=torch.long),
            label_remap=label_remap,
            transforms_dict=transforms_dict,  # transform being used in get_item
        )
        logger = LoggerSingleton.get_logger(EXP_LOGGER_NAME)
        # Whether to return the outlier exposure crops for the contrastive losses
        self.return_two_view_oe_crops = config.dataset.args.return_two_view_oe_crops
        # Outlier exposure feature strategy
        self.oe_sampling_feat_strategy = config.dataset.args.oe_sampling.feature_strategy
        strategy_type = self.oe_sampling_feat_strategy.type
        self.oe_hybrid_full_sample_prob = None
        if strategy_type in {"hybrid_feature_mix", "id_oe_mixup"}:
            self.oe_hybrid_full_sample_prob = self.oe_sampling_feat_strategy.args.get("oe_hybrid_full_sample_prob", 0.5)
            if not 0.0 <= self.oe_hybrid_full_sample_prob <= 1.0:
                raise ValueError("oe_hybrid_full_sample_prob must be within [0, 1].")

        if strategy_type == "id_oe_mixup":
            self.oe_mixup_alpha = self.oe_sampling_feat_strategy.args.get("mixup_alpha", 2.0)
            if self.oe_mixup_alpha <= 0.0:
                raise ValueError("mixup_alpha must be positive.")
            self.oe_mixup_beta = self.oe_sampling_feat_strategy.args.get("mixup_beta", 7.0)
            if self.oe_mixup_beta <= 0.0:
                raise ValueError("mixup_beta must be positive.")
            self.oe_mixup_use_hybrid_seed = self.oe_sampling_feat_strategy.args.get("use_hybrid_feature_seed", True)

        # load OE data
        id_dataset = self.config.dataset.args.name  # e.g., 'ECG5000', 'Epilepsy', etc
        loader_type = config.dataset.args.loader  # 'UCR', 'UEA'
        # Get dataset names map and all dataset names of the same type as ID dataset
        dataset_names_map = {"UCR": UCR_DATASETS_TYPE, "UEA": UEA_DATASETS_TYPE}
        all_datasets = list(dataset_names_map[loader_type].keys())

        # Get the non-ID datasets and datasets of the same type as ID dataset
        non_id_datasets = [d for d in all_datasets if d != id_dataset]
        datasets_of_same_subtype_as_id = filter_non_id_datasets_by_type(
            non_id_datasets, id_dataset, loader_type, use_most_similar_if_no_match=True, logger=logger
        )

        if self.config.dataset.args.same_outlier_exposure_sub_type:
            logger.info("Using OE datasets of the subtype as ID dataset")
            oe_datasets = datasets_of_same_subtype_as_id
        else:
            logger.info("Using OE datasets of different subtype from ID dataset")
            oe_datasets = [d for d in non_id_datasets if d not in datasets_of_same_subtype_as_id]
        logger.info(f"For {loader_type} ID dataset {id_dataset}, loading {len(oe_datasets)} OE datasets: {oe_datasets}")

        # Pick appropriate loader
        loader_map = {"UCR": loader_module.load_ucr, "UEA": loader_module.load_uea}
        if loader_type not in loader_map:
            raise ValueError(f"{TermColors.FAIL}Unsupported far OOD loader type: {loader_type}{TermColors.ENDC}")
        load_function = partial(loader_map[loader_type])

        self.data_oe_list = []
        for dset_oe in oe_datasets:
            oe_config = deepcopy(config)
            oe_config.dataset.args.name = dset_oe
            oe_config.dataset.args.id_classes = None
            oe_config.dataset.args.id_classes_remap = None
            target_feat_len = (
                num_features if self.oe_sampling_feat_strategy.type == "pre_match_features_to_id" else None
            )

            oe_data, _, _, _, _ = loader_module.load_and_preprocess_timeseries_data(
                oe_config,
                dataset_name=dset_oe,
                load_function=load_function,
                id_seq_len=ts_length,
                id_feat_len=target_feat_len,
                oe_feature_strategy=self.oe_sampling_feat_strategy.type,
            )
            self.data_oe_list.append(oe_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the time-series sample and corresponding label at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple of 5 elements:
                - ID sample
                - ID label
                - OE sample or OE view 1
                - OE label or OE view 2
                - OE label (always returned for consistency; dummy if unused)
        """
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)
        if self.transform != self.main_transform:  # used for extracting embedding bank
            return sample, label

        oe_sampling_probs = self._get_oe_sampling_probs()
        if self.oe_sampling_feat_strategy.type == "hybrid_feature_mix":
            oe_sample_raw = self._sample_hybrid_outlier(
                target_features=self.num_features,
                sampling_probs=oe_sampling_probs,
                p_full_sample=self.oe_hybrid_full_sample_prob,
            )
        elif self.oe_sampling_feat_strategy.type == "id_oe_mixup":
            oe_sample_raw = self._sample_id_oe_mixup(
                sampling_probs=oe_sampling_probs,
                mixup_alpha=self.oe_mixup_alpha,
                mixup_beta=self.oe_mixup_beta,
                use_hybrid_seed=self.oe_mixup_use_hybrid_seed,
            )
        else:
            oe_sample_raw = self._sample_auxiliary_series(oe_sampling_probs)

        # Dummy OE label (always returned for API consistency)
        oe_label = torch.tensor(self.num_classes, dtype=label.dtype)

        if self.return_two_view_oe_crops:
            # Two stochastic crops of the same OE sample
            oe_view1, oe_view2 = random_crop_resize_with_overlap(
                oe_sample_raw, scale=self.config.dataset.args.crop_scale, min_overlap=0.001
            )

            if self.transform:
                oe_view1 = self.transform(oe_view1)
                oe_view2 = self.transform(oe_view2)

            return sample, label, oe_view1, oe_view2, oe_label

        else:
            # Single-view OE mode (non-contrastive)
            oe_sample = oe_sample_raw
            if self.transform:
                oe_sample = self.transform(oe_sample)

            return sample, label, oe_sample, oe_label

    def _get_oe_sampling_probs(self) -> np.ndarray:
        """Get square-Root Normalized Probabilities to sample OE datasets fairly."""
        dataset_sizes = np.array([len(dset) for dset in self.data_oe_list], dtype=np.float64)
        if dataset_sizes.size == 0:
            raise ValueError("No OE datasets available for sampling.")
        probs = np.sqrt(dataset_sizes)
        total = probs.sum()
        if total == 0:
            raise ValueError("All OE datasets are empty.")
        return probs / total

    def _sample_auxiliary_series(self, sampling_probs: np.ndarray) -> np.ndarray:
        """Sample a single auxiliary OE time-series from one of the OE datasets."""
        dataset_idx = np.random.choice(len(self.data_oe_list), p=sampling_probs)
        # select which OE dataset to sample from
        dataset = self.data_oe_list[dataset_idx]
        sample_idx = np.random.randint(0, len(dataset))
        # sample random index from the chosen dataset
        return dataset[sample_idx]

    def _sample_hybrid_outlier(
        self, target_features: int, sampling_probs: np.ndarray, p_full_sample: float
    ) -> np.ndarray:
        """
        Build an outlier sample by mixing features pulled from auxiliary datasets until `target_features` is reached.
        Args:
            target_features (int): Number of features to sample for the hybrid outlier.
            sampling_probs (np.ndarray): Sampling probabilities for each auxiliary dataset.
            p_full_sample (float): Probability of taking all features from a sampled OE series.
        """
        outlier_features = []
        while len(outlier_features) < target_features:
            oe_sample = self._sample_auxiliary_series(sampling_probs)
            num_feats = oe_sample.shape[0]
            if num_feats == 0:
                continue

            if np.random.rand() < p_full_sample:
                outlier_features.extend([oe_sample[i].copy() for i in range(num_feats)])
            else:
                feat_idx = np.random.randint(0, num_feats)
                outlier_features.append(oe_sample[feat_idx].copy())

        total_feats = len(outlier_features)
        if total_feats > target_features:
            selected = np.random.choice(total_feats, target_features, replace=False)
            outlier_features = [outlier_features[i] for i in selected]

        np.random.shuffle(outlier_features)
        return np.stack(outlier_features[:target_features], axis=0)

    def _sample_id_oe_mixup(
        self, sampling_probs: np.ndarray, mixup_alpha: float, mixup_beta: float, use_hybrid_seed: bool
    ) -> np.ndarray:
        """
        Mix a randomly drawn ID sample with a sampled OE sample using mixup-style interpolation.

        Args:
            sampling_probs (np.ndarray): Probabilities for sampling OE datasets.
            mixup_alpha (float): Alpha parameter for the Beta distribution used to sample mixup weights.
            mixup_beta (float): Beta parameter for the Beta distribution used to sample mixup weights.
            use_hybrid_seed (bool): Whether to build the OE sample via the hybrid feature mix routine.
        """
        if len(self.data) == 0:
            raise ValueError("Cannot perform ID/OE mixup without ID data.")

        id_idx = np.random.randint(0, len(self.data))
        id_sample = self.data[id_idx]
        if use_hybrid_seed:
            if self.oe_hybrid_full_sample_prob is None:
                raise ValueError("Hybrid seed requested without oe_hybrid_full_sample_prob configured.")
            oe_sample = self._sample_hybrid_outlier(
                target_features=self.num_features,
                sampling_probs=sampling_probs,
                p_full_sample=self.oe_hybrid_full_sample_prob,
            )
        else:
            oe_sample_raw = self._sample_auxiliary_series(sampling_probs)
            oe_sample = np.expand_dims(oe_sample_raw.T, axis=0)
            oe_sample = match_features(oe_sample, target_features=self.num_features)
            oe_sample = np.squeeze(oe_sample, axis=0).T

        mixup_lambda = self._sample_mixup_lambda(alpha=mixup_alpha, beta=mixup_beta)
        return mixup_lambda * id_sample + (1.0 - mixup_lambda) * oe_sample

    @staticmethod
    def _sample_mixup_lambda(alpha: float, beta: float) -> float:
        """Sample a mixup interpolation factor."""
        if alpha <= 0 or beta <= 0:
            return 1.0
        return float(np.random.beta(alpha, beta))
