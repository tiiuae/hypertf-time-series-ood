from ood_detection.augmentations.time_series_funcs import random_crop_resize_views

from .ts_dataset import CustomTimeSeriesDataset


class ContrastiveTimeSeriesDataset(CustomTimeSeriesDataset):
    def __init__(self, *args, **kwargs):
        """
        Contrastive Dataset that returns two augmented views per sample.
        """
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        """
        Returns two augmented views of a sample instead of one.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (view1, view2, label)
        """
        sample = self.data[idx]  # Get raw data (no transforms applied)
        label = self.labels[idx]

        if self.transform == self.main_transform:  # during training
            view1, view2 = random_crop_resize_views(sample, scale=self.config.dataset.args.crop_scale)

            # Apply contrastive-specific augmentations
            view1 = self.transform(view1)
            view2 = self.transform(view2)

            return view1, view2, label
        else:  # used for extracting embedding bank
            sample = self.transform(sample)

            return sample, label
