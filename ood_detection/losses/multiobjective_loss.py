from types import SimpleNamespace

import torch.nn as nn


class MultiObjectiveLoss(nn.Module):
    def __init__(self, loss_dict, lambda_dict, lambda_warmup_epochs_dict):
        """
        Multi-objective loss function combining multiple losses.

        Args:
            loss_dict (dict): Dictionary mapping loss names to instantiated loss functions (nn.Module).
            lambda_dict (dict): Dictionary mapping loss names to their respective weight (lambda).
            lambda_warmup_epochs (dict): Dictionary mapping loss names to number of epochs for lambda warm-up.
        """
        super().__init__()
        self.loss_dict = loss_dict
        self.lambda_dict = lambda_dict
        self.lambda_warmup_epochs_dict = lambda_warmup_epochs_dict

        # Track progress for lambda warm-up schedules
        self.current_epoch = 0
        self.total_epochs = 1

    def set_epoch_progress(self, epoch: int, total_epochs: int | None = None):
        """Update internal epoch pointer so losses can warm up their lambda. Called from trainer."""
        self.current_epoch = max(0, int(epoch))
        if total_epochs is not None:
            self.total_epochs = max(1, int(total_epochs))

    def _scale_lambda(self, loss_name: str) -> float:
        """Linearly scale loss lambda based on current epoch and warmup epochs."""
        base_lambda = self.lambda_dict[loss_name]
        warmup_epochs = self.lambda_warmup_epochs_dict[loss_name]

        if not warmup_epochs or warmup_epochs <= 0:
            return base_lambda

        progress = min(self.current_epoch / float(warmup_epochs), 1.0)
        return base_lambda * progress

    def forward(self, **kwargs: dict):
        """
        Computes the combined loss.

        Kwargs contain:
            logits (torch.Tensor): Model predictions (logits, cosine similarities,
                or projected norm features in case of contrastive).
            temporal_features (torch.Tensor): Per timestamp feature embeddings.
            instance_features (torch.Tensor): Pooled embedding featueres.
            projected_features (torch.Tensor): Features projected using a proj. head from instance features.
            labels (torch.Tensor): Ground-truth labels.

        Returns:
            torch.Tensor: Weighted sum of all the combined losses.
            dict: Individual loss values for logging/debugging.
        """
        kw = SimpleNamespace(**kwargs)
        total_loss = 0.0
        loss_values = {}

        for loss_name, loss_fn in self.loss_dict.items():
            lambda_val = self.lambda_dict.get(loss_name, None)
            assert lambda_val is not None, f"Provide lambda for {loss_name}."

            lambda_val = self._scale_lambda(loss_name)

            if lambda_val == 0:
                continue
            # CE losses
            if loss_name in {"CrossEntropyLoss", "TemperatureScaledCELoss", "RescaledTemperatureCELoss"}:
                loss_value = loss_fn(kw.logits, kw.labels)
            # compactness/separation losses
            elif loss_name == "CompactnessLoss":
                loss_value = loss_fn(kw.logits, kw.labels)
            elif loss_name == "SeparationLoss":
                loss_value = loss_fn()
            # binary class based losses
            elif loss_name == "BCEWithLogitsLoss":
                loss_value = loss_fn(kw.logits, kw.labels)
            # contrastive losses
            elif loss_name == "InfoNCELoss" or loss_name == "NTXentLoss":
                loss_value = loss_fn(projected_features=kw.projected_features, instance_features=kw.instance_features)
            elif loss_name == "SupConLoss":
                loss_value = loss_fn(
                    projected_features=kw.projected_features, instance_features=kw.instance_features, labels=kw.labels
                )
            elif loss_name == "AuxiliaryContrastiveLoss":
                if not all(
                    hasattr(kw, attr)
                    for attr in ["aux_sec_proj_view1", "aux_sec_proj_view2", "id_sec_projected_features"]
                ):
                    continue
                loss_value = loss_fn(kw.aux_sec_proj_view1, kw.aux_sec_proj_view2, kw.id_sec_projected_features)
            elif loss_name == "FFTConsistencyLoss":
                loss_value = loss_fn(projected_features=kw.projected_features)
            else:
                raise NotImplementedError(f"Loss {loss_name} not implemented.")

            loss_values[loss_name] = loss_value.item()
            total_loss += lambda_val * loss_value

        return total_loss
