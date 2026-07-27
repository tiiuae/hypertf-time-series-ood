from contextlib import nullcontext

from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ood_detection.trainers.base_trainer import BaseTrainer
from ood_detection.utils.common import AverageMeter
from ood_detection.utils.model_ops import frozen_stochastic


class AngularAuxiliaryContrastiveTrainer(BaseTrainer):
    def __init__(self, config, device, train_loader, test_loader, class_weights, ood_evaluator=None):
        # Initialize the common trainer components from BaseTrainer.
        train_csv_headers = ["epoch", "loss", "accuracy", "f1", "lr"]
        val_csv_headers = [
            "epoch",
            "loss",
            "accuracy",
            "f1",
            "accuracy_nn",
            "f1_nn",
            "accuracy_prot",
            "f1_prot",
            "pre_proj_dispersion",
            "post_proj_dispersion",
            "pre_proj_compactness",
            "post_proj_compactness",
        ]

        super().__init__(
            config,
            device,
            train_loader,
            test_loader,
            class_weights,
            ood_evaluator,
            train_csv_headers,
            val_csv_headers,
        )

    def train_epoch(self, epoch: int) -> dict:
        """
        Train the model for one epoch with an extra update step for center loss parameters.
        Overrides the BaseTrainer's train_epoch.
        """
        self.model.train()
        # Initialize your loss meter and progress bar as in your BaseTrainer.
        loss_meter = AverageMeter()
        all_labels = []
        all_predictions = []

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Training]", leave=False)

        for batch_idx, (samples, labels, oe_view1, oe_view2, _) in enumerate(self.train_loader):
            samples = samples.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            oe_view1 = oe_view1.to(self.device, non_blocking=True)
            oe_view2 = oe_view2.to(self.device, non_blocking=True)

            # Forward pass on in-distribution data (updates BN stats)
            (
                temporal_features,
                instance_features,
                projected_features,
                logits,
                id_sec_projected_features,
            ) = self.model(samples)

            # Forward pass on auxiliary views
            # freeze stochastic updates like Norms & Dropout for non-ID data if freeze enabled
            freeze = self.config.trainer.args.freeze_stochastic_for_non_id
            with frozen_stochastic(self.model) if freeze else nullcontext():
                *_, aux_sec_proj_view1 = self.model(oe_view1)
                *_, aux_sec_proj_view2 = self.model(oe_view2)

            # Compute loss
            loss_kwargs = {
                "logits": logits,
                "temporal_features": temporal_features,
                "instance_features": instance_features,
                "projected_features": projected_features,
                "labels": labels,
                "id_sec_projected_features": id_sec_projected_features,
                "aux_sec_proj_view1": aux_sec_proj_view1,
                "aux_sec_proj_view2": aux_sec_proj_view2,
            }

            loss = self.loss(**loss_kwargs)

            if not torch.isfinite(loss):
                raise ValueError(f"Loss is NaN or Inf at batch {batch_idx} in epoch {epoch}.")

            # Get predictions
            _, predicted = torch.max(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Zero gradients for both optimizers.
            self.optimizer.zero_grad()

            # Backward pass.
            loss.backward()

            # Update model parameters.
            self.optimizer.step()

            # Optionally update your learning rate scheduler here.
            self._step_lr_scheduler(epoch, batch_idx, len(self.train_loader))

            # Update your loss meter and progress bar (assume similar helper methods exist).
            loss_meter.update(loss.item(), samples.size(0))
            current_lr = self.optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                Loss=f"{loss.item():.4f}", Mean_Loss=f"{loss_meter.avg:.4f}", LR=f"{current_lr:.6f}"
            )

        # Compute accuracy and F1 score
        accuracy = accuracy_score(all_labels, all_predictions) * 100
        f1 = f1_score(all_labels, all_predictions, average="weighted") * 100

        # Log training metrics as needed.
        self.train_metrics_logger.info(f"{epoch},{loss.item()},{accuracy},{f1},{current_lr}", to_console=False)
        metrics = {"loss": loss_meter.avg, "accuracy": accuracy, "f1": f1}
        return metrics

    @torch.no_grad()
    def _run_validation_forward(self, loss_meter: AverageMeter):
        inst_feats_list, proj_feats_list, logits_list, targets = [], [], [], []
        for data, labels in self.test_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            temp_feats, inst_feats, proj_feats, logits, _ = self.model(data)

            loss_kwargs = {
                "logits": logits,
                "temporal_features": temp_feats,
                "instance_features": inst_feats,
                "projected_features": proj_feats,
                "labels": labels,
            }
            loss = self.loss(**loss_kwargs)
            loss_meter.update(loss.item(), data.size(0))
            inst_feats_list.append(inst_feats)
            proj_feats_list.append(proj_feats)
            logits_list.append(logits)
            targets.append(labels)

        return (
            F.normalize(torch.cat(inst_feats_list), p=2, dim=1),
            F.normalize(torch.cat(proj_feats_list), p=2, dim=1),
            torch.cat(logits_list),
            torch.cat(targets),
            loss_meter.avg,
        )

    @torch.no_grad()
    def _extract_train_embeddings(self):
        inst_emb_list, proj_emb_list, labels_list = [], [], []
        for data, labels in self.train_loader_for_val:
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            _, inst_feats, proj_feats, _, _ = self.model(data)
            inst_emb_list.append(inst_feats)
            proj_emb_list.append(proj_feats)
            labels_list.append(labels.cpu())

        return (
            F.normalize(torch.cat(inst_emb_list), p=2, dim=1),
            F.normalize(torch.cat(proj_emb_list), p=2, dim=1),
            torch.cat(labels_list),
        )
