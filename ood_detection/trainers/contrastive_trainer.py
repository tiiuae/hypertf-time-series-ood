import torch
import torch.nn.functional as F
from tqdm import tqdm

from ood_detection.metrics.hyperspherical import (
    compute_compactness,
    compute_dispersion,
    compute_nn_compactness,
    compute_nn_dispersion,
    compute_prototypes,
    eval_classifier_logits,
    eval_nearest_neighbor,
    eval_prototype_classification,
)
from ood_detection.trainers.base_trainer import BaseTrainer
from ood_detection.utils.common import AverageMeter


class ContrastiveTrainer(BaseTrainer):
    def __init__(self, config, device, train_loader, test_loader, class_weights, ood_evaluator=None):
        # Initialize the common trainer components from BaseTrainer.
        train_csv_headers = ["epoch", "loss", "lr"]
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
            "nn_compactness",
            "nn_dispersion",
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

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Training]", leave=False)

        for batch_idx, (view1, view2, labels) in enumerate(self.train_loader):
            view1 = view1.to(self.device, non_blocking=True)
            view2 = view2.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward each view separately
            temp1, inst1, proj1, _, logits1 = self.model(view1)
            temp2, inst2, proj2, _, logits2 = self.model(view2)

            # ---------------------------------------------------------------
            # ⚠️ FIX: Correct concatenation for FFTConsistencyLoss (Solution B)
            # ---------------------------------------------------------------
            if self.model.fft_encoder is not None:
                # proj1 = [raw_v1(1..B); fft_v1(1..B)]
                # proj2 = [raw_v2(1..B); fft_v2(1..B)]
                B = proj1.shape[0] // 2

                # Raw first, then FFT second
                projected_features = torch.cat(
                    [
                        proj1[:B],  # raw view1
                        proj2[:B],  # raw view2
                        proj1[B:],  # fft view1
                        proj2[B:],  # fft view2
                    ],
                    dim=0,
                )

                temporal_features = torch.cat(
                    [
                        temp1[:B],
                        temp2[:B],
                        temp1[B:],
                        temp2[B:],
                    ],
                    dim=0,
                )

                instance_features = torch.cat(
                    [
                        inst1[:B],
                        inst2[:B],
                        inst1[B:],
                        inst2[B:],
                    ],
                    dim=0,
                )

                logits = torch.cat(
                    [
                        logits1[:B],
                        logits2[:B],
                        logits1[B:],
                        logits2[B:],
                    ],
                    dim=0,
                )

                # Duplicate labels for raw+fft
                labels = labels.repeat(2)
            else:
                projected_features = torch.cat([proj1, proj2], dim=0)
                temporal_features = torch.cat([temp1, temp2], dim=0)
                instance_features = torch.cat([inst1, inst2], dim=0)
                logits = torch.cat([logits1, logits2], dim=0)

            # Compute loss
            loss_kwargs = {
                "logits": logits,
                "projected_features": projected_features,
                "temporal_features": temporal_features,
                "instance_features": instance_features,
                "labels": labels,
            }
            loss = self.loss(**loss_kwargs)

            if not torch.isfinite(loss):
                raise ValueError(f"Loss is NaN or Inf at batch {batch_idx} in epoch {epoch}. {loss.item()}")

            # Zero gradients for both optimizers.
            self.optimizer.zero_grad()

            # Backward pass.
            loss.backward()

            # Update model parameters.
            self.optimizer.step()

            # Optionally update your learning rate scheduler here.
            self._step_lr_scheduler(epoch, batch_idx, len(self.train_loader))

            # Update your loss meter and progress bar (assume similar helper methods exist).
            loss_meter.update(loss.item(), view1.size(0))
            current_lr = self.optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                Loss=f"{loss.item():.4f}", Mean_Loss=f"{loss_meter.avg:.4f}", LR=f"{current_lr:.6f}"
            )

        # Log training metrics as needed.
        self.train_metrics_logger.info(f"{epoch},{loss_meter.avg},{current_lr}", to_console=False)
        metrics = {"loss": loss_meter.avg}
        return metrics

    def _fuse_fft(self, inst, proj, logits):
        # inst, proj, logits are shape (2B, D) or (2B, C)
        B = inst.shape[0] // 2

        # instance features: concatenate raw and FFT along channel dimension
        inst_t = inst[:B]
        inst_f = inst[B:]
        inst_final = torch.cat([inst_t, inst_f], dim=1)  # shape (B, 2D)

        # projected features: mean fuse then normalize
        proj_t = proj[:B]
        proj_f = proj[B:]
        proj_final = F.normalize((proj_t + proj_f) / 2, dim=1)

        # logits: fuse by mean
        log_t = logits[:B]
        log_f = logits[B:]
        logits_final = (log_t + log_f) / 2

        return inst_final, proj_final, logits_final

    @torch.no_grad()
    def validate_epoch(self, epoch: int, best_metrics: dict) -> dict:
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()

        loss_meter = AverageMeter()
        progress_bar = tqdm(self.test_loader, desc=f"Epoch {epoch} [Validation]", leave=True)

        #  Step 1: Training embeddings for NN and post-proj prototypes
        train_embeddings, train_proj_embeddings, train_labels = self._extract_train_embeddings()

        #  Step 2: Validation forward pass
        test_instance_embeddings, test_projected_embeddings, test_logits, test_targets, test_loss = (
            self._run_validation_forward(loss_meter)
        )
        pre_proj_prototypes = compute_prototypes(
            train_embeddings, train_labels, self.train_loader.dataset.num_classes, self.device
        )

        #  Step 3: Classification metrics (NN / Proto / Logits)
        accuracy_nn, f1_nn, preds_nn = eval_nearest_neighbor(
            test_instance_embeddings, train_embeddings, train_labels, test_targets
        )
        accuracy_prot, f1_prot, preds_prot = eval_prototype_classification(
            test_instance_embeddings, test_targets, pre_proj_prototypes
        )
        accuracy_logits, f1_logits, preds_cls = eval_classifier_logits(test_logits, test_targets)

        #  Step 4: Compactness & Dispersion
        if self.config.model.args.cosine:
            post_proj_prototypes = F.normalize(self.model.classifier, p=2, dim=1)
        else:
            post_proj_prototypes = compute_prototypes(
                train_proj_embeddings, train_labels, self.train_loader.dataset.num_classes, self.device
            )

        pre_compact = compute_compactness(test_instance_embeddings, test_targets, pre_proj_prototypes, self.device)
        post_compact = compute_compactness(test_projected_embeddings, test_targets, post_proj_prototypes, self.device)
        pre_disp = compute_dispersion(pre_proj_prototypes)
        post_disp = compute_dispersion(post_proj_prototypes)

        # ▶️ Add FAISS-based metrics (Step 4.5)
        nn_compact = compute_nn_compactness(test_instance_embeddings, test_targets)
        nn_disp = compute_nn_dispersion(test_instance_embeddings, test_targets)

        log_metrics = {
            "epoch": epoch,
            "loss": test_loss,
            "accuracy": accuracy_logits,
            "f1": f1_logits,
            "accuracy_nn": accuracy_nn,
            "f1_nn": f1_nn,
            "accuracy_prot": accuracy_prot,
            "f1_prot": f1_prot,
            "pre_proj_dispersion": pre_disp,
            "post_proj_dispersion": post_disp,
            "pre_proj_compactness": pre_compact,
            "post_proj_compactness": post_compact,
            "nn_compactness": nn_compact,
            "nn_dispersion": nn_disp,
        }
        self._log_validation(
            log_metrics,
            console_fields=(
                "epoch",
                "loss",
                "accuracy",
                "accuracy_nn",
                "accuracy_prot",
                "nn_compactness",
                "nn_dispersion",
            ),
        )

        best_metrics["loss"] = min(best_metrics["loss"], test_loss)
        best_metrics["accuracy"] = max(best_metrics["accuracy"], accuracy_nn)
        best_metrics["f1"] = max(best_metrics["f1"], f1_logits)

        progress_bar.set_postfix(
            mean_loss=f"{test_loss:.4f}",
            acc_logits=f"{accuracy_logits:.2f}%",
            f1_logits=f"{f1_logits:.4f}",
            f1_nn=f"{f1_nn:.4f}",
            f1_prot=f"{f1_prot:.4f}",
            pre_disp=f"{pre_disp:.4f}",
            post_disp=f"{post_disp:.4f}",
            pre_comp=f"{pre_compact:.4f}",
            post_comp=f"{post_compact:.4f}",
            nn_comp=f"{nn_compact:.4f}",
            nn_disp=f"{nn_disp:.4f}",
        )

        return {
            "loss": test_loss,
            "accuracy": accuracy_logits,
            "f1": f1_logits,
            "accuracy_prot": accuracy_prot,
            "f1_prot": f1_prot,
            "accuracy_nn": accuracy_nn,
            "pre_proj_dispersion": pre_disp,
            "post_proj_dispersion": post_disp,
            "pre_proj_compactness": pre_compact,
            "post_proj_compactness": post_compact,
            "labels": test_targets.cpu().numpy(),
            "preds": preds_cls.cpu().numpy(),
            "preds_prot": preds_prot.cpu().numpy(),
            "preds_nn": preds_nn.cpu().numpy(),
        }

    @torch.no_grad()
    def _extract_train_embeddings(self):
        inst_emb_list, proj_ce_emb_list, labels_list = [], [], []
        for data, labels in self.train_loader_for_val:
            data, labels = data.to(self.device), labels.to(self.device)
            _, inst, proj_sup, proj_ce, _ = self.model(data)
            proj_ce = proj_sup if proj_ce is None else proj_ce

            if self.model.fft_encoder is not None:
                B = labels.shape[0]
                inst_t, inst_f = inst[:B], inst[B:]
                proj_t, proj_f = proj_ce[:B], proj_ce[B:]
                inst = torch.cat([inst_t, inst_f], dim=1)
                proj_ce = F.normalize((proj_t + proj_f) / 2, dim=1)

            inst_emb_list.append(inst.detach().cpu())
            proj_ce_emb_list.append(proj_ce.detach().cpu())
            labels_list.append(labels.cpu())

        return (
            F.normalize(torch.cat(inst_emb_list).to(self.device), p=2, dim=1),
            F.normalize(torch.cat(proj_ce_emb_list).to(self.device), p=2, dim=1),
            torch.cat(labels_list),
        )

    @torch.no_grad()
    def _run_validation_forward(self, loss_meter):
        instance_feats, proj_ce_feats, logits_list, targets = [], [], [], []
        for data, labels in self.test_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            temp_feats, inst_feats, proj_sup, proj_ce, logits = self.model(data)
            proj_ce = proj_sup if proj_ce is None else proj_ce

            loss_projected_features = torch.cat([proj_ce, proj_ce])

            if self.model.fft_encoder is not None:
                raw_fft_projected_features = proj_ce
                inst_feats, proj_ce, logits = self._fuse_fft(inst_feats, proj_ce, logits)

                # Keep validation metrics on fused features, but preserve the raw/FFT
                # pairing for FFT consistency loss computation.
                if "FFTConsistencyLoss" in getattr(self.loss, "loss_dict", {}):
                    loss_projected_features = raw_fft_projected_features

            proj_ce = F.normalize(proj_ce, dim=1)
            loss = self.loss(
                logits=logits,
                projected_features=loss_projected_features,
                temporal_features=torch.cat([temp_feats, temp_feats]),
                instance_features=torch.cat([inst_feats, inst_feats]),
                labels=labels,
            )
            loss_meter.update(loss.item(), data.size(0))
            instance_feats.append(inst_feats)
            proj_ce_feats.append(proj_ce)
            logits_list.append(logits)
            targets.append(labels)

        return (
            F.normalize(torch.cat(instance_feats), p=2, dim=1),
            F.normalize(torch.cat(proj_ce_feats), p=2, dim=1),
            torch.cat(logits_list),
            torch.cat(targets),
            loss_meter.avg,
        )
