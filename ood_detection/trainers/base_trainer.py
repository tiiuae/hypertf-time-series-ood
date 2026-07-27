import copy
import math
import multiprocessing
import os
import os.path as osp

import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ood_detection.losses import init_loss
from ood_detection.metrics import DEFAULT_FALLBACK_PRECISION, DEFAULT_VAL_METRIC_PRECISIONS
from ood_detection.metrics.hyperspherical import (
    compute_compactness,
    compute_dispersion,
    compute_prototypes,
    eval_classifier_logits,
    eval_linear_svm,
    eval_nearest_neighbor,
    eval_prototype_classification,
)
from ood_detection.metrics.plots import plot_confusion_matrix, plot_train_val_metrics
from ood_detection.models.encoder import init_model
from ood_detection.ood_eval.ood_evaluator import OODEvaluator
from ood_detection.optimizers import init_optimizer
from ood_detection.utils.common import AverageMeter
from ood_detection.utils.logger import (
    EXP_LOGGER_NAME,
    TRAIN_METRICS_LOGGER_NAME,
    VAL_METRICS_LOGGER_NAME,
    LoggerSingleton,
)
from ood_detection.utils.visualization import plot_ood_histograms, plot_ood_stacked


class BaseTrainer:
    BATCH_STEP_SCHEDULERS = {"OneCycleLR", "CosineAnnealingWarmRestarts"}

    def __init__(
        self,
        config: DictConfig,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        class_weights: np.ndarray,
        ood_evaluator: OODEvaluator = None,
        train_csv_headers: list[str] = None,
        val_csv_headers: list[str] = None,
    ):
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.ood_evaluator = ood_evaluator

        if train_csv_headers is None:
            train_csv_headers = ["epoch", "loss", "accuracy", "f1", "lr"]
        if val_csv_headers is None:
            val_csv_headers = [
                "epoch",
                "loss",
                "accuracy",
                "f1",
                "accuracy_nn",
                "f1_nn",
                "accuracy_prot",
                "f1_prot",
                "accuracy_svm",
                "f1_svm",
                "pre_proj_dispersion",
                "post_proj_dispersion",
                "pre_proj_compactness",
                "post_proj_compactness",
            ]

        self.train_csv_headers = list(train_csv_headers)
        self.val_csv_headers = list(val_csv_headers)
        self._val_metric_precisions = DEFAULT_VAL_METRIC_PRECISIONS.copy()

        # Init trainer components
        self.model = init_model(
            self.config,
            self.config.model.type,
            dataset=train_loader.dataset,
            device=self.device,
            verbose=config.verbose,
        ).to(self.device)

        # Init loss
        self.loss = init_loss(
            self.config.loss,
            device,
            num_classes=train_loader.dataset.num_classes,
            feat_dim=self.model.pooled_feature_dim,
            prototypes=self.model.classifier,
            class_weights=class_weights,
        )
        if hasattr(self.loss, "set_epoch_progress"):  # for multi-obj loss with lambda warm-ups
            self.loss.set_epoch_progress(0, self.config.trainer.args.epochs)

        # Init optimizer
        self.optimizer, self.lr_scheduler = init_optimizer(
            self.config.optimizer.type,
            self.config.lr_scheduler.type,
            self.model,
            self.train_loader,
            self.config,
        )

        # get singleton train & metrics logger
        self.logger = LoggerSingleton.get_logger(EXP_LOGGER_NAME)
        self.train_metrics_logger = LoggerSingleton.get_logger(TRAIN_METRICS_LOGGER_NAME)
        self.val_metrics_logger = LoggerSingleton.get_logger(VAL_METRICS_LOGGER_NAME)
        self._warned_unlogged_metrics = False
        # log the csv metric headers
        self.train_metrics_logger.info(",".join(self.train_csv_headers).replace(" ", ""), to_console=False)
        self.val_metrics_logger.info(",".join(self.val_csv_headers).replace(" ", ""), to_console=False)

        # load existing chkp if resume_checkpoint provided
        resume_ckpt_path = self.config.trainer.args.resume_checkpoint
        if resume_ckpt_path:
            model_type = config.get("model_type", "pytorch")
            self.model = self.load_checkpoint(self.model, resume_ckpt_path, model_type).to(device)

        # create train loader with secondary transform for extracting train embeddings during validation
        cpus_per_gpu = max(2, (multiprocessing.cpu_count() - 8) // 8)
        train_workers = min(4, cpus_per_gpu - 1)
        train_dataset_copy = copy.deepcopy(self.train_loader.dataset)
        self.train_loader_for_val = DataLoader(
            train_dataset_copy,
            batch_size=self.config.dataloader.args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=train_workers,
            prefetch_factor=2,
            persistent_workers=(train_workers > 0),
            pin_memory=torch.cuda.is_available(),
        )
        self.train_loader_for_val.dataset.set_secondary_transform()

        if self.config.verbose:
            self.logger.info(f"Model: {self.config.model}")
            self.logger.info(f"Model Type: {self.config.model.type}")
            self.logger.info(f"Loss: {self.loss}")
            self.logger.info(f"Optimizer: {self.optimizer}")
            self.logger.info(f"LR Scheduler: {self.config.lr_scheduler.type}\n")

    def _step_lr_scheduler(self, epoch: int, batch_idx: int, num_batches: int) -> None:
        """Step supported per-batch schedulers with the right convention."""
        if not self.lr_scheduler:
            return

        scheduler_type = self.config.lr_scheduler.type
        if scheduler_type not in self.BATCH_STEP_SCHEDULERS:
            return

        if scheduler_type == "CosineAnnealingWarmRestarts":
            progress = epoch - 1 + (batch_idx + 1) / max(1, num_batches)
            self.lr_scheduler.step(progress)
            return

        self.lr_scheduler.step()

    def load_checkpoint(self, model: nn.Module, file_path: str, model_type: str) -> nn.Module:
        """
        Load checkpoint for model from file_path
        args:
            model: model whose weights are loaded from file_path
            file_path: file_path to checkpoint file with only weights
            model_type: type of model to load weights into ["pytorch", "torchscript"]
        """
        if not osp.isfile(file_path):
            msg = f"'{file_path}' is not a torch weight file."
            raise ValueError(msg)

        if model_type == "pytorch":
            state_dict = (
                torch.load(file_path)
                if self.config.device == "cuda"
                else torch.load(file_path, map_location=torch.device("cpu"))
            )
            model.load_state_dict(state_dict)
        elif model_type == "torchscript":
            model = torch.jit.load(file_path, map_location=self.device)
        else:
            raise ValueError(f"model_type must be either 'pytorch' or 'torchscript', got {model_type}")

        self.logger.info(f"Loaded {model_type} checkpoint: {file_path}")
        return model

    def save_checkpoint(self, model: nn.Module, ckpt_save_name: str = "checkpoint.pth") -> None:
        """
        Checkpoint saver
        args:
            ckpt_save_name: checkpoint file name which is saved inside self.config.experiment_models_dir
        """
        # create checkpoint directory if it doesnt exist
        os.makedirs(self.config.experiment_models_dir, exist_ok=True)
        save_path = osp.join(self.config.experiment_models_dir, ckpt_save_name)
        torch.save(model.state_dict(), save_path)

    def train_epoch(self, epoch: int):
        """
        Train the model for one epoch.

        Args:
            epoch: Current epoch number.
        """
        self.model.train()
        if hasattr(self.optimizer, "train"):  # for ScheduleFree optimizer
            self.optimizer.train()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Training]", leave=False)
        loss_meter = AverageMeter()
        all_labels = []
        all_predictions = []

        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass
            temporal_features, instance_features, projected_features, logits = self.model(data)

            # Compute loss
            loss_kwargs = {
                "logits": logits,
                "temporal_features": temporal_features,
                "instance_features": instance_features,
                "projected_features": projected_features,
                "labels": labels,
            }
            loss = self.loss(**loss_kwargs)
            if not torch.isfinite(loss):
                raise ValueError(f"Loss is NaN or Inf at batch {batch_idx} in epoch {epoch}.")

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Step scheduler
            self._step_lr_scheduler(epoch, batch_idx, len(self.train_loader))

            # Update loss meter
            loss_meter.update(loss.item(), data.size(0))

            # Get predictions and ground truth
            _, predicted = torch.max(logits, dim=1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar description
            current_lr = self.optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                Loss=f"{loss.item():.4f}", Mean_Loss=f"{loss_meter.avg:.4f}", LR=f"{current_lr:.6f}"
            )
        # Compute final metrics
        accuracy = accuracy_score(all_labels, all_predictions) * 100
        f1 = f1_score(all_labels, all_predictions, average="weighted") * 100
        # log train_metrics epoch,loss,accuracy,f1,lr to file only
        self.train_metrics_logger.info(f"{epoch},{loss},{accuracy},{f1},{current_lr}", to_console=False)

        metrics = {"loss": loss_meter.avg, "accuracy": accuracy, "f1": f1}
        return metrics

    @torch.no_grad()
    def _extract_train_embeddings(self):
        inst_emb_list, proj_emb_list, labels_list = [], [], []
        for data, labels in self.train_loader_for_val:
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            _, instance_features, projected_features, _ = self.model(data)

            B = labels.shape[0]
            if self.model.fft_encoder is not None:
                # Split raw / FFT halves
                inst_t = instance_features[:B]
                inst_f = instance_features[B:]

                proj_t = projected_features[:B]
                proj_f = projected_features[B:]

                # Concatenate instance features from both branches
                inst_final = torch.cat([inst_t, inst_f], dim=1)

                # Average and normalize projected features from both branches
                proj_final = F.normalize((proj_t + proj_f) / 2, dim=1)
                # proj_final = proj_t
            else:
                inst_final = instance_features
                proj_final = projected_features

            inst_emb_list.append(inst_final)
            proj_emb_list.append(proj_final)
            labels_list.append(labels.cpu())

        return (
            F.normalize(torch.cat(inst_emb_list), p=2, dim=1),
            F.normalize(torch.cat(proj_emb_list), p=2, dim=1),
            torch.cat(labels_list),
        )

    @torch.no_grad()
    def _run_validation_forward(self, loss_meter: AverageMeter, mode: str = "avg_proto"):
        instance_feats, projected_feats, logits_list, labels_list = [], [], [], []

        for data, labels in self.test_loader:
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            temporal_features, instance_features, projected_features, logits = self.model(data)

            B = labels.shape[0]

            if self.model.fft_encoder is not None:
                # Split raw / FFT halves
                inst_t = instance_features[:B]  # first half are features of the temporal raw signal
                inst_f = instance_features[B:]  # second half are features of the FFT augmented signal

                # Same for projected features
                proj_t = projected_features[:B]
                proj_f = projected_features[B:]

                # Features fusion by concatenation (double feature dimension)
                inst_final = torch.cat([inst_t, inst_f], dim=1)

                # Fuse both branches by averaging and normalizing
                proj_final = F.normalize((proj_t + proj_f) / 2, dim=1)
                # proj_final = proj_t

                # Compute logits of the average projections with respect to class prototypes
                logits_final = self.model._compute_logits(proj_final)
            else:
                # No FFT path → use raw embeddings directly
                inst_final = instance_features
                proj_final = projected_features
                logits_final = logits

            # Loss on fused outputs
            loss_kwargs = {
                "logits": logits_final,
                "temporal_features": temporal_features,
                "instance_features": inst_final,
                "projected_features": projected_features,
                "labels": labels,
            }
            loss = self.loss(**loss_kwargs)
            loss_meter.update(loss.item(), data.size(0))

            # Accumulate fused outputs (one embedding per sample)
            instance_feats.append(inst_final)
            projected_feats.append(proj_final)
            logits_list.append(logits_final)
            labels_list.append(labels)

        return (
            F.normalize(torch.cat(instance_feats), p=2, dim=1),
            F.normalize(torch.cat(projected_feats), p=2, dim=1),
            torch.cat(logits_list),
            torch.cat(labels_list),
            loss_meter.avg,
        )

    def _format_metric_value(self, name: str, value) -> str:
        if value is None:
            return "nan"
        if isinstance(value, torch.Tensor):
            value = value.item()
        if isinstance(value, np.generic):
            value = np.asarray(value).item()

        if isinstance(value, int | float):
            if isinstance(value, float) and not math.isfinite(value):
                return "nan"
            precision = self._val_metric_precisions.get(name, DEFAULT_FALLBACK_PRECISION)
            if precision is None:
                return str(value)
            if precision == 0:
                return str(int(round(value)))
            return f"{value:.{precision}f}"

        return str(value)

    def _log_validation(self, metrics: dict[str, float], console_fields: tuple[str, ...] | None = None) -> None:
        """
        Log validation metrics to the validation CSV file and to the console if desired.

        Args:
            metrics (dict[str, float]): Dictionary of validation metrics.
            console_fields (tuple[str, ...] | None): Fields to log to the console. If None, default fields are used.
        """
        missing = [header for header in self.val_csv_headers if header not in metrics]
        if missing:
            raise KeyError(f"Missing validation metrics for headers: {missing}")
        unlogged_metrics = [metric for metric in metrics if metric not in self.val_csv_headers]
        if unlogged_metrics and not self._warned_unlogged_metrics:
            self.logger.warning(f"Unlogged validation metrics: {unlogged_metrics}")
            self._warned_unlogged_metrics = True

        csv_line = ",".join(self._format_metric_value(name, metrics[name]) for name in self.val_csv_headers)
        self.val_metrics_logger.info(csv_line, to_console=False)

        if console_fields is None:
            console_fields = ("epoch", "loss", "accuracy", "accuracy_prot", "accuracy_nn", "accuracy_svm")

        console_parts = []
        for field in console_fields:
            if field in metrics:
                console_parts.append(f"{field}={self._format_metric_value(field, metrics[field])}")

        if console_parts:
            self.logger.info(f"Val: {', '.join(console_parts)}", to_console=False)

    @torch.no_grad()
    def validate_epoch(self, epoch: int, best_metrics: dict) -> dict:
        """
        Validate the model for one epoch and compute additional metrics (F1 score, accuracy).
        Args:
            epoch: Current epoch number.
            best_metrics: Dictionary to store and update the best metrics.
        """
        self.model.eval()
        if hasattr(self.optimizer, "eval"):  # for ScheduleFree optimizer
            self.optimizer.eval()

        progress_bar = tqdm(self.test_loader, desc=f"Epoch {epoch} [Validation]", leave=True)
        loss_meter = AverageMeter()

        compute_nn_metrics = self.config.trainer.args.eval_nearest_neighbor_classification
        compute_proto_metrics = self.config.trainer.args.eval_prototype_classification

        train_instance_embeddings, train_projected_embeddings, train_labels = self._extract_train_embeddings()
        test_instance_embeddings, test_projected_embeddings, test_logits, test_targets, test_loss = (
            self._run_validation_forward(loss_meter)
        )

        acc_svm, f1_svm, _ = eval_linear_svm(
            test_instance_embeddings,
            train_instance_embeddings,
            train_labels,
            test_targets,
        )

        pre_proj_prototypes = compute_prototypes(
            train_instance_embeddings, train_labels, self.train_loader.dataset.num_classes, self.device
        )

        accuracy_nn, f1_nn, preds_nn = float("nan"), float("nan"), torch.empty(0, dtype=torch.long)
        if compute_nn_metrics:
            accuracy_nn, f1_nn, preds_nn = eval_nearest_neighbor(
                test_instance_embeddings, train_instance_embeddings, train_labels, test_targets
            )
        accuracy_prot, f1_prot, preds_prot = (float("nan"), float("nan"), torch.empty(0, dtype=torch.long))
        if compute_proto_metrics:
            accuracy_prot, f1_prot, preds_prot = eval_prototype_classification(
                test_instance_embeddings, test_targets, pre_proj_prototypes
            )
        accuracy_logits, f1_logits, preds_cls = eval_classifier_logits(test_logits, test_targets)

        if self.config.model.args.cosine:
            classifier = self.model.classifier
            classifier_weights = (
                classifier.weight.detach() if isinstance(classifier, nn.Module) else classifier.detach()
            )
            post_proj_prototypes = F.normalize(classifier_weights, p=2, dim=1)  # classifier prototypes
        else:
            post_proj_prototypes = compute_prototypes(
                train_projected_embeddings,
                train_labels,
                self.train_loader.dataset.num_classes,
                self.device,
            )

        pre_proj_compactness = compute_compactness(
            test_instance_embeddings, test_targets, pre_proj_prototypes, self.device
        )
        post_proj_compactness = compute_compactness(
            test_projected_embeddings, test_targets, post_proj_prototypes, self.device
        )
        pre_proj_dispersion = compute_dispersion(pre_proj_prototypes)
        post_proj_dispersion = compute_dispersion(post_proj_prototypes)

        log_metrics = {
            "epoch": epoch,
            "loss": test_loss,
            "accuracy": accuracy_logits,
            "f1": f1_logits,
            "accuracy_nn": accuracy_nn,
            "f1_nn": f1_nn,
            "accuracy_prot": accuracy_prot,
            "f1_prot": f1_prot,
            "accuracy_svm": acc_svm,
            "f1_svm": f1_svm,
            "pre_proj_dispersion": pre_proj_dispersion,
            "post_proj_dispersion": post_proj_dispersion,
            "pre_proj_compactness": pre_proj_compactness,
            "post_proj_compactness": post_proj_compactness,
        }
        self._log_validation(log_metrics)

        best_metrics["loss"] = min(best_metrics["loss"], test_loss)
        best_metrics["accuracy"] = max(best_metrics["accuracy"], accuracy_logits)
        best_metrics["f1"] = max(best_metrics["f1"], f1_logits)

        progress_bar.set_postfix(
            mean_loss=f"{test_loss:.4f}",
            acc_svm=f"{acc_svm:.2f}%",
            acc_logits=f"{accuracy_logits:.2f}%",
            f1_svm=f"{f1_svm:.4f}",
            f1_logits=f"{f1_logits:.4f}",
            f1_nn="nan" if not math.isfinite(f1_nn) else f"{f1_nn:.4f}",
            f1_prot=f"{f1_prot:.4f}",
            pre_disp=f"{pre_proj_dispersion:.4f}",
            post_disp=f"{post_proj_dispersion:.4f}",
            pre_comp=f"{pre_proj_compactness:.4f}",
            post_comp=f"{post_proj_compactness:.4f}",
        )

        metrics = {
            "loss": test_loss,
            "accuracy": accuracy_logits,
            "f1": f1_logits,
            "accuracy_prot": accuracy_prot,
            "f1_prot": f1_prot,
            "accuracy_nn": accuracy_nn,
            "f1_nn": f1_nn,
            "accuracy_svm": acc_svm,
            "f1_svm": f1_svm,
            "pre_proj_dispersion": pre_proj_dispersion,
            "post_proj_dispersion": post_proj_dispersion,
            "pre_proj_compactness": pre_proj_compactness,
            "post_proj_compactness": post_proj_compactness,
            "labels": test_targets.cpu().numpy(),
            "preds": preds_cls.cpu().numpy(),
            "preds_prot": preds_prot.cpu().numpy(),
            "preds_nn": preds_nn.cpu().numpy() if preds_nn.numel() else np.array([]),
        }
        return metrics

    def fit(self):
        """
        Train and validate the model for all epochs specified in args.
        """
        best_metrics = {"loss": float("inf"), "accuracy": 0.0, "f1": 0.0}
        loss = best_loss = float("inf")
        epochs = self.config.trainer.args.epochs
        val_freq = self.config.trainer.args.val_freq
        save_checkpoints = self.config.trainer.args.save_checkpoints
        ckpt_save_freq = self.config.trainer.args.checkpoint_save_freq
        ood_eval_enabled = self.config.ood_eval.enabled
        ood_eval_freq = self.config.ood_eval.eval_freq
        save_best_loss_ckpt = self.config.trainer.args.save_best_loss_checkpoint

        for epoch in range(1, epochs + 1):
            if hasattr(self.loss, "set_epoch_progress"):  # for multi-obj loss with lambda warm-ups
                self.loss.set_epoch_progress(epoch, epochs)
            self.train_epoch(epoch)
            if epoch % val_freq == 0:
                val_metrics = self.validate_epoch(epoch, best_metrics)
                loss = val_metrics["loss"]

            # save per ckpt_save_freq if save_checkpoints is True
            if save_checkpoints and epoch % ckpt_save_freq == 0:
                self.save_checkpoint(self.model, f"epoch_{epoch}.pth")

            # Save epoch with lowest val loss
            if save_checkpoints and save_best_loss_ckpt and loss <= best_loss:
                self.save_checkpoint(self.model, "best_loss.pth")
                best_loss = loss

            # Evaluate OOD detection every ood_eval_freq epochs and last epoch
            if ood_eval_enabled and (epoch % ood_eval_freq == 0 or epoch == epochs):
                self.ood_evaluator.evaluate(self.model, epoch)

        label_remap = self.train_loader.dataset.label_remap

        if ood_eval_enabled and self.ood_evaluator:
            non_normal_class_ids = [
                cls_id
                for cls_name, cls_id in label_remap.items()
                if cls_name in {"vibration", "shake", "vibrate_shake"}
            ]
            # extract train id, test id and OOD-dataset(s) embeddings, logits, and labels
            all_outputs_dict = self.ood_evaluator._extract_all_outputs(self.model)
            # compute OOD scores for test id and OOD-dataset(s)
            ood_scores = self.ood_evaluator._compute_ood_scores(all_outputs_dict)

            # plot t-sne of the train id, test id and OOD embeddings if enabled
            plot_emb_sets = getattr(self.config.ood_eval, "plot_emb_sets", None)
            if self.config.ood_eval.plot_emb_tsne:
                self.ood_evaluator.plot_embeddings_visualization(
                    all_outputs_dict,
                    method="tsne",
                    savedir=self.config.experiment_plot_dir,
                    metric="cosine" if self.config.model.args.cosine else "euclidean",
                    plot_sets=plot_emb_sets,
                )
            # plot umap of the train id, test id and OOD embeddings if enabled
            if self.config.ood_eval.plot_emb_umap:
                self.ood_evaluator.plot_embeddings_visualization(
                    all_outputs_dict,
                    method="umap",
                    savedir=self.config.experiment_plot_dir,
                    metric="cosine" if self.config.model.args.cosine else "euclidean",
                    plot_sets=plot_emb_sets,
                )

            if self.config.trainer.args.plot_metrics:
                # plot ood metrics line-plot with different ood_methods over epochs
                self.ood_evaluator.plot_ood_metrics(
                    exp_ood_metrics_dir=osp.join(self.config.experiment_metrics_dir, "ood_metrics"),
                    savedir=self.config.experiment_plot_dir,
                )

                # plot ood histogram and stacked histogram
                plot_ood_histograms(
                    ood_scores,
                    bins=100,
                    alpha=0.6,
                    figsize=(15, 5),
                    savepath=osp.join(self.config.experiment_plot_dir, "ood_score_histogram.png"),
                )
                plot_ood_stacked(
                    ood_scores,
                    bins=100,
                    figsize=(15, 5),
                    savepath=osp.join(self.config.experiment_plot_dir, "ood_score_norm_stacked.png"),
                )

        if val_metrics and self.config.trainer.args.plot_metrics:
            # plot confusion matrix with label_remap
            plot_confusion_matrix(
                val_metrics["labels"],
                val_metrics["preds"],
                label_remap=label_remap,
                title=f"Epoch:{epochs} Confusion Matrix",
                save_path=osp.join(self.config.experiment_plot_dir, "confusion_matrix.png"),
            )

            # load train and val metrics & plot them
            train_df = pd.read_csv(osp.join(self.config.experiment_metrics_dir, "train_metrics.csv"))
            val_df = pd.read_csv(osp.join(self.config.experiment_metrics_dir, "val_metrics.csv"))
            plot_train_val_metrics(
                train_df,
                val_df,
                save_path=osp.join(self.config.experiment_plot_dir, "train_val_metrics.png"),
            )

        # Save the updated config to the log root directory with the updated label_remap
        self.config.dataset.args.label_remap = {str(k): v for k, v in label_remap.items()}
        OmegaConf.save(self.config, osp.join(osp.dirname(self.config.experiment_log_dir), "config.yaml"))
