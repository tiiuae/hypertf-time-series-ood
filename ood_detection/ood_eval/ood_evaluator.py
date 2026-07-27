from collections.abc import Callable, Sequence
import os
import os.path as osp

import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ood_detection.datasets.ood_loader import get_far_ood_loaders, get_near_ood_loaders
from ood_detection.metrics.ood_eval import init_metric_function
from ood_detection.metrics.plots import plot_embedding, plot_metrics
from ood_detection.ood_eval.ood_methods import init_ood_eval_method
from ood_detection.utils.logger import EXP_LOGGER_NAME, LoggerSingleton


class OODEvaluator:
    """
    Args:
        config: OmegaConf configuration object.
        train_id_loader: DataLoader for the in-distribution (ID) training dataset (used to build embedding banks).
        test_id_loader: DataLoader for the in-distribution (ID) testing dataset (used for evaluation).
        transforms: Dict of transforms to apply to the data.
    """

    def __init__(
        self,
        config: DictConfig,
        train_id_loader: DataLoader,
        test_id_loader: DataLoader,
        transforms: dict[str, Callable],
    ):
        self.config = config
        self.logger = LoggerSingleton.get_logger(EXP_LOGGER_NAME)
        # For building embedding banks (Create a new dataldr to avoid disturbing the RNG of the orig train loader)
        self.train_id_loader = DataLoader(
            train_id_loader.dataset,
            batch_size=config.dataloader.args.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=False,
            num_workers=4,
        )
        # For evaluating on unseen ID samples
        self.test_id_loader = test_id_loader
        self.transforms = transforms

        # Initialize OOD evaluation methods from config (if any)
        self.ood_eval_methods = []
        self.requires_embeddings = False  # Flag to check if any method needs embedding bank

        if config.ood_eval.methods:
            for ood_eval_method in config.ood_eval.methods:
                ood_eval = init_ood_eval_method(ood_eval_method.type, **ood_eval_method.args)
                self.ood_eval_methods.append(ood_eval)

                # Check if the method requires an embedding bank
                if getattr(ood_eval, "requires_embeddings", False):
                    self.requires_embeddings = True

        # Evaluation metrics (e.g., 'auroc'); default to ['auroc'] if not provided
        self.eval_metrics = config.ood_eval.metrics if hasattr(config.ood_eval, "metrics") else ["auroc"]

        # Load OOD DataLoaders
        self.ood_loaders = self._get_loaders()

    def _get_loaders(self):
        """
        Loads OOD DataLoaders using the helper function.
        """
        ood_type = self.config.ood_eval.data.type
        if ood_type == "far":
            # get ID dataset shape, ID dataset must be of shape [N, features, seq_len]
            _, id_feat_len, id_seq_len = self.train_id_loader.dataset.data.shape
            ood_loaders = get_far_ood_loaders(
                self.config, id_seq_len=id_seq_len, id_feat_len=id_feat_len, transform=self.transforms["ood"]
            )
        elif ood_type == "near":
            ood_loaders = get_near_ood_loaders(self.config, transform=self.transforms["ood"])
        else:
            raise NotImplementedError(f"ood_eval.data.type: '{ood_type}' not implemented. Use 'near' or 'far'")

        return ood_loaders

    @torch.no_grad()
    def _extract_outputs(
        self, model: torch.nn.Module, loader: torch.utils.data.DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract instance embeddings, logits, and labels from a given DataLoader.
        Assumes the model returns (temporal_feats, instance_feats, projected_feats, logits).
        """
        model.eval()
        all_instance_feats = []
        all_logits = []
        all_labels = []
        device = next(model.parameters()).device

        for batch in loader:
            data, labels = batch[0], batch[1]  # handles when loader returns more tuples
            data = data.to(device)

            # Forward pass through the model
            outputs = model(data)

            # Assuming the model returns (temporal_feats, instance_feats, projected_feats, logits)
            if len(outputs) == 4:  # Standard Classifier: (temporal, instance, projected, logits)
                instance_feats, logits = outputs[1], outputs[3]
            elif len(outputs) == 5:  # ContrastiveHyperClassifier: (temporal, instance, proj_sup, proj_ce, logits)
                instance_feats, logits = outputs[1], outputs[4]
            else:
                raise RuntimeError("Unexpected model output structure")

            B = labels.shape[0]

            # Fuse raw and FFT halves if fft_encoder exists
            if hasattr(model, "fft_encoder") and model.fft_encoder is not None:
                # Instance feats shape is (2B, D)
                inst_t = instance_feats[:B]
                inst_f = instance_feats[B:]
                inst_final = torch.cat([inst_t, inst_f], dim=1)

                # Logits shape is (2B, C)
                logits_t = logits[:B]
                logits_f = logits[B:]
                logits_final = (logits_t + logits_f) / 2
            else:
                inst_final = instance_feats
                logits_final = logits

            all_instance_feats.append(inst_final.cpu().numpy().astype("float32"))
            all_logits.append(logits_final.cpu().numpy().astype("float32"))
            all_labels.append(labels.cpu().numpy())

        instance_feats = np.concatenate(all_instance_feats, axis=0)
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return instance_feats, logits, labels

    @torch.no_grad()
    def _extract_all_outputs(self, model: torch.nn.Module) -> dict:
        """
        Extract embeddings, logits, and labels for train ID, test ID, and each OOD dataset.

        Returns:
            A dictionary with keys 'train_id', 'test_id', and 'ood_dataset1', 'ood_dataset2', ...
            Each key contains a dictionary with keys 'embeddings', 'logits', and 'labels'.
            Format:
                {
                    'train_id': {
                        'embeddings': train_id_emb,
                        'logits': train_id_logits,
                        'labels': train_id_labels
                    },
                    'test_id': {
                        'embeddings': test_id_emb,
                        'logits': test_id_logits,
                        'labels': test_id_labels
                    },
                    'ood_dataset1': {
                        'embeddings': ood_dataset1_emb,
                        'logits': ood_dataset1_logits,
                        'labels': ood_dataset1_labels
                    },
                    'ood_dataset2': {
                        'embeddings': ood_dataset2_emb,
                        'logits': ood_dataset2_logits,
                        'labels': ood_dataset2_labels
                    },
                    ...
                }
        """
        outputs = {}

        if self.requires_embeddings:
            # Switch train_id_loader transform to test for extracting bank
            self.train_id_loader.dataset.set_secondary_transform()

            # Process train in-distribution (ID) data for embedding bank
            train_id_emb, train_id_logits, train_id_labels = self._extract_outputs(model, self.train_id_loader)
            outputs["train_id"] = {"embeddings": train_id_emb, "logits": train_id_logits, "labels": train_id_labels}

            # Switch it back to ensure training uses proper transform
            self.train_id_loader.dataset.set_main_transform()

        # Process test in-distribution (ID) data
        test_id_emb, test_id_logits, test_id_labels = self._extract_outputs(model, self.test_id_loader)
        outputs["test_id"] = {"embeddings": test_id_emb, "logits": test_id_logits, "labels": test_id_labels}

        # Process each OOD dataset
        ood_outputs = {}
        for dset_name, loader in tqdm(self.ood_loaders.items(), desc="Extracting OOD outputs from OOD dsets"):
            emb, logits, labels = self._extract_outputs(model, loader)
            ood_outputs[dset_name] = {
                "embeddings": emb,
                "logits": logits,
                "labels": labels,  # prob. dummy labels for OOD
            }

        outputs["ood_datasets"] = ood_outputs
        return outputs

    def _compute_ood_scores(self, outputs: dict) -> dict:
        """
        Compute OOD scores for test ID samples and OOD datasets for each configured evaluation method.

        Args:
            outputs (dict): Output dictionary from _extract_all_outputs containing embeddings and logits.

        Returns:
            ood_score_results (dict): OOD scores for each evaluation method.
                Format:
                {
                    'MethodName': {
                        'test_id': [scores],
                        'ood_dataset1': [scores],
                        'ood_dataset2': [scores],
                        ...
                    }
                }
        """
        ood_score_results = {}

        # Extract relevant embeddings and logits
        train_id_embeddings = outputs.get("train_id", {}).get("embeddings", None)
        train_id_labels = outputs.get("train_id", {}).get("labels", None)
        test_id_embeddings = outputs["test_id"]["embeddings"]
        test_id_logits = outputs["test_id"]["logits"]
        ood_datasets = outputs["ood_datasets"]

        # Per OOD eval method (e.g., MSP, Energy, CosineNN)
        for ood_method in tqdm(self.ood_eval_methods, desc="Computing OOD scores for OOD methods"):
            ood_method_scores = {}

            # Score Test ID Samples
            if ood_method.requires_embeddings:
                # For methods like CosineNN that need the embedding bank
                test_id_scores = ood_method.compute_score(
                    bank=train_id_embeddings, labels=train_id_labels, embeddings=test_id_embeddings
                )
            else:
                # For methods like MSP that use logits
                test_id_scores = ood_method.compute_score(test_id_logits)

            # Store Test ID scores
            ood_method_scores["test_id"] = test_id_scores

            # Score OOD Datasets
            for ood_dset_name, ood_data in ood_datasets.items():
                ood_embeddings = ood_data["embeddings"]
                ood_logits = ood_data["logits"]

                if ood_method.requires_embeddings:
                    ood_scores = ood_method.compute_score(
                        bank=train_id_embeddings, labels=train_id_labels, embeddings=ood_embeddings
                    )
                else:
                    ood_scores = ood_method.compute_score(ood_logits)

                # Store OOD scores
                ood_method_scores[ood_dset_name] = ood_scores

            #  Store All Scores for Current Method
            ood_score_results[str(ood_method)] = ood_method_scores

        return ood_score_results

    def _compute_metrics(self, ood_results) -> dict:
        """
        Compute evaluation metrics for OOD detection.

        Args:
            ood_results (dict): OOD scores for each evaluation method.
                Format:
                {
                    'MethodName': {
                        'test_id': [scores],
                        'ood_dataset1': [scores],
                        'ood_dataset2': [scores],
                        ...
                    }
                }

        Returns:
            metric_results (dict): Evaluation metrics for each method and dataset.
                Format:
                {
                    'MethodName': {
                        'metric_name': {
                            'ood_dataset1': value,
                            'ood_dataset2': value,
                            'All_Combined': value,
                            'All_Averaged': value
                        }
                    }
                }
        """
        metric_results = {}

        for method, scores in tqdm(ood_results.items(), desc="Computing eval metrics for OOD methods"):
            metric_results[method] = {}

            for metric in self.eval_metrics:
                metric_func = init_metric_function(metric)
                if not metric_func:
                    print(f"Unknown evaluation metric: {metric}")
                    continue

                metric_results[method][metric] = {}

                # Extract Test ID scores
                id_scores = scores["test_id"]

                # Prepare to combine all OOD scores
                combined_ood_scores = []

                # Compute metric for each OOD dataset
                for ood_dset_name, ood_scores in scores.items():
                    if ood_dset_name == "test_id":
                        continue  # Skip test ID

                    # Compute the metric (e.g., AUROC) for ID vs. current OOD dataset
                    metric_value = metric_func(id_scores, ood_scores) * 100
                    metric_results[method][metric][ood_dset_name] = metric_value

                    # Accumulate OOD scores for combined evaluation
                    combined_ood_scores.extend(ood_scores)

                # Compute metric for ID vs. all OOD combined
                combined_metric = metric_func(id_scores, np.array(combined_ood_scores)) * 100
                metric_results[method][metric]["All_Combined"] = combined_metric

                # Compute metric for ID vs. OOD averaged over all datasets
                ood_metric_values = [
                    v for k, v in metric_results[method][metric].items() if k not in ["All_Combined", "All_Averaged"]
                ]
                average_metric = sum(ood_metric_values) / len(ood_metric_values)
                metric_results[method][metric]["All_Averaged"] = average_metric

        return metric_results

    def _log_results(self, epoch: int, ood_metrics: dict) -> None:
        """
        Log and store Out-of-Distribution (OOD) evaluation results.

        Args:
            epoch (int): Current epoch number during evaluation.
            ood_metrics (dict): Dictionary containing OOD evaluation metrics.
                Format:
                {
                    'MethodName': {
                        'MetricName': {
                            'DatasetName1': metric_value,
                            'DatasetName2': metric_value,
                            'All_Combined': combined_metric_value,
                            'All_Averaged': averaged_metric_value
                        },
                        ...
                    },
                    ...
                }
                Example:
                {
                    'MSP()': {'auroc': {'SVHN': 68.13,'LSUN': 81.64,'All_Averaged': 75.20}},
                    'CosineNN(k=5)': {'auroc': {'SVHN': 70.45,'LSUN': 84.12,'All_Averaged': 77.30}}
                }
        """
        # Log only 'All_Averaged' OOD results to the console
        summary_metrics = {}
        for ood_method, metric_details in ood_metrics.items():
            summary_metrics[ood_method] = {}
            for metric_name, dataset_scores in metric_details.items():
                if "All_Averaged" in dataset_scores:
                    summary_metrics[ood_method][metric_name] = round(dataset_scores["All_Averaged"], 4)

        # Print the summarized 'All_Averaged' metrics
        self.logger.info(f"Val: Averaged OOD results over all OOD dsets (Epoch {epoch}):")
        for method, metric_dict in summary_metrics.items():
            metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in metric_dict.items())
            self.logger.info(f"  {method} - {metric_str}")

        # Prepare directory to save OOD metrics
        exp_ood_metrics_dir = osp.join(self.config.experiment_metrics_dir, "ood_metrics")
        os.makedirs(exp_ood_metrics_dir, exist_ok=True)

        # Iterate over each OOD method and log results
        for ood_method, metric_details in ood_metrics.items():
            # Extract method name for the filename
            ood_method_name = ood_method.lower().split("(")[0]
            metric_save_path = osp.join(exp_ood_metrics_dir, f"{ood_method_name}.csv")

            # Collect unique OOD datasets across all metrics
            ood_dataset_names = set.union(*[set(metric_scores.keys()) for metric_scores in metric_details.values()])

            # Prepare data rows for logging
            data = []
            for ood_dset in ood_dataset_names:
                row = {"epoch": epoch, "ood_dataset": ood_dset}
                for metric_name, metric_scores in metric_details.items():
                    row[metric_name] = metric_scores.get(ood_dset, None)
                data.append(row)

            # Convert to DataFrame and append to CSV
            df = pd.DataFrame(data)
            df.sort_values(by="ood_dataset", ascending=True, inplace=True)
            df.to_csv(
                metric_save_path, index=False, encoding="utf-8", mode="a", header=not osp.exists(metric_save_path)
            )

    def plot_ood_metrics(self, exp_ood_metrics_dir: str, savedir: str):
        """
        Load OOD metrics from CSV files and plot AUROC scores over epochs.

        Args:
            exp_ood_metrics_dir (str): Dir containing metric csv files with ood_method names
            savepath (str): Path to save ood_metrics graph
        """
        ood_metrics_paths = []  # store paths to ood_metric files
        for ood_eval_method in self.config.ood_eval.methods:
            ood_method_name = ood_eval_method.type.lower().split("(")[0]
            metric_save_path = osp.join(exp_ood_metrics_dir, f"{ood_method_name}.csv")

            if osp.exists(metric_save_path):
                ood_metrics_paths.append(metric_save_path)

        ood_data = {}  # dict to store data from each method
        for path in ood_metrics_paths:
            if not osp.exists(path):
                print(f"Warning: {path} does not exist. Skipping.")
                continue

            # Extract method name from filename
            method_name = osp.splitext(osp.basename(path))[0]
            df = pd.read_csv(path)
            ood_data[method_name] = df

        for metric in self.config.ood_eval.metrics:
            # plot metrics for all ood methods in the same graph
            plot_metrics(
                ood_data,
                x_col="epoch",
                y_col=metric,
                ylabel=metric,
                title=f"OOD method {metric} over epochs ",
                savepath=osp.join(savedir, f"ood_{metric}.png"),
            )

    def plot_embeddings_visualization(
        self,
        all_outputs_dict: dict,
        method: str,
        savedir: str,
        metric: str,
        plot_sets: Sequence[str] | None = None,
    ) -> None:
        """
        Generate t-SNE/UMAP visualizations for train ID, test ID, and OOD embeddings.

        Args:
            all_outputs_dict (dict): Dict with embeddings, logits, and labels for train_id, test_id, and OOD-dataset(s)
            method: The method to use for embedding visualization. (eg. tsne/umap)
            savedir: Directory to save the generated visualizations.
            metric: Metric to use for embedding visualization. (eg. euclidean/cosine)
            plot_sets: Optional iterable describing which subsets to plot. Valid entries are
                {'train_id', 'test_id', 'train_id_and_ood', 'test_id_and_ood'}.
                Defaults to ('train_id', 'train_id_and_ood', 'test_id_and_ood') to preserve previous behaviour.
        """
        if not self.requires_embeddings:
            self.logger.warning(f"No ID embeddings present. Skipping embedding {method} plotting.")
            return
        assert method in ["tsne", "umap"], "Method must be either 'tsne' or 'umap'"
        assert metric in ["euclidean", "cosine"], "Metric must be either 'euclidean' or 'cosine'"
        self.logger.info(f"Extracting embeddings for {method} plots.")

        valid_sets = {
            "train_id": "Train ID embeddings",
            "test_id": "Test ID embeddings",
            "train_id_and_ood": "Train ID embeddings combined with OOD",
            "test_id_and_ood": "Test ID embeddings combined with OOD",
        }
        selected_sets = (
            tuple(plot_sets)
            if plot_sets is not None
            else (
                "train_id",
                "train_id_and_ood",
                "test_id_and_ood",
            )
        )
        unknown_sets = set(selected_sets).difference(valid_sets)
        if unknown_sets:
            raise ValueError(f"Unknown plotting sets requested: {unknown_sets}. Valid options: {tuple(valid_sets)}")

        needs_train = any(k in selected_sets for k in ("train_id", "train_id_and_ood"))
        needs_test = any(k in selected_sets for k in ("test_id", "test_id_and_ood"))
        needs_ood = any(k in selected_sets for k in ("train_id_and_ood", "test_id_and_ood"))

        train_embs = train_labels = train_label_remap = None
        if needs_train:
            train_embs = all_outputs_dict["train_id"]["embeddings"]
            train_labels = all_outputs_dict["train_id"]["labels"]
            train_label_remap = self.train_id_loader.dataset.label_remap

        test_embs = test_labels = test_label_remap = None
        if needs_test:
            test_embs = all_outputs_dict["test_id"]["embeddings"]
            test_labels = all_outputs_dict["test_id"]["labels"]
            test_label_remap = self.test_id_loader.dataset.label_remap

        ood_embs = ood_labels = ood_label_remap = None
        if needs_ood:
            ood_emb_list = []
            ood_labels_list = []
            for _, ood_data in all_outputs_dict["ood_datasets"].items():
                ood_emb_list.append(ood_data["embeddings"])
                # Convert OOD labels from [0, 1, 2, ...] to [-1, -2, -3, ...]
                ood_labels_list.append(-(ood_data["labels"] + 1))

            ood_embs = np.vstack(ood_emb_list)
            ood_labels = np.concatenate(ood_labels_list)
            ood_label_remap = {
                k: -(v + 1) for loader in self.ood_loaders.values() for k, v in loader.dataset.label_remap.items()
            }

        if "train_id" in selected_sets and train_embs is not None:
            plot_embedding(
                train_embs,
                train_labels,
                method,
                train_label_remap,
                title=f"{method} Plot: Train ID Embeddings",
                metric=metric,
                savepath=osp.join(savedir, f"{method}_train_id_emb.png"),
            )

        if "test_id" in selected_sets and test_embs is not None:
            plot_embedding(
                test_embs,
                test_labels,
                method,
                test_label_remap,
                title=f"{method} Plot: Test ID Embeddings",
                metric=metric,
                savepath=osp.join(savedir, f"{method}_test_id_emb.png"),
            )

        if "train_id_and_ood" in selected_sets and train_embs is not None and needs_ood:
            plot_embedding(
                np.vstack((train_embs, ood_embs)),
                np.concatenate((train_labels, ood_labels)),
                method,
                {**train_label_remap, **ood_label_remap},
                title=f"{method} Plot: Train ID Classes + OOD Embeddings",
                metric=metric,
                savepath=osp.join(savedir, f"{method}_train_id_and_ood_embs.png"),
            )

        if "test_id_and_ood" in selected_sets and test_embs is not None and needs_ood:
            plot_embedding(
                np.vstack((test_embs, ood_embs)),
                np.concatenate((test_labels, ood_labels)),
                method,
                {**test_label_remap, **ood_label_remap},
                title=f"{method} Plot: Test ID Classes + OOD Embeddings",
                metric=metric,
                savepath=osp.join(savedir, f"{method}_test_id_and_ood_emb.png"),
            )

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, epoch: int) -> None:
        """
        Run the full OOD evaluation pipeline.
        """
        outputs = self._extract_all_outputs(model)
        ood_scores = self._compute_ood_scores(outputs)
        ood_metrics = self._compute_metrics(ood_scores)
        self._log_results(epoch, ood_metrics)
