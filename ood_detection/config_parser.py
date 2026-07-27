"""
CLI config parsing module with OmegaConf and YAML support
Refactored for less duplication and clearer validation flow.
"""

import argparse
from datetime import datetime
import os
import os.path as osp
from typing import Any

from dotenv import load_dotenv
from omegaconf import DictConfig, ListConfig, OmegaConf
import yaml

from .utils.common import get_git_revision_hash

load_dotenv()  # looks for .env in the current working directory


def _validate_cosine_losses(config: DictConfig) -> None:
    is_cosine = config.model.args.cosine
    loss_cfg = config.loss
    loss_type = loss_cfg.type
    loss_args = loss_cfg.get("args", {})

    # losses that only make sense when cosine=True
    cosine_only = {"CompactnessLoss", "SeparationLoss", "RescaledTemperatureCELoss"}

    # CELoss temperature rules when cosine=False
    def _check_temp_ok(args: dict) -> None:
        if args.get("temperature", 1.0) != 1.0:
            raise ValueError("When model.args.cosine is False, CE-like temperatures must be 1.0.")

    if loss_type == "MultiObjectiveLoss":
        loss_list = loss_args.get("losses", [])
        lambdas = loss_args.get("lambdas", [])
        lambda_warmup_epochs = loss_args.get("lambda_warmup_epochs", 0)
        if not isinstance(loss_list, list | ListConfig):
            raise ValueError("'loss.args.losses' must be a list.")
        if not isinstance(lambdas, list | ListConfig):
            raise ValueError("'loss.args.lambdas' must be a list.")
        if not isinstance(lambda_warmup_epochs, list | ListConfig):
            raise ValueError("'loss.args.lambda_warmup_epochs' must be a list.")
        if not all(isinstance(epoch, int) for epoch in lambda_warmup_epochs):
            raise ValueError("'loss.args.lambda_warmup_epochs' must be a list of ints.")
        if (len(loss_list) == len(lambdas) == len(lambda_warmup_epochs)) is False:
            raise ValueError("losses, lambdas, and lambda_warmup_epochs must have same length.")
        if sum(lambdas) < 0:
            raise ValueError("Sum of loss lambdas must be >= 0.")
        for i, loss_item in enumerate(loss_list):
            lt = loss_item.type
            la = loss_item.get("args", {})

            if lambdas[i] < 0:
                raise ValueError(f"Lambda for loss '{lt}' must be >= 0.")

            if not is_cosine:
                if lt in cosine_only:
                    raise ValueError(f"model.args.cosine=False is incompatible with {lt}.")
                if lt in {"TemperatureScaledCELoss", "RescaledTemperatureCELoss"}:
                    _check_temp_ok(la)

    else:
        if not is_cosine:
            if loss_type in cosine_only:
                raise ValueError(f"model.args.cosine=False is incompatible with {loss_type}.")
            if loss_type in {"TemperatureScaledCELoss", "RescaledTemperatureCELoss"}:
                _check_temp_ok(loss_args)


def _validate_dataset_id_remap(config: DictConfig) -> None:
    ds_args = config.dataset.args
    if "id_classes" in ds_args and "id_classes_remap" in ds_args:
        if len(ds_args.id_classes) != len(ds_args.id_classes_remap):
            raise ValueError("'dataset.args.id_classes' length must match 'dataset.args.id_classes_remap' length.")


def _validate_contrastive_trainer(config: DictConfig) -> None:
    def _validate_cosine_and_ce_loss_type(cosine: bool, loss_type: str) -> bool:
        if cosine and loss_type == "CrossEntropyLoss":
            raise ValueError(
                "ContrastiveTrainer + cosine=True should use RescaledTemperatureCELoss, not CrossEntropyLoss."
            )
        if not cosine and loss_type == "RescaledTemperatureCELoss":
            raise ValueError(
                "ContrastiveTrainer + cosine=False should use CrossEntropyLoss, not RescaledTemperatureCELoss."
            )

    if config.trainer.type != "ContrastiveTrainer":
        return

    if config.dataset.type != "ContrastiveTimeSeriesDataset":
        raise ValueError(
            f"ContrastiveTrainer requires dataset.type=ContrastiveTimeSeriesDataset. Found {config.dataset.type}."
        )

    loss_type = config.loss.type
    if loss_type == "MultiObjectiveLoss":
        for loss in config.loss.args.losses:
            _validate_cosine_and_ce_loss_type(config.model.args.cosine, loss.type)
    else:  # single loss
        _validate_cosine_and_ce_loss_type(config.model.args.cosine, loss_type)


def _validate_aux_trainer_and_dataset(config: DictConfig) -> None:
    if config.trainer.type not in {
        "AngularAuxiliaryContrastiveTrainer",
    }:
        return

    if config.dataset.type != "AuxiliaryOutlierExposureTimeSeriesDataset":
        raise ValueError(
            f"{config.trainer.type} requires dataset.type=AuxiliaryOutlierExposureTimeSeriesDataset. "
            f"Found {config.dataset.type}."
        )

    if config.ood_eval.enabled:
        same_sub = config.dataset.args.same_outlier_exposure_sub_type
        far_sub = config.ood_eval.data.far_ood_subtype
        if same_sub and far_sub not in {"diff", "all"}:
            raise ValueError(
                "ood_eval.data.far_ood_subtype must be 'diff' or 'all' when same_outlier_exposure_sub_type=True."
            )
        if not same_sub and far_sub not in {"same", "all"}:
            raise ValueError(
                "ood_eval.data.far_ood_subtype must be 'same' or 'all' when same_outlier_exposure_sub_type=False."
            )


def _validate_oe_based_losses(config: DictConfig) -> None:
    def _validate_oe_losses(loss_type: str, trainer: str, dataset: str):
        if loss_type not in {"AuxiliaryContrastiveLoss"}:
            return
        assert trainer in {"AngularAuxiliaryContrastiveTrainer"}
        assert dataset in {"AuxiliaryOutlierExposureTimeSeriesDataset"}

    loss_type = config.loss.type
    if loss_type == "MultiObjectiveLoss":
        for loss in config.loss.args.losses:
            _validate_oe_losses(loss.type, config.trainer.type, config.dataset.type)
    else:  # single loss
        _validate_oe_losses(loss_type, config.trainer.type, config.dataset.type)


def _validate_ood_eval(config: DictConfig) -> None:
    if not config.ood_eval.enabled:
        return

    data_cfg = config.ood_eval.data
    if data_cfg.type == "far" and data_cfg.far_ood_subtype not in {"same", "diff", "all"}:
        raise ValueError("ood_eval.data.far_ood_subtype must be one of 'same', 'diff', or 'all'.")


def _validate_augmentations(config: DictConfig) -> None:
    required = {"Standardize", "TsToTensor"}
    for split in ("train", "test"):
        aug_list = config.dataset.augmentations.get(split, [])
        aug_types = {aug["type"] for aug in aug_list}
        missing = required - aug_types
        if missing:
            raise ValueError(f"Missing required augmentations {missing} in '{split}' pipeline.")


def validate_config(config: DictConfig) -> None:
    """
    Top-level config validation entrypoint.
    Keeps individual rules small and targeted.
    """
    _validate_cosine_losses(config)
    _validate_dataset_id_remap(config)
    _validate_contrastive_trainer(config)
    _validate_aux_trainer_and_dataset(config)
    _validate_ood_eval(config)
    _validate_augmentations(config)
    _validate_oe_based_losses(config)


# ---------------------------
# main config wrapper
# ---------------------------


def parse_omegaconf_primitive(val_str: str) -> Any:
    """
    Parse CLI override value into the right Python type using YAML + OmegaConf.
    Examples:
      "3" -> 3
      "3.14" -> 3.14
      "true" -> True
      "[1,2]" -> [1, 2]
      "${oc.env:HOME}" -> resolved env
    """
    wrapped = OmegaConf.create({"_val_": yaml.safe_load(val_str)})
    return OmegaConf.to_container(wrapped, resolve=True)["_val_"]


class CustomDictConfig(DictConfig):
    """
    A wrapper around OmegaConf's DictConfig to extend its functionality.
    Handles additional tasks like setting up directories, logging, and
    applying runtime modifications.
    Args:
        config: DictConfig object with configurations.
        verbose: Verbosity flag for training.
        modification: Additional key-value pairs to override in config.
    """

    def __init__(self, config: DictConfig, verbose: bool = False, modification: dict | None = None):
        super().__init__(config)

        # Apply any modifications to the self configuration
        if modification:
            # Removes keys that have None as values
            modification = {k: v for k, v in modification.items() if v is not None}
            for k, v in modification.items():
                OmegaConf.update(self, k, v, merge=True)
        # any cfgs should be received from self not config now
        validate_config(self)  # raises error if config is invalid

        self.verbose = verbose
        self.git_hash = get_git_revision_hash()

        # Set directories for saving logs, metrics, and models
        trial_id = datetime.now().strftime(r"%Y%m%d_%H%M%S_%f")
        save_root = osp.join(
            self.save_dir, self.experiment_name, self.dataset.args.loader, self.dataset.args.name, "trial_" + trial_id
        )
        _log_dir = osp.join(save_root, "logs")
        _metrics_dir = osp.join(save_root, "metrics")
        _plot_dir = osp.join(save_root, "plots")
        _models_dir = osp.join(save_root, "models")

        # Create necessary directories
        os.makedirs(_log_dir, exist_ok=True)
        os.makedirs(_metrics_dir, exist_ok=True)
        os.makedirs(_plot_dir, exist_ok=True)
        os.makedirs(_models_dir, exist_ok=True)

        # Save the updated config to the save_root directory
        OmegaConf.save(self, osp.join(save_root, "config.yaml"))
        # Assign updated log, metrics, and models dir after saving config
        self.experiment_log_dir = _log_dir
        self.experiment_metrics_dir = _metrics_dir
        self.experiment_plot_dir = _plot_dir
        self.experiment_models_dir = _models_dir

    @classmethod
    def from_args(cls, args: argparse.Namespace, modification: dict | None = None, add_all_args: bool = True):
        """
        Initialize this class from CLI arguments. Used in train, test.
        Args:
            args: Parsed CLI arguments.
            modification: Key-value pair to override in config.
                          Can have nested structure separated by periods(.)
                          e.g. {"key1":"val1", "key2.sub_key2":"val2"}
            add_all_args: Add all args to modification
                          that are not alr present as top-level keys.
        """
        modification = modification if modification else {}
        # Add all args to modification from args
        if add_all_args:
            # only check top-level keys
            mod_keys = {k.rsplit(".")[0] for k in modification}
            for arg, value in vars(args).items():
                # add new keys not present in orig yaml config
                if arg not in mod_keys and arg not in {"override"}:
                    modification[arg] = value

        # Load configuration from YAML
        config = OmegaConf.load(args.config)
        # Apply dotlist overrides (-o)
        if args.override:
            OmegaConf.set_struct(config, True)  # Enable strict mode to disallow unknown keys
            for override in args.override:
                if "=" not in override:
                    raise ValueError(f"Invalid override format: {override}. Expected format: key=value")
                key, val_str = override.split("=", 1)
                OmegaConf.update(config, key, parse_omegaconf_primitive(val_str))
            OmegaConf.set_struct(config, False)  # Disable strict mode to allow runtime modifications later

        return cls(config, args.verbose, modification)

    def __str__(self):
        return OmegaConf.to_yaml(self)
