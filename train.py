"""
Training script with OmegaConf and YAML support.
Run with: python train.py --cfg YAML_CONFIG_PATH
"""

import argparse
import time

from ood_detection.config_parser import CustomDictConfig
from ood_detection.datasets.loader import get_loaders
from ood_detection.ood_eval.ood_evaluator import OODEvaluator
from ood_detection.trainers import init_trainer
from ood_detection.utils.common import init_environment
from ood_detection.utils.logger import (
    EXP_LOGGER_NAME,
    TRAIN_METRICS_LOGGER_NAME,
    VAL_METRICS_LOGGER_NAME,
    LoggerSingleton,
)


def get_config_from_args() -> CustomDictConfig:
    """Get CustomDictConfig object from argparse."""
    parser = argparse.ArgumentParser(description="PyTorch Training with YAML configs via OmegaConf")
    # primary required cli args
    parser.add_argument(
        "--cfg", "--config", type=str, dest="config", required=True, help="Config file path e.g. configs/ucr.yaml, etc."
    )
    parser.add_argument(
        "-d", "--dataset", type=str, dest="dataset", help="Dataset name based on the loader. eg. ECG200, Epilepsy."
    )
    parser.add_argument(
        "-l", "--loader", type=str, dest="loader", choices=["UCR", "UEA"], help="Dataset loader type."
    )
    parser.add_argument(
        "-n",
        "--experiment_name",
        type=str,
        dest="experiment_name",
        help="Identifier to annotate train experiment checkpoints & logs. (Check yaml for default)",
    )
    parser.add_argument(
        "-o",
        "--override",
        type=str,
        nargs="+",
        dest="override",
        default=None,
        help="Override config params. Must match keys in YAML config. "
        "e.g. -o seed=1 dataset.type=NewDataType model.layers=[64,128,256] model.layers[2]=512 (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Run training in verbose mode (default: %(default)s)",
    )

    # Add additional arguments here
    # WARNING: These will override the config params with the same name in the YAML file.
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        type=str,
        dest="resume_checkpoint",
        help="Path to resume checkpoint. Overrides `resume_checkpoint` in config. (default: None)",
    )
    parser.add_argument("--batch-size", type=int, help="The batch size (e.g. 8)")
    parser.add_argument("--epochs", type=int, help="The number of epochs (e.g. 100)")
    parser.add_argument("--lr", type=float, dest="lr", help="Learning rate for the optimizer (e.g. 0.01)")
    parser.add_argument(
        "--device",
        type=str,
        dest="device",
        choices=["cpu", "cuda"],
        help="Device type (defaults to 'cuda' from yaml config)",
    )
    args = parser.parse_args()

    # To override key-value params from YAML file,
    # match the YAML kv structure for any additional args above
    # keys-val pairs can have nested structure separated by colons
    yaml_modification = {
        "dataset.args.name": args.dataset,
        "dataset.args.loader": args.loader,
        "experiment_name": args.experiment_name,
        "trainer.args.resume_checkpoint": args.resume_checkpoint,
        "dataloader.args.batch_size": args.batch_size,
        "trainer.args.epochs": args.epochs,
        "optimizer.args.lr": args.lr,
        "device": args.device,
    }
    # get custom omegaconf DictConfig-like obj
    cfg = CustomDictConfig.from_args(args, yaml_modification)
    return cfg


def main():
    config = get_config_from_args()

    # Initialized Singleton loggers only ONCE using directories from config
    LoggerSingleton(
        name=EXP_LOGGER_NAME,
        log_dir=config.experiment_log_dir,
        log_file_name="train.txt",
        fmt="%(levelname)s: %(message)s",
    )
    LoggerSingleton(
        name=TRAIN_METRICS_LOGGER_NAME, log_dir=config.experiment_metrics_dir, log_file_name="train_metrics.csv"
    )
    LoggerSingleton(
        name=VAL_METRICS_LOGGER_NAME, log_dir=config.experiment_metrics_dir, log_file_name="val_metrics.csv"
    )

    start = time.perf_counter()
    # Initialize environment (device, seed, reproducibility)
    device, generator = init_environment(
        config.gpu, seed=config.seed, device=config.device, reproducible=config.reproducible
    )

    # Load in-distribution training and testing data loaders
    train_loader, test_loader, transforms, class_weights = get_loaders(config, generator)

    # Initialize OOD Evaluator if enabled
    ood_evaluator = None
    if config.ood_eval.enabled:
        ood_evaluator = OODEvaluator(
            config,
            train_id_loader=train_loader,
            test_id_loader=test_loader,
            transforms=transforms,
        )

    # Initialize the trainer, passing the OOD evaluator (if any)
    trainer = init_trainer(
        config.trainer.type,
        config=config,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        ood_evaluator=ood_evaluator,
    )
    trainer.fit()

    end = time.perf_counter()
    trainer.logger.info(f"Training completed in {end - start:.3f} seconds.")


if __name__ == "__main__":
    main()
