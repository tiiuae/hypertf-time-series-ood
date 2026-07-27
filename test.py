import argparse
import os
import time

from ood_detection.config_parser import CustomDictConfig
from ood_detection.datasets.loader import get_loaders
from ood_detection.ood_eval.ood_evaluator import OODEvaluator
from ood_detection.trainers import init_trainer
from ood_detection.utils.common import TermColors, init_environment
from ood_detection.utils.logger import (
    EXP_LOGGER_NAME,
    TRAIN_METRICS_LOGGER_NAME,
    VAL_METRICS_LOGGER_NAME,
    LoggerSingleton,
)
from ood_detection.utils.visualization import plot_ood_histograms, plot_ood_stacked


def get_config_from_args():
    """Parse command-line arguments and load configuration."""
    parser = argparse.ArgumentParser(description="Test a trained model on a dataset")
    parser.add_argument(
        "--cfg",
        "--config",
        type=str,
        dest="config",
        required=True,
        help="Config file path for the trained model, e.g., experiments/trial.yaml",
    )
    parser.add_argument(
        "-n",
        "--experiment_name",
        type=str,
        dest="experiment_name",
        help="Identifier for the experiment (optional, check YAML for default)",
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
        "-v", "--verbose", action="store_true", dest="verbose", default=False, help="Run testing in verbose mode"
    )
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        type=str,
        dest="resume_checkpoint",
        required=True,
        help="Path to the trained model checkpoint to be tested",
    )
    parser.add_argument(
        "--mt",
        "--model_type",
        type=str,
        dest="model_type",
        default="pytorch",
        choices=["pytorch", "torchscript"],
        help="Model type to use for testing. Default is pytorch",
    )
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="Device to use for testing")
    args = parser.parse_args()

    # To override key-value params from YAML file,
    # match the YAML kv structure for any additional args above
    # keys-val pairs can have nested structure separated by colons
    yaml_modification = {
        "trainer.args.resume_checkpoint": args.resume_checkpoint,
        "device": args.device,
    }

    # Load configuration
    cfg = CustomDictConfig.from_args(args, yaml_modification)
    return cfg


def main():
    """Main function to test the model."""
    config = get_config_from_args()

    # Initialized Singleton loggers only ONCE using directories from config
    LoggerSingleton(
        name=EXP_LOGGER_NAME,
        log_dir=config.experiment_log_dir,
        log_file_name="test.txt",
        fmt="%(levelname)s: %(message)s",
    )
    LoggerSingleton(
        name=TRAIN_METRICS_LOGGER_NAME, log_dir=config.experiment_metrics_dir, log_file_name="train_metrics.csv"
    )
    LoggerSingleton(
        name=VAL_METRICS_LOGGER_NAME, log_dir=config.experiment_metrics_dir, log_file_name="val_metrics.csv"
    )
    logger = LoggerSingleton.get_logger(EXP_LOGGER_NAME)

    start = time.perf_counter()
    # Initialize environment (device, seed, reproducibility)
    device, generator = init_environment(
        config.gpu, seed=config.seed, device=config.device, reproducible=config.reproducible
    )

    # Load in-distribution training and testing data
    train_loader, test_loader, transforms, class_weights = get_loaders(config, generator=generator)

    # Initialize OOD Evaluator if enabled
    ood_evaluator = None
    if config.ood_eval.enabled:
        ood_evaluator = OODEvaluator(
            config, train_id_loader=train_loader, test_id_loader=test_loader, transforms=transforms
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

    # Run model evaluation
    trainer.validate_epoch(epoch=0, best_metrics={"loss": float("inf"), "accuracy": 0.0, "f1": 0.0})

    # Evaluate OOD detection if enabled
    if config.ood_eval.enabled and ood_evaluator:
        logger.info("Running OOD evaluation...")
        logger.warning(
            f"{TermColors.WARN}OOD eval results may not exactly match those achieved when running train.py, "
            f"if shuffle and drop_last are set to True in dataloaders.{TermColors.ENDC}"
        )
        ood_evaluator.evaluate(trainer.model, epoch=0)

    all_outputs_dict, ood_scores = None, None

    plot_emb_sets = getattr(config.ood_eval, "plot_emb_sets", ["train_id", "test_id"])
    if config.ood_eval.plot_emb_tsne:
        if all_outputs_dict is None:
            all_outputs_dict = trainer.ood_evaluator._extract_all_outputs(trainer.model)
        trainer.ood_evaluator.plot_embeddings_visualization(
            all_outputs_dict,
            method="tsne",
            savedir=trainer.config.experiment_plot_dir,
            metric="cosine" if trainer.config.model.args.cosine else "euclidean",
            plot_sets=plot_emb_sets,
        )

    if config.ood_eval.plot_emb_umap:
        if all_outputs_dict is None:
            all_outputs_dict = trainer.ood_evaluator._extract_all_outputs(trainer.model)
        trainer.ood_evaluator.plot_embeddings_visualization(
            all_outputs_dict,
            method="umap",
            savedir=trainer.config.experiment_plot_dir,
            metric="cosine" if trainer.config.model.args.cosine else "euclidean",
            plot_sets=plot_emb_sets,
        )

    if config.trainer.args.plot_metrics:
        # plot ood metrics line-plot with different ood_methods over epochs
        trainer.ood_evaluator.plot_ood_metrics(
            exp_ood_metrics_dir=os.path.join(trainer.config.experiment_metrics_dir, "ood_metrics"),
            savedir=trainer.config.experiment_plot_dir,
        )

        all_outputs_dict = trainer.ood_evaluator._extract_all_outputs(trainer.model) if all_outputs_dict is None else all_outputs_dict
        ood_scores = trainer.ood_evaluator._compute_ood_scores(all_outputs_dict) if ood_scores is None else ood_scores

        # plot ood histogram and stacked histogram
        plot_ood_histograms(
            ood_scores,
            bins=100,
            alpha=0.6,
            figsize=(15, 5),
            savepath=os.path.join(trainer.config.experiment_plot_dir, "ood_score_histogram.png"),
        )
        plot_ood_stacked(
            ood_scores,
            bins=100,
            figsize=(15, 5),
            savepath=os.path.join(trainer.config.experiment_plot_dir, "ood_score_norm_stacked.png"),
        )

    end = time.perf_counter()
    logger.info(f"Testing completed in {end - start:.3f} seconds.")


if __name__ == "__main__":
    main()
