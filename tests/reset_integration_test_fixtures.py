"""
Overwrite existing integration test fixture metrics directories to align with the current codebase.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

from ood_detection.utils.common import get_latest_folder

TMP_PATH = "/tmp/pytest"


def run_train_and_generate_outputs(config_path: Path, dataset_name: str, use_cuda: bool = True):
    # Auto-select device
    cuda_visible_devices = "0" if torch.cuda.is_available() and use_cuda else ""
    print("Using", "GPU: CUDA DEVICE 0" if cuda_visible_devices else "CPU")

    # Setup paths
    save_dir = Path(TMP_PATH)
    exp_name = "pytest_integration_test_fixture_reset"
    exp_root = save_dir / exp_name / dataset_name

    if exp_root.exists():
        shutil.rmtree(exp_root)

    # Run training
    cmd = [
        "python",
        "train.py",
        "--cfg",
        str(config_path),
        "-n",
        exp_name,
        "-o",
        "device=cpu" if cuda_visible_devices == "" else "device=cuda",
        "trainer.args.save_checkpoints=False",
        "trainer.args.plot_metrics=False",
        f"save_dir={save_dir}",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    subprocess.check_call(cmd, env=env)

    # Locate experiment folder
    latest_exp_dir = get_latest_folder(exp_root)
    latest_metrics_dir = latest_exp_dir / "metrics"

    shutil.copytree(
        latest_metrics_dir, config_path.parent / ("metrics_gpu" if use_cuda else "metrics_cpu"), dirs_exist_ok=True
    )


def gen_and_replace_metrics(use_cuda: bool):
    cfg_dset_fix_tuple = [
        (
            Path(__file__).parent / "fixtures" / "UEA_auxiliary_contrastive_RacketSports" / "uea_aux_contrastive.yaml",
            "UEA/RacketSports",
        ),
        (
            Path(__file__).parent / "fixtures" / "UEA_euclidean_Epilepsy" / "uea_euclidean.yaml",
            "UEA/Epilepsy",
        ),
        (
            Path(__file__).parent / "fixtures" / "UEA_supcon_euclidean_Epilepsy" / "uea_supcon_euc.yaml",
            "UEA/Epilepsy",
        ),
        (
            Path(__file__).parent / "fixtures" / "UEA_supcon_hyper_RacketSports" / "uea_supcon_hyper.yaml",
            "UEA/RacketSports",
        ),
    ]
    for i, cfg_dset_fix in enumerate(cfg_dset_fix_tuple):
        config_path, dataset_name = cfg_dset_fix
        # gen gpu metrics
        run_train_and_generate_outputs(
            config_path=config_path,
            dataset_name=dataset_name,
            use_cuda=use_cuda,
        )
        print(
            f"########## {i + 1} out of {len(cfg_dset_fix_tuple)} fixtures reset successfully for cuda=={use_cuda} mode ##########"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Reset PyTest Integration Test Fixtures (generate metrics for CPU/GPU)"
    )
    parser.add_argument(
        "devices",
        nargs="+",
        choices=["cpu", "gpu"],
        help=("Specify one or more devices to generate metrics. "
             "Options: cpu, gpu. Example: python reset_integration_test_fixtures.py.py cpu gpu"),
    )
    args = parser.parse_args()

    for device in args.devices:
        use_cuda = device == "gpu"
        print(f"Generating metrics for: {device.upper()}")
        t0 = time.perf_counter()
        gen_and_replace_metrics(use_cuda)
        t1 = time.perf_counter()
        print(f"Total time taken: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
