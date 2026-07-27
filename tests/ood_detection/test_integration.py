import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

import pandas as pd
import pytest
import torch

from ood_detection.utils.common import get_latest_folder

required_version = (3, 11, 9)
current_version = sys.version_info[:3]


if current_version != required_version:
    warnings.warn(
        f"Python version mismatch for PyTest, Integration tests may fail: running {current_version}, expected {required_version}",
        UserWarning,
        stacklevel=2,
    )


def assert_csv_almost_equal(actual: Path, expected: Path, atol=1e-3):
    assert actual.exists(), f"Missing test file: {actual}"
    assert expected.exists(), f"Missing reference file: {expected}"

    actual_df = pd.read_csv(actual)
    expected_df = pd.read_csv(expected)

    pd.testing.assert_frame_equal(
        actual_df.sort_index(axis=1),
        expected_df.sort_index(axis=1),
        check_exact=False,
        atol=atol,
        obj=f"CSV mismatch (left=actual, right=expected/reference): {actual.name}",
    )


def run_train_and_compare_outputs(
    tmp_path: Path,
    config_path: Path,
    dataset_name: str,
    fixture_dirname: str,
    expected_file_keys: list[str],
):
    # Auto-select device
    cuda_visible_devices = "0" if torch.cuda.is_available() else ""
    print("Using", "GPU: CUDA DEVICE 0" if cuda_visible_devices else "CPU")

    # Setup paths
    save_dir = tmp_path
    exp_name = "pytest_exp"
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
        "trainer.args.save_checkpoints=False",
        "trainer.args.plot_metrics=False",
        f"save_dir={save_dir}",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    subprocess.check_call(cmd, env=env)

    # Locate experiment folder
    latest_exp_dir = get_latest_folder(exp_root)
    assert latest_exp_dir, "No experiment directory found."

    # Check output directories
    for subdir in ["logs", "metrics", "models"]:
        path = latest_exp_dir / subdir
        assert path.exists(), f"Missing {subdir} directory: {path}"

    # Setup actual and expected file paths
    metrics_dir = latest_exp_dir / "metrics"
    ood_dir = metrics_dir / "ood_metrics"
    actual_files = {
        "val_metrics": metrics_dir / "val_metrics.csv",
        "ood_maha": ood_dir / "mahalanobisscore.csv",
        "ood_ml": ood_dir / "maxlogitscore.csv",
        "ood_msp": ood_dir / "mspscore.csv",
        "ood_nn": ood_dir / "nearestneighborscore.csv",
        "ood_ec": ood_dir / "embeddingcenterscore.csv",
        "ood_gen": ood_dir / "genscore.csv",
    }

    ref_base = (
        Path(__file__).parent.parent
        / "fixtures"
        / fixture_dirname
        / ("metrics_gpu" if cuda_visible_devices == "0" else "metrics_cpu")
    )
    reference_files = {key: ref_base / actual_files[key].relative_to(metrics_dir) for key in expected_file_keys}

    for key in expected_file_keys:
        assert_csv_almost_equal(actual_files[key], reference_files[key])


# === TESTS ===


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Skipped in GitHub Actions CI")
def test_train_py_uea_auxiliary_contrastive_racketsports(tmp_path: Path, uea_auxiliary_contrastive_config_path: Path):
    run_train_and_compare_outputs(
        tmp_path=tmp_path,
        config_path=uea_auxiliary_contrastive_config_path,
        dataset_name="UEA/RacketSports",
        fixture_dirname="UEA_auxiliary_contrastive_RacketSports",
        expected_file_keys=[
            "val_metrics",
            "ood_ml",
            "ood_nn",
            "ood_maha"
        ],
    )


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Skipped in GitHub Actions CI")
def test_train_py_uea_euclidean_epilepsy(tmp_path: Path, uea_euclidean_config_path: Path):
    run_train_and_compare_outputs(
        tmp_path=tmp_path,
        config_path=uea_euclidean_config_path,
        dataset_name="UEA/Epilepsy",
        fixture_dirname="UEA_euclidean_Epilepsy",
        expected_file_keys=[
            "val_metrics",
            "ood_maha",
            "ood_msp",
            "ood_nn",
            "ood_ec"
        ],
    )


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Skipped in GitHub Actions CI")
def test_train_py_uea_supcon_euclidean_epilepsy(tmp_path: Path, uea_supcon_euclidean_config_path: Path):
    run_train_and_compare_outputs(
        tmp_path=tmp_path,
        config_path=uea_supcon_euclidean_config_path,
        dataset_name="UEA/Epilepsy",
        fixture_dirname="UEA_supcon_euclidean_Epilepsy",
        expected_file_keys=[
            "val_metrics",
            "ood_nn",
            "ood_maha"
        ],
    )


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Skipped in GitHub Actions CI")
def test_train_py_uea_contrastive_supcon_hyper_racketsports(
    tmp_path: Path, uea_contrastive_supcon_hyper_config_path: Path
):
    run_train_and_compare_outputs(
        tmp_path=tmp_path,
        config_path=uea_contrastive_supcon_hyper_config_path,
        dataset_name="UEA/RacketSports",
        fixture_dirname="UEA_supcon_hyper_RacketSports",
        expected_file_keys=[
            "val_metrics",
            "ood_maha",
            "ood_nn"
        ],
    )
