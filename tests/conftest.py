from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_seed():
    """
    Fix numpy's random seed for reproducible augmentation tests.
    """
    np.random.seed(42)


@pytest.fixture
def simple_series():
    """
    2 features × 10 timesteps series for basic transforms.
    """
    return np.arange(20, dtype=np.float32).reshape(2, 10)


@pytest.fixture
def batch_series():
    """
    Batch of 3 samples, 1 channel, 10 timesteps for permutation tests.
    """
    data = np.arange(10, dtype=np.float32)
    # shape: (3, 1, 10)
    return np.tile(data, (3, 1)).reshape(3, 1, 10)


@pytest.fixture
def ucr_angular_config_path() -> Path:
    """
    Sample configuration for UCR angular ECG5000 dataset.
    """
    return Path(__file__).parent / "fixtures" / "UCR_angular_ECG5000" / "ucr.yaml"


@pytest.fixture
def uea_auxiliary_contrastive_config_path() -> Path:
    """
    Sample configuration for UEA auxiliary contrastive RacketSports dataset.
    """
    return Path(__file__).parent / "fixtures" / "UEA_auxiliary_contrastive_RacketSports" / "uea_aux_contrastive.yaml"


@pytest.fixture
def uea_euclidean_config_path() -> Path:
    """
    Sample configuration for UEA euclidean Epilepsy dataset.
    """
    return Path(__file__).parent / "fixtures" / "UEA_euclidean_Epilepsy" / "uea_euclidean.yaml"


@pytest.fixture
def ucr_center_loss_config_path() -> Path:
    """
    Sample configuration for UCR center loss euclidean ECG5000 dataset.
    """
    return Path(__file__).parent / "fixtures" / "UCR_center_loss_ECG5000" / "ucr.yaml"


@pytest.fixture
def uea_supcon_euclidean_config_path() -> Path:
    """
    Sample configuration for UEA contrastive supcon euclidean Epilepsy dataset.
    """
    return Path(__file__).parent / "fixtures" / "UEA_supcon_euclidean_Epilepsy" / "uea_supcon_euc.yaml"


@pytest.fixture
def uea_contrastive_supcon_hyper_config_path() -> Path:
    """
    Sample configuration for UEA contrastive supcon hyper RacketSports dataset.
    """
    return Path(__file__).parent / "fixtures" / "UEA_supcon_hyper_RacketSports" / "uea_supcon_hyper.yaml"
