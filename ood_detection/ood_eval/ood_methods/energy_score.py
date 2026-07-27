"""
Energy score for OOD Score: https://arxiv.org/pdf/2010.03759
https://github.com/wetliu/energy_ood/blob/master/utils/score_calculation.py
"""

import numpy as np
from scipy.special import logsumexp


class EnergyScore:
    """
    Computes the energy score for OOD detection using NumPy.
    Args:
        temperature: Temperature hyperparameter.
    """

    def __init__(self, temperature: float = 1.0):
        self.t = temperature
        self.requires_embeddings = False

    def __repr__(self):
        return f"EnergyScore(temperature={self.t})"

    def compute_score(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute the energy score from logits.

        Args:
            logits (np.ndarray): Logits array of shape (N_samples, N_classes).

        Returns:
            np.ndarray: Energy scores of shape (N_samples,).
        """
        energy_scores = -self.t * logsumexp(logits / self.t, axis=1)
        return energy_scores.astype(np.float32)
