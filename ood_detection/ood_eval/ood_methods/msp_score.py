"""
Maximum Softmax Probability (MSP) for OOD Score
"""

import numpy as np
from scipy.special import softmax


class MSPScore:
    """
    Returns the maximum softmax probability as the OOD score.
    """

    def __init__(self):
        self.requires_embeddings = False

    def __repr__(self):
        return "MSPScore()"

    def compute_score(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute OOD scores using Maximum Softmax Probability.

        Args:
            logits (np.ndarray): Raw logits (NumPy array) with shape (N_samples, N_classes).

        Returns:
            np.ndarray: OOD scores (negative of max softmax probability for each sample).
        """
        # Apply softmax along the class dimension using scipy's softmax
        softmax_scores = softmax(logits, axis=1)

        # Compute the maximum softmax probability for each sample
        max_softmax = np.max(softmax_scores, axis=1)

        # OOD score is negative of the max softmax probability
        ood_scores = -max_softmax

        return ood_scores
