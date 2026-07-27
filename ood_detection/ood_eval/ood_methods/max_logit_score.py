import numpy as np


class MaxLogitScore:
    """
    Maximum Logit (MaxLogit) OOD Scoring.
    Uses the maximum logit value directly as the OOD score.
    """

    def __init__(self):
        self.requires_embeddings = False

    def __repr__(self):
        return "MaxLogitScore()"

    def compute_score(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute OOD scores using Maximum Logit.

        Args:
            logits (np.ndarray): Raw logits (NumPy array) with shape (N_samples, N_classes).

        Returns:
            np.ndarray: OOD scores (negative of max logit for each sample).
        """
        # Directly return negative max logits as OOD scores
        return -np.max(logits, axis=1)
