from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingScore(ABC):
    """
    Base class for OOD methods that require embeddings and labels.
    """

    def __init__(self):
        self.requires_embeddings = True

    @abstractmethod
    def compute_score(
        self, bank: np.ndarray, labels: np.ndarray, embeddings: np.ndarray, recompute: bool = True
    ) -> np.ndarray:
        """
        Compute OOD scores based on embeddings and labels.

        Args:
            bank (np.ndarray): Training embeddings. Shape: (N_train, D)
            labels (np.ndarray): Training labels. Shape: (N_train,)
            embeddings (np.ndarray): Embeddings to score (test ID or OOD). Shape: (N_test, D)
            recompute (bool): Whether to recompute necessary structures (index, inverse matrices, etc.)

        Returns:
            np.ndarray: OOD scores.
        """
        pass
