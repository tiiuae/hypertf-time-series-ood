import faiss
import numpy as np
from scipy.spatial.distance import cdist

from .base_embedding_score import BaseEmbeddingScore
from .score_utils import compute_class_centers


class EmbeddingCenterScore(BaseEmbeddingScore):
    """
    OOD Scoring based on distance to class centers.
    Computes the distance of embeddings to the nearest class center.
    """

    def __init__(self, distance_metric: str = "euclidean"):
        """
        Args:
            distance_metric (str): Distance metric to use ('euclidean', 'cosine', etc.).
        """
        super().__init__()
        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        self.distance_metric = distance_metric
        self.class_centers = None

    def __repr__(self):
        return f"EmbeddingCenterScore(metric={self.distance_metric})"

    def compute_score(
        self, bank: np.ndarray, labels: np.ndarray, embeddings: np.ndarray, recompute: bool = True
    ) -> np.ndarray:
        """
        Compute OOD scores based on distance to the nearest class center.

        Args:
            bank (np.ndarray): Training embeddings. Shape: (N_train_samples, D)
            labels (np.ndarray): Training labels. Shape: (N_train_samples,)
            embeddings (np.ndarray): Embeddings to score (test ID or OOD). Shape: (N_test, D)
            recompute (bool): Whether to always recompute the class means.

        Returns:
            np.ndarray: OOD scores (higher distance => higher OOD likelihood).
        """
        if recompute or self.class_centers is None:
            if self.distance_metric == "cosine":
                # Normalize deep copy of bank
                bank = np.ascontiguousarray(bank.copy(), dtype=bank.dtype)
                faiss.normalize_L2(bank)
            # Step 1: Compute class centers
            self.class_centers = compute_class_centers(bank, labels)

        if self.distance_metric == "cosine":
            # Normalize deep copy of embeddings
            embeddings = np.ascontiguousarray(embeddings.copy(), dtype=embeddings.dtype)
            faiss.normalize_L2(embeddings)

        # Step 2: Compute distances between embeddings and class centers
        distances = cdist(embeddings, self.class_centers, metric=self.distance_metric)  # Shape: (N_test, N_classes)

        # Step 3: Use the minimum distance to any class center as the OOD score
        min_distances = np.min(distances, axis=1)  # Shape: (N_test,)

        return min_distances  # Higher distance => Higher OOD likelihood
