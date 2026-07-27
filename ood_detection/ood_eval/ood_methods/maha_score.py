import numpy as np
from sklearn.covariance import EmpiricalCovariance

from .base_embedding_score import BaseEmbeddingScore
from .score_utils import compute_class_centers


class MahalanobisScore(BaseEmbeddingScore):
    """
    Mahalanobis Distance-based OOD Scoring using sklearn's EmpiricalCovariance.
    Computes the Mahalanobis distance of embeddings to class centers.
    """

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize (bool): If True, normalize the embeddings before computing the Mahalanobis distance.

        If normalized, we are computing mahalanobis++ distance
        """
        super().__init__()
        self.normalize = normalize
        self.cov_estimator = EmpiricalCovariance(assume_centered=False)
        self.precision_matrix = None
        self.class_means = None

    def __repr__(self):
        return f"MahalanobisScore(normalize={self.normalize})"

    def fit_covariance(self, bank_embeddings: np.ndarray):
        """
        Fit the covariance matrix using EmpiricalCovariance with regularization.

        Args:
            bank_embeddings (np.ndarray): Training embeddings. Shape: (N_samples, D)
        """
        if self.normalize:
            bank_embeddings = bank_embeddings / np.linalg.norm(bank_embeddings, axis=1, keepdims=True)
        # Fit the covariance estimator on the original data
        self.cov_estimator.fit(bank_embeddings)

        # Get the covariance matrix
        covariance_matrix = self.cov_estimator.covariance_

        # Apply regularization to the covariance matrix
        regularization_term = 1e-6 * np.eye(covariance_matrix.shape[0])
        covariance_matrix_reg = covariance_matrix + regularization_term

        # Compute the precision matrix from the regularized covariance
        self.precision_matrix = np.linalg.inv(covariance_matrix_reg)

    def compute_score(
        self, bank: np.ndarray, labels: np.ndarray, embeddings: np.ndarray, recompute: bool = True
    ) -> np.ndarray:
        """
        Compute OOD scores using Mahalanobis distance.

        Args:
            bank (np.ndarray): Training embeddings. Shape: (N_train_samples, D)
            labels (np.ndarray): Training labels. Shape: (N_train_samples,)
            embeddings (np.ndarray): Test embeddings to score. Shape: (N_test_samples, D)
            recompute (bool): Whether to always recompute the precision matrix and class means.

        Returns:
            np.ndarray: OOD scores (higher score => higher OOD likelihood).
        """
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        if recompute or self.precision_matrix is None or self.class_means is None:
            if self.normalize:
                bank = bank / np.linalg.norm(bank, axis=1, keepdims=True)
            # Step 1: Compute class means
            self.class_means = compute_class_centers(bank, labels)
            # Step 2: Fit covariance estimator
            self.fit_covariance(bank)

        # Step 3: Compute Mahalanobis scores
        ood_scores = []
        for emb in embeddings:
            # Compute Mahalanobis distance to each class mean using precision matrix
            scores = []
            for class_mean in self.class_means:
                diff = emb - class_mean
                mahalanobis_dist_sq = np.dot(np.dot(diff.T, self.precision_matrix), diff)
                # Apply scaling factor (-0.5) as in probabilistic modeling
                score = mahalanobis_dist_sq
                scores.append(score)

            # OOD score is the minimum Mahalanobis score (higher means more OOD)
            ood_scores.append(min(scores))

        return np.array(ood_scores)
