import faiss
import numpy as np

from .base_embedding_score import BaseEmbeddingScore


class NearestNeighborScore(BaseEmbeddingScore):
    """
    Nearest Neighbor (NN) OOD Scoring using FAISS (CPU).
    Supports both cosine similarity and Euclidean distance to compare samples
    against an embedding bank built from train ID data.
    """

    def __init__(self, k: int | None = None, k_perc: float | None = None, distance_metric: str = "cosine"):
        """
        Args:
            k (int, optional): Number of nearest neighbors to consider for scoring.
            k_perc (float, optional): Fraction (0, 1] applied to the smallest class size to derive k.
            distance_metric (str): Distance metric to use ('cosine' or 'euclidean').
        """
        super().__init__()
        if k is not None and k_perc is not None:
            raise ValueError("Provide either `k` or `k_perc`, not both.")
        if k_perc is not None and not (0 < k_perc <= 1):
            raise ValueError("`k_perc` must be within (0, 1].")
        if k is None and k_perc is None:
            k = 1  # default behaviour maintains backwards compatibility
        if k is not None and k < 1:
            raise ValueError("`k` must be >= 1.")
        self.k = k
        self.k_perc = k_perc
        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        self.distance_metric = distance_metric
        self.index = None
        self._current_bank_signature: tuple[int, tuple[int, ...], str] | None = None
        self._cached_k: int | None = None
        self._cached_k_key: tuple[int, int] | None = None

    def __repr__(self):
        k_repr = f"k_perc={self.k_perc}" if self.k_perc is not None else f"k={self.k}"
        return f"NearestNeighborScore({k_repr}, metric={self.distance_metric})"

    def build_index(self, bank: np.ndarray):
        dimension = bank.shape[1]
        if self.distance_metric == "cosine":
            # normalize bank copy instead of bank in-place
            bank = np.ascontiguousarray(bank.copy(), dtype=bank.dtype)
            faiss.normalize_L2(bank)
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        # Add bank embeddings to FAISS index
        self.index.add(bank)

    def compute_score(
        self, bank: np.ndarray, labels: np.ndarray | None, embeddings: np.ndarray, recompute: bool = True
    ) -> np.ndarray:
        """
        Compute OOD scores based on nearest neighbor distances/similarities using FAISS.

        Args:
            bank (np.ndarray): Embedding bank built from train ID samples. Shape: (N_train, D)
            labels (np.ndarray): Training labels used to compute k when k_perc is set. Shape: (N_train,)
            embeddings (np.ndarray): Embeddings to score (test ID or OOD). Shape: (N_test, D)
            recompute (bool): Whether to always recompute the indexes

        Returns:
            np.ndarray: OOD scores.
                - For cosine similarity: lower scores indicate higher OOD likelihood (negative similarity).
                - For Euclidean distance: higher scores indicate higher OOD likelihood.
        """
        bank_signature = self._bank_signature(bank)
        if recompute or self.index is None or bank_signature != self._current_bank_signature:
            self.build_index(bank)
            self._current_bank_signature = bank_signature
            self._invalidate_k_cache()

        if self.distance_metric == "cosine":
            # normalize embeddings copy instead of bank in-place
            embeddings = np.ascontiguousarray(embeddings.copy(), dtype=embeddings.dtype)
            faiss.normalize_L2(embeddings)  # normalize in place

        effective_k = self._resolve_k(labels, bank.shape[0])

        # Search for top-k nearest neighbors
        distances_or_similarities, _ = self.index.search(embeddings, effective_k)  # Shape: (N_test, k)

        # Compute average over top-k neighbors
        avg_topk = np.mean(distances_or_similarities, axis=1)  # Shape: (N_test,)

        # Compute OOD scores
        if self.distance_metric == "cosine":
            # Higher similarity => more in-distribution, (-1 is most OOD, 1 is most ID)
            # so scale cos. sim. from [1, -1] to [0, 1] for ood scoring
            ood_scores = (1 - avg_topk) / 2
        else:  # Euclidean
            # Higher distance => more likely to be OOD
            ood_scores = avg_topk

        return ood_scores

    def _resolve_k(self, labels: np.ndarray | None, bank_size: int) -> int:
        """
        Determine the effective k given the training labels and configuration.
        Falls back to the provided k when k_perc is not specified.
        """
        if self.k_perc is None:
            # Cap k to available bank size to avoid FAISS runtime errors.
            return min(self.k, bank_size)

        if labels is None:
            raise ValueError("train_id labels are required to compute k from k_perc.")

        cache_key = (id(labels), bank_size)
        if self._cached_k is not None and self._cached_k_key == cache_key:
            return self._cached_k

        _, counts = np.unique(labels, return_counts=True)
        min_class_size = counts.min()
        computed_k = max(1, int(np.floor(min_class_size * self.k_perc)))
        computed_k = min(computed_k, bank_size)

        self._cached_k = computed_k
        self._cached_k_key = cache_key

        return computed_k

    def _invalidate_k_cache(self) -> None:
        self._cached_k = None
        self._cached_k_key = None

    @staticmethod
    def _bank_signature(bank: np.ndarray) -> tuple[int, tuple[int, ...], str]:
        """
        Create a lightweight signature for the bank so FAISS index rebuilds when the data changes.
        """
        ptr = bank.__array_interface__["data"][0]
        return (ptr, bank.shape, bank.dtype.str)
