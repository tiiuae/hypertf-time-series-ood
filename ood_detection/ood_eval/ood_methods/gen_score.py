"""
Generalized ENtropy (GEN) score: https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.pdf
https://github.com/wetliu/energy_ood
"""

import numpy as np


class GENScore:
    """
    Computes the Generalized Entropy (GEN) score for OOD detection using NumPy.

    Intuition:
        - Instead of using the raw energy or entropy of logits, the GEN score measures
          how "abnormally shaped" the softmax confidence distribution is.
        - It raises probabilities and their complements to a power gamma, emphasizing
          mid-confidence regions (where model uncertainty behaves differently on OOD data).
        - Only the top-M probabilities (determined by a fraction of total classes)
          are used to compute this score to reduce noise from the tail.

    Formula:
        probs = softmax(logits / T)
        probs_topM = top-M probabilities per sample
        raw_score = sum_i [p_i^γ * (1 - p_i)^γ]
        GEN_score = -raw_score

    Args:
        gamma (float): Curvature hyperparameter in the generalized entropy function.
                       Controls sensitivity to mid-confidence probabilities.
                       Recommended: 0.1 (default)
        top_m_ratio (float): Fraction of top classes to include in the calculation (0 < ratio ≤ 1).
                             For example, 0.1 means top 10% of the classes.
                             M = max(1, floor(top_m_ratio * num_classes))
        temperature (float): Temperature scaling for the softmax normalization.
                             Higher T makes probabilities more uniform.
                             Recommended: 1.0 (default)
    """

    def __init__(self, gamma: float = 0.1, top_m_ratio: float = 0.1, temperature: float = 1.0):
        # Sanity checks for hyperparameters
        if not (0 < top_m_ratio <= 1):
            raise ValueError("top_m_ratio must be between (0, 1].")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")

        # Hyperparameters
        self.gamma = float(gamma)
        self.top_m_ratio = float(top_m_ratio)
        self.t = float(temperature)
        self.requires_embeddings = False

    def __repr__(self):
        pct = int(self.top_m_ratio * 100)
        return f"GENScore(gamma={self.gamma}, top_m_ratio={pct}%, temperature={self.t})"

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Numerically stable softmax implementation using max subtraction trick.

        Args:
            x (np.ndarray): Input logits array.
            axis (int): Axis along which to compute softmax.

        Returns:
            np.ndarray: Softmax probabilities along the given axis.
        """
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x)
        return ex / np.sum(ex, axis=axis, keepdims=True)

    def compute_score(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute the Generalized Entropy (GEN) OOD scores from model logits.

        Args:
            logits (np.ndarray): Logits array of shape (N_samples, N_classes).

        Returns:
            np.ndarray: GEN OOD scores of shape (N_samples,).
        """
        # Validate input shape
        if logits.ndim != 2:
            raise ValueError(f"logits must be 2D, got shape {logits.shape}")

        # Step 1: Apply softmax with temperature scaling
        probs = self._softmax(logits / self.t, axis=1)

        # Step 2: Determine dynamic Top-M based on number of classes
        n_classes = probs.shape[1]
        M = max(1, int(np.floor(self.top_m_ratio * n_classes)))

        # Step 3: Extract top-M probabilities per sample
        if n_classes <= M:
            # If M covers all classes, just sort the full array
            probs_topM = np.sort(probs, axis=1)
        else:
            # np.partition for fast selection of top-M elements (unordered)
            kth = n_classes - M
            topM_unordered = np.partition(probs, kth=kth, axis=1)[:, -M:]
            # Sort ascending to match np.sort(...)[..., -M:] semantics from the GEN reference code
            probs_topM = np.sort(topM_unordered, axis=1)

        # Step 4: Compute the GEN raw term (sum of p^γ * (1-p)^γ)
        p = np.clip(probs_topM, 0.0, 1.0)  # ensure numerical stability
        raw = np.sum((p**self.gamma) * ((1.0 - p) ** self.gamma), axis=1)

        # Step 5: Negate for consistency with OOD scoring convention
        # (higher = more ID-like, lower = more OOD-like)
        scores = -raw

        return scores.astype(np.float32)
