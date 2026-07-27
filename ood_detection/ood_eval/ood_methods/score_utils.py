import numpy as np


def compute_class_centers(bank_embeddings: np.ndarray, bank_labels: np.ndarray) -> np.ndarray:
    """
    Compute the center (mean embedding) for each class.

    Args:
        bank_embeddings (np.ndarray): Embeddings from the training set. Shape: (N_samples, D)
        bank_labels (np.ndarray): Corresponding class labels. Shape: (N_samples,)

    Returns:
        np.ndarray: Class centers. Shape: (N_classes, D)
    """
    unique_classes = np.unique(bank_labels)
    centers = []
    for cls in unique_classes:
        cls_embeddings = bank_embeddings[bank_labels == cls]
        center = np.mean(cls_embeddings, axis=0)
        centers.append(center)
    class_centers = np.vstack(centers).astype("float32")  # Shape: (N_classes, D)
    return class_centers
