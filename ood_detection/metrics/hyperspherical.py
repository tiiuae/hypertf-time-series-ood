import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import torch
import torch.nn.functional as F


def compute_dispersion(prototypes: torch.Tensor, angular: bool = False) -> float:
    """
    Compute dispersion: Measures inter-class separation.
    - Higher values indicate better separation.
    - Range: 0 to 1.
    """
    num_classes = prototypes.size(0)

    # Normalize prototypes to compute cosine similarity
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)  # (num_classes, feat_dim)

    # Compute cosine similarity matrix
    cosine_sim_matrix = torch.matmul(prototypes_norm, prototypes_norm.T)  # (num_classes, num_classes)

    # Mask to exclude diagonal (self-similarity)
    mask = torch.eye(num_classes, device=prototypes.device).bool()
    cosine_sim_matrix = cosine_sim_matrix.masked_fill(mask, float("nan"))

    if angular:
        # Compute angular distances (arccos of cosine similarities)
        # Clamp values to avoid numerical issues with arccos
        clamped_cosine = torch.clamp(cosine_sim_matrix, -1.0 + 1e-7, 1.0 - 1e-7)
        angular_distances = torch.acos(clamped_cosine)  # In radians

        # Convert to degrees for better interpretability
        angular_distances_deg = torch.rad2deg(angular_distances)

        # Compute metrics
        avg_angular_distance = torch.nanmean(angular_distances_deg)
        min_angular_distance = torch.nanmin(angular_distances_deg)

        return avg_angular_distance
    else:
        # Compute average pairwise cosine similarity
        avg_cosine_sim = torch.nanmean(cosine_sim_matrix)

        # Compute minimum pairwise cosine similarity
        # Replace NaNs with +inf to ignore them during min computation
        clean_cosine_sim_matrix = torch.nan_to_num(cosine_sim_matrix, nan=float("inf"))
        min_cosine_sim = torch.min(clean_cosine_sim_matrix)

        # Separation Score: Shift cosine similarity to [0,1]
        separation_score = (1 - avg_cosine_sim) / 2

        return separation_score


def compute_compactness(
    all_embeddings: torch.Tensor, all_targets: torch.Tensor, prototypes: torch.Tensor, device: torch.device
) -> float:
    """
    Compute compactness: Measures intra-class similarity.
    - Higher values indicate tighter class clusters.
    - Range: 0 to 1.
    """
    num_classes = prototypes.shape[0]

    # Normalize embeddings and prototypes to unit vectors (for cosine similarity)
    all_embeddings = F.normalize(all_embeddings.to(device), p=2, dim=1)
    prototypes = F.normalize(prototypes.to(device), p=2, dim=1)

    # Collect cosine similarities between each feature and its class prototype
    cosine_similarities = []
    for class_id in range(num_classes):
        # Select embeddings belonging to the current class
        class_mask = all_targets == class_id
        class_embeddings = all_embeddings[class_mask]

        if class_embeddings.size(0) == 0:
            continue  # Skip empty classes

        # Compute cosine similarity via matrix multiplication
        sim = torch.matmul(class_embeddings, prototypes[class_id])

        # Map similarity to [0, 1]
        mapped_sim = (sim + 1) / 2
        cosine_similarities.extend(mapped_sim.tolist())

    # Compute average compactness
    compactness_score = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0

    return compactness_score


def compute_prototypes(
    emb: torch.Tensor, targets: torch.Tensor, num_classes: int, device: torch.device
) -> torch.Tensor:
    """Return L2-normalized class prototypes for the given embeddings."""
    # class prototypes are the mean of instance embeddings
    raw = [emb[targets == c].mean(0) for c in range(num_classes)]
    raw = [p for p in raw if not torch.isnan(p).any()]
    if not raw:
        raise ValueError("No valid prototypes computed; check targets/num_classes.")
    prototypes = torch.stack(raw)
    return F.normalize(prototypes.to(device), p=2, dim=1)


def eval_nearest_neighbor(
    test_emb: torch.Tensor, train_emb: torch.Tensor, train_labels: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float, torch.Tensor]:
    """
    Evaluate the nearest neighbor (NN) classification accuracy and F1 score.

    Args:
    - test_emb (torch.Tensor): Embeddings to classify.
    - train_emb (torch.Tensor): Embeddings of the training set.
    - train_labels (torch.Tensor): Labels of the training set.
    - targets (torch.Tensor): Labels of the test set.

    Returns:
    - acc (float): Accuracy of the NN classifier.
    - f1 (float): F1 score of the NN classifier.
    - preds (torch.Tensor): Predicted labels.
    """
    sims = torch.mm(test_emb, train_emb.T)
    nn_indices = torch.argmax(sims, dim=1)
    preds = train_labels[nn_indices.cpu()]
    acc = accuracy_score(targets.cpu(), preds.cpu()) * 100
    f1 = f1_score(targets.cpu(), preds.cpu(), average="weighted") * 100
    return acc, f1, preds


def eval_linear_svm(
    test_emb: torch.Tensor,
    train_emb: torch.Tensor,
    train_labels: torch.Tensor,
    targets: torch.Tensor,
    C: float = 1.0,
    max_iter: int = 10000,
) -> tuple[float, float, torch.Tensor]:
    """
    Evaluate a Linear SVM classifier trained on training embeddings.

    Args:
    - test_emb (torch.Tensor): Test embeddings (N_test, D)
    - train_emb (torch.Tensor): Train embeddings (N_train, D)
    - train_labels (torch.Tensor): Labels for train embeddings (N_train,)
    - targets (torch.Tensor): Ground truth labels for test set (N_test,)
    - C (float): SVM regularization strength
    - max_iter (int): Max iterations for SVM solver

    Returns:
    - acc (float): Accuracy in percent
    - f1 (float): Weighted F1 score in percent
    - preds (torch.Tensor): Predicted labels as a tensor
    """
    # Convert to numpy
    Xtr = train_emb.cpu().numpy()
    Xte = test_emb.cpu().numpy()
    ytr = train_labels.cpu().numpy()
    yte = targets.cpu().numpy()

    # Standardize features
    scaler = StandardScaler()
    Xtr_std = scaler.fit_transform(Xtr)
    Xte_std = scaler.transform(Xte)

    # Train Linear SVM
    clf = LinearSVC(C=C, max_iter=max_iter)
    clf.fit(Xtr_std, ytr)

    # Predict
    preds_np = clf.predict(Xte_std)

    # Metrics
    acc = accuracy_score(yte, preds_np) * 100
    f1 = f1_score(yte, preds_np, average="weighted") * 100

    # Convert predictions back to torch.Tensor
    preds = torch.from_numpy(preds_np).long()

    return acc, f1, preds


def eval_prototype_classification(
    emb: torch.Tensor,
    targets: torch.Tensor,
    prototypes: torch.Tensor,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """
    Evaluate prototype classification accuracy and F1 score.

    Args:
    - emb (torch.Tensor): Embeddings to classify.
    - targets (torch.Tensor): Labels of the test set.
    - prototypes: torch.Tensor: Class prototypes.

    Returns:
    - acc (float): Accuracy of the prototype classifier.
    - f1 (float): F1 score of the prototype classifier.
    - preds (torch.Tensor): Predicted labels.
    """
    sims = torch.mm(emb, prototypes.T)
    preds = torch.argmax(sims, dim=1)
    acc = accuracy_score(targets.cpu(), preds.cpu()) * 100
    f1 = f1_score(targets.cpu(), preds.cpu(), average="weighted") * 100
    return acc, f1, preds


def eval_classifier_logits(logits, targets):
    """
    Evaluate classifier predictions based on logits and targets.

    Args:
    - logits (torch.Tensor): The predicted logits from the classifier.
    - targets (torch.Tensor): The true labels.

    Returns:
    - acc (float): Accuracy of the classifier predictions as a percentage.
    - f1 (float): Weighted F1 score of the classifier predictions as a percentage.
    - preds (torch.Tensor): Predicted labels obtained by taking the argmax of logits.
    """
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(targets.cpu(), preds.cpu()) * 100
    f1 = f1_score(targets.cpu(), preds.cpu(), average="weighted") * 100
    return acc, f1, preds


@torch.no_grad()
def compute_nn_compactness(embeddings: torch.Tensor, targets: torch.Tensor, k: int = None) -> float:
    """
    Compactness using FAISS — average similarity to k nearest neighbors of the same class.
    Higher = more compact.
    If k is None, it is set to 10% of the smallest class size.
    """
    embeddings = F.normalize(embeddings, dim=1)
    targets = targets.cpu()
    N = embeddings.size(0)

    # Set k dynamically if not provided
    if k is None:
        class_sizes = [(targets == cls).sum().item() for cls in targets.unique()]
        min_class_size = min(class_sizes)
        k = max(1, int(min_class_size * 0.2))  # 20% of smallest class

    # Cosine similarity matrix: [N x N]
    sim_matrix = embeddings @ embeddings.T  # (N, N)

    compact_scores = []

    for cls in targets.unique():
        class_mask = targets == cls
        idxs = class_mask.nonzero(as_tuple=True)[0]

        if len(idxs) <= k:
            continue

        class_sims = sim_matrix[idxs][:, idxs]  # intra-class sim matrix

        # Mask out diagonal (self-similarity)
        mask = torch.eye(len(idxs), dtype=torch.bool, device=embeddings.device)
        class_sims = class_sims.masked_fill(mask, float("-inf"))

        # For each sample, get top-k sims within the class
        topk_sims, _ = torch.topk(class_sims, k=k, dim=1, largest=True)
        compact_scores.append(topk_sims.mean().item())

    return float(np.mean(compact_scores)) if compact_scores else 0.0


@torch.no_grad()
def compute_nn_dispersion(embeddings: torch.Tensor, targets: torch.Tensor, k: int = None) -> float:
    """
    Dispersion using FAISS — average similarity to k nearest neighbors from other classes.
    Higher = more dispersed.
    If k is None, it is set to 10% of the smallest class size.
    """
    embeddings = F.normalize(embeddings, dim=1)
    targets = targets.cpu()
    N = embeddings.size(0)

    # Set k dynamically if not provided
    if k is None:
        class_sizes = [(targets == cls).sum().item() for cls in targets.unique()]
        min_class_size = min(class_sizes)
        k = max(1, int(min_class_size * 0.2))  # 20% of smallest class

    # Cosine similarity matrix: [N x N]
    sim_matrix = embeddings @ embeddings.T  # (N, N)

    dispersion_scores = []

    for i in range(N):
        label_i = targets[i]
        # Mask to only keep other-class similarities
        mask = targets != label_i
        sim_row = sim_matrix[i]  # (N,)
        other_class_sims = sim_row[mask]  # similarities to other-class samples

        if other_class_sims.numel() < k:
            continue

        topk_sims, _ = torch.topk(other_class_sims, k=k, largest=True)
        dispersion_scores.append(topk_sims.mean().item())

    return float(np.mean(dispersion_scores)) if dispersion_scores else 0.0
