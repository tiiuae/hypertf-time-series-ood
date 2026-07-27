import torch
from torch import nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Temperature-Scaled InfoNCE Loss for Contrastive Learning.
    Reference: https://github.com/Willtl/firm/blob/master/losses.py
    """

    def __init__(
        self, temperature: float = 0.0333, base_temperature: float = 0.0333, on_instance_features: bool = False
    ):
        super(InfoNCELoss, self).__init__()
        self.n_views = 2
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.on_instance_features = on_instance_features

    def forward(self, projected_features: torch.Tensor, instance_features: torch.Tensor = None):
        # Select features based on config flag
        features = instance_features if self.on_instance_features else projected_features
        features = F.normalize(features, dim=1)

        assert features.size(0) % self.n_views == 0, "NTXentLoss expects feature embeddings in pairs."
        f1, f2 = features.chunk(self.n_views, dim=0)  # Split projected features into two views

        # Compute the cosine similarity
        cos_similarity = torch.mm(f1, f2.t())

        # Scale the cosine similarities by the temperature
        logits = cos_similarity / self.temperature

        # Labels are the indices themselves since the diagonal corresponds to the positive examples
        labels = torch.arange(logits.size(0), device=logits.device)

        # Calculate the cross-entropy loss
        loss = F.cross_entropy(logits, labels) * (self.base_temperature / self.temperature)

        return loss


class NTXentLoss(nn.Module):
    """
    Computes the Normalized Temperature-scaled Cross-Entropy (NT-Xent) Loss.
    Reference: https://github.com/Willtl/firm/blob/master/losses.py
    """

    def __init__(
        self, temperature: float = 0.0333, base_temperature: float = 0.0333, on_instance_features: bool = False
    ):
        super().__init__()
        self.n_views = 2
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.cel = nn.CrossEntropyLoss(reduction="mean")
        self.on_instance_features = on_instance_features

    def forward(self, projected_features: torch.Tensor, instance_features: torch.Tensor = None):
        # Select features based on config flag
        features = instance_features if self.on_instance_features else projected_features
        features = F.normalize(features, dim=1)

        assert features.size(0) % self.n_views == 0, "Number of samples should be divisible by the number of views."
        batch_size = features.size(0) // self.n_views

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)

        # Construct labels matrix
        labels = torch.arange(batch_size, device=features.device).repeat(2)
        labels = (labels[:, None] == labels[None, :]).float()

        # Mask self-similarity (diagonal elements)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels.masked_select(~mask).view(labels.shape[0], -1)

        # Mask out diagonal (self-similarity)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # Extract positive and negative similarities
        positives = similarity_matrix[labels > 0].view(labels.shape[0], -1)
        negatives = similarity_matrix[labels == 0].view(similarity_matrix.shape[0], -1)

        # Construct logits and targets
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        # Apply temperature scaling and compute loss
        logits = logits / self.temperature
        loss = self.cel(logits, labels) * (self.base_temperature / self.temperature)
        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss with Class Weights https://arxiv.org/pdf/2004.11362.pdf"""

    def __init__(
        self,
        temperature: str = 0.0333,
        base_temperature: float = 0.0333,
        contrast_mode: str = "all",
        class_weights: list[float] = None,
        on_instance_features: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode
        self.class_weights = class_weights
        self.on_instance_features = on_instance_features

    def forward(
        self,
        projected_features: torch.Tensor,
        labels: torch.Tensor = None,
        instance_features: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        device = projected_features.device
        features = instance_features if self.on_instance_features else projected_features
        features = F.normalize(features, dim=1)

        # Ensure features are shaped correctly: [batch_size, n_views, feature_dim]
        batch_size = features.shape[0] // 2
        feature_dim = features.shape[1]
        features = features.view(batch_size, 2, feature_dim)

        # Initialize `mask` based on `labels` correctly
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # Compute similarity scores
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask and mask-out self-contrast cases
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive pairs
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)

        # Apply class weights if available
        if self.class_weights is not None:
            # Get class weights corresponding to each label
            weights = self.class_weights[labels.squeeze()].to(device)  # Shape: [batch_size]
            weights = weights.repeat(anchor_count)  # Expand to match anchor count
            weighted_mask = mask * weights.unsqueeze(1)  # Apply weights to mask

            mean_log_prob_pos = (weighted_mask * log_prob).sum(1) / (weighted_mask.sum(1) + 1e-9)
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Apply base temperature scaling
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
