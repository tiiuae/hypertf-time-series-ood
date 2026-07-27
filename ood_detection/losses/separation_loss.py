import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        prototypes: torch.Tensor,
        loss_type: str = "row-wise-log-sum-exp",
        temperature: float = 0.03333333333333333,
    ):
        """
        Separation loss to enhance inter-class discrepancy by maximizing distance between class prototypes.

        Args:
            num_classes (int): Number of classes.
            prototypes (Tensor): Tensor containing class prototype vectors (num_classes x feat_dim).
            loss_type (str): Type of separation loss: 'linear', 'quadratic', 'global-log-sum-exp', or 'row-wise-log-sum-exp'.
            temperature (float): Scaling factor used in 'log-sum-exp' mode.
        """
        super().__init__()
        self.num_classes = num_classes
        self.prototypes = prototypes  # expected to be nn.Parameter or torch.Tensor
        self.loss_type = loss_type.lower()
        self.temperature = temperature
        assert self.loss_type in [
            "linear",
            "quadratic",
            "global-log-sum-exp",
            "row-wise-log-sum-exp",
        ], f"Unsupported loss_type '{self.loss_type}'"

    def forward(self):
        """
        Compute the separation loss based on current class prototypes.
        Returns:
            torch.Tensor: Scalar separation loss
        """
        assert self.prototypes is not None, "Prototypes must be set before calling forward."
        assert self.prototypes.shape[0] == self.num_classes, "Number of prototypes must match num_classes."

        # Normalize prototypes
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)  # (num_classes, feat_dim)

        # Compute cosine similarity between prototypes
        cosine_sim_matrix = torch.matmul(prototypes_norm, prototypes_norm.T)  # (C, C)

        # Mask the diagonal (self-similarity)
        mask = torch.eye(self.num_classes, device=cosine_sim_matrix.device).bool()

        if self.loss_type == "linear":
            cosine_sim_matrix = cosine_sim_matrix.masked_fill(mask, 0.0)
            shifted_cos_sim = cosine_sim_matrix + 1  # target is -1
            loss = torch.sum(shifted_cos_sim) / (self.num_classes * (self.num_classes - 1))

        elif self.loss_type == "quadratic":
            cosine_sim_matrix = cosine_sim_matrix.masked_fill(mask, 0.0)
            shifted_cos_sim = cosine_sim_matrix + 1
            loss = torch.sum(shifted_cos_sim**2) / (self.num_classes * (self.num_classes - 1))

        elif self.loss_type == "global-log-sum-exp":
            cosine_sim_matrix = cosine_sim_matrix.masked_fill(mask, -float("inf"))
            exp_scaled = torch.exp(cosine_sim_matrix * (1 / self.temperature))  # shape: (C, C)
            loss = torch.log(torch.sum(exp_scaled) / (self.num_classes * (self.num_classes - 1)))

        elif self.loss_type == "row-wise-log-sum-exp":
            cosine_sim_matrix = cosine_sim_matrix.masked_fill(mask, -float("inf"))
            exp_scaled = torch.exp(cosine_sim_matrix * (1 / self.temperature))  # shape: (C, C)
            row_sums = exp_scaled.sum(dim=1)  # sum over j ≠ i
            loss = torch.log(row_sums / (self.num_classes - 1))  # (C,)
            loss = loss.mean()

        return loss
