import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTConsistencyLoss(nn.Module):
    """
    Enforces similarity between raw and FFT-projected features.
    Assumes features are concatenated as:

        features = torch.cat([raw, fft], dim=0)

    with equal batch sizes.
    """

    def __init__(self, loss_type="l2"):
        super().__init__()
        if loss_type not in ["l2", "cosine"]:
            raise ValueError("loss_type must be 'l2' or 'cosine'")
        self.loss_type = loss_type

    def forward(self, projected_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            projected_features: Tensor [2B, D]
        """
        B = projected_features.size(0) // 2
        raw = projected_features[:B]
        fft = projected_features[B:]

        if self.loss_type == "l2":
            return F.mse_loss(raw, fft)

        if self.loss_type == "cosine":
            sim = F.cosine_similarity(raw, fft, dim=1)
            return (1 - sim).mean()
