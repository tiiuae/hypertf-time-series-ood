import torch
import torch.nn as nn


class CompactnessLoss(nn.Module):
    def __init__(self, num_classes, loss_type="linear"):
        """
        Compactness loss with different penalty types.

        Args:
            num_classes (int): Number of classes.
            loss_type (str): Type of compactness penalty. Options: ['linear', 'log', 'exponential', 'quadratic'].
        """
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type.lower()

    def forward(self, logits, labels):
        """
        Compute compactness loss.

        Args:
            logits (torch.Tensor): Cosine similarity values between features and class prototypes.
                                   Shape: [batch_size, num_classes]
            labels (torch.Tensor): Ground truth labels. Shape: [batch_size]

        Returns:
            torch.Tensor: Compactness loss.
        """
        cos_sim = logits.gather(1, labels.view(-1, 1)).squeeze()

        if self.loss_type == "linear":
            comp_loss = torch.mean(1 - cos_sim)
        elif self.loss_type == "log":
            cos_sim_shifted = (cos_sim + 1) / 2
            comp_loss = torch.mean(-torch.log(cos_sim_shifted + 1e-6))
        elif self.loss_type == "quadratic":
            comp_loss = torch.mean((1 - cos_sim) ** 2)
        else:
            raise ValueError(
                f"Invalid loss_type: {self.loss_type}. Choose from ['linear', 'log', 'exponential', 'quadratic']."
            )

        return comp_loss
