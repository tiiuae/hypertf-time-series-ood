import torch
import torch.nn as nn


class TemperatureScaledCELoss(nn.Module):
    """CrossEntropyLoss with temperature scaling and automatic logits doubling handling."""

    def __init__(self, temperature: float = 0.1, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.temperature = temperature
        self.class_weights = class_weights
        # Initialize the CrossEntropyLoss
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (2*B, C) where B is batch size, C is num classes
            labels: Tensor of shape (B,) containing class labels
        """
        # Check if logits are already doubled (2*B)
        if logits.shape[0] == 2 * labels.shape[0]:
            # Logits are already doubled, use as is
            doubled_labels = labels.repeat(2)
            scaled_logits = logits / self.temperature
            loss = self.ce_loss(scaled_logits, doubled_labels)
        else:
            # Logits are not doubled, scale and compute loss normally
            scaled_logits = logits / self.temperature
            loss = self.ce_loss(scaled_logits, labels)

        return loss


class RescaledTemperatureCELoss(TemperatureScaledCELoss):
    """
    CrossEntropyLoss with temperature scaling and optional base temperature for loss rescaling.
    Automatically handles doubled logits without requiring doubled labels.
    """

    def __init__(
        self,
        temperature: float = 0.03333333333333333,
        base_temperature: float = 0.03333333333333333,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(temperature=temperature, class_weights=class_weights)
        self.base_temperature = base_temperature

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = super().forward(logits, labels)
        return (self.temperature / self.base_temperature) * loss
