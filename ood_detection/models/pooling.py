import torch
import torch.nn as nn


class NonePool(nn.Module):
    @property
    def output_multiplier(self):
        return 1  # no change

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # just return features as they are
        return x


class MaxPool(nn.Module):
    @property
    def output_multiplier(self):
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=-1).values


class MeanPool(nn.Module):
    @property
    def output_multiplier(self):
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1)


class StdPool(nn.Module):
    @property
    def output_multiplier(self):
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.std(dim=-1)


class PPVPool(nn.Module):
    def __init__(self, threshold: float = 1e-6):
        super().__init__()
        self.threshold = threshold

    @property
    def output_multiplier(self):
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = (x > self.threshold).float()
        return pos.mean(dim=-1)


class MaxMeanPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = MeanPool()
        self.max = MaxPool()

    @property
    def output_multiplier(self):
        return 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_feat = self.avg(x)
        max_feat = self.max(x)
        return torch.cat([max_feat, avg_feat], dim=1)


class MaxPPVPool(nn.Module):
    def __init__(self, threshold: float = 1e-8):
        super().__init__()
        self.max = MaxPool()
        self.ppv = PPVPool(threshold)

    @property
    def output_multiplier(self):
        return 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_feat = self.max(x)
        ppv_feat = self.ppv(x)
        return torch.cat([max_feat, ppv_feat], dim=1)


class MaxAvgPPVPool(nn.Module):
    def __init__(self, threshold: float = 1e-6):
        super().__init__()
        self.max = MaxPool()
        self.avg = MeanPool()
        self.ppv = PPVPool(threshold)

    @property
    def output_multiplier(self):
        return 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_feat = self.max(x)
        avg_feat = self.avg(x)
        ppv_feat = self.ppv(x)
        return torch.cat([max_feat, avg_feat, ppv_feat], dim=1)


class MaxAvgStdPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.max = MaxPool()
        self.avg = MeanPool()
        self.std = StdPool()

    @property
    def output_multiplier(self):
        return 3  # max + avg + std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_feat = self.max(x)
        avg_feat = self.avg(x)
        std_feat = self.std(x)
        return torch.cat([max_feat, avg_feat, std_feat], dim=1)


class AttentionPool(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    @property
    def output_multiplier(self):
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        x_t = x.transpose(1, 2)  # (batch, seq_len, channels)
        scores = self.attn(x_t)  # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        pooled = (x_t * weights).sum(dim=1)  # (batch, channels)
        return pooled


class MaxMeanAttentionPool(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.max = MaxPool()
        self.mean = MeanPool()
        self.attn = AttentionPool(input_dim)

    @property
    def output_multiplier(self):
        return 3  # max + mean + attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_feat = self.max(x)
        mean_feat = self.mean(x)
        attn_feat = self.attn(x)
        return torch.cat([max_feat, mean_feat, attn_feat], dim=1)


class MaxMeanPPVStdPool(nn.Module):
    def __init__(self, threshold: float = 1e-6):
        super().__init__()
        self.max = MaxPool()
        self.mean = MeanPool()
        self.ppv = PPVPool(threshold)
        self.std = StdPool()

    @property
    def output_multiplier(self):
        return 4  # max + mean + ppv + std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_feat = self.max(x)
        mean_feat = self.mean(x)
        ppv_feat = self.ppv(x)
        std_feat = self.std(x)
        return torch.cat([max_feat, mean_feat, ppv_feat, std_feat], dim=1)


def get_pooling(pooling_type: str, feature_dim: int) -> nn.Module:
    pooling_type = pooling_type.lower()
    if pooling_type == "none":
        return NonePool()
    elif pooling_type == "max":
        return MaxPool()
    elif pooling_type == "avg":
        return MeanPool()
    elif pooling_type == "ppv":
        return PPVPool()
    elif pooling_type == "max_avg":
        return MaxMeanPool()
    elif pooling_type == "max_ppv":
        return MaxPPVPool()
    elif pooling_type == "max_avg_ppv":
        return MaxAvgPPVPool()
    elif pooling_type == "max_avg_std":
        return MaxAvgStdPool()
    elif pooling_type == "max_avg_attention":
        return MaxMeanAttentionPool(feature_dim)
    elif pooling_type == "max_avg_ppv_std":
        return MaxMeanPPVStdPool()
    elif pooling_type == "gru":
        return GRUPooling(input_dim=feature_dim, hidden_dim=feature_dim, bidirectional=False)
    else:
        raise ValueError(f"Unsupported pooling type: {pooling_type}")


def get_pooling_output_dim(base_feature_dim: int, pooling: nn.Module) -> int:
    """Calculate the output dimension after pooling"""
    if hasattr(pooling, "output_multiplier"):
        return base_feature_dim * pooling.output_multiplier
    else:
        # Default to 1 for unknown pooling types
        return base_feature_dim


class GRUPooling(nn.Module):
    """
    Learns to aggregate temporal features using a GRU instead of static pooling.
    Input: [B, D, T]
    Output: [B, H] (or [B, 2H] if bidirectional)
    """

    def __init__(self, input_dim, hidden_dim=None, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim or input_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1,
        )

    @property
    def output_multiplier(self):
        """Match interface used by all pooling classes"""
        return 2 if self.bidirectional else 1

    def forward(self, x):
        # x: [B, D, T]
        x = x.transpose(1, 2)  # -> [B, T, D]
        _, h_n = self.gru(x)  # h_n: [num_layers * num_dirs, B, H]

        if self.bidirectional:
            h_forward = h_n[-2]  # [B, H]
            h_backward = h_n[-1]  # [B, H]
            h_last = torch.cat([h_forward, h_backward], dim=1)  # [B, 2H]
        else:
            h_last = h_n[-1]  # [B, H]

        return h_last
