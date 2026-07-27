import torch
import torch.nn as nn
import torch.nn.functional as F


class NegativeFeatureQueue(nn.Module):
    def __init__(self, dim: int, capacity: int = 4096) -> None:
        super().__init__()
        self.capacity = capacity
        self.register_buffer("queue", torch.zeros(capacity, dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("size", torch.zeros(1, dtype=torch.long))

    def get(self) -> torch.Tensor:
        return self.queue[: int(self.size)]

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor) -> None:
        if feats.numel() == 0:
            return

        # --- short‑circuit when we receive >= capacity elements
        if feats.size(0) >= self.capacity:
            self.queue.copy_(feats[-self.capacity :])
            self.ptr.zero_()
            self.size.fill_(self.capacity)
            return
        # -------------------------------------------------------

        n = feats.size(0)
        ptr = int(self.ptr)
        end = (ptr + n) % self.capacity

        if end < ptr:  # wrap‑around
            first = self.capacity - ptr
            self.queue[ptr:] = feats[:first]
            self.queue[:end] = feats[first:]
        else:  # contiguous
            self.queue[ptr:end] = feats

        self.ptr[0] = end
        self.size[0] = min(self.capacity, int(self.size) + n)

    def is_full(self) -> bool:
        return int(self.size) == self.capacity


class AuxiliaryContrastiveLoss(nn.Module):
    """
    Auxiliary contrastive loss (cross‑entropy form) with an internal queue
    for in‑distribution negatives.

    Forward signature unchanged:
        loss = loss_fn(aux_view1, aux_view2, id_feats)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        base_temperature: float | None = None,
        feature_dim: int = 256,
        queue_capacity: int = 1024,
        detach_id: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature) if base_temperature is not None else float(temperature)
        assert self.base_temperature > 0, "`base_temperature` must be > 0"

        self.detach_id = detach_id

        # Disable the queue cleanly when capacity ≤ 0
        self.use_queue = queue_capacity > 0
        if self.use_queue:
            self.neg_queue = NegativeFeatureQueue(feature_dim, queue_capacity)
        else:
            self.register_buffer("dummy", torch.empty(0))

    def forward(
        self,
        aux_sec_proj_view1: torch.Tensor,  # z'   [B, D]
        aux_sec_proj_view2: torch.Tensor,  # z''  [B, D]
        id_sec_projected_features: torch.Tensor,  # z⁻   [N_in, D]
    ) -> torch.Tensor:
        # 1. ℓ2‑normalise
        z1 = F.normalize(aux_sec_proj_view1, dim=1)  # [B, D]
        z2 = F.normalize(aux_sec_proj_view2, dim=1)  # [B, D]
        z_id_batch = F.normalize(id_sec_projected_features, dim=1)  # [N_in, D]

        # 2. Assemble *all* negatives: queue + current batch
        if self.use_queue:
            neg_queue_feats = self.neg_queue.get()  # [M, D] (may be 0)
            if neg_queue_feats.numel():
                neg_queue_feats = neg_queue_feats.to(z1.device, non_blocking=True)
                all_negs = torch.cat([z_id_batch, neg_queue_feats], dim=0)  # [N_in+M, D]
            else:
                all_negs = z_id_batch  # [N_in, D] or empty
        else:
            all_negs = z_id_batch

        # 3. Build logits matrix
        logits_aux = torch.matmul(z1, z2.T)  # [B, B]

        z_neg_eff = all_negs.detach() if self.detach_id else all_negs
        logits_neg = (
            torch.empty(z1.size(0), 0, device=z1.device)
            if z_neg_eff.numel() == 0
            else torch.matmul(z1, z_neg_eff.T)  # [B, N_neg]
        )

        logits = torch.cat([logits_aux, logits_neg], dim=1) / self.temperature
        target = torch.arange(z1.size(0), device=z1.device)

        ce_loss = F.cross_entropy(logits, target)
        loss = ce_loss * (self.temperature / self.base_temperature)

        # 4. After computing the loss, enqueue current‑batch ID features
        if self.use_queue:
            with torch.no_grad():
                self.neg_queue.enqueue(z_id_batch.detach())

        return loss
