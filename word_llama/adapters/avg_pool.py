import torch
from torch import nn


class AvgPool(nn.Module):
    def __init__(self, key="x", norm: bool = False):
        super().__init__()
        self.key = key
        self.norm = norm

    @staticmethod
    def avg_pool(
        x: torch.Tensor, attention_mask: torch.Tensor, norm: bool = False
    ) -> torch.Tensor:
        # Mask
        mask = attention_mask.unsqueeze(dim=-1)

        # Average pool with mask
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        if norm:
            norms = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
            x = x / norms

        return x

    def forward(self, tensors):
        x = self.avg_pool(tensors[self.key], tensors["attention_mask"], norm=self.norm)
        tensors.update({"sentence_embedding": x})
        return tensors

    def save(self, *args, **kwargs):
        pass  # nothing to save
