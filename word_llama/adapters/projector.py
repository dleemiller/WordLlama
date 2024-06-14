from torch import nn


class Projector(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, tensors) -> dict:
        return {
            "x": self.proj(tensors["x"]),
            "attention_mask": tensors["attention_mask"],
        }
