from torch import nn


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, tensors) -> dict:
        tensors.update({"x": self.mlp(tensors["token_embeddings"])})
        return tensors
