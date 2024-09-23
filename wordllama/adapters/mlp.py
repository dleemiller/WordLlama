import os

from torch import nn
import safetensors.torch as st


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, tensors) -> dict:
        tensors.update({"x": self.mlp(tensors["token_embeddings"])})
        return tensors

    def save(self, filepath: str, **kwargs):
        """Save the model's state_dict using safetensors.

        Args:
            filepath (str): The path where the model should be saved.
        """
        # Ensure tensors are on CPU and converted to the required format for safetensors
        {k: v.cpu() for k, v in self.state_dict().items()}
        metadata = {
            "model": "MLP",
            # "in_dim": self.mlp[0].in_features,
            # "out_dim": self.mlp[2].out_features,
            # "hidden_dim": self.mlp[0].out_features
        }
        st.save_model(
            model=self,
            filename=os.path.join(filepath, "mlp.safetensors"),
            metadata=metadata,
        )
