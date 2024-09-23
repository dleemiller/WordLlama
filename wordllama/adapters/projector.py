import os
from torch import nn
import safetensors.torch as st


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim, key="token_embeddings"):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.key = key

    def forward(self, tensors) -> dict:
        tensors.update({"x": self.proj(tensors[self.key])})
        return tensors

    def save(self, filepath: str, **kwargs):
        """Save the model's state_dict using safetensors.

        Args:
            filepath (str): The path where the model should be saved.
        """
        # Ensure tensors are on CPU and converted to the required format for safetensors
        {k: v.cpu() for k, v in self.state_dict().items()}
        metadata = {
            "model": "Projector",
        }
        st.save_model(
            model=self,
            filename=os.path.join(filepath, "projector.safetensors"),
            metadata=metadata,
        )
