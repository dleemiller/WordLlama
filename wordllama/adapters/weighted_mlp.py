import os
from nltk.corpus import stopwords

import torch
from torch import nn
import torch.nn.functional as F
import safetensors.torch as st


class WeightedMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, tokenizer, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        # avg pool weight initialization
        # Initialize stopword weights to a lower value
        n_vocab = len(tokenizer.vocab)
        self.weights = nn.Parameter(torch.ones(n_vocab))
        stopword_list = set(stopwords.words("english"))
        stopword_ids = [
            tokenizer.vocab[word] for word in stopword_list if word in tokenizer.vocab
        ]
        with torch.no_grad():
            for stopword_id in stopword_ids:
                self.weights[stopword_id] = 0.1

    def forward(self, tensors) -> dict:
        token_ids = tensors["token_ids"]
        weights = F.gelu(
            self.weights[token_ids]
        )  # use gelu to limit negative contribution of weights
        weighted_embeddings = self.mlp(tensors["token_embeddings"]) * weights.unsqueeze(
            -1
        )
        tensors.update({"x": weighted_embeddings})
        return tensors

    def save(self, filepath: str, **kwargs):
        """Save the model's state_dict using safetensors.

        Args:
            filepath (str): The path where the model should be saved.
        """
        # Ensure tensors are on CPU and converted to the required format for safetensors
        {k: v.cpu() for k, v in self.state_dict().items()}
        metadata = {
            "model": "WeightedMLP",
            # "in_dim": self.mlp[0].in_features,
            # "out_dim": self.mlp[2].out_features,
            # "hidden_dim": self.mlp[0].out_features
        }
        st.save_model(
            model=self,
            filename=os.path.join(filepath, "weighted_mlp.safetensors"),
            metadata=metadata,
        )
