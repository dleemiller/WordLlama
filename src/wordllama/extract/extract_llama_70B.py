import os

import safetensors.torch

from ..config import Config
from ..embedding.word_llama_embedding import WordLlamaEmbedding


def extract_llama_70B(filepath: str, tensor: str = "model-00001-of-00030.safetensors"):
    """
    Args:
        filepath: the path to your huggingface model dir (default ~/.config/huggingface/...)
        tensor: the tensor file containing the token embeddings
    """
    tensor_path = os.path.join(filepath, tensor)
    with safetensors.torch.safe_open(tensor_path, "pt") as f:
        embed = f.get_tensor("model.embed_tokens.weight")

    wl = WordLlamaEmbedding(Config.llama3_70B)
    wl.load_state_dict({"embedding.weight": embed})
    safetensors.torch.save_model(wl.half(), "llama3_70B_embedding.safetensors")
