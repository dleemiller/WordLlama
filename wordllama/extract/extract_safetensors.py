import os
import safetensors.torch

from ..config import Config
from ..embedding.word_llama_embedding import WordLlamaEmbedding


def extract_safetensors(
    config,
    filepath: str,
    outname: str,
    tensor: str = "model-00001-of-00030.safetensors",
):
    """
    Args:
        filepath: the path to your huggingface model dir (default ~/.config/huggingface/...)
        tensor: the tensor file containing the token embeddings
    """
    tensor_path = os.path.join(filepath, tensor)
    with safetensors.torch.safe_open(tensor_path, "pt") as f:
        embed = f.get_tensor("model.embed_tokens.weight")

    wl = WordLlamaEmbedding(config)
    wl.load_state_dict({"embedding.weight": embed})
    safetensors.torch.save_model(wl.half(), f"{outname}.safetensors")
