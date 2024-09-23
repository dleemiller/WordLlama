import safetensors.torch

from ..config import Config
from ..embedding.word_llama_embedding import WordLlamaEmbedding


def extract_safetensors(
    config_name: str,
    tensor_path: str = "model-00001-of-00030.safetensors",  # example for llama3 70B
    key: str = "model.embed_tokens.weight",
):
    """
    Args:
        config_name: name of the model configuration (see Config object)
        tensor_path: the path to your safetensors file
        key: the tensor key name (eg. model.embed_tokens.weight)
    """
    config = getattr(Config, config_name)
    with safetensors.torch.safe_open(tensor_path, "pt") as f:
        embed = f.get_tensor(key)
        assert embed.size(0) == config.model.n_vocab

    wl = WordLlamaEmbedding(config)
    wl.load_state_dict({"embedding.weight": embed})
    safetensors.torch.save_model(wl.half(), f"{config_name}.safetensors")
