# wordllama/__init__.py
import pathlib
import logging
from .wordllama import WordLlama
from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load(
    config_name: str = "llama3_70B",
    weights_dir: str = "weights",
    binary: bool = False,
    dim: int = 64,
):
    """
    Load the WordLlama model.

    Parameters:
    - config_name (str): The name of the configuration to load.
    - weights_dir (str): Directory where weight files are stored. Default is 'weights'.
    - binary (bool): Whether to load the binary version of the weights.
    - dim (int): The dimensionality of the model to load. Options: [64, 128, 256, 512, 1024].

    Returns:
    - WordLlama: The loaded WordLlama model.

    Raises:
    - ValueError: If the configuration is not found.
    - FileNotFoundError: If the weights file is not found.

    Examples:
    ---------
    >>> model = load(config_name="llama3_70B", dim=256)
    >>> model = load(config_name="mixtral", dim=1024, binary=True)
    >>> model = load(config_name="llama3_8B", dim=64, weights_dir="custom_weights")
    """
    logger.info(f"Loading configuration for: {config_name}")
    config = getattr(Config, config_name)

    suffix = "_binary" if binary else ""
    weights_file_name = f"{config_name}_{dim}{suffix}.safetensors"

    weights_file_path = pathlib.Path(weights_dir) / weights_file_name
    if not weights_file_path.exists():
        raise FileNotFoundError(
            f"Weights file '{weights_file_path}' not found in directory '{weights_dir}'."
        )

    logger.info(f"Loading weights from: {weights_file_path}")
    return WordLlama.build(weights_file_path, config)


def load_training(weights, config):
    from wordllama.embedding.word_llama_embedding import WordLlamaEmbedding

    return WordLlamaEmbedding.build(weights, config)
