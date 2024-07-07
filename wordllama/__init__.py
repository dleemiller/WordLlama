# wordllama/__init__.py

import pathlib
import logging
import toml
from .wordllama import WordLlama

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Dynamic configuration mapping
def load_config(dim, config_dir="wordllama/config"):
    config_file = pathlib.Path(config_dir) / f"wordllama_{dim}.toml"
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    with open(config_file, "r") as f:
        config = toml.load(f)
    return config


def load(
    dim: int = 64,
    binary: bool = False,
    weights_dir: str = "weights",
    config_dir: str = "wordllama/config",
):
    """
    Load the WordLlama model.

    Parameters:
    - dim (int): The dimensionality of the model to load. Options: [64, 128, 256, 512, 1024].
    - binary (bool): Whether to load the binary version of the weights.
    - weights_dir (str): Directory where weight files are stored. Default is 'weights'.
    - config_dir (str): Directory where configuration files are stored. Default is 'wordllama/config'.

    Returns:
    - WordLlama: The loaded WordLlama model.

    Raises:
    - ValueError: If the configuration is not found.
    - FileNotFoundError: If the weights file is not found.

    Examples:
    ---------
    >>> model = load(dim=256)
    >>> model = load(dim=1024, binary=True)
    >>> model = load(dim=64, weights_dir="custom_weights")
    """
    logger.info(f"Loading configuration for dimension: {dim}")
    config = load_config(dim, config_dir=config_dir)

    suffix = "_binary" if binary else ""
    weights_file_name = f"wordllama_{dim}{suffix}.safetensors"

    weights_file_path = pathlib.Path(weights_dir) / weights_file_name
    if not weights_file_path.exists():
        raise FileNotFoundError(
            f"Weights file '{weights_file_path}' not found in directory '{weights_dir}'."
        )

    logger.info(f"Loading weights from: {weights_file_path}")
    return WordLlama.build(weights_file_path, config)
