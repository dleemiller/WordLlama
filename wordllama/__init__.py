# wordllama/__init__.py
import pathlib
import logging
from .wordllama import WordLlama
from .config import Config, WordLlamaConfig

from typing import Union, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load(
    config: Union[str, WordLlamaConfig] = "l2_supercat",
    weights_dir: str = None,
    binary: bool = False,
    dim: int = 256,
    trunc_dim: Optional[int] = None,
):
    """
    Load the WordLlama model.

    Parameters:
    - config (str or WordLlamaConfig): The configuration object or the name of the configuration to load.
    - weights_dir (str): Directory where weight files are stored. If None, defaults to the 'weights' directory in the current module directory.
    - binary (bool): Whether to load the binary version of the weights.
    - dim (int): The dimensionality of the model to load. Options: [64, 128, 256, 512, 1024].

    Returns:
    - WordLlama: The loaded WordLlama model.

    Raises:
    - ValueError: If the configuration is not found.
    - FileNotFoundError: If the weights file is not found.

    Examples:
    ---------
    >>> model = load(config="llama3_70B", dim=256)
    >>> model = load(config="mixtral", dim=1024, binary=True)
    >>> model = load(config="llama3_8B", dim=64, weights_dir="custom_weights")
    """
    if isinstance(config, str):
        config_obj = Config._configurations.get(config, None)
        if config_obj is None:
            raise ValueError(f"Configuration '{config}' not found.")
    elif isinstance(config, WordLlamaConfig):
        config_obj = config  # Direct use of passed config object
    else:
        raise ValueError(
            "Invalid configuration type provided. It must be either a string or an instance of WordLlamaConfig."
        )

    assert (
        dim in config_obj.matryoshka.dims
    ), f"Model dimension must be one of matryoshka dims in config: {config_obj.matryoshka_dims}"
    if trunc_dim is not None:
        assert (
            trunc_dim <= dim
        ), f"Cannot truncate to dimension lower than model dimension ({trunc_dim} > {dim})"
        assert trunc_dim in config_obj.matryoshka.dims

    # Set default weights_dir to the 'weights' subdirectory in the current module directory if not provided
    if weights_dir is None:
        weights_dir = pathlib.Path(__file__).parent / "weights"
    else:
        weights_dir = pathlib.Path(weights_dir)

    suffix = "_binary" if binary else ""
    weights_file_name = f"{config}_{dim}{suffix}.safetensors"

    weights_file_path = weights_dir / weights_file_name
    if not weights_file_path.exists():
        raise FileNotFoundError(
            f"Weights file '{weights_file_path}' not found in directory '{weights_dir}'."
        )

    logger.info(f"Loading weights from: {weights_file_path}")
    return WordLlama.build(weights_file_path, config_obj, trunc_dim=trunc_dim)


def load_training(weights, config, dims=None):
    from wordllama.embedding.word_llama_embedding import WordLlamaEmbedding

    return WordLlamaEmbedding.build(weights, config, dims=dims)
