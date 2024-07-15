# wordllama/__init__.py
import toml
import pathlib
import logging
from .wordllama import WordLlama
from .config import Config, WordLlamaConfig

from typing import Union, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_version():
    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    pyproject_content = toml.load(pyproject_path)
    return pyproject_content["project"]["version"]


__version__ = get_version()


def load_training(weights, config, dims=None):
    from wordllama.embedding.word_llama_embedding import WordLlamaEmbedding

    return WordLlamaEmbedding.build(weights, config, dims=dims)
