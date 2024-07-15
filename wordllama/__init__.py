# wordllama/__init__.py
import pathlib
import logging
from .wordllama import WordLlama
from .config import Config, WordLlamaConfig

from typing import Union, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training(weights, config, dims=None):
    from wordllama.embedding.word_llama_embedding import WordLlamaEmbedding

    return WordLlamaEmbedding.build(weights, config, dims=dims)
