"""
WordLlama: A package for embedding and training word representations.

This package provides tools for working with word embeddings, including
the WordLlama class for managing embeddings and associated configurations.
"""

import logging

from .wordllama import WordLlama
from .config import Config, WordLlamaConfig
from ._version import __version__

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ["WordLlama", "Config", "WordLlamaConfig", "load_training", "__version__"]


def load_training(weights, config, dims=None):
    """
    Load a WordLlamaEmbedding model for training.

    This function requires additional dependencies. If they're not installed,
    an ImportError will be raised.
    """
    try:
        from .embedding.word_llama_embedding import WordLlamaEmbedding

        return WordLlamaEmbedding.build(weights, config, dims=dims)
    except ImportError:
        logger.error(
            "Required dependencies for training are not installed. "
            "Please install wordllama with the 'train' extra."
        )
        raise
