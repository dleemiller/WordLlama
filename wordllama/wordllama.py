import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer
from typing import Union, List
import warnings
import pathlib
import logging
from wordllama.config import Config, WordLlamaConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WordLlama:
    def __init__(self, config: WordLlamaConfig):
        self.config = config
        self.tokenizer_kwargs = self.config.tokenizer.dict()

        # Load the tokenizer
        self.tokenizer = Tokenizer.from_pretrained(self.config.model.hf_model_id)
        self.tokenizer.enable_padding(length=self.tokenizer_kwargs["max_length"])

    @classmethod
    def build(cls, weights_file, config: WordLlamaConfig):
        with safe_open(weights_file, framework="np", device="cpu") as f:
            cls.embedding = f.get_tensor("embedding.weight")

        return cls(config)

    def tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.tokenizer.encode_batch(texts)

    def embed(self, texts, norm=False, binarize=False, pack=True, return_np=True):
        # Tokenize the texts
        encoded_texts = self.tokenize(texts)
        input_ids = np.array([enc.ids for enc in encoded_texts], dtype=np.int32)
        attention_mask = np.array(
            [enc.attention_mask for enc in encoded_texts], dtype=np.int32
        )

        # Clamp out-of-bounds input_ids
        input_ids = np.clip(input_ids, 0, self.embedding.shape[0] - 1)

        # Compute the embeddings
        x = self.embedding[input_ids]

        # Apply average pooling
        x = self.avg_pool(x, attention_mask, norm=norm)

        if binarize:
            x = x > 0
            if pack:
                x = np.packbits(x, axis=-1)
                x = x.view(np.uint32)  # Change to uint32

        if return_np:
            return x

        return x.tolist()

    @staticmethod
    def avg_pool(x, mask, norm=False):
        if norm:
            x = x / np.linalg.norm(x + 1e-9, axis=-1, keepdims=True)
        return np.sum(x * mask[..., np.newaxis], axis=1) / np.maximum(
            mask.sum(axis=1, keepdims=True) + 1e-9, 1
        )

    @staticmethod
    def hamming_similarity(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculate the Hamming similarity between a single vector and one or more vectors.

        Parameters:
        - a (np.ndarray): A single dimension vector of dtype np.uint32.
        - b (np.ndarray): A single dimension vector or a 2D array of dtype np.uint32 with shape (batch_size, vec_dim).

        Returns:
        - Union[float, np.ndarray]: The Hamming similarity as a float if b is a single dimension vector,
                                    or a numpy array of floats if b is a 2D array.
        """
        assert a.ndim == 1, "a must be a single dimension vector"
        assert a.dtype == np.uint32, "a must be of dtype np.uint32"
        assert b.dtype == np.uint32, "b must be of dtype np.uint32"

        if b.ndim == 1:
            b = np.expand_dims(b, axis=0)
        assert b.ndim == 2, "b must be a single dimension vector or a 2D array"

        max_dist = a.size * 32

        # Calculate Hamming distance
        xor_result = np.bitwise_xor(a, b)
        dist = np.sum(np.unpackbits(xor_result.view(np.uint8), axis=1), axis=1)

        return 1.0 - dist / max_dist

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculate the cosine similarity between a single vector and one or more vectors.

        Parameters:
        - a (np.ndarray): A single dimension vector of dtype float16, float32, or float64.
        - b (np.ndarray): A single dimension vector or a 2D array of dtype float16, float32, or float64.

        Returns:
        - Union[float, np.ndarray]: The cosine similarity as a float if b is a single dimension vector,
                                    or a numpy array of floats if b is a 2D array.
        """
        assert a.dtype in (
            np.float16,
            np.float32,
            np.float64,
        ), "Input vectors must be of type float16, float32, or float64."
        assert b.dtype in (
            np.float16,
            np.float32,
            np.float64,
        ), "Input vectors must be of type float16, float32, or float64."
        epsilon = 1e-9  # Small value to prevent division by zero

        if b.ndim == 1:
            b = np.expand_dims(b, axis=0)
        assert b.ndim == 2, "b must be a single dimension vector or a 2D array"

        assert a.ndim == 1
        a = np.expand_dims(a, axis=0)

        # Normalize the vectors
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)

        # Calculate cosine similarity
        cosine_sim = np.dot(a, b.T) / (a_norm * b_norm.T + epsilon)
        return cosine_sim.flatten()

    def similarity(self, text1, text2, use_hamming=False):
        """
        Compare two strings and return their similarity score.

        Parameters:
        - text1 (str): The first text.
        - text2 (str): The second text.
        - use_hamming (bool): If True, use Hamming similarity. Otherwise, use cosine similarity.

        Returns:
        - float: The similarity score.
        """
        if use_hamming:
            embedding1 = self.embed(text1, binarize=True, pack=True)
            embedding2 = self.embed(text2, binarize=True, pack=True)
            return self.hamming_similarity(embedding1[0], embedding2[0]).item()
        else:
            embedding1 = self.embed(text1)
            embedding2 = self.embed(text2)
            return self.cosine_similarity(embedding1[0], embedding2[0]).item()

    def rank(self, query, docs, use_hamming=False):
        """
        Rank a list of documents based on their similarity to a query.

        Parameters:
        - query (str): The query text.
        - docs (list of str): The list of document texts.
        - use_hamming (bool): If True, use Hamming similarity. Otherwise, use cosine similarity.

        Returns:
        - list of tuple: A list of (doc, score) tuples, sorted by score in descending order.
        """
        if use_hamming:
            query_embedding = self.embed(query, binarize=True, pack=True)
            doc_embeddings = self.embed(docs, binarize=True, pack=True)
            scores = self.hamming_similarity(query_embedding[0], doc_embeddings)
        else:
            query_embedding = self.embed(query)
            doc_embeddings = self.embed(docs)
            scores = self.cosine_similarity(query_embedding[0], doc_embeddings)

        similarities = list(zip(docs, scores.tolist()))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
