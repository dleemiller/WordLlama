import numpy as np
from tokenizers import Tokenizer
from typing import Union, List
import logging

from wordllama.config import WordLlamaConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WordLlamaInference:
    def __init__(
        self, embedding: np.array, config: WordLlamaConfig, tokenizer: Tokenizer
    ):
        self.embedding = embedding
        self.config = config
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = self.config.tokenizer.model_dump()

        # Default settings for all inference
        self.tokenizer.enable_padding()
        self.tokenizer.no_truncation()

    def tokenize(self, texts: Union[str, List[str]]) -> List:
        """
        Tokenize input texts using the configured tokenizer.

        Args:
            texts (Union[str, List[str]]): Single string or list of strings to tokenize.

        Returns:
            List: List of tokenized and encoded text data.
        """
        if isinstance(texts, str):
            texts = [texts]
        else:
            assert isinstance(texts, list), "Input texts must be str or List[str]"

        return self.tokenizer.encode_batch(
            texts, is_pretokenized=False, add_special_tokens=False
        )

    def embed(
        self,
        texts: Union[str, List[str]],
        norm: bool = False,
        binarize: bool = False,
        pack: bool = True,
        return_np: bool = True,
    ) -> Union[np.ndarray, List]:
        """
        Generate embeddings for input texts with options for normalization and binarization.

        Args:
            texts (Union[str, List[str]]): Texts to embed.
            norm (bool): Apply normalization to embeddings.
            binarize (bool): Convert embeddings to binary format.
            pack (bool): Pack binary data into bits.
            return_np (bool): Return result as a numpy array if True, otherwise as a list.

        Returns:
            Union[np.ndarray, List]: Embeddings as numpy array or list.
        """
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
    def avg_pool(x: np.ndarray, mask: np.ndarray, norm: bool = False) -> np.ndarray:
        """
        Apply average pooling to the embeddings.

        Args:
            x (np.ndarray): The input embeddings.
            mask (np.ndarray): The attention mask indicating which tokens to consider.
            norm (bool): Whether to normalize the embeddings.

        Returns:
            np.ndarray: The pooled embeddings.
        """
        x = np.sum(x * mask[..., np.newaxis], axis=1) / np.maximum(
            mask.sum(axis=1, keepdims=True) + 1e-9, 1
        )

        if norm:
            x = x / np.linalg.norm(x + 1e-9, axis=-1, keepdims=True)
        return x

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

        return 1.0 - 2.0 * (dist / max_dist)

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
