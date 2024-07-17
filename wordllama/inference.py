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
        self,
        embedding: np.array,
        config: WordLlamaConfig,
        tokenizer: Tokenizer,
        binary: bool = False,
    ):
        self.binary = binary
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
        return_np: bool = True,
        pool_embeddings: bool = True,
    ) -> Union[np.ndarray, List]:
        """
        Generate embeddings for input texts with options for normalization and binarization.

        Args:
            texts (Union[str, List[str]]): Texts to embed.
            norm (bool): Apply normalization to embeddings.
            return_np (bool): Return result as a numpy array if True, otherwise as a list.
            pool_embeddings (bool): Apply average pooling to embeddings.

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
        if not pool_embeddings:
            return x

        # Apply average pooling
        x = self.avg_pool(x, attention_mask)

        if norm:
            x = self.normalize_embeddings(x)

        if self.binary:
            x = x > 0
            x = np.packbits(x, axis=-1)
            x = x.view(np.uint32)  # Change to uint32

        if return_np:
            return x

        return x.tolist()

    @staticmethod
    def avg_pool(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply average pooling to the embeddings.

        Args:
            x (np.ndarray): The input embeddings.
            mask (np.ndarray): The attention mask indicating which tokens to consider.

        Returns:
            np.ndarray: The pooled embeddings.
        """
        x = np.sum(x * mask[..., np.newaxis], axis=1) / np.maximum(
            mask.sum(axis=1, keepdims=True), 1
        )

        return x

    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit vectors.

        Args:
            embeddings (np.ndarray): The input embeddings.

        Returns:
            np.ndarray: Normalized embeddings.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    @staticmethod
    def hamming_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calculate the Hamming similarity between vectors.

        Parameters:
        - a (np.ndarray): A 2D array of dtype np.uint32.
        - b (np.ndarray): A 2D array of dtype np.uint32.

        Returns:
        - np.ndarray: A 2D array of Hamming similarity scores.
        """
        max_dist = a.shape[1] * 32

        # Calculate Hamming distance
        xor_result = np.bitwise_xor(a[:, np.newaxis], b)
        dist = np.sum(np.unpackbits(xor_result.view(np.uint8), axis=2), axis=2)

        return 1.0 - 2.0 * (dist / max_dist)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calculate the cosine similarity between vectors.

        Parameters:
        - a (np.ndarray): A 2D array of dtype float16, float32, or float64.
        - b (np.ndarray): A 2D array of dtype float16, float32, or float64.

        Returns:
        - np.ndarray: A 2D array of cosine similarity scores.
        """
        # Normalize the vectors
        a = WordLlamaInference.normalize_embeddings(a)
        b = WordLlamaInference.normalize_embeddings(b)

        # Calculate cosine similarity
        return a @ b.T

    def vector_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calculate the similarity between vectors based on the binary attribute.

        Parameters:
        - a (np.ndarray): A 1D or 2D array of vectors.
        - b (np.ndarray): A 1D or 2D array of vectors.

        Returns:
        - np.ndarray: A 2D array of similarity scores.
        """
        if a.ndim == 1:
            a = np.expand_dims(a, axis=0)
        if b.ndim == 1:
            b = np.expand_dims(b, axis=0)

        assert a.ndim == 2, "a must be a 2D array"
        assert b.ndim == 2, "b must be a 2D array"

        if self.binary:
            return self.hamming_similarity(a, b)
        else:
            return self.cosine_similarity(a, b)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compare two strings and return their similarity score.

        Parameters:
        - text1 (str): The first text.
        - text2 (str): The second text.

        Returns:
        - float: The similarity score.
        """
        embedding1 = self.embed(text1)
        embedding2 = self.embed(text2)
        return self.vector_similarity(embedding1[0], embedding2[0]).item()

    def rank(self, query: str, docs: List[str]) -> List[tuple]:
        """
        Rank a list of documents based on their similarity to a query.

        Parameters:
        - query (str): The query text.
        - docs (list of str): The list of document texts.

        Returns:
        - list of tuple: A list of (doc, score) tuples, sorted by score in descending order.
        """
        assert isinstance(query, str), "Query must be a string"
        query_embedding = self.embed(query)
        doc_embeddings = self.embed(docs)
        scores = self.vector_similarity(query_embedding[0], doc_embeddings)

        scores = scores.squeeze()
        similarities = list(zip(docs, scores.tolist()))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    def deduplicate(
        self, docs: List[str], threshold: float = 0.9, batch_size: int = 100
    ) -> List[str]:
        """
        Deduplicate a list of documents based on similarity threshold.

        Args:
            docs (List[str]): List of document texts to deduplicate.
            threshold (float): Similarity threshold for deduplication.
            batch_size (int): Batch size for processing embeddings.

        Returns:
            List[str]: Deduplicated list of document texts.
        """
        # Embed all documents
        doc_embeddings = self.embed(docs, norm=not self.binary)

        num_embeddings = doc_embeddings.shape[0]
        duplicate_indices = set()
        seen_docs = set()

        for i in range(0, num_embeddings, batch_size):
            start_i = i
            end_i = min(i + batch_size, num_embeddings)
            batch_i = doc_embeddings[start_i:end_i]

            for j in range(
                i, num_embeddings, batch_size
            ):  # Start from i to avoid redundant comparisons
                start_j = j
                end_j = min(j + batch_size, num_embeddings)
                batch_j = doc_embeddings[start_j:end_j]

                sim_matrix = self.vector_similarity(batch_i, batch_j)

                # Find indices where similarity exceeds the threshold
                sim_indices = np.argwhere(sim_matrix > threshold)
                for idx in sim_indices:
                    if idx[0] + start_i != idx[1] + start_j:  # Ignore self-comparison
                        doc_idx_1 = idx[0] + start_i
                        doc_idx_2 = idx[1] + start_j
                        if doc_idx_2 not in seen_docs:
                            seen_docs.add(doc_idx_1)
                            duplicate_indices.add(doc_idx_2)

        # Filter out embeddings that are not in duplicate_indices
        unique_docs = [
            doc for idx, doc in enumerate(docs) if idx not in duplicate_indices
        ]
        return unique_docs

    def topk(self, query: str, candidates: List[str], k: int = 3) -> List[str]:
        """
        Retrieve the top-k documents based on their similarity to the query.

        Parameters:
        - query (str): The query text.
        - candidates (list of str): The list of candidate document texts.
        - k (int): The number of top documents to return.

        Returns:
        - list of str: The top-k document texts.
        """
        assert (
            len(candidates) > k
        ), f"Number of candidates ({len(candidates)}) must be greater than k ({k})"
        ranked_docs = self.rank(query, candidates)
        return [doc for doc, score in ranked_docs[:k]]

    def filter(
        self, query: str, candidates: List[str], threshold: float = 0.3
    ) -> List[str]:
        """
        Filter documents based on their similarity to the query.

        Parameters:
        - query (str): The query text.
        - candidates (list of str): The list of candidate document texts.
        - threshold (float): The similarity threshold for filtering.

        Returns:
        - list of str: The filtered document texts.
        """
        query_embedding = self.embed(query)
        candidate_embeddings = self.embed(candidates)
        similarity_scores = self.vector_similarity(
            query_embedding[0], candidate_embeddings
        ).squeeze()

        filtered_docs = [
            doc
            for doc, score in zip(candidates, similarity_scores)
            if score > threshold
        ]
        return filtered_docs
