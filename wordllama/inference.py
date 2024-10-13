import numpy as np
from tokenizers import Tokenizer
from typing import Union, List, Tuple, Optional
import logging

from .algorithms import (
    kmeans_clustering,
    hamming_distance,
    binarize_and_packbits,
    process_batches_cy,
)
from .algorithms.semantic_splitter import SemanticSplitter
from .config import WordLlamaConfig
from .mode_decorators import dense_only

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WordLlamaInference:
    def __init__(
        self,
        embedding: np.ndarray,
        config: WordLlamaConfig,
        tokenizer: Tokenizer,
        binary: bool = False,
    ):
        """Initialize WordLlamaInference with embeddings and configurations.

        Args:
            embedding (np.ndarray): The embedding matrix of shape (vocab_size, embedding_dim).
            config (WordLlamaConfig): The configuration object.
            tokenizer (Tokenizer): The tokenizer to use for encoding texts.
            binary (bool, optional): Whether to use binary embeddings. Defaults to False.
        """
        self.binary = binary
        self.embedding = np.ascontiguousarray(embedding.astype(np.float32))
        self.config = config
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = self.config.tokenizer.model_dump()

        # Default settings for all inference
        self.tokenizer.enable_padding()
        self.tokenizer.no_truncation()

    def tokenize(self, texts: Union[str, List[str]]) -> List:
        """Tokenize input texts using the configured tokenizer.

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
        batch_size: int = 32,
    ) -> Union[np.ndarray, List]:
        """Generate embeddings for input texts with optional normalization and binarization.

        Args:
            texts (Union[str, List[str]]): Text or list of texts to embed.
            norm (bool, optional): If True, normalize embeddings to unit vectors. Defaults to False.
            return_np (bool, optional): If True, return embeddings as a NumPy array; otherwise, return as a list. Defaults to True.
            pool_embeddings (bool, optional): If True, apply average pooling to token embeddings. Defaults to True.
            batch_size (int, optional): Number of texts to process in each batch. Defaults to 32.

        Returns:
            Union[np.ndarray, List]: Embeddings as a NumPy array or a list, depending on `return_np`.
        """
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise TypeError("Input 'texts' must be a string or a list of strings")

        if texts and not isinstance(texts[0], str):
            raise TypeError("All elements in 'texts' must be strings")

        # Preallocate final embeddings array
        num_texts = len(texts)
        embedding_dim = self.embedding.shape[1]
        np_type = np.float32 if not self.binary else np.uint64
        pooled_embeddings = np.empty((num_texts, embedding_dim), dtype=np_type)

        for i in range(0, num_texts, batch_size):
            chunk = texts[i : i + batch_size]

            # Tokenize the texts
            encoded_texts = self.tokenize(chunk)
            input_ids = np.array([enc.ids for enc in encoded_texts], dtype=np.int32)
            attention_mask = np.array(
                [enc.attention_mask for enc in encoded_texts], dtype=np.float32
            )

            # Clamp out-of-bounds input_ids
            np.clip(input_ids, 0, self.embedding.shape[0] - 1, out=input_ids)

            # Compute embeddings in batch
            batch_embeddings = self.embedding[input_ids]

            # Apply average pooling to the batch
            if pool_embeddings:
                batch_embeddings = self.avg_pool(batch_embeddings, attention_mask)

            # Normalize embeddings after pooling
            if norm:
                batch_embeddings /= np.linalg.norm(
                    batch_embeddings, axis=1, keepdims=True
                )

            # Binarize embeddings
            if self.binary:
                batch_embeddings = binarize_and_packbits(batch_embeddings)

            # Store the computed embeddings in preallocated array
            pooled_embeddings[i : i + batch_embeddings.shape[0]] = batch_embeddings

        # Return embeddings as NumPy array or list based on user preference
        if return_np:
            return pooled_embeddings

        return pooled_embeddings.tolist()

    @staticmethod
    def avg_pool(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply average pooling to token embeddings using an attention mask.

        Args:
            x (np.ndarray): Token embeddings of shape (batch_size, seq_length, embedding_dim).
            mask (np.ndarray): Attention mask of shape (batch_size, seq_length), indicating which tokens to include.

        Returns:
            np.ndarray: Pooled embeddings of shape (batch_size, embedding_dim).
        """
        # Ensure mask is float32 to avoid promotion
        mask_sum = np.maximum(mask.sum(axis=1, keepdims=True), 1.0).astype(np.float32)
        return np.sum(x * mask[..., np.newaxis], axis=1, dtype=np.float32) / mask_sum

    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors.

        Args:
            embeddings (np.ndarray): The input embeddings.

        Returns:
            np.ndarray: Normalized embeddings.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    @staticmethod
    def hamming_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate the Hamming similarity between binary vectors.

        Args:
            a (np.ndarray): A 2D array of binary vectors (dtype np.uint64).
            b (np.ndarray): A 2D array of binary vectors (dtype np.uint64).

        Returns:
            np.ndarray: A 2D array containing the Hamming similarity scores between vectors in `a` and `b`.
        """
        max_dist = a.shape[1] * 64

        # Calculate Hamming distance
        dist = hamming_distance(a, b).astype(np.float32)
        return 1.0 - 2.0 * (dist / max_dist)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate the cosine similarity between dense vectors.

        Args:
            a (np.ndarray): A 2D array of dense vectors.
            b (np.ndarray): A 2D array of dense vectors.

        Returns:
            np.ndarray: A 2D array containing the cosine similarity scores between vectors in `a` and `b`.
        """
        # Normalize the vectors
        if a.shape == b.shape and (a == b).all():
            a = WordLlamaInference.normalize_embeddings(a)
            b = a
        else:
            a = WordLlamaInference.normalize_embeddings(a)
            b = WordLlamaInference.normalize_embeddings(b)

        # Calculate cosine similarity
        return a @ b.T

    def vector_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate similarity between vectors, using either Hamming or cosine similarity.

        Args:
            a (np.ndarray): A 1D or 2D array of vectors.
            b (np.ndarray): A 1D or 2D array of vectors.

        Returns:
            np.ndarray: A 2D array of similarity scores between vectors in `a` and `b`.
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
        """Compute the similarity score between two texts.

        Args:
            text1 (str): The first text.
            text2 (str): The second text.

        Returns:
            float: The similarity score between `text1` and `text2`.
        """
        embedding1 = self.embed(text1)
        embedding2 = self.embed(text2)
        return self.vector_similarity(embedding1[0], embedding2[0]).item()

    def rank(self, query: str, docs: List[str], sort: bool = True) -> List[Tuple[str, float]]:
        """Rank documents based on their similarity to a query.

        Result may be sorted by similarity score in descending order, or not (see `sort` parameter)

        Args:
            query (str): The query text.
            docs (List[str]): The list of document texts to rank.
            sort (bool): Sort documents by similarity, or not (respect the order in `docs`)

        Returns:
            List[Tuple[str, float]]: A list of tuples `(doc, score)`.
        """
        assert isinstance(query, str), "Query must be a string"
        assert (
            isinstance(docs, list) and len(docs) > 1
        ), "Docs must be a list of 2 more more strings."
        query_embedding = self.embed(query)
        doc_embeddings = self.embed(docs)
        scores = self.vector_similarity(query_embedding[0], doc_embeddings)

        scores = np.atleast_1d(scores.squeeze())
        similarities = list(zip(docs, scores.tolist()))
        if sort:
            similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    def deduplicate(
        self, docs: List[str], threshold: float = 0.9, batch_size: Optional[int] = None
    ) -> List[str]:
        """Deduplicate documents based on a similarity threshold.

        Args:
            docs (List[str]): List of documents to deduplicate.
            threshold (float, optional): Similarity threshold above which documents are considered duplicates. Defaults to 0.9.
            batch_size (Optional[int], optional): Batch size for processing embeddings. Defaults to None.

        Returns:
            List[str]: A list of unique documents after deduplication.
        """
        doc_embeddings = self.embed(docs, norm=not self.binary)

        if batch_size is None:
            batch_size = 500 if self.binary else 5000
        duplicate_indices = process_batches_cy(
            doc_embeddings, threshold, batch_size, self.vector_similarity
        )

        unique_docs = [
            doc for idx, doc in enumerate(docs) if idx not in duplicate_indices
        ]
        return unique_docs

    def topk(self, query: str, candidates: List[str], k: int = 3) -> List[str]:
        """Retrieve the top-k most similar documents to a query.

        Args:
            query (str): The query text.
            candidates (List[str]): The list of candidate documents.
            k (int, optional): The number of top documents to return. Defaults to 3.

        Returns:
            List[str]: The top-k documents most similar to the query.
        """
        assert (
            len(candidates) > k
        ), f"Number of candidates ({len(candidates)}) must be greater than k ({k})"
        ranked_docs = self.rank(query, candidates)
        return [doc for doc, score in ranked_docs[:k]]

    def filter(
        self, query: str, candidates: List[str], threshold: float = 0.3
    ) -> List[str]:
        """Filter documents to include only those similar to the query above a threshold.

        Args:
            query (str): The query text.
            candidates (List[str]): The list of candidate documents.
            threshold (float, optional): The similarity threshold for filtering. Defaults to 0.3.

        Returns:
            List[str]: The documents that have a similarity score above the threshold.
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

    @dense_only
    def cluster(
        self,
        docs: List[str],
        k: int,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        n_init: int = 10,
        min_iterations: int = 5,
        random_state=None,
    ) -> Tuple[List[int], float]:
        """Cluster documents into `k` clusters using KMeans clustering.

        Args:
            docs (List[str]): The list of documents to cluster.
            k (int): The number of clusters.
            max_iterations (int, optional): Maximum number of iterations for the clustering algorithm. Defaults to 100.
            tolerance (float, optional): Convergence tolerance. Defaults to 1e-4.
            n_init (int, optional): Number of times the algorithm is run with different centroid seeds. Defaults to 10.
            min_iterations (int, optional): Minimum number of iterations before checking for convergence. Defaults to 5.
            random_state (Optional[int or np.random.RandomState], optional): Random state for reproducibility. Defaults to None.

        Returns:
            Tuple[List[int], float]: A tuple containing a list of cluster labels for each document and the final inertia (sum of squared distances to cluster centers).
        """
        embeddings = self.embed(docs, norm=True)
        assert isinstance(docs, list), "`docs` must be a list of strings"
        assert len(docs) >= k, "number of clusters cannot be larger than len(docs)"
        assert isinstance(docs[0], str), "`docs` must be a list of strings"

        cluster_labels, inertia = kmeans_clustering(
            embeddings,
            k,
            max_iterations=max_iterations,
            tolerance=tolerance,
            n_init=n_init,
            min_iterations=min_iterations,
            random_state=random_state,
        )
        return cluster_labels, inertia

    @dense_only
    def split(
        self,
        text: str,
        target_size: int = 1536,
        window_size: int = 3,
        poly_order: int = 2,
        savgol_window: int = 3,
        cleanup_size: int = 24,
        intermediate_size: int = 96,
        return_minima: bool = False,
    ) -> List[str]:
        """Split text into semantically coherent chunks.

        Args:
            text (str): The input text to split.
            target_size (int, optional): Desired size of text chunks. Defaults to 1536.
            window_size (int, optional): Window size for similarity matrix averaging. Defaults to 3.
            poly_order (int, optional): Polynomial order for Savitzky-Golay filter. Defaults to 2.
            savgol_window (int, optional): Window size for Savitzky-Golay filter. Defaults to 3.
            cleanup_size (int, optional): Size for cleanup operations. Defaults to 24.
            intermediate_size (int, optional): Intermediate size for initial splitting. Defaults to 96.
            return_minima (bool, optional): If True, return the indices of minima instead of chunks. Defaults to False.

        Returns:
            List[str]: List of semantically split text chunks.
        """
        # Split text
        lines = SemanticSplitter.split(
            text,
            target_size=target_size,
            intermediate_size=intermediate_size,
            cleanup_size=cleanup_size,
        )

        # Embed lines and normalize
        embeddings = self.embed(lines, norm=True)

        # Reconstruct text with similarity signals
        return SemanticSplitter.reconstruct(
            lines,
            embeddings,
            target_size=target_size,
            window_size=window_size,
            poly_order=poly_order,
            savgol_window=savgol_window,
            return_minima=return_minima,
        )
