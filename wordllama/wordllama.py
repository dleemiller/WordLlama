import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer
import warnings
import pathlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WordLlama:
    tokenizer_kwargs = {
        "return_tensors": "np",
        "return_attention_mask": True,
        "max_length": 1024,
        "padding": "max_length",
        "truncation": True,
        "add_special_tokens": False,  # don't need without context
    }

    def __init__(self, config, tokenizer_kwargs=None):
        self.config = config
        if tokenizer_kwargs:
            self.tokenizer_kwargs = tokenizer_kwargs

        # Load the tokenizer
        self.tokenizer = Tokenizer.from_pretrained(config["model"]["hf_model_id"])
        self.tokenizer.enable_padding(length=self.tokenizer_kwargs["max_length"])

        # Load the embeddings from safetensors
        with safe_open(
            config["model"]["embedding_path"], framework="np", device="cpu"
        ) as f:
            self.embedding = f.get_tensor("embedding.weight")

    @classmethod
    def build(cls, weights_file, config):
        config["model"]["embedding_path"] = str(weights_file)
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
        if norm:
            x = self.avg_pool(x, attention_mask)
        else:
            x = np.sum(x * attention_mask[..., np.newaxis], axis=1) / np.maximum(
                attention_mask.sum(axis=1, keepdims=True), 1
            )

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
            x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        return np.sum(x * mask[..., np.newaxis], axis=1) / np.maximum(
            mask.sum(axis=1, keepdims=True), 1
        )

    @staticmethod
    def hamming_similarity(a, b):
        assert a.shape == b.shape
        assert a.ndim == 1
        assert a.dtype == np.uint32
        assert b.dtype == np.uint32

        max_dist = a.size * 32

        # Calculate Hamming distance
        dist = np.sum([bin(x).count("1") for x in a ^ b])

        return 1.0 - dist / max_dist

    @staticmethod
    def cosine_similarity(a, b):
        assert a.shape == b.shape, "Input vectors must have the same shape."
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

        epsilon = 1e-8  # Small value to prevent division by zero

        if a.ndim == 1:
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            return np.dot(a, b) / (a_norm * b_norm + epsilon)
        elif a.ndim == 2:
            a_norm = np.linalg.norm(a, axis=1)
            b_norm = np.linalg.norm(b, axis=1)
            return np.sum(a * b, axis=1) / (a_norm * b_norm + epsilon)
        else:
            raise ValueError("Input arrays must be either 1D or 2D.")

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
        embedding1 = self.embed(text1)
        embedding2 = self.embed(text2)

        if use_hamming:
            embedding1 = np.packbits(embedding1 > 0, axis=-1).view(np.uint32)
            embedding2 = np.packbits(embedding2 > 0, axis=-1).view(np.uint32)
            return self.hamming_similarity(embedding1[0], embedding2[0])
        else:
            return self.cosine_similarity(embedding1[0], embedding2[0])

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
        query_embedding = self.embed(query)
        doc_embeddings = self.embed(docs)

        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            if use_hamming:
                doc_embedding = np.packbits(doc_embedding > 0, axis=-1).view(np.uint32)
                query_embedding_packed = np.packbits(query_embedding > 0, axis=-1).view(
                    np.uint32
                )
                score = self.hamming_similarity(
                    query_embedding_packed[0], doc_embedding
                )
            else:
                score = self.cosine_similarity(query_embedding[0], doc_embedding)
            similarities.append((docs[i], score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
