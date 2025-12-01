import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wordllama.inference import WordLlamaInference

np.random.seed(42)


class TestWordLlamaInference(unittest.TestCase):
    @patch("wordllama.inference.Tokenizer.from_pretrained")
    def setUp(self, mock_tokenizer):
        np.random.seed(42)

        # Mock the tokenizer
        self.mock_tokenizer = MagicMock()

        def mock_encode_batch(texts, *args, **kwargs):
            return [MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1]) for _ in texts]

        self.mock_tokenizer.encode_batch.side_effect = mock_encode_batch
        mock_tokenizer.return_value = self.mock_tokenizer

        self.model = WordLlamaInference(
            embedding=np.random.rand(32000, 64),
            tokenizer=self.mock_tokenizer,
        )

    @patch.object(
        WordLlamaInference,
        "embed",
        return_value=np.array(
            [[0.1] * 64, [0.1] * 64, np.random.rand(64), [0.1] * 64], dtype=np.float32
        ),
    )
    def test_deduplicate_cosine(self, mock_embed):
        docs = ["doc1", "doc1_dup", "a second document that is different", "doc1_dup2"]
        deduplicated_docs = self.model.deduplicate(docs, threshold=0.9)
        assert len(deduplicated_docs) == 2
        assert "doc1" in deduplicated_docs
        assert "a second document that is different" in deduplicated_docs

    @patch.object(
        WordLlamaInference,
        "embed",
        return_value=np.array(
            [
                [0.1] * 64,
                np.concatenate([np.random.rand(32), np.zeros(32)], axis=0),
                np.concatenate([np.zeros(32), np.random.rand(32)]),
            ],
            dtype=np.float32,
        ),
    )
    def test_deduplicate_no_duplicates(self, mock_embed):
        docs = ["doc1", "doc2", "doc3"]
        deduplicated_docs = self.model.deduplicate(docs, threshold=0.9)
        assert len(deduplicated_docs) == 3
        assert "doc1" in deduplicated_docs
        assert "doc2" in deduplicated_docs
        assert "doc3" in deduplicated_docs

    @patch.object(
        WordLlamaInference,
        "embed",
        return_value=np.array([[0.1] * 64, [0.1] * 64, [0.1] * 64], dtype=np.float32),
    )
    def test_deduplicate_all_duplicates(self, mock_embed):
        docs = ["doc1", "doc1_dup", "doc1_dup2"]
        deduplicated_docs = self.model.deduplicate(docs, threshold=0.9)
        assert len(deduplicated_docs) == 1
        assert "doc1" in deduplicated_docs

    @patch.object(
        WordLlamaInference,
        "embed",
        return_value=np.array([[0.1] * 64, [0.1] * 64, [0.1] * 64], dtype=np.float32),
    )
    def test_deduplicate_return_indices(self, mock_embed):
        docs = ["doc1", "doc1_dup", "doc1_dup2"]
        duplicated_idx = self.model.deduplicate(docs, return_indices=True, threshold=0.9)
        assert len(duplicated_idx) == 2
        assert 1 in duplicated_idx
        assert 2 in duplicated_idx

    def test_tokenize(self):
        tokens = self.model.tokenize("test string")
        self.mock_tokenizer.encode_batch.assert_called_with(
            ["test string"], is_pretokenized=False, add_special_tokens=False
        )
        assert len(tokens) == 1

    def test_embed(self):
        embeddings = self.model.embed("test string", return_np=True)
        assert embeddings.shape == (1, 64)

    def test_cluster_fails_binary(self):
        self.model.binary = True
        with pytest.raises(ValueError):
            self.model.cluster(["a", "b", "c"])

    def test_split_fails_binary(self):
        self.model.binary = True
        with pytest.raises(ValueError):
            self.model.split("a" * 1000)

    def test_similarity_cosine(self):
        def mock_encode_batch(texts, *args, **kwargs):
            return [MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1]) for _ in texts]

        self.mock_tokenizer.encode_batch.side_effect = mock_encode_batch
        sim_score = self.model.similarity("test string 1", "test string 2")
        assert isinstance(sim_score, float)

    def test_similarity_hamming(self):
        def mock_encode_batch(texts, *args, **kwargs):
            return [MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1]) for _ in texts]

        self.mock_tokenizer.encode_batch.side_effect = mock_encode_batch

        self.model.binary = True
        sim_score = self.model.similarity("test string 1", "test string 2")
        assert isinstance(sim_score, float)

    def test_rank_cosine(self):
        def mock_encode_batch(texts, *args, **kwargs):
            return [
                MagicMock(ids=[i + 1, i + 2, i + 3], attention_mask=[1, 1, 1])
                for i, _ in enumerate(texts)
            ]

        self.mock_tokenizer.encode_batch.side_effect = mock_encode_batch

        # Mock embeddings to be slightly different for each document
        def mock_embed(texts, *args, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            embeddings = []
            for i, _text in enumerate(texts):
                embedding = np.zeros(64, dtype=np.float32)
                embedding[1 if len(texts) == 1 else i] = 1
                embeddings.append(embedding)
            return np.vstack(embeddings)

        self.model.embed = mock_embed

        docs = ["doc1", "doc2", "doc3"]
        ranked_docs = self.model.rank("test query", docs)
        assert len(ranked_docs) == len(docs)
        assert all(isinstance(score, float) for doc, score in ranked_docs)
        assert ranked_docs[0] == ("doc2", 1.0)

        # test turning off sorting
        unsorted_docs = self.model.rank("test query", docs, sort=False)
        assert len(unsorted_docs) == len(docs)
        assert all(isinstance(score, float) for doc, score in unsorted_docs)
        assert unsorted_docs[1] == ("doc2", 1.0)

    def test_rank_hamming(self):
        def mock_encode_batch(texts, *args, **kwargs):
            return [MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1]) for _ in texts]

        self.mock_tokenizer.encode_batch.side_effect = mock_encode_batch
        docs = ["doc1", "doc2", "doc3"]

        with patch.object(self.model, "vector_similarity") as mock_hamming:
            mock_hamming.return_value = np.array([[0.5, 0.7, 0.6]])
            self.model.binary = True
            ranked_docs = self.model.rank("test query", docs)

            mock_hamming.assert_called_once()  # check setting binary calls hamming
            assert len(ranked_docs) == len(docs)
            assert all(isinstance(score, float) for doc, score in ranked_docs)

    def test_instantiate_with_truncation(self):
        truncated_embedding = np.random.rand(128256, 32)
        truncated_model = WordLlamaInference(
            embedding=truncated_embedding,
            tokenizer=self.mock_tokenizer,
        )
        assert truncated_model.embedding.shape[1] == 32

    def test_error_on_wrong_embedding_type(self):
        with pytest.raises(TypeError):
            self.model.embed(np.array([1, 2]))

    def test_binarization_and_packing(self):
        self.model.binary = True
        binary_output = self.model.embed("test string")
        assert isinstance(binary_output, np.ndarray)
        assert binary_output.dtype == np.uint64

    def test_normalization_effect(self):
        normalized_output = self.model.embed("test string", norm=True)
        norm = np.linalg.norm(normalized_output)
        assert norm == pytest.approx(1, abs=1e-5)


if __name__ == "__main__":
    unittest.main()
