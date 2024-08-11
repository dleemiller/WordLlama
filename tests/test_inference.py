import unittest
from unittest.mock import patch, MagicMock, create_autospec
import numpy as np
from wordllama.inference import WordLlamaInference
from wordllama.config import (
    WordLlamaConfig,
    WordLlamaModel,
    TokenizerConfig,
    TrainingConfig,
    MatryoshkaConfig,
    TokenizerInferenceConfig,
)


class TestWordLlamaInference(unittest.TestCase):
    @patch("wordllama.inference.Tokenizer.from_pretrained")
    def setUp(self, mock_tokenizer):
        # Mock the tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.encode_batch.return_value = [
            MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1])
        ]
        mock_tokenizer.return_value = self.mock_tokenizer

        # Example config using Pydantic models
        tokenizer_inference_config = TokenizerInferenceConfig(
            use_local_config=True, config_filename="tokenizer_config.json"
        )

        model_config = WordLlamaModel(
            n_vocab=32000,
            dim=64,
            n_layers=12,
            n_heads=12,
            hf_model_id="meta-llama/Meta-Llama-3-8B",
            pad_token="",
        )

        tokenizer_config = TokenizerConfig(
            return_tensors="pt",
            return_attention_mask=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            inference=tokenizer_inference_config,
        )

        training_config = TrainingConfig(
            output_dir="output/matryoshka_sts_custom",
            num_train_epochs=2,
            per_device_train_batch_size=512,
            warmup_steps=256,
            evaluation_strategy="steps",
            eval_steps=250,
            save_steps=1000,
            fp16=True,
            include_num_input_tokens_seen=False,
            learning_rate=0.01,
            multi_dataset_batch_sampler="PROPORTIONAL",
            binarizer_ste="tanh",
        )

        matryoshka_config = MatryoshkaConfig(dims=[1024, 512, 256, 128, 64])

        self.config = WordLlamaConfig(
            model=model_config,
            tokenizer=tokenizer_config,
            training=training_config,
            matryoshka=matryoshka_config,
        )

        self.model = WordLlamaInference(
            embedding=np.random.rand(32000, 64),
            config=self.config,
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
        self.assertEqual(len(deduplicated_docs), 2)
        self.assertIn("doc1", deduplicated_docs)
        self.assertIn("a second document that is different", deduplicated_docs)

    # @patch.object(
    #    WordLlamaInference,
    #    "embed",
    #    return_value=np.array(
    #        [[1, 2, 3], [1, 2, 3], [4, 5, 6], [3, 2, 3]], dtype=np.uint64
    #    ),
    # )
    # def test_deduplicate_hamming(self, mock_embed):
    #    docs = ["doc1", "doc1_dup", "doc2", "doc1_dup2"]
    #    self.model.binary = True
    #    deduplicated_docs = self.model.deduplicate(docs, threshold=0.9)
    #    self.assertEqual(len(deduplicated_docs), 2)
    #    self.assertIn("doc1", deduplicated_docs)
    #    self.assertIn("doc2", deduplicated_docs)

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
        self.assertEqual(len(deduplicated_docs), 3)
        self.assertIn("doc1", deduplicated_docs)
        self.assertIn("doc2", deduplicated_docs)
        self.assertIn("doc3", deduplicated_docs)

    @patch.object(
        WordLlamaInference,
        "embed",
        return_value=np.array([[0.1] * 64, [0.1] * 64, [0.1] * 64], dtype=np.float32),
    )
    def test_deduplicate_all_duplicates(self, mock_embed):
        docs = ["doc1", "doc1_dup", "doc1_dup2"]
        deduplicated_docs = self.model.deduplicate(docs, threshold=0.9)
        self.assertEqual(len(deduplicated_docs), 1)
        self.assertIn("doc1", deduplicated_docs)

    def test_tokenize(self):
        self.mock_tokenizer.encode_batch.return_value = [
            MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1])
        ]
        tokens = self.model.tokenize("test string")
        self.mock_tokenizer.encode_batch.assert_called_with(
            ["test string"], is_pretokenized=False, add_special_tokens=False
        )
        self.assertEqual(len(tokens), 1)

    def test_embed(self):
        self.mock_tokenizer.encode_batch.return_value = [
            MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1])
        ]
        embeddings = self.model.embed("test string", return_np=True)
        self.assertEqual(embeddings.shape, (1, 64))

    def test_similarity_cosine(self):
        self.mock_tokenizer.encode_batch.return_value = [
            MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1]),
            MagicMock(ids=[4, 5, 6], attention_mask=[1, 1, 1]),
        ]
        sim_score = self.model.similarity("test string 1", "test string 2")
        self.assertTrue(isinstance(sim_score, float))

    def test_similarity_hamming(self):
        self.mock_tokenizer.encode_batch.return_value = [
            MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1]),
            MagicMock(ids=[4, 5, 6], attention_mask=[1, 1, 1]),
        ]

        self.model.binary = True
        sim_score = self.model.similarity("test string 1", "test string 2")
        self.assertTrue(isinstance(sim_score, float))

    def test_vector_similarity(self):
        self.mock_tokenizer.encode_batch.return_value = [
            MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1]),
            MagicMock(ids=[4, 5, 6], attention_mask=[1, 1, 1]),
        ]

        with patch.object(self.model, "cosine_similarity") as mock_cosine:
            mock_cosine.return_value = np.array([[0.5]])
            self.model.binary = False
            sim_score = self.model.similarity("test string 1", "test string 2")
            self.assertTrue(isinstance(sim_score, float))
            mock_cosine.assert_called_once()

        with patch.object(self.model, "hamming_similarity") as mock_hamming:
            mock_hamming.return_value = np.array([[0.5]])
            self.model.binary = True
            sim_score = self.model.similarity("test string 1", "test string 2")
            self.assertTrue(isinstance(sim_score, float))
            mock_hamming.assert_called_once()

    def test_rank_cosine(self):
        self.mock_tokenizer.encode_batch.return_value = [
            MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1]),
            MagicMock(ids=[4, 5, 6], attention_mask=[1, 1, 1]),
            MagicMock(ids=[7, 8, 9], attention_mask=[1, 1, 1]),
        ]
        docs = ["doc1", "doc2", "doc3"]
        ranked_docs = self.model.rank("test query", docs)
        self.assertEqual(len(ranked_docs), len(docs))
        self.assertTrue(all(isinstance(score, float) for doc, score in ranked_docs))

    def test_rank_hamming(self):
        self.mock_tokenizer.encode_batch.return_value = [
            MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1]),
            MagicMock(ids=[4, 5, 6], attention_mask=[1, 1, 1]),
            MagicMock(ids=[7, 8, 9], attention_mask=[1, 1, 1]),
        ]
        docs = ["doc1", "doc2", "doc3"]

        with patch.object(self.model, "hamming_similarity") as mock_hamming:
            mock_hamming.return_value = np.array([[0.5, 0.7, 0.6]])
            self.model.binary = True
            ranked_docs = self.model.rank("test query", docs)

            mock_hamming.assert_called_once()  # check setting binary calls hamming
            self.assertEqual(len(ranked_docs), len(docs))
            self.assertTrue(all(isinstance(score, float) for doc, score in ranked_docs))

    def test_instantiate_with_truncation(self):
        truncated_embedding = np.random.rand(128256, 32)
        truncated_model = WordLlamaInference(
            embedding=truncated_embedding,
            config=self.config,
            tokenizer=self.mock_tokenizer,
        )
        self.assertEqual(truncated_model.embedding.shape[1], 32)

    def test_error_on_wrong_embedding_type(self):
        with self.assertRaises(AssertionError):
            self.model.embed(np.array([1, 2]))

    def test_binarization_and_packing(self):
        self.model.binary = True
        binary_output = self.model.embed("test string")
        self.assertIsInstance(binary_output, np.ndarray)
        self.assertEqual(binary_output.dtype, np.uint64)

    def test_normalization_effect(self):
        normalized_output = self.model.embed("test string", norm=True)
        norm = np.linalg.norm(normalized_output)
        self.assertAlmostEqual(norm, 1, places=5)

    def test_cosine_similarity_direct(self):
        vec1 = np.random.rand(1, 64)
        vec2 = np.random.rand(1, 64)
        result = WordLlamaInference.cosine_similarity(vec1, vec2)
        self.assertIsInstance(result.item(), float)

    def test_hamming_similarity_direct(self):
        vec1 = np.expand_dims(np.random.randint(2, size=64, dtype=np.uint64), axis=0)
        vec2 = np.expand_dims(np.random.randint(2, size=64, dtype=np.uint64), axis=0)
        result = WordLlamaInference.hamming_similarity(vec1, vec2)
        self.assertIsInstance(result.item(), float)


if __name__ == "__main__":
    unittest.main()
