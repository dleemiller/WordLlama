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
            embedding=np.random.rand(128256, 64),
            config=self.config,
            tokenizer=self.mock_tokenizer,
        )

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
        sim_score = self.model.similarity(
            "test string 1", "test string 2", use_hamming=False
        )
        self.assertTrue(isinstance(sim_score, float))

    def test_similarity_hamming(self):
        self.mock_tokenizer.encode_batch.return_value = [
            MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1]),
            MagicMock(ids=[4, 5, 6], attention_mask=[1, 1, 1]),
        ]
        sim_score = self.model.similarity(
            "test string 1", "test string 2", use_hamming=True
        )
        self.assertTrue(isinstance(sim_score, float))

    def test_rank_cosine(self):
        self.mock_tokenizer.encode_batch.return_value = [
            MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1]),
            MagicMock(ids=[4, 5, 6], attention_mask=[1, 1, 1]),
            MagicMock(ids=[7, 8, 9], attention_mask=[1, 1, 1]),
        ]
        docs = ["doc1", "doc2", "doc3"]
        ranked_docs = self.model.rank("test query", docs, use_hamming=False)
        self.assertEqual(len(ranked_docs), len(docs))
        self.assertTrue(all(isinstance(score, float) for doc, score in ranked_docs))

    def test_rank_hamming(self):
        self.mock_tokenizer.encode_batch.return_value = [
            MagicMock(ids=[1, 2, 3], attention_mask=[1, 1, 1]),
            MagicMock(ids=[4, 5, 6], attention_mask=[1, 1, 1]),
            MagicMock(ids=[7, 8, 9], attention_mask=[1, 1, 1]),
        ]
        docs = ["doc1", "doc2", "doc3"]
        ranked_docs = self.model.rank("test query", docs, use_hamming=True)
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
        binary_output = self.model.embed("test string", binarize=True, pack=True)
        self.assertIsInstance(binary_output, np.ndarray)
        self.assertEqual(binary_output.dtype, np.uint32)

    def test_normalization_effect(self):
        normalized_output = self.model.embed("test string", norm=True)
        norm = np.linalg.norm(normalized_output)
        self.assertAlmostEqual(norm, 1, places=5)

    def test_cosine_similarity_direct(self):
        vec1 = np.random.rand(64)
        vec2 = np.random.rand(64)
        result = WordLlamaInference.cosine_similarity(vec1, vec2)
        self.assertIsInstance(result.item(), float)

    def test_hamming_similarity_direct(self):
        vec1 = np.random.randint(2, size=64, dtype=np.uint32)
        vec2 = np.random.randint(2, size=64, dtype=np.uint32)
        result = WordLlamaInference.hamming_similarity(vec1, vec2)
        self.assertIsInstance(result.item(), float)


if __name__ == "__main__":
    unittest.main()
