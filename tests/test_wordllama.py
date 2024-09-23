import unittest
from unittest.mock import patch, MagicMock, mock_open, create_autospec
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from wordllama.wordllama import WordLlama, WordLlamaInference
from wordllama.config import (
    WordLlamaConfig,
    TokenizerConfig,
    MatryoshkaConfig,
    WordLlamaModel,
    TrainingConfig,
    TokenizerInferenceConfig,
)


class TestWordLlama(unittest.TestCase):

    def setUp(self):
        self.config_name = "l2_supercat"
        self.dim = 256
        self.binary = False
        self.trunc_dim = None

        tokenizer_inference_config = create_autospec(TokenizerInferenceConfig)
        tokenizer_inference_config.use_local_config = True
        tokenizer_inference_config.config_filename = "tokenizer_config.json"

        model_config = create_autospec(WordLlamaModel)
        model_config.n_vocab = 32000
        model_config.dim = 4096
        model_config.n_layers = 12
        model_config.n_heads = 12
        model_config.hf_model_id = "dummy-model-id"
        model_config.pad_token = 0

        tokenizer_config = create_autospec(TokenizerConfig)
        tokenizer_config.inference = tokenizer_inference_config
        tokenizer_config.return_tensors = True
        tokenizer_config.return_attention_mask = True
        tokenizer_config.max_length = 512
        tokenizer_config.padding = "max_length"
        tokenizer_config.truncation = True
        tokenizer_config.add_special_tokens = True

        training_config = create_autospec(TrainingConfig)
        training_config.learning_rate = 0.001
        training_config.batch_size = 32
        training_config.epochs = 10

        matryoshka_config = create_autospec(MatryoshkaConfig)
        matryoshka_config.dims = [64, 128, 256, 512, 1024]

        self.config = WordLlamaConfig(
            model=model_config,
            tokenizer=tokenizer_config,
            training=training_config,
            matryoshka=matryoshka_config,
        )

    @patch("wordllama.wordllama.Path.open", new_callable=mock_open)
    @patch("wordllama.wordllama.Path.exists", autospec=True)
    @patch("wordllama.wordllama.Path.mkdir", autospec=True)
    @patch("wordllama.wordllama.requests.get", autospec=True)
    def test_download_file_from_hf(
        self, mock_get, mock_mkdir, mock_exists, mock_file_open
    ):
        mock_exists.return_value = False
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        WordLlama.download_file_from_hf(
            repo_id="dummy-repo",
            filename="dummy-file",
            cache_dir=Path("/dummy/cache"),
            force_download=True,
            token="dummy-token",
        )
        mock_get.assert_called_once_with(
            "https://huggingface.co/dummy-repo/resolve/main/dummy-file",
            headers={"Authorization": "Bearer dummy-token"},
            stream=True,
        )
        mock_file_open.assert_called_once_with("wb")
        mock_file_open().write.assert_any_call(b"chunk1")
        mock_file_open().write.assert_any_call(b"chunk2")

    @patch(
        "wordllama.wordllama.WordLlama.download_file_from_hf",
        return_value=Path("/dummy/cache/l2_supercat_256.safetensors"),
    )
    @patch("wordllama.wordllama.Path.exists", side_effect=[False, True, True])
    def test_check_and_download_model(self, mock_exists, mock_download):
        weights_file_path = WordLlama.check_and_download_model(
            config_name=self.config_name,
            dim=self.dim,
            binary=self.binary,
            weights_dir=Path("/dummy/weights"),
            cache_dir=Path("/dummy/cache"),
        )
        self.assertEqual(
            weights_file_path, Path("/dummy/cache/l2_supercat_256.safetensors")
        )

    @patch(
        "wordllama.wordllama.WordLlama.download_file_from_hf",
        return_value=Path("/dummy/cache/tokenizers/tokenizer_config.json"),
    )
    @patch("wordllama.wordllama.Path.exists", side_effect=[False, True, True])
    def test_check_and_download_tokenizer(self, mock_exists, mock_download):
        tokenizer_file_path = WordLlama.check_and_download_tokenizer(
            config_name=self.config_name
        )
        self.assertEqual(
            tokenizer_file_path, Path("/dummy/cache/tokenizers/tokenizer_config.json")
        )

    @patch(
        "wordllama.wordllama.Tokenizer.from_pretrained",
        return_value=MagicMock(spec=Tokenizer),
    )
    @patch(
        "wordllama.wordllama.tokenizer_from_file",
        return_value=MagicMock(spec=Tokenizer),
    )
    @patch("wordllama.wordllama.Path.exists", return_value=True)
    def test_load_tokenizer(
        self, mock_exists, mock_tokenizer_from_file, mock_from_pretrained
    ):
        tokenizer = WordLlama.load_tokenizer(
            Path("/dummy/cache/tokenizers/tokenizer_config.json"), self.config
        )
        mock_tokenizer_from_file.assert_called_once_with(
            Path("/dummy/cache/tokenizers/tokenizer_config.json")
        )
        mock_from_pretrained.assert_not_called()
        self.assertIsInstance(tokenizer, Tokenizer)

    @patch(
        "wordllama.wordllama.WordLlama.check_and_download_model",
        return_value=Path("/dummy/cache/l2_supercat_256.safetensors"),
    )
    @patch(
        "wordllama.wordllama.WordLlama.check_and_download_tokenizer",
        return_value=Path("/dummy/cache/tokenizers/l2_supercat_tokenizer_config.json"),
    )
    @patch(
        "wordllama.wordllama.WordLlama.load_tokenizer",
        return_value=MagicMock(spec=Tokenizer),
    )
    @patch("wordllama.wordllama.safe_open", autospec=True)
    def test_load(
        self,
        mock_safe_open,
        mock_load_tokenizer,
        mock_check_tokenizer,
        mock_check_model,
    ):
        mock_safe_open.return_value.__enter__.return_value.get_tensor.return_value = (
            np.random.rand(256, 4096)
        )

        model = WordLlama.load(
            config=self.config_name,
            weights_dir=Path("/dummy/weights"),
            cache_dir=Path("/dummy/cache"),
            binary=self.binary,
            dim=self.dim,
            trunc_dim=self.trunc_dim,
        )

        self.assertIsInstance(model, WordLlamaInference)
        mock_check_model.assert_called_once_with(
            config_name="l2_supercat",
            dim=256,
            binary=False,
            weights_dir=Path("/dummy/weights"),
            cache_dir=Path("/dummy/cache"),
            disable_download=False,
        )
        mock_check_tokenizer.assert_called_once_with(
            config_name="l2_supercat", disable_download=False
        )
        mock_load_tokenizer.assert_called_once()
        mock_safe_open.assert_called_once_with(
            Path("/dummy/cache/l2_supercat_256.safetensors"),
            framework="np",
            device="cpu",
        )
        self.assertIsInstance(model, WordLlamaInference)


if __name__ == "__main__":
    unittest.main()
