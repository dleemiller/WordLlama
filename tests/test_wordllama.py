import unittest
from unittest.mock import patch, MagicMock, mock_open, call
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

        # Mock TokenizerInferenceConfig
        tokenizer_inference_config = MagicMock(spec=TokenizerInferenceConfig)
        tokenizer_inference_config.use_local_config = True
        tokenizer_inference_config.config_filename = "tokenizer_config.json"

        # Mock WordLlamaModel
        model_config = MagicMock(spec=WordLlamaModel)
        model_config.n_vocab = 32000
        model_config.dim = 4096
        model_config.n_layers = 12
        model_config.n_heads = 12
        model_config.hf_model_id = "dummy-model-id"
        model_config.pad_token = 0

        # Mock TokenizerConfig
        tokenizer_config = MagicMock(spec=TokenizerConfig)
        tokenizer_config.inference = tokenizer_inference_config
        tokenizer_config.return_tensors = True
        tokenizer_config.return_attention_mask = True
        tokenizer_config.max_length = 512
        tokenizer_config.padding = "max_length"
        tokenizer_config.truncation = True
        tokenizer_config.add_special_tokens = True

        # Mock TrainingConfig
        training_config = MagicMock(spec=TrainingConfig)
        training_config.learning_rate = 0.001
        training_config.batch_size = 32
        training_config.epochs = 10

        # Mock MatryoshkaConfig
        matryoshka_config = MagicMock(spec=MatryoshkaConfig)
        matryoshka_config.dims = [64, 128, 256, 512, 1024]

        # Assemble WordLlamaConfig
        self.config = WordLlamaConfig(
            config_name="test",
            model=model_config,
            tokenizer=tokenizer_config,
            training=training_config,
            matryoshka=matryoshka_config,
        )

    @patch("wordllama.wordllama.requests.get", autospec=True)
    @patch("wordllama.wordllama.Path.mkdir", autospec=True)
    @patch("wordllama.wordllama.Path.exists", autospec=True)
    @patch("wordllama.wordllama.Path.open", new_callable=mock_open)
    def test_resolve_file_downloads_if_not_found(
        self, mock_file_open, mock_exists, mock_mkdir, mock_get
    ):
        """
        Test that resolve_file downloads the file from Hugging Face
        when it does not exist in project root or cache.
        """
        # Setup mocks
        # First, project_root_path.exists() returns False
        # Then, cache_path.exists() returns False
        # Therefore, it should attempt to download
        mock_exists.side_effect = [False, False]

        # Mock the GET request
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Call resolve_file for weights
        weights_path = WordLlama.resolve_file(
            config_name=self.config_name,
            dim=self.dim,
            binary=self.binary,
            file_type="weights",
            cache_dir=Path("/dummy/cache"),
            disable_download=False,
        )

        # Assert that the file was attempted to be downloaded
        mock_get.assert_called_once_with(
            "https://huggingface.co/dleemiller/word-llama-l2-supercat/resolve/main/l2_supercat_256.safetensors",
            stream=True,
        )

        # Assert that mkdir was called to create the cache directory
        mock_mkdir.assert_called_once_with(
            Path("/dummy/cache/weights"), parents=True, exist_ok=True
        )

        # Assert that the file was written to cache_path
        mock_file_open.assert_called_once_with("wb")
        handle = mock_file_open()
        handle.write.assert_has_calls([call(b"chunk1"), call(b"chunk2")])

        # Assert the returned path is correct
        self.assertEqual(
            weights_path, Path("/dummy/cache/weights/l2_supercat_256.safetensors")
        )

    @patch.object(WordLlama, "resolve_file", autospec=True)
    def test_load_with_default_cache_dir(self, mock_resolve_file):
        """
        Test that load uses the default cache directory when cache_dir is not provided.
        """
        # Setup mock for resolve_file
        default_cache_dir = WordLlama.DEFAULT_CACHE_DIR
        weights_path = default_cache_dir / "weights" / "l2_supercat_256.safetensors"
        tokenizer_path = (
            default_cache_dir / "tokenizers" / "l2_supercat_tokenizer_config.json"
        )
        mock_resolve_file.side_effect = [weights_path, tokenizer_path]

        # Mock tokenizer and weights loading
        with patch(
            "wordllama.wordllama.WordLlama.load_tokenizer",
            return_value=MagicMock(spec=Tokenizer),
        ) as mock_load_tokenizer, patch(
            "wordllama.wordllama.safe_open", autospec=True
        ) as mock_safe_open:
            # Mock the tensor returned by safe_open
            mock_tensor = MagicMock()
            mock_tensor.__getitem__.return_value = np.random.rand(256, 4096)
            mock_safe_open.return_value.__enter__.return_value.get_tensor.return_value = (
                mock_tensor
            )

            # Call load without specifying cache_dir
            model = WordLlama.load(
                config=self.config,
                binary=self.binary,
                dim=self.dim,
                trunc_dim=self.trunc_dim,
                cache_dir=default_cache_dir,
            )

            # Assert resolve_file was called twice: once for weights, once for tokenizer
            expected_calls = [
                call(
                    config_name="test",
                    dim=self.dim,
                    binary=self.binary,
                    file_type="weights",
                    cache_dir=default_cache_dir,
                    disable_download=True,
                ),
                call(
                    config_name="test",
                    dim=self.dim,
                    binary=False,
                    file_type="tokenizer",
                    cache_dir=default_cache_dir,
                    disable_download=True,
                ),
            ]
            mock_resolve_file.assert_has_calls(expected_calls, any_order=False)
            self.assertEqual(mock_resolve_file.call_count, 2)

            # Assert load_tokenizer was called with correct path
            mock_load_tokenizer.assert_called_once_with(
                tokenizer_path,
                self.config,
            )

            # Assert safe_open was called with the weights path
            mock_safe_open.assert_called_once_with(
                weights_path,
                framework="np",
                device="cpu",
            )

            # Assert the returned model is an instance of WordLlamaInference
            self.assertIsInstance(model, WordLlamaInference)

    @patch.object(WordLlama, "resolve_file", autospec=True)
    def test_load_with_custom_cache_dir(self, mock_resolve_file):
        """
        Test that load correctly handles various custom cache_dir inputs.
        """
        # Define different cache_dir inputs
        cache_dirs = {
            "tilde": "~/tmp_cache",
            "relative": "tmp",
            "relative_dot": "./tmp",
            "absolute": "/tmp/cache_dir",
        }

        # Expected resolved paths
        expected_resolved_dirs = {
            "tilde": Path("~/tmp_cache").expanduser(),
            "relative": Path("tmp").resolve(),
            "relative_dot": Path("./tmp").resolve(),
            "absolute": Path("/tmp/cache_dir"),
        }

        for key, cache_dir_input in cache_dirs.items():
            with self.subTest(cache_dir=key):
                # Reset mocks
                mock_resolve_file.reset_mock()

                # Setup mock for resolve_file
                weights_path = (
                    expected_resolved_dirs[key]
                    / "weights"
                    / "l2_supercat_256.safetensors"
                )
                tokenizer_path = (
                    expected_resolved_dirs[key]
                    / "tokenizers"
                    / "l2_supercat_tokenizer_config.json"
                )
                mock_resolve_file.side_effect = [weights_path, tokenizer_path]

                # Mock tokenizer and weights loading
                with patch(
                    "wordllama.wordllama.WordLlama.load_tokenizer",
                    return_value=MagicMock(spec=Tokenizer),
                ) as mock_load_tokenizer, patch(
                    "wordllama.wordllama.safe_open", autospec=True
                ) as mock_safe_open:
                    # Mock the tensor returned by safe_open
                    mock_tensor = MagicMock()
                    mock_tensor.__getitem__.return_value = np.random.rand(256, 4096)
                    mock_safe_open.return_value.__enter__.return_value.get_tensor.return_value = (
                        mock_tensor
                    )

                    # Call load with custom cache_dir
                    model = WordLlama.load(
                        config=self.config,
                        cache_dir=cache_dir_input,
                        binary=self.binary,
                        dim=self.dim,
                        trunc_dim=self.trunc_dim,
                    )

                    # Assert resolve_file was called twice with the correct cache_dir
                    expected_calls = [
                        call(
                            # WordLlama,
                            config_name="test",
                            dim=self.dim,
                            binary=self.binary,
                            file_type="weights",
                            cache_dir=expected_resolved_dirs[key],
                            disable_download=True,
                        ),
                        call(
                            # WordLlama,
                            config_name="test",
                            dim=self.dim,
                            binary=False,
                            file_type="tokenizer",
                            cache_dir=expected_resolved_dirs[key],
                            disable_download=True,
                        ),
                    ]
                    mock_resolve_file.assert_has_calls(expected_calls, any_order=False)
                    self.assertEqual(mock_resolve_file.call_count, 2)

                    # Assert load_tokenizer was called with correct path
                    mock_load_tokenizer.assert_called_once_with(
                        tokenizer_path,
                        self.config,
                    )

                    # Assert safe_open was called with the weights path
                    mock_safe_open.assert_called_once_with(
                        weights_path,
                        framework="np",
                        device="cpu",
                    )

                    # Assert the returned model is an instance of WordLlamaInference
                    self.assertIsInstance(model, WordLlamaInference)

    @patch.object(WordLlama, "resolve_file", autospec=True)
    def test_load_with_disable_download(self, mock_resolve_file):
        """
        Test that load raises FileNotFoundError when files are missing and downloads are disabled.
        """
        # Setup mocks to simulate files not existing and downloads disabled
        mock_resolve_file.side_effect = FileNotFoundError("File not found")

        # Call load with disable_download=True and expect FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            WordLlama.load(
                config=self.config,
                cache_dir=Path("/dummy/cache"),
                binary=self.binary,
                dim=self.dim,
                trunc_dim=self.trunc_dim,
                disable_download=True,
            )

        # Assert resolve_file was called twice: once for weights, once for tokenizer
        expected_calls = [
            call(
                config_name="test",
                dim=self.dim,
                binary=self.binary,
                file_type="weights",
                cache_dir=Path("/dummy/cache"),
                disable_download=True,
            )
        ]
        mock_resolve_file.assert_has_calls(expected_calls, any_order=False)
        self.assertEqual(mock_resolve_file.call_count, 1)

    @patch.object(WordLlama, "resolve_file", autospec=True)
    def test_load_with_truncated_dimension(self, mock_resolve_file):
        """
        Test that load correctly handles trunc_dim parameter.
        """
        # Setup mock for resolve_file
        weights_path = Path("/dummy/cache/weights/l2_supercat_256.safetensors")
        tokenizer_path = Path(
            "/dummy/cache/tokenizers/l2_supercat_tokenizer_config.json"
        )
        mock_resolve_file.side_effect = [weights_path, tokenizer_path]

        # Mock tokenizer and weights loading
        with patch(
            "wordllama.wordllama.WordLlama.load_tokenizer",
            return_value=MagicMock(spec=Tokenizer),
        ) as mock_load_tokenizer, patch(
            "wordllama.wordllama.safe_open", autospec=True
        ) as mock_safe_open:
            # Mock the tensor returned by safe_open
            mock_tensor = MagicMock()
            mock_tensor.__getitem__.return_value = np.random.rand(256, 4096)
            mock_safe_open.return_value.__enter__.return_value.get_tensor.return_value = (
                mock_tensor
            )

            # Call load with trunc_dim
            model = WordLlama.load(
                config=self.config,
                cache_dir=Path("/dummy/cache"),
                binary=self.binary,
                dim=self.dim,
                trunc_dim=128,
            )

            # Assert resolve_file was called twice
            expected_calls = [
                call(
                    config_name="test",
                    dim=self.dim,
                    binary=self.binary,
                    file_type="weights",
                    cache_dir=Path("/dummy/cache"),
                    disable_download=True,
                ),
                call(
                    config_name="test",
                    dim=self.dim,
                    binary=False,
                    file_type="tokenizer",
                    cache_dir=Path("/dummy/cache"),
                    disable_download=True,
                ),
            ]
            mock_resolve_file.assert_has_calls(expected_calls, any_order=False)
            self.assertEqual(mock_resolve_file.call_count, 2)

            # Assert load_tokenizer was called with correct path
            mock_load_tokenizer.assert_called_once_with(
                tokenizer_path,
                self.config,
            )

            # Assert safe_open was called with the weights path
            mock_safe_open.assert_called_once_with(
                weights_path,
                framework="np",
                device="cpu",
            )

            # Assert the returned model is an instance of WordLlamaInference
            self.assertIsInstance(model, WordLlamaInference)

            # Assert that the embedding was truncated
            mock_tensor.__getitem__.assert_called_with((slice(None), slice(None, 128)))

    @patch(
        "wordllama.wordllama.Tokenizer.from_pretrained",
        return_value=MagicMock(spec=Tokenizer),
    )
    @patch.object(WordLlama, "resolve_file", autospec=True)
    def test_load_tokenizer_fallback(self, mock_resolve_file, mock_from_pretrained):
        """
        Test that load_tokenizer falls back to Hugging Face if local config is not found.
        """
        # Setup mocks
        # First call for weights, second call for tokenizer
        weights_path = Path("/dummy/cache/weights/l2_supercat_256.safetensors")
        tokenizer_path = Path(
            "/dummy/cache/tokenizers/l2_supercat_tokenizer_config.json"
        )
        mock_resolve_file.side_effect = [weights_path, tokenizer_path]

        # Simulate tokenizer config does not exist by patching Path.exists
        with patch(
            "wordllama.wordllama.Path.exists", side_effect=[False, False]
        ), patch("wordllama.wordllama.safe_open", autospec=True) as mock_safe_open:
            # Mock the tensor returned by safe_open
            mock_tensor = MagicMock()
            mock_tensor.__getitem__.return_value = np.random.rand(256, 4096)
            mock_safe_open.return_value.__enter__.return_value.get_tensor.return_value = (
                mock_tensor
            )

            # Call load
            model = WordLlama.load(
                config=self.config,
                cache_dir=Path("/dummy/cache"),
                binary=self.binary,
                dim=self.dim,
                trunc_dim=self.trunc_dim,
            )

            # Assert resolve_file was called twice: weights and tokenizer
            expected_calls = [
                call(
                    config_name="test",
                    dim=self.dim,
                    binary=self.binary,
                    file_type="weights",
                    cache_dir=Path("/dummy/cache"),
                    disable_download=True,
                ),
                call(
                    config_name="test",
                    dim=self.dim,
                    binary=False,
                    file_type="tokenizer",
                    cache_dir=Path("/dummy/cache"),
                    disable_download=True,
                ),
            ]
            mock_resolve_file.assert_has_calls(expected_calls, any_order=False)
            self.assertEqual(mock_resolve_file.call_count, 2)

            # Assert Tokenizer.from_pretrained was called since local config was not found
            mock_from_pretrained.assert_called_once_with("dummy-model-id")

            # Assert safe_open was called with the weights path
            mock_safe_open.assert_called_once_with(
                weights_path,
                framework="np",
                device="cpu",
            )

            # Assert the returned model is an instance of WordLlamaInference
            self.assertIsInstance(model, WordLlamaInference)


if __name__ == "__main__":
    unittest.main()
