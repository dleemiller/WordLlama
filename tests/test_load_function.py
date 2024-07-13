import unittest
from unittest.mock import patch, MagicMock
import pathlib
from wordllama import load, Config


class TestLoadFunction(unittest.TestCase):
    @patch("wordllama.pathlib.Path.exists")
    @patch("wordllama.WordLlama.build")
    def test_load(self, mock_build, mock_exists):
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_build.return_value = mock_model

        with patch.object(Config, "_configurations", MagicMock()):
            mock_config = MagicMock()
            mock_config.matryoshka.dims = [64, 128, 256, 512]
            Config._configurations.get.return_value = mock_config
            model = load(
                config="l2_supercat",
                dim=256,
                binary=False,
                weights_dir="weights",
                trunc_dim=64,
            )

        suffix = "_binary" if False else ""
        weights_file_name = f"l2_supercat_256{suffix}.safetensors"
        weights_file_path = pathlib.Path("weights") / weights_file_name

        mock_exists.assert_called_with()
        mock_build.assert_called_with(weights_file_path, mock_config, trunc_dim=64)
        self.assertEqual(model, mock_model)

    @patch("wordllama.pathlib.Path.exists")
    def test_load_file_not_found(self, mock_exists):
        mock_exists.return_value = False

        with self.assertRaises(FileNotFoundError):
            load(config="l2_supercat", dim=256, binary=False, weights_dir="weights")

    def test_load_config_not_found(self):
        with self.assertRaises(ValueError):
            load(
                config="non_existent_config",
                dim=256,
                binary=False,
                weights_dir="weights",
            )


if __name__ == "__main__":
    unittest.main()
