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

        with patch.object(Config, "llama3_70B", MagicMock()):
            model = load(
                config_name="llama3_70B", dim=64, binary=False, weights_dir="weights"
            )
            config = getattr(Config, "llama3_70B")

        suffix = "_binary" if False else ""
        weights_file_name = f"llama3_70B_64{suffix}.safetensors"
        weights_file_path = pathlib.Path("weights") / weights_file_name

        mock_exists.assert_called_with()
        mock_build.assert_called_with(weights_file_path, config)
        self.assertEqual(model, mock_model)

    @patch("wordllama.pathlib.Path.exists")
    def test_load_file_not_found(self, mock_exists):
        mock_exists.return_value = False

        with self.assertRaises(FileNotFoundError):
            load(config_name="llama3_70B", dim=64, binary=False, weights_dir="weights")

    def test_load_config_not_found(self):
        with self.assertRaises(AttributeError):
            load(
                config_name="non_existent_config",
                dim=64,
                binary=False,
                weights_dir="weights",
            )


if __name__ == "__main__":
    unittest.main()
