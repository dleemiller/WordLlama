import logging
import requests
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List
from tokenizers import Tokenizer
from safetensors import safe_open

from .inference import WordLlamaInference
from .config import Config, WordLlamaConfig
from .tokenizers import tokenizer_from_file

logger = logging.getLogger(__name__)


@dataclass
class ModelURI:
    repo_id: str
    available_dims: List[int]
    binary_dims: List[int]
    tokenizer_config: Optional[str]


class WordLlama:
    l2_supercat = ModelURI(
        repo_id="dleemiller/word-llama-l2-supercat",
        available_dims=[64, 128, 256, 512, 1024],
        binary_dims=[64, 128, 256, 512, 1024],
        tokenizer_config="l2_supercat_tokenizer_config.json",
    )

    l3_supercat = ModelURI(
        repo_id="dleemiller/wordllama-l3-supercat",
        available_dims=[64, 128, 256, 512, 1024],
        binary_dims=[64, 128, 256, 512, 1024],
        tokenizer_config="l3_supercat_tokenizer_config.json",
    )

    @staticmethod
    def get_filename(config_name: str, dim: int, binary: bool = False) -> str:
        """
        Generate the filename for the model weights or binary file.

        Args:
            config_name (str): The name of the configuration.
            dim (int): The dimensionality of the model.
            binary (bool): Whether the file is binary.

        Returns:
            str: The generated filename.
        """
        suffix = "" if not binary else "_binary"
        return f"{config_name}_{dim}{suffix}.safetensors"

    @staticmethod
    def get_cache_dir(is_tokenizer_config: bool = False) -> Path:
        """
        Get the cache directory path for weights or tokenizer configuration.

        Args:
            is_tokenizer_config (bool, optional): If True, return the tokenizer cache directory.

        Returns:
            Path: The path to the cache directory.
        """
        base_cache_dir = Path.home() / ".cache" / "wordllama"
        return base_cache_dir / ("tokenizers" if is_tokenizer_config else "weights")

    @staticmethod
    def download_file_from_hf(
        repo_id: str,
        filename: str,
        cache_dir: Optional[Path] = None,
        force_download: bool = False,
        token: Optional[str] = None,
    ) -> Path:
        """
        Download a file from a Hugging Face model repository and cache it locally.

        Args:
            repo_id (str): The repository ID on Hugging Face (e.g., 'user/repo').
            filename (str): The name of the file to download.
            cache_dir (Path, optional): The directory to cache the downloaded file. Defaults to the appropriate cache directory.
            force_download (bool, optional): If True, force download the file even if it exists in the cache.
            token (str, optional): The Hugging Face token for accessing private repositories.

        Returns:
            Path: The path to the cached file.
        """
        if cache_dir is None:
            cache_dir = WordLlama.get_cache_dir()

        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_file_path = cache_dir / filename

        if not force_download and cached_file_path.exists():
            logger.debug(f"File {filename} exists in cache. Using cached version.")
            return cached_file_path

        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        logger.info(f"Downloading {filename} from {url}")

        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        with cached_file_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.debug(f"File {filename} downloaded and cached at {cached_file_path}")

        return cached_file_path

    @classmethod
    def check_and_download_model(
        cls,
        config_name: str,
        dim: int,
        binary: bool = False,
        weights_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        disable_download: bool = False,
    ) -> Path:
        """
        Check if model weights exist locally, if not, download them.

        Args:
            config_name (str): The name of the model configuration.
            dim (int): The dimensionality of the model.
            binary (bool, optional): Whether the file is binary. Defaults to False.
            weights_dir (Path, optional): Directory where weight files are stored. If None, defaults to 'weights' directory in the current module directory.
            cache_dir (Path, optional): Directory where cached files are stored. Defaults to the appropriate cache directory.
            disable_download (bool, optional): Disable downloads for models not in cache.

        Returns:
            Path: Path to the model weights file.
        """
        if weights_dir is None:
            weights_dir = Path(__file__).parent / "weights"

        if cache_dir is None:
            cache_dir = cls.get_cache_dir()

        filename = cls.get_filename(config_name=config_name, dim=dim, binary=binary)
        weights_file_path = weights_dir / filename

        if not weights_file_path.exists():
            logger.debug(
                f"Weights file '{filename}' not found in '{weights_dir}'. Checking cache directory..."
            )
            weights_file_path = cache_dir / filename
            if not weights_file_path.exists():
                if disable_download:
                    raise FileNotFoundError(
                        f"Weights file '{filename}' not found and downloads are disabled."
                    )

                model_uri = getattr(cls, config_name)
                if binary:
                    assert (
                        dim in model_uri.binary_dims
                    ), f"Dimension must be one of {model_uri.binary_dims}"
                else:
                    assert (
                        dim in model_uri.available_dims
                    ), f"Dimension must be one of {model_uri.available_dims}"

                logger.debug(
                    f"Weights file '{filename}' not found in cache directory '{cache_dir}'. Downloading..."
                )
                weights_file_path = cls.download_file_from_hf(
                    repo_id=model_uri.repo_id, filename=filename
                )

        if not weights_file_path.exists():
            raise FileNotFoundError(
                f"Weights file '{weights_file_path}' not found in directory '{weights_dir}' or cache '{cache_dir}'."
            )

        return weights_file_path

    @classmethod
    def check_and_download_tokenizer(
        cls, config_name: str, disable_download: bool = False
    ) -> Path:
        """
        Check if tokenizer configuration exists locally, if not, download it.

        Args:
            config_name (str): The name of the model configuration.
            tokenizer_filename (str): The filename of the tokenizer configuration.
            disable_download (bool, optional): Disable downloading for models not in cache.

        Returns:
            Path: Path to the tokenizer configuration file.
        """
        model_uri = getattr(cls, config_name)
        cache_dir = cls.get_cache_dir(is_tokenizer_config=True)
        tokenizer_file_path = cache_dir / model_uri.tokenizer_config

        if not tokenizer_file_path.exists():
            if disable_download:
                raise FileNotFoundError(
                    f"Weights file '{tokenizer_file_path}' not found and downloads are disabled."
                )

            logger.debug(
                f"Tokenizer config '{model_uri.tokenizer_config}' not found in cache directory '{cache_dir}'. Downloading..."
            )

            tokenizer_file_path = cls.download_file_from_hf(
                repo_id=model_uri.repo_id,
                filename=model_uri.tokenizer_config,
                cache_dir=cache_dir,
            )

        if not tokenizer_file_path.exists():
            raise FileNotFoundError(
                f"Tokenizer config file '{tokenizer_file_path}' not found in cache '{cache_dir}'."
            )

        return tokenizer_file_path

    @classmethod
    def load(
        cls,
        config: Union[str, WordLlamaConfig] = "l2_supercat",
        weights_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        binary: bool = False,
        dim: int = 256,
        trunc_dim: Optional[int] = None,
        disable_download: bool = False,
    ) -> WordLlamaInference:
        """
        Load the WordLlama model.

        Args:
            config (Union[str, WordLlamaConfig], optional): The configuration object or the name of the configuration to load. Defaults to "l2_supercat".
            weights_dir (Optional[Path], optional): Directory where weight files are stored. If None, defaults to 'weights' directory in the current module directory. Defaults to None.
            cache_dir (Optional[Path], optional): Directory where cached files are stored. Defaults to ~/.cache/wordllama/weights. Defaults to None.
            binary (bool, optional): Whether to load the binary version of the weights. Defaults to False.
            dim (int, optional): The dimensionality of the model to load. Options: [64, 128, 256, 512, 1024]. Defaults to 256.
            trunc_dim (Optional[int], optional): The dimension to truncate the model to. Must be less than or equal to 'dim'. Defaults to None.
            disable_download(bool, optional): Turn off downloading models from huggingface when local model is not cached.

        Returns:
            WordLlamaInference: The loaded WordLlama model.

        Raises:
            ValueError: If the configuration is not found or dimensions are invalid.
            FileNotFoundError: If the weights file is not found in either the weights directory or cache directory.
        """
        if isinstance(config, str):
            config_obj = Config._configurations.get(config, None)
            if config_obj is None:
                raise ValueError(f"Configuration '{config}' not found.")
        elif isinstance(config, WordLlamaConfig):
            config_obj = config
        else:
            raise ValueError(
                "Invalid configuration type provided. It must be either a string or an instance of WordLlamaConfig."
            )

        assert (
            dim in config_obj.matryoshka.dims
        ), f"Model dimension must be one of matryoshka dims in config: {config_obj.matryoshka.dims}"
        if trunc_dim is not None:
            assert (
                trunc_dim <= dim
            ), f"Cannot truncate to dimension lower than model dimension ({trunc_dim} > {dim})"
            assert trunc_dim in config_obj.matryoshka.dims

        # Check and download model weights
        weights_file_path = cls.check_and_download_model(
            config_name=config,
            dim=dim,
            binary=binary,
            weights_dir=weights_dir,
            cache_dir=cache_dir,
            disable_download=disable_download,
        )

        # Check and download tokenizer config if necessary
        tokenizer_file_path = cls.check_and_download_tokenizer(
            config_name=config,
            disable_download=disable_download,
        )

        # Load the tokenizer
        tokenizer = cls.load_tokenizer(tokenizer_file_path, config_obj)

        # Load the model weights
        with safe_open(weights_file_path, framework="np", device="cpu") as f:
            embedding = f.get_tensor("embedding.weight")
            if trunc_dim:  # truncate dimension
                embedding = embedding[:, 0:trunc_dim]

        logger.debug(f"Loading weights from: {weights_file_path}")
        return WordLlamaInference(embedding, config_obj, tokenizer, binary=binary)

    @staticmethod
    def load_tokenizer(tokenizer_file_path: Path, config: WordLlamaConfig) -> Tokenizer:
        """
        Load the tokenizer from a local file or from the Hugging Face repository.
        First, it checks the default path, then the cache directory.

        Args:
            tokenizer_file_path (Path): The path to the tokenizer configuration file.
            config (WordLlamaConfig): The configuration for the WordLlama model.

        Returns:
            Tokenizer: An instance of the Tokenizer class.
        """
        if (
            config.tokenizer.inference is not None
            and config.tokenizer.inference.use_local_config
        ):
            # Check in the default path
            if tokenizer_file_path.exists():
                logger.debug(
                    f"Loading tokenizer from default path: {tokenizer_file_path}"
                )
                return tokenizer_from_file(tokenizer_file_path)

            warnings.warn(
                f"Tokenizer config file not found in both default and cache paths. Falling back to Hugging Face model: {config.model.hf_model_id}"
            )

        # Load from Hugging Face if local config is not used or not found
        return Tokenizer.from_pretrained(config.model.hf_model_id)
