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


logger = logging.getLogger(__name__)


@dataclass
class ModelURI:
    repo_id: str
    available_dims: List[int]
    binary_dims: List[int]
    tokenizer_config: Optional[str]


class WordLlama:
    """
    The WordLlama class is responsible for managing model weights and tokenizer configurations.
    It handles the resolution of file paths, caching, and downloading from Hugging Face repositories.
    """

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

    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "wordllama"

    @staticmethod
    def get_filename(config_name: str, dim: int, binary: bool = False) -> str:
        """
        Generate the filename for the model weights or binary file.

        Args:
            config_name (str): The name of the configuration.
            dim (int): The dimensionality of the model.
            binary (bool): Indicates whether the file is binary.

        Returns:
            str: The constructed filename.
        """
        suffix = "_binary" if binary else ""
        return f"{config_name}_{dim}{suffix}.safetensors"

    @staticmethod
    def get_tokenizer_filename(config_name: str) -> str:
        """
        Retrieve the tokenizer configuration filename based on the configuration name.

        Args:
            config_name (str): The name of the configuration.

        Returns:
            str: The tokenizer configuration filename.
        """
        model_uri = getattr(WordLlama, config_name)
        return model_uri.tokenizer_config

    @classmethod
    def get_file_path(
        cls,
        file_type: str,  # 'weights' or 'tokenizer'
        cache_dir: Optional[Path] = None,
    ) -> Path:
        """
        Determine the directory path for weights or tokenizer files.

        Args:
            file_type (str): Specifies the type of file ('weights' or 'tokenizer').
            cache_dir (Path, optional): Custom cache directory. Defaults to None.

        Returns:
            Path: The resolved directory path for the specified file type.
        """
        cache_dir = cache_dir or cls.DEFAULT_CACHE_DIR
        sub_dir = "tokenizers" if file_type == "tokenizer" else "weights"
        return cache_dir / sub_dir

    @classmethod
    def resolve_file(
        cls,
        config_name: str,
        dim: int,
        binary: bool,
        file_type: str,  # 'weights' or 'tokenizer'
        cache_dir: Optional[Path] = None,
        disable_download: bool = False,
    ) -> Path:
        """
        Resolve the file path by checking the project root and cache directories.
        If the file is not found, download it from Hugging Face to the cache directory.

        Args:
            config_name (str): The name of the configuration.
            dim (int): The dimensionality of the model (irrelevant for tokenizers).
            binary (bool): Indicates whether the weights file is binary (irrelevant for tokenizers).
            file_type (str): Specifies the type of file ('weights' or 'tokenizer').
            cache_dir (Path, optional): Custom cache directory. Defaults to None.
            disable_download (bool): If True, prevents downloading files not found locally.

        Returns:
            Path: The resolved file path.

        Raises:
            FileNotFoundError: If the file is not found locally and downloads are disabled.
            ValueError: If an invalid file_type is provided.
        """
        if file_type == "weights":
            filename = cls.get_filename(config_name, dim, binary)
        elif file_type == "tokenizer":
            filename = cls.get_tokenizer_filename(config_name)
        else:
            raise ValueError("file_type must be either 'weights' or 'tokenizer'.")

        project_root_path = Path(__file__).parent / "wordllama" / file_type / filename
        cache_path = cls.get_file_path(file_type, cache_dir) / filename

        # Check in project root directory
        if project_root_path.exists():
            logger.debug(f"Found {file_type} file in project root: {project_root_path}")
            return project_root_path

        # Check in cache directory
        if cache_path.exists():
            logger.debug(f"Found {file_type} file in cache: {cache_path}")
            return cache_path

        if disable_download:
            raise FileNotFoundError(
                f"{file_type.capitalize()} file '{filename}' not found in project root or cache, and downloads are disabled."
            )

        # Download from Hugging Face
        model_uri = getattr(cls, config_name)
        repo_id = model_uri.repo_id
        download_dir = cache_path.parent
        download_dir.mkdir(parents=True, exist_ok=True)

        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        logger.info(
            f"Downloading {file_type} file '{filename}' from Hugging Face repository '{repo_id}'."
        )

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(
                f"Failed to download {file_type} file '{filename}' from '{url}': {e}"
            )
            raise FileNotFoundError(
                f"Failed to download {file_type} file '{filename}' from '{url}'."
            ) from e

        with cache_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)

        logger.debug(f"Downloaded {file_type} file and cached at {cache_path}")
        return cache_path

    @classmethod
    def load(
        cls,
        config: Union[str, WordLlamaConfig] = "l2_supercat",
        cache_dir: Optional[Union[Path, str]] = None,
        binary: bool = False,
        dim: int = 256,
        trunc_dim: Optional[int] = None,
        disable_download: bool = False,
    ) -> WordLlamaInference:
        """
        Load the WordLlama model by resolving and loading the necessary weights and tokenizer files.

        Weights:
            1. Check in project root (`project_root / wordllama / weights / ...`).
            2. If not found, check in `cache_dir / weights / ...`.
            3. If still not found, download from Hugging Face to `cache_dir / weights / ...`.
        Tokenizer:
            1. Check in project root (`project_root / wordllama / tokenizers / ...`).
            2. If not found, check in `cache_dir / tokenizers / ...`.
            3. If still not found, download from Hugging Face to `cache_dir / tokenizers / ...`.

        Args:
            config (Union[str, WordLlamaConfig], optional):
                The configuration name or a custom instance of WordLlamaConfig to load.
                Defaults to "l2_supercat".
            cache_dir (Optional[Path], optional):
                The directory to use for caching files.
                If None, defaults to `~/.cache/wordllama`.
                Can be set to a custom path as needed.
            binary (bool, optional):
                Indicates whether to load the binary version of the weights.
                Defaults to False.
            dim (int, optional):
                The dimensionality of the model to load.
                Must be one of the available dimensions specified in the configuration.
                Defaults to 256.
            trunc_dim (Optional[int], optional):
                The dimension to truncate the model to.
                Must be less than or equal to 'dim' and one of the available dimensions.
                Defaults to None.
            disable_download (bool, optional):
                If True, prevents downloading files from Hugging Face if they are not found locally.
                Defaults to False.

        Returns:
            WordLlamaInference: An instance of WordLlamaInference containing the loaded model.

        Raises:
            ValueError:
                - If the provided configuration is invalid or not found.
                - If the specified dimensions are invalid.
            FileNotFoundError:
                - If the required files are not found locally and downloads are disabled.
                - If downloading fails due to network issues or invalid URLs.
        """
        # Resolve configuration
        if isinstance(config, str):
            config_obj = Config._configurations.get(config)
            if config_obj is None:
                raise ValueError(f"Configuration '{config}' not found.")
            config_name = config
        elif isinstance(config, WordLlamaConfig):
            config_obj = config
            config_name = getattr(config, "config_name")
            disable_download = True  # disable for custom config
            logger.debug("Downloads are disabled for custom configs.")
        else:
            raise ValueError(
                "Invalid configuration type provided. It must be either a string or an instance of WordLlamaConfig."
            )

        # Validate dimensions
        if dim not in config_obj.matryoshka.dims:
            raise ValueError(
                f"Model dimension must be one of {config_obj.matryoshka.dims}"
            )
        if trunc_dim is not None:
            if trunc_dim > dim:
                raise ValueError(
                    f"Cannot truncate to a higher dimension ({trunc_dim} > {dim})"
                )
            if trunc_dim not in config_obj.matryoshka.dims:
                raise ValueError(
                    f"Truncated dimension must be one of {config_obj.matryoshka.dims}"
                )

        if cache_dir and isinstance(cache_dir, str):
            cache_dir = Path(cache_dir).expanduser()  # Expand ~ to the home dir
            cache_dir = cache_dir.resolve(strict=False)  # Resolve to absolute path

        # Resolve and load weights
        weights_file_path = cls.resolve_file(
            config_name=config_name,
            dim=dim,
            binary=binary,
            file_type="weights",
            cache_dir=cache_dir,
            disable_download=disable_download,
        )

        # Resolve and load tokenizer
        tokenizer_file_path = cls.resolve_file(
            config_name=config_name,
            dim=dim,
            binary=False,
            file_type="tokenizer",
            cache_dir=cache_dir,
            disable_download=disable_download,
        )

        # Load tokenizer
        tokenizer = cls.load_tokenizer(tokenizer_file_path, config_obj)

        # Load model weights
        with safe_open(weights_file_path, framework="np", device="cpu") as f:
            embedding = f.get_tensor("embedding.weight")
            if trunc_dim:
                embedding = embedding[:, :trunc_dim]

        logger.debug(f"Loading weights from: {weights_file_path}")
        return WordLlamaInference(embedding, config_obj, tokenizer, binary=binary)

    @staticmethod
    def load_tokenizer(tokenizer_file_path: Path, config: WordLlamaConfig) -> Tokenizer:
        """
        Load the tokenizer from a local configuration file or fallback to Hugging Face.

        The method first attempts to load the tokenizer using the local configuration if specified.
        If the local configuration is not found or not used, it falls back to loading the tokenizer
        from the Hugging Face repository.

        Args:
            tokenizer_file_path (Path): The path to the tokenizer configuration file.
            config (WordLlamaConfig): The configuration object containing tokenizer settings.

        Returns:
            Tokenizer: An instance of the Tokenizer class initialized with the appropriate configuration.
        """
        if config.tokenizer.inference and config.tokenizer.inference.use_local_config:
            if tokenizer_file_path.exists():
                logger.debug(
                    f"Loading tokenizer from local config: {tokenizer_file_path}"
                )
                return Tokenizer.from_file(str(tokenizer_file_path))
            else:
                warnings.warn(
                    f"Local tokenizer config not found at {tokenizer_file_path}. Falling back to Hugging Face."
                )

        # Fallback to Hugging Face
        logger.debug(f"Loading tokenizer from Hugging Face: {config.model.hf_model_id}")
        return Tokenizer.from_pretrained(config.model.hf_model_id)
