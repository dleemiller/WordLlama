import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
from safetensors import safe_open
from tokenizers import Tokenizer

from .inference import WordLlamaInference
from .config import ModelURI, WordLlamaModels, Model2VecModels


logger = logging.getLogger(__name__)


class WordLlama:
    """
    The WordLlama class is responsible for managing model weights and tokenizer configurations.

    It handles the resolution of file paths, caching, and downloading from Hugging Face repositories.
    """

    _wordllama = WordLlamaModels
    _model2vec = Model2VecModels
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "wordllama"

    @classmethod
    def list_configs(cls) -> Dict[str, List[str]]:
        """
        List the available configurations.

        Returns:
            a dict of configuration names
        """
        return {
            "wordllama": cls._wordllama.list_configs(),
            "model2vec": cls._model2vec.list_configs(),
        }

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
    def get_tokenizer_filename(model_uri: ModelURI) -> str:
        """
        Retrieve the tokenizer configuration filename based on the configuration name.

        Args:
            model_uri (ModelURI): The model URI containing tokenizer configuration info.

        Returns:
            str: The tokenizer configuration filename.
        """
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
        model_uri: ModelURI,
        dim: int,
        binary: bool,
        file_type: str,  # 'weights' or 'tokenizer'
        cache_dir: Optional[Path] = None,
        remote_filename: Optional[str] = None,
        disable_download: bool = False,
    ) -> Path:
        """
        Resolve the file path by checking the project root and cache directories.
        If the file is not found, download it from Hugging Face to the cache directory.

        Args:
            config_name (str): The name of the configuration.
            model_uri (ModelURI): The model URI containing repository and file info.
            dim (int): The dimensionality of the model (irrelevant for tokenizers).
            binary (bool): Indicates whether the weights file is binary (irrelevant for tokenizers).
            file_type (str): Specifies the type of file ('weights' or 'tokenizer').
            cache_dir (Path, optional): Custom cache directory. Defaults to None.
            remote_filename (Optional[str]): A specific filename to use when downloading instead
                of the default. Defaults to None.
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
            filename = cls.get_tokenizer_filename(model_uri)
        else:
            raise ValueError("file_type must be either 'weights' or 'tokenizer'.")

        project_root_path = Path(__file__).parent / file_type / filename
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
                f"{file_type.capitalize()} file '{filename}' not found in project root or cache, "
                f"and downloads are disabled."
            )

        # Download from Hugging Face
        repo_id = model_uri.repo_id
        download_dir = cache_path.parent
        download_dir.mkdir(parents=True, exist_ok=True)

        # check if need to use different remote filename
        if remote_filename:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{remote_filename}"
        else:
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
    def load_m2v(
        cls,
        config: str,
        cache_dir: Optional[Union[Path, str]] = None,
        disable_download: bool = False,
    ) -> WordLlamaInference:
        """
        Load the Model2Vec model configuration.

        Args:
            config (str): The name of the Model2Vec configuration.
            cache_dir (Optional[Union[Path, str]]): The directory to use for caching files.
            disable_download (bool): If True, prevents downloading if files are not found locally.

        Returns:
            WordLlamaInference: The loaded Model2Vec inference object.
        """
        model_uri = getattr(cls._model2vec, config)
        return cls._load(
            config_name=config,
            model_uri=model_uri,
            cache_dir=cache_dir,
            binary=False,
            dim=model_uri.available_dims[0],
            trunc_dim=None,
            disable_download=disable_download,
            tensor_key=model_uri.tensor_key,
        )

    @classmethod
    def load(
        cls,
        config: Optional[Union[str, ModelURI]] = "l2_supercat",
        cache_dir: Optional[Union[Path, str]] = None,
        binary: bool = False,
        dim: int = 256,
        trunc_dim: Optional[int] = None,
        disable_download: bool = False,
    ) -> WordLlamaInference:
        """
        Load the WordLlama model by resolving and loading the necessary weights and tokenizer files.

        Weights resolution order:
        1. project_root/wordllama/weights/...
        2. cache_dir/weights/...
        3. Hugging Face download -> cache_dir/weights/...

        Tokenizer resolution order:
        1. project_root/wordllama/tokenizers/...
        2. cache_dir/tokenizers/...
        3. Hugging Face download -> cache_dir/tokenizers/...

        Args:
            config (Optional[Union[str, ModelURI]]): The configuration name or a custom
                ModelURI instance to load. Defaults to "l2_supercat".
            cache_dir (Optional[Union[Path, str]]): The directory to use for caching files.
                Defaults to None, which expands to `~/.cache/wordllama`.
            binary (bool): Indicates whether to load the binary version of the weights.
                Defaults to False.
            dim (int): The dimensionality of the model to load. Must be one of
                the available dimensions. Defaults to 256.
            trunc_dim (Optional[int]): The dimension to truncate the model to. Must be <= dim.
                Defaults to None.
            disable_download (bool): If True, prevents downloading files from Hugging Face
                if they are not found locally. Defaults to False.

        Returns:
            WordLlamaInference: An instance of WordLlamaInference containing the loaded model.

        Raises:
            ValueError: If the provided configuration is invalid or if dimensions are invalid.
            FileNotFoundError: If required files are missing locally and downloads are disabled,
                or if a download fails.
        """
        if isinstance(config, ModelURI):
            config_name = "custom"
            model_uri = config
            disable_download = True
            logger.debug("Downloads are disabled for custom configs.")
        else:
            config_name = config
            model_uri = getattr(cls._wordllama, config)

        # Validate dimensions
        if dim not in model_uri.available_dims:
            raise ValueError(
                f"Model dimension must be one of {model_uri.available_dims}"
            )
        if trunc_dim is not None:
            if trunc_dim > dim:
                raise ValueError(
                    f"Cannot truncate to a higher dimension ({trunc_dim} > {dim})"
                )
            if trunc_dim not in model_uri.available_dims:
                raise ValueError(
                    f"Truncated dimension must be one of {model_uri.available_dims}"
                )

        return cls._load(
            config_name=config_name,
            model_uri=model_uri,
            cache_dir=cache_dir,
            binary=binary,
            dim=dim,
            trunc_dim=trunc_dim,
            disable_download=disable_download,
            tensor_key="embedding.weight",
        )

    @classmethod
    def _load(
        cls,
        config_name: str,
        model_uri: ModelURI,
        cache_dir: Optional[Union[Path, str]] = None,
        binary: bool = False,
        dim: int = 256,
        trunc_dim: Optional[int] = None,
        tensor_key: str = "embedding.weight",
        disable_download: bool = False,
    ) -> WordLlamaInference:
        """
        Internal method to load the model weights and tokenizer, then return a WordLlamaInference.

        Args:
            config_name (str): The name of the configuration.
            model_uri (ModelURI): The model URI containing repository and file information.
            cache_dir (Optional[Union[Path, str]]): The directory to use for caching files.
            binary (bool): Indicates whether to load the binary version of the weights.
            dim (int): The dimensionality of the model to load.
            trunc_dim (Optional[int]): The dimension to truncate the model to.
            tensor_key (str): The tensor key in the safetensor file (default: "embedding.weight").
            disable_download (bool): If True, prevents downloading files if not found locally.

        Returns:
            WordLlamaInference: The loaded WordLlamaInference object.
        """
        if cache_dir and isinstance(cache_dir, str):
            cache_dir = Path(cache_dir).expanduser()  # Expand ~ to the home dir
            cache_dir = cache_dir.resolve(strict=False)  # Resolve to absolute path

        # Resolve and load weights
        weights_file_path = cls.resolve_file(
            config_name=config_name,
            model_uri=model_uri,
            dim=dim,
            binary=binary,
            file_type="weights",
            cache_dir=cache_dir,
            remote_filename=model_uri.remote_filename,
            disable_download=disable_download,
        )

        # Resolve and load tokenizer
        tokenizer_file_path = cls.resolve_file(
            config_name=config_name,
            model_uri=model_uri,
            dim=dim,
            binary=False,
            file_type="tokenizer",
            cache_dir=cache_dir,
            remote_filename=model_uri.remote_tokenizer_filename,
            disable_download=disable_download,
        )

        # Load tokenizer
        tokenizer = cls.load_tokenizer(
            tokenizer_file_path, hf_model_id=model_uri.tokenizer_fallback
        )

        # Load model weights
        with safe_open(weights_file_path, framework="np", device="cpu") as f:
            embedding = f.get_tensor(tensor_key)
            if trunc_dim:
                embedding = embedding[:, :trunc_dim]

        logger.debug(f"Loading weights from: {weights_file_path}")
        return WordLlamaInference(embedding, tokenizer, binary=binary)

    @staticmethod
    def load_tokenizer(
        tokenizer_file_path: Path,
        hf_model_id: Optional[str] = None,
        use_local_if_exists: bool = True,
    ) -> Tokenizer:
        """
        Load the tokenizer from a local configuration file or fallback to Hugging Face.

        Args:
            tokenizer_file_path (Path): The path to the tokenizer configuration file.
            hf_model_id (Optional[str]): A Hugging Face model ID to load if local config is unavailable.
            use_local_if_exists (bool): If True, tries to load local tokenizer config first.

        Returns:
            Tokenizer: An instance of the Tokenizer class initialized with the configuration.

        Raises:
            FileNotFoundError: If the tokenizer is not found locally and no Hugging Face model ID is provided.
        """
        if use_local_if_exists:
            # Check in the default path
            if tokenizer_file_path.exists():
                logger.debug(
                    f"Loading tokenizer from local config_name: {tokenizer_file_path}"
                )
                return Tokenizer.from_file(str(tokenizer_file_path))
            else:
                warnings.warn(
                    f"Local tokenizer config not found at {tokenizer_file_path}. "
                    "Falling back to Hugging Face."
                )

        if hf_model_id:
            logger.debug(f"Loading tokenizer from Hugging Face: {hf_model_id}")
            return Tokenizer.from_pretrained(hf_model_id)
        else:
            raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_file_path}")
