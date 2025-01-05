from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelURI:
    repo_id: str
    available_dims: List[int]
    binary_dims: List[int]
    tokenizer_config: Optional[str]
    remote_filename: Optional[str] = None
    remote_tokenizer_filename: Optional[str] = None
    tensor_key: str = "embedding.weight"
    tokenizer_fallback: Optional[str] = None


class WordLlamaModels:

    @classmethod
    def list_configs(cls) -> List[str]:
        """
        Return a list of configuration names defined as `ModelURI` instances in the class.

        Returns:
            List[str]: A list of configuration attribute names.
        """
        return [
            name for name, value in cls.__dict__.items() if isinstance(value, ModelURI)
        ]

    l2_supercat = ModelURI(
        repo_id="dleemiller/word-llama-l2-supercat",
        available_dims=[64, 128, 256, 512, 1024],
        binary_dims=[64, 128, 256, 512, 1024],
        tokenizer_config="l2_supercat_tokenizer_config.json",
        tokenizer_fallback="meta-llama/Llama-2-70b-hf",
    )

    l3_supercat = ModelURI(
        repo_id="dleemiller/wordllama-l3-supercat",
        available_dims=[64, 128, 256, 512, 1024],
        binary_dims=[64, 128, 256, 512, 1024],
        tokenizer_config="l3_supercat_tokenizer_config.json",
        tokenizer_fallback="meta-llama/Meta-Llama-3.1-405B",
    )


class Model2VecModels:

    @classmethod
    def list_configs(cls) -> List[str]:
        """
        Return a list of configuration names defined as `ModelURI` instances in the class.

        Returns:
            List[str]: A list of configuration attribute names.
        """
        return [
            name for name, value in cls.__dict__.items() if isinstance(value, ModelURI)
        ]

    potion_base_8m = ModelURI(
        repo_id="minishlab/potion-base-8M",
        available_dims=[256],
        binary_dims=[],
        tokenizer_config="m2v_potion_8m_tokenizer_config.json",
        remote_filename="model.safetensors",
        remote_tokenizer_filename="tokenizer.json",
        tensor_key="embeddings",
        tokenizer_fallback="minishlab/potion-base-8M",
    )

    potion_base_4m = ModelURI(
        repo_id="minishlab/potion-base-4M",
        available_dims=[128],
        binary_dims=[],
        tokenizer_config="m2v_potion_4m_tokenizer_config.json",
        remote_filename="model.safetensors",
        remote_tokenizer_filename="tokenizer.json",
        tensor_key="embeddings",
        tokenizer_fallback="minishlab/potion-base-4M",
    )

    potion_base_2m = ModelURI(
        repo_id="minishlab/potion-base-2M",
        available_dims=[64],
        binary_dims=[],
        tokenizer_config="m2v_potion_2m_tokenizer_config.json",
        remote_filename="model.safetensors",
        remote_tokenizer_filename="tokenizer.json",
        tensor_key="embeddings",
        tokenizer_fallback="minishlab/potion-base-2M",
    )

    m2v_multilingual = ModelURI(
        repo_id="minishlab/M2V_multilingual_output",
        available_dims=[256],
        binary_dims=[],
        tokenizer_config="m2v_multilingual_output_tokenizer_config.json",
        remote_filename="model.safetensors",
        remote_tokenizer_filename="tokenizer.json",
        tensor_key="embeddings",
        tokenizer_fallback="minishlab/M2V_multilingual_output",
    )

    m2v_glove = ModelURI(
        repo_id="minishlab/M2V_base_glove",
        available_dims=[256],
        binary_dims=[],
        tokenizer_config="m2v_glove_tokenizer_config.json",
        remote_filename="model.safetensors",
        remote_tokenizer_filename="tokenizer.json",
        tensor_key="embeddings",
        tokenizer_fallback="minishlab/M2V_base_glove",
    )

    m2v_glove_subword = ModelURI(
        repo_id="minishlab/M2V_base_glove_subword",
        available_dims=[256],
        binary_dims=[],
        tokenizer_config="m2v_glove_subword_tokenizer_config.json",
        remote_filename="model.safetensors",
        remote_tokenizer_filename="tokenizer.json",
        tensor_key="embeddings",
        tokenizer_fallback="minishlab/M2V_base_glove_subword",
    )
