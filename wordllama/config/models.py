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


class WordLlamaModels:
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


class Model2VecModels:
    potion_base_8m = ModelURI(
        repo_id="minishlab/potion-base-8M",
        available_dims=[256],
        binary_dims=[],
        tokenizer_config="m2v_potion_8m_tokenizer_config.json",
        remote_filename="model.safetensors",
        remote_tokenizer_filename="tokenizer.json",
        tensor_key="embeddings",
    )

    potion_base_4m = ModelURI(
        repo_id="minishlab/potion-base-4M",
        available_dims=[128],
        binary_dims=[],
        tokenizer_config="m2v_potion_4m_tokenizer_config.json",
        remote_filename="model.safetensors",
        remote_tokenizer_filename="tokenizer.json",
        tensor_key="embeddings",
    )

    potion_base_2m = ModelURI(
        repo_id="minishlab/potion-base-2M",
        available_dims=[64],
        binary_dims=[],
        tokenizer_config="m2v_potion_2m_tokenizer_config.json",
        remote_filename="model.safetensors",
        remote_tokenizer_filename="tokenizer.json",
        tensor_key="embeddings",
    )

    m2v_multilingual = ModelURI(
        repo_id="minishlab/M2V_multilingual_output",
        available_dims=[256],
        binary_dims=[],
        tokenizer_config="m2v_multilingual_output_tokenizer_config.json",
        remote_filename="model.safetensors",
        remote_tokenizer_filename="tokenizer.json",
        tensor_key="embeddings",
    )

    m2v_glove = ModelURI(
        repo_id="minishlab/M2V_base_glove",
        available_dims=[256],
        binary_dims=[],
        tokenizer_config="m2v_glove_tokenizer_config.json",
        remote_filename="model.safetensors",
        remote_tokenizer_filename="tokenizer.json",
        tensor_key="embeddings",
    )

    m2v_glove_subword = ModelURI(
        repo_id="minishlab/M2V_base_glove_subword",
        available_dims=[256],
        binary_dims=[],
        tokenizer_config="m2v_glove_subword_tokenizer_config.json",
        remote_filename="model.safetensors",
        remote_tokenizer_filename="tokenizer.json",
        tensor_key="embeddings",
    )
