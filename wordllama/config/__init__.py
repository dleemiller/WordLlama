import toml
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Optional


class TokenizerInferenceConfig(BaseModel):
    use_local_config: Optional[bool] = False
    config_filename: Optional[str] = None


class TokenizerConfig(BaseModel):
    return_tensors: str
    return_attention_mask: bool
    max_length: int
    padding: str
    truncation: bool
    add_special_tokens: bool
    inference: Optional[TokenizerInferenceConfig] = None


class TrainingConfig(BaseModel):
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    warmup_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: int
    fp16: bool
    include_num_input_tokens_seen: bool
    learning_rate: float
    multi_dataset_batch_sampler: str
    binarizer_ste: str


class MatryoshkaConfig(BaseModel):
    dims: List[int]


class WordLlamaModel(BaseModel):
    n_vocab: int
    dim: int
    hf_model_id: str
    pad_token: str
    is_encoder: bool = False


class WordLlamaConfig(BaseModel):
    model: WordLlamaModel
    tokenizer: TokenizerConfig
    training: TrainingConfig
    matryoshka: MatryoshkaConfig


class Config:
    _configurations: Dict[str, WordLlamaConfig] = {}

    @classmethod
    def setup(cls):
        """Load configurations from TOML files and set them as class attributes."""
        cls._configurations = cls.load_configurations()
        for config_name, config in cls._configurations.items():
            setattr(cls, config_name, config)  # Set as class attributes for easy access

    @staticmethod
    def load_configurations() -> Dict[str, WordLlamaConfig]:
        """Load configurations from TOML files within the same directory as this script."""
        config_dir = Path(__file__).parent
        configs = {}
        for config_file in config_dir.glob("*.toml"):
            config_data = toml.load(config_file)
            config_name = config_file.stem  # Filename without extension
            configs[config_name] = WordLlamaConfig(**config_data)
        return configs


Config.setup()
