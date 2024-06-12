import toml

from enum import Enum, auto
from pathlib import Path
from pydantic import BaseModel
from typing import Dict


class WordLlamaModel(BaseModel):
    n_vocab: int
    dim: int
    hf_model_id: str


class WordLlamaConfig(BaseModel):
    model: WordLlamaModel


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
            configs[config_name] = WordLlamaConfig.model_validate(config_data)
        return configs


Config.setup()
