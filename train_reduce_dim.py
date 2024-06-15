import logging
from datetime import datetime
from typing import Tuple, Dict, List
from sentence_transformers import (
    SentenceTransformerTrainingArguments,
    SentenceTransformer,
)

from word_llama import load, Config
from word_llama.adapters import MLP, AvgPool
from word_llama.trainers.reduce_dimension import ReduceDimension


class ReduceDimensionConfig:
    """Configuration for Dimension Reduction."""

    matryoshka_dims: List[int] = [2048, 1024, 512, 256, 128, 64]
    tokenizer_kwargs: Dict[str, any] = {
        "return_tensors": "pt",
        "return_attention_mask": True,
        "max_length": 128,
        "padding": "max_length",
        "truncation": True,
        "add_special_tokens": True,
    }
    training_datasets: Dict[str, str] = {
        "train": ("sentence-transformers/all-nli", "triplet"),
        "eval": ("sentence-transformers/all-nli", "triplet"),
        "sts_validation": ("sentence-transformers/stsb",),
    }
    training_args: SentenceTransformerTrainingArguments = (
        SentenceTransformerTrainingArguments(
            output_dir=f"output/matryoshka_sts_custom_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            num_train_epochs=5,
            per_device_train_batch_size=512,
            warmup_steps=0,
            evaluation_strategy="steps",
            eval_steps=64,
            save_steps=32,
            fp16=True,
        )
    )

    def __init__(
        self,
        model_config=Config.llama3_70B,
        model_path="llama3_70B_embedding.safetensors",
        device="cuda",
    ):
        self.model_path = model_path
        self.model_config = model_config
        self.device = device
        self.model = self.build_model()

    def build_model(self) -> SentenceTransformer:
        wl = load(self.model_path, self.model_config)
        proj = MLP(
            self.model_config.model.dim, 2048, hidden_dim=self.model_config.model.dim
        )
        pool = AvgPool()
        return SentenceTransformer(modules=[wl, proj, pool], device=self.device)


if __name__ == "__main__":
    config = ReduceDimensionConfig()
    cst = ReduceDimension(config)
    cst.train()
