import logging
from datetime import datetime
from typing import Tuple, Dict, List
from sentence_transformers import (
    SentenceTransformerTrainingArguments,
    SentenceTransformer,
)
from sentence_transformers.training_args import MultiDatasetBatchSamplers
from datasets import load_dataset

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
        "train": {
            "all-nli": load_dataset(
                "sentence-transformers/all-nli", "triplet", split="train"
            ),
            "all-nli-score": load_dataset(
                "sentence-transformers/all-nli", "pair-score", split="train"
            ),
            "msmarco": load_dataset(
                "sentence-transformers/msmarco-bm25", "triplet", split="train"
            ),
            "stsb": load_dataset("sentence-transformers/stsb", split="train"),
            "quora_duplicates": load_dataset(
                "sentence-transformers/quora-duplicates", "pair", split="train"
            ),
            "natural_questions": load_dataset(
                "sentence-transformers/natural-questions", split="train"
            ),
        },
        "eval": {
            "all-nli": load_dataset(
                "sentence-transformers/all-nli", "triplet", split="dev"
            ),
            "stsb": load_dataset("sentence-transformers/stsb", split="test"),
        },
        "sts_validation": ("sentence-transformers/stsb",),
    }
    loss_types = {
        "all-nli": "mnrl",
        "all-nli-score": "cosent",
        "msmarco": "mnrl",
        "stsb": "cosent",
        "quora_duplicates": "mnrl",
        "natural_questions": "mnrl",
    }
    training_args: SentenceTransformerTrainingArguments = SentenceTransformerTrainingArguments(
        output_dir=f"output/matryoshka_sts_custom_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        num_train_epochs=2,
        per_device_train_batch_size=512,
        warmup_steps=0,
        evaluation_strategy="steps",
        eval_steps=128,
        save_steps=512,
        fp16=True,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
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
