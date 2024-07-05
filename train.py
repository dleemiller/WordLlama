import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List
from sentence_transformers import (
    SentenceTransformerTrainingArguments,
    SentenceTransformer,
)
from sentence_transformers.training_args import MultiDatasetBatchSamplers
from datasets import load_dataset

from word_llama import load, Config
from word_llama.adapters import AvgPool, WeightedProjector, Binarizer
from word_llama.trainers.reduce_dimension import ReduceDimension


class ReduceDimensionConfig:
    """Configuration for Dimension Reduction."""

    matryoshka_dims: List[int] = [1024, 512, 256, 128, 64]
    tokenizer_kwargs: Dict[str, any] = {
        "return_tensors": "pt",
        "return_attention_mask": True,
        "max_length": 128,
        "padding": "max_length",
        "truncation": True,
        "add_special_tokens": True,
    }

    @classmethod
    def load_datasets(cls):
        cls.training_datasets = {
            "train": {
                "all-nli": load_dataset(
                    "sentence-transformers/all-nli", "triplet", split="train"
                ),
                #"all-nli-score": load_dataset(
                #    "sentence-transformers/all-nli", "pair-score", split="train"
                #),
                "msmarco": load_dataset(
                    "sentence-transformers/msmarco-bm25", "triplet", split="train"
                ),
                "msmarco2": load_dataset(
                    "sentence-transformers/msmarco-msmarco-distilbert-base-tas-b", "triplet", split="train"
                ),
                "hotpotqa": load_dataset("sentence-transformers/hotpotqa", "triplet", split="train"),
                "nli-for-simcse": load_dataset("sentence-transformers/nli-for-simcse", "triplet", split="train"),
                "mr-tydi": load_dataset("sentence-transformers/mr-tydi", "en-triplet", split="train"),
                #"stsb": load_dataset("sentence-transformers/stsb", split="train"),
                "compression": load_dataset("sentence-transformers/sentence-compression", split="train"),
                "agnews": load_dataset("sentence-transformers/agnews", split="train"),
                "gooaq": load_dataset("sentence-transformers/gooaq", split="train"),
                #"flikr": load_dataset("sentence-transformers/flickr30k-captions", split="train"),
                "yahoo": load_dataset("sentence-transformers/yahoo-answers", "title-question-answer-pair", split="train"),
                "eli5": load_dataset("sentence-transformers/eli5", split="train"),
                "specter": load_dataset("sentence-transformers/specter", "triplet", split="train"),
                "quora_duplicates": load_dataset(
                    "sentence-transformers/quora-duplicates", "pair", split="train"
                ),
                #"wikianswers_duplicates": load_dataset("sentence-transformers/wikianswers-duplicates", split="train[0:1000000]"),
                #"paq": load_dataset("sentence-transformers/paq", split="train[0:1000000]"),
                "amazon-qa": load_dataset("sentence-transformers/amazon-qa", split="train[0:1000000]"),
                #"s2orc": load_dataset("sentence-transformers/s2orc", "title-abstract-pair", split="train[0:1000000]"),
                "squad": load_dataset("sentence-transformers/squad", split="train"),
                "stackexchange_bbp": load_dataset("sentence-transformers/stackexchange-duplicates", "body-body-pair", split="train"),
                "stackexchange_ttp": load_dataset("sentence-transformers/stackexchange-duplicates", "title-title-pair", split="train"),
                "stackexchange_ppp": load_dataset("sentence-transformers/stackexchange-duplicates", "post-post-pair", split="train"),
                "quora_triplets": load_dataset(
                    "sentence-transformers/quora-duplicates", "triplet", split="train"
                ),
                "natural_questions": load_dataset(
                    "sentence-transformers/natural-questions", split="train"
                ),
                "altlex": load_dataset("sentence-transformers/altlex", split="train"),
            },
            "eval": {
                "all-nli": load_dataset(
                    "sentence-transformers/all-nli", "triplet", split="dev"
                ),
                "stsb": load_dataset("sentence-transformers/stsb", split="test"),
            },
            "sts_validation": ("sentence-transformers/stsb",),
        }


    training_args: SentenceTransformerTrainingArguments = SentenceTransformerTrainingArguments(
        output_dir=f"output/matryoshka_sts_custom_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        num_train_epochs=2,
        per_device_train_batch_size=512,
        warmup_steps=256,
        evaluation_strategy="steps",
        eval_steps=250,
        save_steps=1000,
        fp16=True,
        include_num_input_tokens_seen=False,
        learning_rate=1e-2,
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
        self.load_datasets()
        self.model = self.build_model()

    def build_model(self) -> SentenceTransformer:
        wl = load(self.model_path, self.model_config)
        wl.tokenizer_kwargs = self.tokenizer_kwargs
        max_dim = max(self.matryoshka_dims)
        #proj = WeightedMLP(self.model_config.model.dim, max_dim, tokenizer=wl.tokenizer, hidden_dim=4096)
        proj = WeightedProjector(self.model_config.model.dim, max_dim, tokenizer=wl.tokenizer)
        return SentenceTransformer(modules=[wl, proj, AvgPool(), Binarizer(ste="tanh")], device=self.device)

def check_checkpoint(path):
    """Check if the checkpoint file exists at the path."""
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"The checkpoint path {path} does not exist.")
    return p

def save(model_config, dims, checkpoint, outfile):
    import torch
    import safetensors.torch
    from word_llama import WordLlamaEmbedding

    p = check_checkpoint(checkpoint)
    proj_path = p / "1_WeightedProjector" / "weighted_projector.safetensors"
    wl = load("llama3_70B_embedding.safetensors", model_config).eval()
    target = WordLlamaEmbedding(Config.wordllama_64)
    #proj = WeightedMLP(8192, 1024, tokenizer=wl.tokenizer, hidden_dim=6144).eval()
    #proj = WeightedProjector(model_config.model.dim, 1024, tokenizer=wl.tokenizer)
 
    safetensors.torch.load_model(proj, proj_path)
    with torch.no_grad():
        x = proj({"token_embeddings": wl.embedding.weight, "token_ids":torch.arange(len(wl.tokenizer.vocab))})
        target.embedding.weight = torch.nn.Parameter(x["x"][:, 0:dims])
    print(f"Saving to: {outfile}")
    safetensors.torch.save_model(target.half(), outfile)


if __name__ == "__main__":
    import argparse

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Reduce Dimension Model Operations")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Add a sub-parser for the 'train' command
    parser_train = subparsers.add_parser('train', help='Train the model')

    # Add a sub-parser for the 'save' command
    parser_save = subparsers.add_parser('save', help='Save the model')
    parser_save.add_argument('--dims', type=int, required=True, help='Dimensions to reduce the model to before saving. Use one of the Matryoksha dims')
    parser_save.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint")
    parser_save.add_argument('--outfile', type=str, required=True, help='File path to save the model')

    # Parse the arguments
    args = parser.parse_args()
    model_config = Config.llama3_70B

    # Execute based on the command
    if args.command == 'train':
        config = ReduceDimensionConfig()
        trainer = ReduceDimension(config)
        trainer.train()

    elif args.command == 'save':
        assert args.dims in ReduceDimensionConfig.matryoshka_dims
        save(model_config, dims=args.dims, checkpoint=args.checkpoint, outfile=args.outfile)

    else:
        parser.print_help()
