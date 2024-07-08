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

from wordllama import load_training, Config
from wordllama.adapters import AvgPool, WeightedProjector, Binarizer
from wordllama.trainers.reduce_dimension import ReduceDimension


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
                # "all-nli-score": load_dataset(
                #    "sentence-transformers/all-nli", "pair-score", split="train"
                # ),
                "msmarco": load_dataset(
                    "sentence-transformers/msmarco-bm25", "triplet", split="train"
                ),
                "msmarco2": load_dataset(
                    "sentence-transformers/msmarco-msmarco-distilbert-base-tas-b",
                    "triplet",
                    split="train",
                ),
                "hotpotqa": load_dataset(
                    "sentence-transformers/hotpotqa", "triplet", split="train"
                ),
                "nli-for-simcse": load_dataset(
                    "sentence-transformers/nli-for-simcse", "triplet", split="train"
                ),
                "mr-tydi": load_dataset(
                    "sentence-transformers/mr-tydi", "en-triplet", split="train"
                ),
                # "stsb": load_dataset("sentence-transformers/stsb", split="train"),
                "compression": load_dataset(
                    "sentence-transformers/sentence-compression", split="train"
                ),
                "agnews": load_dataset("sentence-transformers/agnews", split="train"),
                "gooaq": load_dataset("sentence-transformers/gooaq", split="train"),
                # "flikr": load_dataset("sentence-transformers/flickr30k-captions", split="train"),
                "yahoo": load_dataset(
                    "sentence-transformers/yahoo-answers",
                    "title-question-answer-pair",
                    split="train",
                ),
                "eli5": load_dataset("sentence-transformers/eli5", split="train"),
                "specter": load_dataset(
                    "sentence-transformers/specter", "triplet", split="train"
                ),
                "quora_duplicates": load_dataset(
                    "sentence-transformers/quora-duplicates", "pair", split="train"
                ),
                # "wikianswers_duplicates": load_dataset("sentence-transformers/wikianswers-duplicates", split="train[0:1000000]"),
                # "paq": load_dataset("sentence-transformers/paq", split="train[0:1000000]"),
                "amazon-qa": load_dataset(
                    "sentence-transformers/amazon-qa", split="train[0:1000000]"
                ),
                # "s2orc": load_dataset("sentence-transformers/s2orc", "title-abstract-pair", split="train[0:1000000]"),
                "squad": load_dataset("sentence-transformers/squad", split="train"),
                "stackexchange_bbp": load_dataset(
                    "sentence-transformers/stackexchange-duplicates",
                    "body-body-pair",
                    split="train",
                ),
                "stackexchange_ttp": load_dataset(
                    "sentence-transformers/stackexchange-duplicates",
                    "title-title-pair",
                    split="train",
                ),
                "stackexchange_ppp": load_dataset(
                    "sentence-transformers/stackexchange-duplicates",
                    "post-post-pair",
                    split="train",
                ),
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
        binarize=False,
        binarize_ste="tanh",  # tanh or ste works well
        norm=False,
        device="cuda",
    ):
        self.model_path = model_path
        self.model_config = model_config
        self.device = device
        self.load_datasets()
        self.model = self.build_model()
        self.binarize = binarize  # train with binarize using straight through estimator
        self.binarize_ste = binarize_ste  # option to set binarizer: ste, tanh, reste
        self.norm = norm

    def build_model(self) -> SentenceTransformer:
        wl = load_training(self.model_path, self.model_config)
        wl.tokenizer_kwargs = self.tokenizer_kwargs
        max_dim = max(self.matryoshka_dims)

        # setup modules for sentence transformer
        # best results using weighted projector
        modules = [
            wl,
            WeightedProjector(
                self.model_config.model.dim, max_dim, tokenizer=wl.tokenizer
            ),
            AvgPool(norm=self.norm),
        ]

        # if binarizing, set
        if self.binarize:
            modules.append(Binarizer(ste=self.binarize_ste))

        # train
        return SentenceTransformer(modules=modules, device=self.device)


def check_checkpoint(path):
    """Check if the checkpoint file exists at the path."""
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"The checkpoint path {path} does not exist.")
    return p


def save(model_config, dims, checkpoint, outfile):
    """
    Saves model reduced model weights.
    """
    import torch
    import safetensors.torch
    from word_llama import WordLlamaEmbedding

    p = check_checkpoint(checkpoint)
    proj_path = p / "1_WeightedProjector" / "weighted_projector.safetensors"
    wl = load("llama3_70B_embedding.safetensors", model_config).eval()

    config_map = {  # load a configuration for reduced dimension model
        64: Config.wordllama_64,
        128: Config.wordllama_128,
        256: Config.wordllama_256,
        512: Config.wordllama_512,
        1024: Config.wordllama_1024,
    }
    target = WordLlamaEmbedding(config_map[dims])

    # load the projector weights
    max_dim = max(ReduceDimensionConfig.matryoshka_dims)
    proj = WeightedProjector(model_config.model.dim, max_dim, tokenizer=wl.tokenizer)
    safetensors.torch.load_model(proj, proj_path)

    # inference the trained model
    with torch.no_grad():
        x = proj(
            {
                "token_embeddings": wl.embedding.weight,
                "token_ids": torch.arange(len(wl.tokenizer.vocab)),
            }
        )

        # truncate the matryoshka embedding dimension to the target
        target.embedding.weight = torch.nn.Parameter(x["x"][:, 0:dims])

    # save file
    print(f"Saving to: {outfile}")
    safetensors.torch.save_model(target.half(), outfile)


if __name__ == "__main__":
    import argparse

    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Train a weighted projection model using sentence transformers and Matryoshka Embeddings"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Add a sub-parser for the 'train' command
    parser_train = subparsers.add_parser("train", help="Train the model")
    parser_train.add_argument(
        "--binarize",
        action="store_true",
        default=False,
        help="Train with binarization using straight through estimator",
    )
    parser_train.add_argument(
        "--norm", action="store_true", default=False, help="Norm after pooling"
    )

    # Add a sub-parser for the 'save' command
    parser_save = subparsers.add_parser("save", help="Save the model")
    parser_save.add_argument(
        "--dims",
        type=int,
        required=True,
        help="Dimensions to reduce the model to before saving. Use one of the Matryoksha dims",
    )
    parser_save.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint"
    )
    parser_save.add_argument(
        "--outfile", type=str, required=True, help="File path to save the model"
    )

    # Parse the arguments
    args = parser.parse_args()
    # model_config = Config.llama3_70B
    model_config = Config.mixtral

    # Execute based on the command
    if args.command == "train":
        config = ReduceDimensionConfig(
            model_path="mixtral.safetensors", model_config=model_config, binarize=args.binarize, norm=args.norm
        )
        trainer = ReduceDimension(config)
        trainer.train()

    elif args.command == "save":
        assert args.dims in ReduceDimensionConfig.matryoshka_dims
        save(
            model_config,
            dims=args.dims,
            checkpoint=args.checkpoint,
            outfile=args.outfile,
        )

    else:
        parser.print_help()
