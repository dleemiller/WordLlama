import torch
import safetensors.torch
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers import (
    SentenceTransformerTrainingArguments,
    SentenceTransformer,
)
from sentence_transformers.training_args import MultiDatasetBatchSamplers

from wordllama.trainers.reduce_dimension import ReduceDimension
from wordllama import load_training, Config
from wordllama.adapters import AvgPool, WeightedProjector, Binarizer
from dataset_loader import load_datasets


class ReduceDimensionConfig:
    """Configuration for Dimension Reduction."""

    def __init__(self, config_name: str):
        self.config = getattr(Config, config_name)
        self.config_name = config_name

        # Load Matryoshka dimensions from config
        self.matryoshka_dims = self.config.matryoshka.dims

        # Load tokenizer kwargs from config
        self.tokenizer_kwargs = self.config.tokenizer.dict()
        training_args = self.config.training
        self.model_path = f"{config_name}.safetensors"
        self.device = "cuda"
        self.binarize = False
        self.binarize_ste = training_args.binarizer_ste
        self.norm = False
        self.model = self.build_model()

        # Load training datasets
        self.training_datasets = load_datasets()

        # Load training arguments from config
        self.training_args = SentenceTransformerTrainingArguments(
            output_dir=f"{training_args.output_dir}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            num_train_epochs=training_args.num_train_epochs,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            warmup_steps=training_args.warmup_steps,
            evaluation_strategy=training_args.evaluation_strategy,
            eval_steps=training_args.eval_steps,
            save_steps=training_args.save_steps,
            fp16=training_args.fp16,
            include_num_input_tokens_seen=training_args.include_num_input_tokens_seen,
            learning_rate=training_args.learning_rate,
            multi_dataset_batch_sampler=(
                MultiDatasetBatchSamplers.PROPORTIONAL
                if training_args.multi_dataset_batch_sampler == "PROPORTIONAL"
                else MultiDatasetBatchSamplers.ROUND_ROBIN
            ),
        )

    def build_model(self) -> SentenceTransformer:
        wl = load_training(self.model_path, self.config)
        wl.tokenizer_kwargs = self.tokenizer_kwargs
        max_dim = max(self.matryoshka_dims)

        # setup modules for sentence transformer
        # best results using weighted projector
        modules = [
            wl,
            WeightedProjector(self.config.model.dim, max_dim, tokenizer=wl.tokenizer),
            AvgPool(norm=self.norm),
        ]

        # if binarizing, set
        if self.binarize:
            modules.append(Binarizer(ste=self.binarize_ste))

        # train
        return SentenceTransformer(modules=modules, device=self.device)

    def save(self, checkpoint: Path, outdir: Path):
        """
        Saves model reduced model weights for each dimension in matryoshka_dims.
        """
        wl = load_training(checkpoint / self.model_path, self.config).eval()

        # load the projector weights
        max_dim = max(self.matryoshka_dims)
        proj_path = (
            checkpoint / "1_WeightedProjector" / "weighted_projector.safetensors"
        )
        proj = WeightedProjector(self.config.model.dim, max_dim, tokenizer=wl.tokenizer)
        safetensors.torch.load_model(proj, proj_path)

        for dims in self.matryoshka_dims:
            target = WordLlamaEmbedding(
                WordLlamaModel(
                    n_vocab=self.config.model.n_vocab,
                    dim=dims,
                    hf_model_id=self.config.model.hf_model_id,
                    pad_token=self.config.model.pad_token,
                )
            )

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
            outfile = outdir / f"{self.config_name}_{dims}.safetensors"
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
        "--checkpoint", type=str, required=True, help="Path to the checkpoint"
    )
    parser_save.add_argument(
        "--outdir", type=str, required=True, help="Directory to save the models"
    )

    # Parse the arguments
    args = parser.parse_args()
    # model_config = Config.llama3_70B
    model_config = Config.mixtral

    # Execute based on the command
    if args.command == "train":
        config = ReduceDimensionConfig("mixtral")
        config.binarize = args.binarize
        config.norm = args.norm
        trainer = ReduceDimension(config)
        trainer.train()

    elif args.command == "save":
        config = ReduceDimensionConfig("mixtral")
        config.save(checkpoint=Path(args.checkpoint), outdir=Path(args.outdir))

    else:
        parser.print_help()
