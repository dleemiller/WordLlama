import argparse
import logging

from pathlib import Path
import pandas as pd
from word_llama.classifier.linear_svc import WordLlamaLinearSVC, LinearSVCConfig


def setup_logging():
    # Configure logging to write to stdout
    logging.basicConfig(
        level=logging.INFO,  # Set to DEBUG for more verbose output
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],  # This directs logs to stdout
    )


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Train classifiers using Word Llama embeddings"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Select a classifier trainer"
    )

    # sub-command for LinearSVC
    svc_parser = subparsers.add_parser(
        "linear_svc", help="Train a LinearSVC classifier"
    )
    for field in LinearSVCConfig.__annotations__.keys():
        default_value = getattr(LinearSVCConfig(), field)
        svc_parser.add_argument(
            f"--{field}",
            type=type(default_value),
            default=default_value,
            help=f"Set {field}",
        )
    svc_parser.add_argument(
        "filepath", type=Path, help="The path to the pretrained Word Llama model"
    )
    svc_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device to use for tensor operations ("cpu" or "cuda")',
    )

    return parser


def train_linear_svc(args):
    logger = logging.getLogger("train_linear_svc")
    logger.info("Starting training for LinearSVC")
    config = LinearSVCConfig(
        **{k: v for k, v in vars(args).items() if k in LinearSVCConfig.__annotations__}
    )
    classifier = WordLlamaLinearSVC.build(
        filepath=args.filepath, config=config, device=args.device
    )

    dataframe = pd.read_parquet("data/training_data.pqt").dropna()
    logger.info(f"Loaded {len(dataframe)} rows, training...")
    test_df = classifier.train(dataframe)
    report = classifier.evaluate(test_df)

    logger.info("Training complete!")
    logger.info(f"Classification report:\n{report}")
    classifier.save_model()


def main():
    setup_logging()
    parser = setup_parser()
    args = parser.parse_args()

    if args.command == "linear_svc":
        train_linear_svc(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
