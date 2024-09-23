import logging

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    losses,
)
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SequentialEvaluator,
    SimilarityFunction,
)


logger = logging.getLogger(__name__)


class ReduceDimension:
    """
    Use linear projection or MLP with Matryoshka loss to train a smaller embedding
    that can be truncated.
    """

    def __init__(self, config, device="cuda"):
        self.config = config
        self.configure_logging()
        self.model = config.model
        self.train_loss = self.setup_loss()
        self.dev_evaluator = self.setup_evaluator()
        self.trainer = self.initialize_trainer()

    def configure_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )

    def setup_loss(self):
        return losses.MatryoshkaLoss(
            model=self.model,
            loss=losses.MultipleNegativesRankingLoss(model=self.model),
            matryoshka_dims=self.config.matryoshka_dims,
        )

    def setup_evaluator(self) -> SequentialEvaluator:
        stsb_eval_dataset = load_dataset(
            *self.config.training_datasets["sts_validation"], split="validation"
        )
        evaluators = [
            EmbeddingSimilarityEvaluator(
                sentences1=stsb_eval_dataset["sentence1"],
                sentences2=stsb_eval_dataset["sentence2"],
                scores=stsb_eval_dataset["score"],
                main_similarity=SimilarityFunction.COSINE,
                name=f"sts-dev-{dim}",
                truncate_dim=dim,
            )
            for dim in self.config.matryoshka_dims
        ]
        return SequentialEvaluator(
            evaluators, main_score_function=lambda scores: scores[0]
        )

    def initialize_trainer(self) -> SentenceTransformerTrainer:
        return SentenceTransformerTrainer(
            model=self.model,
            args=self.config.training_args,
            train_dataset=self.config.training_datasets["train"],
            eval_dataset=self.config.training_datasets["eval"],
            evaluator=self.dev_evaluator,
            loss=self.train_loss,
        )

    def train(self) -> None:
        self.trainer.train()
        self.save_model()

    def save_model(self) -> None:
        self.model.save(self.config.training_args.output_dir)
