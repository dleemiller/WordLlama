import logging

from typing import Tuple, Dict, List
from datasets import load_dataset, DatasetDict
from sentence_transformers import (
    SentenceTransformer,
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
        self.train_dataset, self.eval_dataset = self.load_datasets()
        self.train_loss = self.setup_loss()
        self.dev_evaluator = self.setup_evaluator()
        self.trainer = self.initialize_trainer()

    def configure_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )

    # def initialize_model(self) -> SentenceTransformer:
    #     wl = load(self.config.model_path, Config.llama3_70B)
    #     proj = MLP(8192, 2048, hidden_dim=8192)
    #     pool = AvgPool()
    #     return SentenceTransformer(modules=[wl, proj, pool], tokenizer_kwargs=self.config.tokenizer_kwargs, device="cuda")

    def load_datasets(self) -> Tuple[DatasetDict, DatasetDict]:
        train_dataset = load_dataset(
            *self.config.training_datasets["train"], split="train"
        )
        eval_dataset = load_dataset(*self.config.training_datasets["eval"], split="dev")
        return train_dataset, eval_dataset

    def setup_loss(self):
        inner_train_loss = losses.MultipleNegativesRankingLoss(model=self.model)
        return losses.MatryoshkaLoss(
            model=self.model,
            loss=inner_train_loss,
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
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            evaluator=self.dev_evaluator,
            loss=self.train_loss,
        )

    def train(self) -> None:
        self.trainer.train()
        self.save_model()

    def save_model(self) -> None:
        self.model.save(self.config.training_args.output_dir)
