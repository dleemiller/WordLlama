# ruff: noqa: E402
from __future__ import annotations
import os

# Set environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

import mteb
import logging

from functools import partial
from typing import Any, List
from wordllama import load_training, Config
import numpy as np
from more_itertools import chunked

from mteb.model_meta import ModelMeta

logger = logging.getLogger(__name__)


TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    ## "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    ## "DBPedia",
    ## "FEVER",
    "FiQA2018",
    ## "HotpotQA",
    ## "MSMARCO",
    "NFCorpus",
    ## "NQ",
    ## "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]


class WordLlamaWrapper:
    def __init__(
        self, model_name: str, config, embed_dim: int | None = None, **kwargs
    ) -> None:
        self._model_name = model_name
        self._embed_dim = embed_dim
        print(model_name)
        self.model = load_training(model_name, config, dims=embed_dim).to("cuda")

    def encode(self, sentences: List[str], batch_size=512, **kwargs: Any) -> np.ndarray:
        all_embeddings = []

        for chunk in chunked(sentences, batch_size):
            embed_chunk = (
                self.model.embed(chunk, return_pt=True, norm=True)
                .to("cpu")
                .detach()
                .numpy()
            )
            all_embeddings.append(embed_chunk)

        # Concatenate all chunks into a single numpy array
        concatenated_embeddings = np.concatenate(all_embeddings, axis=0)
        return concatenated_embeddings


if __name__ == "__main__":

    TASK_LIST = (
        TASK_LIST_CLASSIFICATION
        + TASK_LIST_CLUSTERING
        + TASK_LIST_PAIR_CLASSIFICATION
        + TASK_LIST_RERANKING
        + TASK_LIST_RETRIEVAL
        + TASK_LIST_STS
    )

    # all tasks
    from datetime import datetime

    CONFIG_NAME = "l2_supercat"
    DIMS = 512
    BINARY = ""
    wordllama = ModelMeta(
        name=f"wordllama_{CONFIG_NAME}",
        revision="1",
        release_date="2024-07-09",
        languages=["eng-Latn"],
        loader=partial(
            WordLlamaWrapper,
            f"wordllama/weights/{CONFIG_NAME}_{DIMS}{BINARY}.safetensors",
            config=getattr(Config, CONFIG_NAME),
            embed_dim=DIMS,
        ),
        max_tokens=512,
        embed_dim=DIMS,
        open_source=True,
        distance_metric="cosine",
    )

    # tasks = MTEB_MAIN_EN  # or use a specific benchmark
    model = wordllama.load_model()
    evaluation = mteb.MTEB(tasks=TASK_LIST)
    results = evaluation.run(
        model,
        output_folder=f"wordllama_{CONFIG_NAME}_{DIMS}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        overwrite_results=True,
        verbosity=3,
        raise_error=False,
        trust_remote_code=True,
    )
