from __future__ import annotations
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

import mteb
import logging
from functools import partial
from typing import Any
from word_llama import load, Config
import numpy as np
from more_itertools import chunked

from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts

logger = logging.getLogger(__name__)


config_map = {
    64: Config.wordllama_64,
    128: Config.wordllama_128,
    256: Config.wordllama_256,
    512: Config.wordllama_512,
    1024: Config.wordllama_1024,
}


class WordLlamaWrapper:
    def __init__(
        self, model_name: str, config, embed_dim: int | None = None, **kwargs
    ) -> None:
        self._model_name = model_name
        self._embed_dim = embed_dim
        self.model = load(model_name, config).to("cuda")

    def encode(self, sentences: List[str], batch_size=256, **kwargs: Any) -> np.ndarray:
        all_embeddings = []

        for chunk in chunked(sentences, batch_size):
            embed_chunk = (
                self.model.embed(chunk, return_pt=True).to("cpu").detach().numpy()
            )
            all_embeddings.append(embed_chunk)

        # Concatenate all chunks into a single numpy array
        concatenated_embeddings = np.concatenate(all_embeddings, axis=0)
        return concatenated_embeddings


if __name__ == "__main__":
    # all tasks
    from mteb.benchmarks import MTEB_MAIN_EN
    from datetime import datetime

    DIMS = 1024
    wordllama = ModelMeta(
        name="wordllama",
        revision="1",
        release_date="2024-07-05",
        languages=["eng-Latn"],
        loader=partial(
            WordLlamaWrapper,
            f"weights/wordllama_{DIMS}_binary.safetensors",
            config_map[DIMS],
            embed_dim=DIMS,
        ),
        max_tokens=512,
        embed_dim=DIMS,
        open_source=True,
        distance_metric="cosine",
    )

    tasks = MTEB_MAIN_EN  # or use a specific benchmark
    model = wordllama.load_model()
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        model,
        output_folder=f"wordllama_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        overwrite_results=True,
        verbosity=3,
        raise_error=True,
        trust_remote_code=True,
    )
