import pandas as pd
import numpy as np
import torch
import logging
from typing import List

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from joblib import dump
from tqdm import tqdm
from dataclasses import dataclass

from word_llama import load


logger = logging.getLogger(__name__)


@dataclass
class LinearSVCConfig:
    C: float = 1.0
    max_iter: int = 1000
    penalty: str = "l2"
    loss: str = "squared_hinge"
    test_size: float = 0.1
    max_length: int = 64
    random_state: int = 42
    batch_size: int = 2048


class WordLlamaLinearSVC:
    def __init__(self, config: LinearSVCConfig):
        self.config = config
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    LinearSVC(
                        C=config.C,
                        penalty=config.penalty,
                        loss=config.loss,
                        max_iter=config.max_iter,
                        random_state=config.random_state,
                    ),
                ),
            ]
        )

    @classmethod
    def build(cls, filepath: str, config: LinearSVCConfig, device: str = "cpu"):
        logger.info(f"Loading WordLlama from {filepath}")
        cls.wl = load(filepath)  # load wordllama
        cls.wl.to(device)
        return cls(config)

    def text_to_embedding(self, texts: List[str]) -> np.array:
        out = self.wl.embed(texts)
        torch.cuda.empty_cache()
        return out.cpu().numpy()

    def batch_generator(self, dataframe):
        dataframe = shuffle(dataframe, random_state=self.config.random_state)
        for i in range(0, len(dataframe), self.config.batch_size):
            batch = dataframe.iloc[i : i + self.config.batch_size]
            yield batch["text"].to_list(), batch["label"]

    def train(self, dataframe):
        train_df, test_df = train_test_split(
            dataframe,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=dataframe["label"],
        )
        X_train, y_train = [], []

        logger.info("Embedding texts using WordLlama...")
        for texts, labels in tqdm(self.batch_generator(train_df), desc="Training"):
            embeddings = self.text_to_embedding(texts)
            X_train.append(embeddings)
            y_train.append(labels)

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)

        logger.info(f"Fitting {self.__class__}...")
        self.model.fit(X_train, y_train)

        return test_df

    def evaluate(self, dataframe):
        X_test, y_test = [], []

        for texts, labels in tqdm(self.batch_generator(dataframe), desc="Evaluating"):
            embeddings = self.text_to_embedding(texts)
            X_test.append(embeddings)
            y_test.append(labels)

        X_test = np.vstack(X_test)
        y_test = np.concatenate(y_test)
        predictions = self.model.predict(X_test)
        report = classification_report(
            y_test, predictions, target_names=["No Profanity", "Profanity"]
        )

        return report

    def save_model(self, filename="wordllama_linearsvc.joblib"):
        dump(self.model, filename)
