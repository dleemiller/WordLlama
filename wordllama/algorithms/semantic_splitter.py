import numpy as np
from typing import List
from itertools import chain
from .find_local_minima import window_average, find_local_minima
from .splitter import constrained_coalesce, split_sentences


class SemanticSplitter:
    """A class for semantically splitting and reconstructing text."""

    @staticmethod
    def flatten(nested_list: List[List]) -> List:
        """Flatten a list of lists into a single list."""
        return list(chain.from_iterable(nested_list))

    @staticmethod
    def constrained_split(text: str, target_size: int) -> List[str]:
        """
        Split text into chunks of approximately target_size.

        Parameters:
        - text (str): The text to split.
        - target_size (int): The target size for each chunk.

        Returns:
        - List[str]: List of text chunks.
        """
        sentences = split_sentences(text)
        return constrained_coalesce(sentences, target_size, separator=" ")

    @classmethod
    def split(cls, text: str, target_size: int, initial_split_size: int) -> List[str]:
        """
        Split the input text into chunks.

        Parameters:
        - text (str): The input text to split.
        - target_size (int): The target size for final chunks.
        - initial_split_size (int): The initial size for splitting on newlines.

        Returns:
        - List[str]: List of text chunks.
        """
        lines = constrained_coalesce(
            text.splitlines(), initial_split_size, separator="\n"
        )
        chunks = [
            cls.constrained_split(line, target_size)
            if len(line) > target_size
            else [line]
            for line in lines
        ]
        chunks = cls.flatten(chunks)
        return [chunk for chunk in chunks if chunk.strip()]

    @classmethod
    def reconstruct(
        cls,
        lines: List[str],
        x_sim: np.ndarray,
        target_size: int,
        window_size: int,
        poly_order: int,
        savgol_window: int,
        max_score_pct: float = 0.4,
    ) -> List[str]:
        """
        Reconstruct text chunks based on semantic similarity.

        Parameters:
        - lines (List[str]): List of text chunks to reconstruct.
        - x_sim (np.ndarray): Cross-similarity matrix of text chunks.
        - target_size (int): Target size for final chunks.
        - window_size (int): Window size for similarity matrix averaging.
        - poly_order (int): Polynomial order for Savitzky-Golay filter.
        - savgol_window (int): Window size for Savitzky-Golay filter.

        Returns:
        - List[str]: List of semantically split text chunks.
        """
        sim_avg = window_average(x_sim, window_size)
        x = np.arange(len(sim_avg))
        roots, y = find_local_minima(
            x, sim_avg, poly_order=poly_order, window_size=savgol_window
        )
        split_points = np.round(roots).astype(int).tolist()

        # filter to minima within bottom 40% of similarity scores
        (x_idx,) = np.where(y < np.quantile(sim_avg, max_score_pct))
        split_points = [x for i, x in enumerate(split_points) if i in x_idx]

        # reconstruct using the minima as boundaries for coalesce
        # this ensures that any semantic boundaries are respected
        chunks = []
        start = 0
        for end in split_points + [len(lines)]:
            chunk = constrained_coalesce(lines[start:end], target_size)
            chunks.extend(chunk)
            start = end

        chunks = constrained_coalesce(chunks, target_size)
        return chunks
