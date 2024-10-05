import numpy as np
from typing import List, Tuple, Optional, Union
from itertools import chain
from .find_local_minima import find_local_minima, windowed_cross_similarity
from .splitter import (
    constrained_batches,
    constrained_coalesce,
    split_sentences,
    reverse_merge,
)


class SemanticSplitter:
    """
    A class for semantically splitting text.

    This class provides methods to split text into chunks based on semantic similarity
    and reconstruct them while maintaining semantic coherence.
    """

    @staticmethod
    def flatten(nested_list: List[List[any]]) -> List[any]:
        """
        Flatten a list of lists into a single list.

        Args:
            nested_list (List[List[any]]): A list of lists to be flattened.

        Returns:
            List[any]: A flattened list containing all elements from the nested lists.
        """
        return list(chain.from_iterable(nested_list))

    @staticmethod
    def constrained_split(
        text: str,
        target_size: int,
        separator: str = " ",
        min_size: int = 24,
    ) -> List[str]:
        """
        Split text into chunks of approximately target_size.

        Args:
            text (str): The text to split.
            target_size (int): The target size for each chunk.
            separator (str, optional): The separator to use when joining text. Defaults to " ".
            min_size (int, optional): The minimum size for each chunk. Defaults to 24.

        Returns:
            List[str]: List of text chunks.
        """
        assert target_size > min_size, "Target size must be larger than minimum size."

        sentences = split_sentences(text)
        sentences = constrained_coalesce(sentences, target_size, separator=separator)
        sentences = reverse_merge(sentences, n=min_size, separator=separator)
        return sentences

    @classmethod
    def split(
        cls,
        text: str,
        target_size: int,
        cleanup_size: int = 24,
        intermediate_size: int = 96,
    ) -> List[str]:
        """
        Split the input text into chunks based on semantic coherence.

        Args:
            text (str): The input text to split.
            target_size (int): The target size for final chunks.
            cleanup_size (int, optional): The minimum size for cleaning up small chunks. Defaults to 24.
            intermediate_size (int, optional): The initial size for splitting on newlines. Defaults to 96.

        Returns:
            List[str]: List of text chunks.
        """
        assert (
            target_size > intermediate_size
        ), "Target size must be larger than intermediate size."
        assert (
            intermediate_size > cleanup_size
        ), "Intermediate size must be larger than cleanup size."

        lines = text.splitlines()
        lines = constrained_coalesce(lines, intermediate_size, separator="\n")
        lines = reverse_merge(lines, n=cleanup_size, separator="\n")

        chunks = [
            cls.constrained_split(
                line, target_size, min_size=cleanup_size, separator=" "
            )
            if len(line) > target_size
            else [line]
            for line in lines
        ]

        chunks = cls.flatten(chunks)
        return list(filter(lambda x: bool(x.strip()), chunks))

    @classmethod
    def reconstruct(
        cls,
        lines: List[str],
        norm_embed: np.ndarray,
        target_size: int,
        window_size: int,
        poly_order: int,
        savgol_window: int,
        max_score_pct: float = 0.4,
        return_minima: bool = False,
    ) -> Union[List[str], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Reconstruct text chunks based on semantic similarity.

        Args:
            lines (List[str]): List of text chunks to reconstruct.
            norm_embed (np.ndarray): Normalized embeddings of the text chunks.
            target_size (int): Target size for final chunks.
            window_size (int): Window size for similarity matrix averaging.
            poly_order (int): Polynomial order for Savitzky-Golay filter.
            savgol_window (int): Window size for Savitzky-Golay filter.
            max_score_pct (float, optional): Maximum percentile of similarity scores to consider. Defaults to 0.4.
            return_minima (bool, optional): If True, return minima information instead of reconstructed text. Defaults to False.

        Returns:
            Union[List[str], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
                If return_minima is False, returns a list of reconstructed text chunks.
                If return_minima is True, returns a tuple of (roots, y, sim_avg).

        Raises:
            AssertionError: If the number of texts doesn't equal the number of embeddings.
        """
        assert (
            len(lines) == norm_embed.shape[0]
        ), "Number of texts must equal number of embeddings"

        sim_avg = windowed_cross_similarity(norm_embed, window_size)
        roots, y = find_local_minima(
            sim_avg, poly_order=poly_order, window_size=savgol_window
        )

        if return_minima:
            return roots, y, sim_avg

        (x_idx,) = np.where(y < np.quantile(sim_avg, max_score_pct))
        split_points = [int(x) for i, x in enumerate(roots.tolist()) if i in x_idx]

        chunks = []
        start = 0
        for end in split_points + [len(lines)]:
            chunk = constrained_coalesce(lines[start:end], target_size)
            chunks.extend(chunk)
            start = end

        return list(
            map("".join, constrained_batches(lines, max_size=target_size, strict=False))
        )
