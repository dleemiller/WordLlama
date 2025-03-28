import difflib
import re
from itertools import chain
from typing import List, Tuple, Union

import numpy as np

from .find_local_minima import find_local_minima, windowed_cross_similarity
from .splitter import (
    constrained_batches,
    constrained_coalesce,
    split_sentences,
    reverse_merge,
)

MAX_SEARCH_RANGE = 500


def build_normalized_mapping(text: str) -> Tuple[str, List[int]]:
    """
    Build a normalized version of `text` by collapsing sequences of whitespace
    into a single space. Also create a mapping from positions in the normalized
    string back to positions in the original text.
    """
    normalized_chars = []
    mapping = (
        []
    )  # mapping[i] is the index in `text` corresponding to normalized_chars[i]
    in_whitespace = False
    for idx, c in enumerate(text):
        if c.isspace():
            if not in_whitespace:
                normalized_chars.append(" ")
                mapping.append(
                    idx
                )  # record the starting index of this whitespace block
                in_whitespace = True
        else:
            normalized_chars.append(c)
            mapping.append(idx)
            in_whitespace = False
    norm_text = "".join(normalized_chars).strip()
    # Adjust mapping if we stripped leading spaces
    leading = 0
    while leading < len(normalized_chars) and normalized_chars[leading] == " ":
        leading += 1
    return norm_text, mapping[leading:]


def fuzzy_find(
    norm_S: str, norm_chunk: str, start_index: int, threshold: float = 0.8
) -> int:
    """
    Search for norm_chunk in norm_S starting at start_index using a sliding window,
    but only search up to a maximum range to leverage sequential order.

    Returns the best matching start index if the similarity ratio exceeds the threshold;
    otherwise returns -1.
    """
    best_ratio = 0.0
    best_idx = -1
    window_size = len(norm_chunk)
    # Limit the search window to MAX_SEARCH_RANGE characters beyond start_index.
    search_end = min(len(norm_S) - window_size + 1, start_index + MAX_SEARCH_RANGE)
    for i in range(start_index, search_end):
        candidate = norm_S[i : i + window_size]
        ratio = difflib.SequenceMatcher(None, norm_chunk, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
        # Early exit if nearly identical
        if best_ratio > 0.98:
            break
    if best_ratio >= threshold:
        return best_idx
    return -1


def postprocess_chunks(original_text: str, chunks: List[str]) -> List[str]:
    """
    Given the original text and the list of chunks (which may have altered whitespace),
    re-align the chunks to match the original text exactly.

    This function first normalizes the original text and each chunk, then searches for
    the normalized chunk in the normalized original text starting at a moving offset.
    If an exact match is not found, a fuzzy matching search is used within a restricted
    search range.

    Returns a list of corrected chunks
    """
    norm_S, mapping = build_normalized_mapping(original_text)
    corrected_chunks = []
    current_norm_index = 0

    for chunk in chunks:
        # Normalize the chunk (collapse whitespace)
        norm_chunk = re.sub(r"\s+", " ", chunk).strip()
        idx = norm_S.find(norm_chunk, current_norm_index)
        if idx == -1:
            # Fall back to fuzzy matching within the next MAX_SEARCH_RANGE characters.
            idx = fuzzy_find(norm_S, norm_chunk, current_norm_index, threshold=0.8)
            if idx == -1:
                raise ValueError(
                    f"Unable to find chunk: {norm_chunk!r} in normalized text."
                )
        # Map normalized start/end to original text indices.
        orig_start = mapping[idx]
        # Mapping for the last character; add 1 because Python slicing is exclusive at the end.
        orig_end = mapping[idx + len(norm_chunk) - 1] + 1
        corrected_chunk = original_text[orig_start:orig_end]
        corrected_chunks.append(corrected_chunk)
        # Update current_norm_index to start the next search immediately after the current chunk.
        current_norm_index = idx + len(norm_chunk)

    return corrected_chunks


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
            (
                cls.constrained_split(
                    line, target_size, min_size=cleanup_size, separator=" "
                )
                if len(line) > target_size
                else [line]
            )
            for line in lines
        ]

        chunks = cls.flatten(chunks)
        return list(filter(lambda x: bool(x.strip()), chunks))

    @classmethod
    def reconstruct(
        cls,
        text: str,
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
            text (str): Original string to reference for reconstruction
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

        chunks = list(
            map("".join, constrained_batches(lines, max_size=target_size, strict=False))
        )

        return postprocess_chunks(text, chunks)
