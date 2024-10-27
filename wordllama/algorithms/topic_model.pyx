# topic_model.pyx

# Cython directives for optimization
# Disable bounds checking and wraparound for speed
# Ensure that array accesses are safe
# These can also be set in setup.py if preferred
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import cython
from cython cimport boundscheck, wraparound, cdivision
import numpy as np
cimport numpy as np

from collections import Counter
from itertools import islice
import tqdm

# Type definitions for clarity and performance
ctypedef np.float64_t FLOAT
ctypedef np.int64_t INT

# Function to generate n-grams (remains in Python for flexibility)
def generate_ngrams(token_ids, n=4):
    """
    Generate n-grams from the list of token ids.

    Parameters:
    - token_ids: List of token IDs.
    - n: The number of tokens in each n-gram.

    Returns:
    - A generator of n-grams.
    """
    return zip(*(islice(token_ids, i, None) for i in range(n)))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list combine_overlapping_ngrams_cython(list ngrams_with_scores, int n, int k):
    """
    Combine overlapping n-grams within the top-k window.

    Overlapping is defined as sharing (len(ngram) - 1) tokens.

    Parameters:
    - ngrams_with_scores: List of tuples ((ngram), score) sorted by score descending.
    - n: The length of the n-grams.
    - k: The desired number of combined n-grams.

    Returns:
    - List of combined n-grams with their aggregated scores.
    """
    cdef list combined_ngrams = []
    cdef set used_token_ids = set()
    cdef int i, j, overlap, idx
    cdef tuple ngram
    cdef float score
    cdef tuple existing_ngram
    cdef float existing_score
    cdef tuple merged_ngram
    cdef float merged_score
    cdef int token_id
    cdef bint overlap_flag

    for i in range(len(ngrams_with_scores)):
        if len(combined_ngrams) >= k:
            break
        ngram, score = ngrams_with_scores[i]

        # Check if any token_id is already used
        overlap_flag = False
        for token_id in ngram:
            if token_id in used_token_ids:
                overlap_flag = True
                break

        if not overlap_flag:
            combined_ngrams.append((ngram, score))
            for token_id in ngram:
                used_token_ids.add(token_id)
        else:
            # Attempt to merge with existing n-grams
            for idx in range(len(combined_ngrams)):
                existing_ngram, existing_score = combined_ngrams[idx]
                overlap = 0
                for j in range(1, n):
                    # Replace negative indices with positive indices
                    # existing_ngram[-j:] -> existing_ngram[n - j :]
                    # ngram[:j] remains the same
                    if existing_ngram[n - j :] == ngram[:j]:
                        overlap = j
                if overlap == n - 1:
                    # Merge the n-grams
                    # Replace ngram[-1] with ngram[n - 1]
                    merged_ngram = existing_ngram + (ngram[n - 1],)
                    merged_score = existing_score + score  # Aggregation method
                    combined_ngrams[idx] = (merged_ngram, merged_score)
                    used_token_ids.add(ngram[n - 1])
                    break

    return combined_ngrams[:k]


def top_k_token_ngrams(texts, wl, int k=10, int n=3):
    """
    Extract the top-k non-overlapping n-grams from the texts.

    Parameters:
    - texts: List of texts (each text is a string).
    - wl: Language model with tokenizer and embeddings.
    - k: Number of top n-grams to return.
    - n: The number of tokens in each n-gram.

    Returns:
    - List of top-k n-grams with their scores.
    """
    # Use Python's Counter since it's optimized and efficient
    trigram_counter = Counter()
    # Ensure wl.embedding is a NumPy array of type float64
    magnitudes_np = np.linalg.norm(wl.embedding, axis=1, keepdims=True)
    # Cast to float64 to match FLOAT
    magnitudes = magnitudes_np.astype(np.float64)
    cdef np.ndarray[FLOAT, ndim=2] magnitudes_c = magnitudes

    # Iterate over each tokenized text (list of token ids)
    for batch in tqdm.tqdm(texts, desc="Processing texts"):
        tokenized_text = wl.tokenize([batch])
        for x in tokenized_text:
            ngrams = generate_ngrams(x.ids, n)
            trigram_counter.update(ngrams)

    # Get the top 10 * k most common n-grams
    ngrams = trigram_counter.most_common(10 * k)
    importances = []
    counts = []
    cdef tuple ngram
    cdef int count
    cdef float importance
    cdef int i

    for ngram, count in ngrams:
        importance = 0.0
        for token_id in ngram:
            importance += magnitudes_c[token_id, 0]
        importances.append(importance)
        counts.append(count)

    iar = np.array(importances, dtype=np.float64)
    counts_arr = np.array(counts, dtype=np.float64)

    # Compute scores in Python using NumPy's optimized functions
    scores = []
    sorted_iar = np.sort(iar)
    sorted_counts = np.sort(counts_arr)
    for i in range(len(ngrams)):
        p0 = np.searchsorted(sorted_iar, iar[i], side='right') / len(iar)
        p1 = np.searchsorted(sorted_counts, counts_arr[i], side='right') / len(counts_arr)
        score = (p0 + p1) / 2.0
        scores.append(score)

    # Combine ngrams with their scores
    ngrams_with_scores = list(zip([ngram for ngram, _ in ngrams], scores))

    # Sort ngrams by score in descending order
    ngrams_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Combine overlapping ngrams using the Cython-optimized function
    combined_ngrams = combine_overlapping_ngrams_cython(ngrams_with_scores, n, k)

    return combined_ngrams

