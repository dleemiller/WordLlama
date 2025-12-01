# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from .vector_similarity cimport vector_similarity

cdef extern from "math.h":
    double sqrt(double)

ctypedef fused embedding_dtype:
    np.float32_t
    np.uint64_t

def deduplicate_embeddings(np.ndarray[embedding_dtype, ndim=2] doc_embeddings,
                           double threshold, int batch_size):
    """
    Identify duplicate document indices based on vector similarity.

    Parameters:
        doc_embeddings (np.ndarray): 2D array of embeddings (float32 or uint64).
        threshold (double): Similarity threshold to consider duplicates.
        batch_size (int): Number of embeddings to process per batch.

    Returns:
        set: Set of duplicate document indices.
    """
    cdef int num_embeddings = doc_embeddings.shape[0]
    cdef set duplicate_indices = set()
    cdef int i, j, start_i, end_i, start_j, end_j
    cdef np.ndarray batch_i
    cdef np.ndarray batch_j
    cdef np.ndarray sim_matrix
    cdef np.ndarray[np.int64_t, ndim=2] sim_indices
    cdef int doc_idx_1, doc_idx_2
    cdef bint binary_flag

    # Determine if embeddings are binary based on dtype
    if doc_embeddings.dtype == np.float32:
        binary_flag = False
    elif doc_embeddings.dtype == np.uint64:
        binary_flag = True
    else:
        raise TypeError("Unsupported embedding dtype. Only float32 and uint64 are supported.")

    for i in range(0, num_embeddings, batch_size):
        start_i = i
        end_i = min(i + batch_size, num_embeddings)
        batch_i = doc_embeddings[start_i:end_i]

        for j in range(i, num_embeddings, batch_size):
            start_j = j
            end_j = min(j + batch_size, num_embeddings)
            batch_j = doc_embeddings[start_j:end_j]

            # Compute similarity matrix
            sim_matrix = vector_similarity(batch_i, batch_j, binary=binary_flag)
            sim_matrix = np.triu(sim_matrix)

            # Find indices where similarity exceeds the threshold
            sim_indices = np.argwhere(sim_matrix > threshold)

            if sim_indices.size > 0:
                # Add document indices in bulk
                doc_idx_1_bulk = sim_indices[:, 0] + start_i
                doc_idx_2_bulk = sim_indices[:, 1] + start_j

                # Only apply self-comparison filtering if i == j
                if i == j:
                    # Filter out self-comparisons (where doc_idx_1 == doc_idx_2)
                    mask = doc_idx_1_bulk != doc_idx_2_bulk
                    doc_idx_1_bulk = doc_idx_1_bulk[mask]
                    doc_idx_2_bulk = doc_idx_2_bulk[mask]

                # Filter out already duplicated documents from doc_idx_1 using np.where
                not_in_duplicates_mask = np.where(
                    np.isin(doc_idx_1_bulk, list(duplicate_indices), invert=True)
                )[0]

                doc_idx_1_bulk = doc_idx_1_bulk[not_in_duplicates_mask]
                doc_idx_2_bulk = doc_idx_2_bulk[not_in_duplicates_mask]

                # Add the second document index in bulk using a set
                duplicate_indices.update(doc_idx_2_bulk)

    return duplicate_indices

