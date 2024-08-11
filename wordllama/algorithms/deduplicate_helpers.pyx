# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
from numpy cimport PyArray_DIMS

ctypedef fused embedding_dtype:
    np.uint32_t
    np.uint64_t
    np.float32_t
    np.float64_t

def process_batches_cy(np.ndarray[embedding_dtype, ndim=2] doc_embeddings, 
                       double threshold, int batch_size, vector_similarity):
    cdef int num_embeddings = PyArray_DIMS(doc_embeddings)[0]
    cdef set duplicate_indices = set()
    cdef set seen_docs = set()
    cdef int i, j, start_i, end_i, start_j, end_j
    cdef np.ndarray[embedding_dtype, ndim=2] batch_i, batch_j
    cdef np.ndarray[np.float32_t, ndim=2] sim_matrix
    cdef np.ndarray[np.int64_t, ndim=2] sim_indices
    cdef int doc_idx_1, doc_idx_2
    
    for i in range(0, num_embeddings, batch_size):
        start_i = i
        end_i = min(i + batch_size, num_embeddings)
        batch_i = doc_embeddings[start_i:end_i]
        for j in range(i, num_embeddings, batch_size):
            start_j = j
            end_j = min(j + batch_size, num_embeddings)
            batch_j = doc_embeddings[start_j:end_j]
            sim_matrix = vector_similarity(batch_i, batch_j)
            sim_indices = np.argwhere(sim_matrix > threshold)
            for idx in sim_indices:
                if idx[0] + start_i != idx[1] + start_j:
                    doc_idx_1 = idx[0] + start_i
                    doc_idx_2 = idx[1] + start_j
                    if doc_idx_2 not in seen_docs:
                        seen_docs.add(doc_idx_1)
                        duplicate_indices.add(doc_idx_2)
    
    return duplicate_indices

