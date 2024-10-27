import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound

# Memoryviews allow fast access to arrays
@boundscheck(False)  # Disable bounds checking for performance
@wraparound(False)   # Disable negative index wraparound for performance
cpdef void optimized_search(
    np.ndarray[np.int32_t, ndim=1] query_idx, 
    np.ndarray[np.float32_t, ndim=2] similarity_matrix,
    np.ndarray[np.float32_t, ndim=1] idf_vector,
    np.ndarray[np.float32_t, ndim=1] scores,
    list tokenized_texts,
    float k1, float b, float avg_doc_len, int top_k):

    cdef int i, j, doc_length
    cdef float alpha, beta, fq_sum, score_sum
    cdef np.ndarray[np.int32_t, ndim=1] doc_idx
    cdef np.ndarray[np.float32_t, ndim=1] fq
    cdef np.ndarray[np.int32_t, ndim=2] mesh1, mesh2
    cdef np.ndarray[np.float32_t, ndim=1] temp_score

    alpha = k1 + 1

    # Loop through the documents (tokenized_texts)
    for i in range(len(tokenized_texts)):
        doc = tokenized_texts[i]
        doc_length = np.sum(doc.attention_mask)  # Fast sum via NumPy
        beta = k1 * (1 - b + b * (doc_length / avg_doc_len))
        
        doc_idx = np.array([x for x in doc.ids if x > 0], dtype=np.int32)
        
        # Meshgrid to compute query-document term interactions
        mesh1, mesh2 = np.meshgrid(query_idx, doc_idx, indexing='ij')
        
        # Access similarity_matrix via memoryviews
        fq = similarity_matrix[mesh1, mesh2].sum(axis=0)
        
        # Reshape idf_vector[query_idx] to enable broadcasting
        temp_score = (idf_vector[query_idx].reshape((-1, 1)) * ((fq * alpha) / (fq + beta))).sum(axis=0)
        
        # Sum the score for the document
        scores[i] = np.sum(temp_score)

