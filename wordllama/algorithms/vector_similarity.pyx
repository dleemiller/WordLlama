# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from numpy cimport (
    uint64_t,
    float32_t,
    uint32_t,
    uint8_t
)

np.import_array()

cpdef object hamming_distance(np.ndarray[np.uint64_t, ndim=2, mode='c'] a,
                              np.ndarray[np.uint64_t, ndim=2, mode='c'] b):
    """
    Compute the Hamming distance between two arrays of binary vectors.

    Parameters:
        a (np.ndarray): A 2D array of binary vectors (dtype uint64).
        b (np.ndarray): A 2D array of binary vectors (dtype uint64).

    Returns:
        np.ndarray: A 2D array containing the Hamming distances.
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t m = b.shape[0]
    cdef Py_ssize_t width = a.shape[1]

    if not a.flags.c_contiguous or not b.flags.c_contiguous:
        raise ValueError("Input arrays must be C-contiguous")

    cdef np.ndarray[np.uint32_t, ndim=2, mode='c'] distance = np.zeros((n, m), dtype=np.uint32)
    cdef np.ndarray[np.uint64_t, ndim=1] a_row
    cdef np.ndarray[np.uint64_t, ndim=2] xor_result
    cdef np.ndarray[np.uint8_t, ndim=2] popcounts
    cdef np.ndarray[np.uint32_t, ndim=1] distances_i

    for i in range(n):
        a_row = a[i, :]

        # XOR 'a_row' and all rows in 'b'
        xor_result = np.bitwise_xor(a_row[np.newaxis, :], b)

        # Compute popcounts
        popcounts = np.bitwise_count(xor_result)

        # Sum to get Hamming distance
        distances_i = np.sum(popcounts, axis=1, dtype=np.uint32)
        distance[i, :] = distances_i

    return distance

cpdef object vector_similarity(
    object a,
    object b,
    bint binary
):
    """
    Calculate similarity between vectors, using either Hamming or cosine similarity.

    Parameters:
        a (np.ndarray): A 1D or 2D array of vectors.
        b (np.ndarray): A 1D or 2D array of vectors.
        binary (bool): If True, use Hamming similarity; otherwise, use cosine similarity.

    Returns:
        np.ndarray: A 2D array of similarity scores between vectors in `a` and `b`.
    """
    cdef object dist = None
    cdef object similarity = None
    cdef float32_t max_distance = 0.0

    # Ensure inputs are 2D
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]

    if binary:
        if a.dtype != np.uint64:
            a_binary = a.astype(np.uint64, copy=False)
        else:
            a_binary = a

        if b.dtype != np.uint64:
            b_binary = b.astype(np.uint64, copy=False)
        else:
            b_binary = b

        # Compute Hamming distance
        dist = hamming_distance(a_binary, b_binary)

        max_distance = a_binary.shape[1] * 64
        if max_distance == 0:
            raise ValueError("Binary vectors must have at least one bit")
        
        # convert to similarity
        similarity = 1.0 - 2.0 * (dist / max_distance).astype(np.float32)

        return similarity
    else:
        if a.dtype != np.float32:
            a_dense = a.astype(np.float32, copy=False)
        else:
            a_dense = a

        if b.dtype != np.float32:
            b_dense = b.astype(np.float32, copy=False)
        else:
            b_dense = b

        # Normalize embeddings
        norms_a = np.linalg.norm(a_dense, axis=1, keepdims=True)
        norms_b = np.linalg.norm(b_dense, axis=1, keepdims=True)

        # div by zero check
        norms_a = np.where(norms_a == 0, 1.0, norms_a)
        norms_b = np.where(norms_b == 0, 1.0, norms_b)

        a_normalized = a_dense / norms_a
        b_normalized = b_dense / norms_b

        # Compute cosine similarity
        similarity = np.dot(a_normalized, b_normalized.T).astype(np.float32)

        return similarity

cpdef object binarize_and_packbits(object x):
    """
    Binarize and pack bits from a 2D float array.

    Parameters:
        x (np.ndarray): A 2D array of floats.

    Returns:
        np.ndarray: A 2D array of packed bits viewed as uint64.
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n, m, packed_length
    cdef object packed_x
    cdef uint8_t[:, :] packed_view
    cdef float32_t[:, :] x_view
    cdef int bit_position

    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    # Check dims
    if x.ndim != 2:
        raise ValueError("Input array must be 2D")

    # Ensure dtype is float32
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)

    # Assign memoryview
    x_view = x

    # Get shape
    n = x.shape[0]
    m = x.shape[1]

    # Compute packed length
    packed_length = (m + 7) // 8

    # Initialize packed_x
    packed_x = np.zeros((n, packed_length), dtype=np.uint8)
    packed_view = packed_x  # to view

    # Iterate and set bits
    for i in range(n):
        for j in range(m):
            if x_view[i, j] > 0:
                bit_position = 7 - (j % 8)
                packed_view[i, j // 8] |= (1 << bit_position)

    # Return as uint64 view
    return packed_x.view(np.uint64)

