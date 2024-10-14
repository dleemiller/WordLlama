# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from numpy cimport (
    uint64_t,
    float32_t,
    uint32_t,
    uint8_t
)

np.import_array()

cdef extern from *:
    """
    #if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    #include <x86intrin.h>
    static inline int popcount(uint64_t x) {
        return __builtin_popcountll(x);
    }
    #elif defined(__GNUC__) && (defined(__ARM_NEON) || defined(__aarch64__))
    #include <arm_neon.h>
    static inline int popcount(uint64_t x) {
        // No direct 64-bit popcount in NEON, need to split into two 32-bit parts
        uint32_t lo = (uint32_t)x;
        uint32_t hi = (uint32_t)(x >> 32);
        return vaddv_u8(vcnt_u8(vcreate_u8(lo))) + vaddv_u8(vcnt_u8(vcreate_u8(hi)));
    }
    #else
    static inline int popcount(uint64_t x) {
        x = x - ((x >> 1) & 0x5555555555555555);
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
        x = x + (x >> 8);
        x = x + (x >> 16);
        x = x + (x >> 32);
        return x & 0x0000007F;
    }
    #endif
    """
    int popcount(uint64_t x) nogil

cpdef object hamming_distance(object a, object b):
    """
    Compute the Hamming distance between two arrays of binary vectors.

    Parameters:
        a (np.ndarray): A 2D array of binary vectors (dtype uint64).
        b (np.ndarray): A 2D array of binary vectors (dtype uint64).

    Returns:
        np.ndarray: A 2D array containing the Hamming distances.
    """
    cdef Py_ssize_t i, j, k
    cdef int dist
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t m = b.shape[0]
    cdef Py_ssize_t width = a.shape[1]
    
    # Allocate distance array
    distance = np.zeros((n, m), dtype=np.uint32)
    
    # Create a typed memoryview
    cdef uint32_t[:, :] distance_view = distance

    # Ensure contiguous
    if not a.flags.c_contiguous or not b.flags.c_contiguous:
        raise ValueError("Input arrays must be C-contiguous")

    # Create typed memoryviews
    cdef uint64_t[:, :] a_view = a
    cdef uint64_t[:, :] b_view = b

    for i in range(n):
        for j in range(m):
            dist = 0
            for k in range(width):
                dist += popcount(a_view[i, k] ^ b_view[j, k])
            distance_view[i, j] = dist

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

