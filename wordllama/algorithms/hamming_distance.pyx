# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from numpy cimport uint8_t, int32_t, uint64_t, PyArrayObject, PyArray_DIMS
from libc.stdint cimport uint64_t

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

def binarize_and_packbits(np.ndarray[float, ndim=2] x):
    cdef int i, j
    cdef int n = x.shape[0]
    cdef int m = x.shape[1]
    cdef int packed_length = (m + 7) // 8
    cdef np.ndarray[uint8_t, ndim=2] packed_x = np.zeros((n, packed_length), dtype=np.uint8)
    
    for i in range(n):
        for j in range(m):
            if x[i, j] > 0:
                packed_x[i, j // 8] |= (1 << (7 - (j % 8)))
    
    return packed_x.view(np.uint64)

cpdef np.ndarray[int32_t, ndim=2] hamming_distance(np.ndarray[uint64_t, ndim=2] a, np.ndarray[uint64_t, ndim=2] b):
    cdef Py_ssize_t i, j, k
    cdef int dist
    cdef Py_ssize_t n = PyArray_DIMS(a)[0]
    cdef Py_ssize_t m = PyArray_DIMS(b)[0]
    cdef Py_ssize_t width = PyArray_DIMS(a)[1]
    cdef np.ndarray[int32_t, ndim=2] distance = np.zeros((n, m), dtype=np.int32)
    
    # Calculate Hamming distance
    for i in range(n):
        for j in range(m):
            dist = 0
            for k in range(width):
                dist += popcount(a[i, k] ^ b[j, k])
            distance[i, j] = dist
    
    return distance

