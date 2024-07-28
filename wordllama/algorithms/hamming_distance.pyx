# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from numpy cimport int32_t, uint32_t, uint8_t, PyArrayObject, PyArray_DIMS
from libc.stdint cimport uint32_t, uint8_t

np.import_array()

cdef extern from *:
    """
    #if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    #include <x86intrin.h>
    static inline int popcount(uint32_t x) {
        return __builtin_popcount(x);
    }
    #elif defined(__GNUC__) && (defined(__ARM_NEON) || defined(__aarch64__))
    #include <arm_neon.h>
    static inline int popcount(uint32_t x) {
        return vaddv_u8(vcnt_u8(vcreate_u8(x)));
    }
    #else
    static inline int popcount(uint32_t x) {
        x = x - ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        x = (x + (x >> 4)) & 0x0F0F0F0F;
        x = x + (x >> 8);
        x = x + (x >> 16);
        return x & 0x0000003F;
    }
    #endif
    """
    int popcount(uint32_t x) nogil

cpdef np.ndarray[int32_t, ndim=2] hamming_distance(np.ndarray[uint32_t, ndim=2] a, np.ndarray[uint32_t, ndim=2] b):
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

