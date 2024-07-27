# cython: language_level=3
import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, uint8_t

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

def hamming_distance(np.ndarray[np.uint32_t, ndim=2] a, np.ndarray[np.uint32_t, ndim=2] b):
    cdef int i, j, k, dist
    cdef int n = a.shape[0]
    cdef int m = b.shape[0]
    cdef int width = a.shape[1]
    cdef np.ndarray[np.int32_t, ndim=2] distance = np.zeros((n, m), dtype=np.int32)
    
    # Calculate Hamming distance
    for i in range(n):
        for j in range(m):
            dist = 0
            for k in range(width):
                dist += popcount(a[i, k] ^ b[j, k])
            distance[i, j] = dist
    
    return distance
