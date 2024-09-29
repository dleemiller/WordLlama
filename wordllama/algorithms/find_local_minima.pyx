# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport abs as c_abs
from libcpp.vector cimport vector
from math import factorial

ctypedef np.float32_t DTYPE_t

np.import_array()

cdef np.ndarray[double, ndim=1] savitzky_golay(double[:] y, int window_size, int order, int deriv=0, double rate=1.0):
    cdef int n = y.shape[0]
    cdef int half_window = (window_size - 1) // 2
    cdef np.ndarray[double, ndim=2] b = np.empty((window_size, order + 1), dtype=np.float64)
    cdef double[:, :] b_view = b
    cdef int i, j, k
    cdef double x
    
    for i in range(window_size):
        x = i - half_window
        for j in range(order + 1):
            b_view[i, j] = x ** j
    
    cdef np.ndarray[double, ndim=1] m = np.linalg.pinv(b)[deriv] * rate**deriv * factorial(deriv)
    cdef double[:] m_view = m
    
    cdef np.ndarray[double, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef double[:] result_view = result
    
    for i in range(n):
        result_view[i] = 0
        for j in range(window_size):
            k = i - half_window + j
            if k < 0:
                result_view[i] += m_view[j] * (2 * y[0] - y[c_abs(k)])
            elif k >= n:
                result_view[i] += m_view[j] * (2 * y[n-1] - y[2*n-k-2])
            else:
                result_view[i] += m_view[j] * y[k]
    
    return result

cpdef tuple find_local_minima(x, y, int window_size=11, int poly_order=2, int dec=1):
    cdef np.ndarray[double, ndim=1] x_double, y_double
    
    if isinstance(x, np.ndarray) and x.dtype != np.float64:
        x_double = x.astype(np.float64)
    else:
        x_double = np.asarray(x, dtype=np.float64)
    
    if isinstance(y, np.ndarray) and y.dtype != np.float64:
        y_double = y.astype(np.float64)
    else:
        y_double = np.asarray(y, dtype=np.float64)
    
    return _find_local_minima_impl(x_double, y_double, window_size, poly_order, dec)

cdef tuple _find_local_minima_impl(double[:] x, double[:] y, int window_size, int poly_order, int dec):
    cdef int n = x.shape[0]
    cdef int n_dec = n // dec
    cdef np.ndarray[double, ndim=1] x_dec = np.empty(n_dec, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] y_dec = np.empty(n_dec, dtype=np.float64)
    cdef int i
    
    for i in range(n_dec):
        x_dec[i] = x[i * dec]
        y_dec[i] = y[i * dec]
    
    cdef np.ndarray[double, ndim=1] dy = savitzky_golay(y_dec, window_size, poly_order, deriv=1)
    cdef np.ndarray[double, ndim=1] ddy = savitzky_golay(y_dec, window_size, poly_order, deriv=2)
    
    cdef vector[int] minima
    cdef double[:] dy_view = dy
    cdef double[:] ddy_view = ddy
    
    for i in range(1, n_dec - 1):
        if dy_view[i-1] < 0 and dy_view[i+1] > 0 and ddy_view[i] > 0:
            minima.push_back(i)
    
    cdef int n_minima = minima.size()
    cdef np.ndarray[double, ndim=1] x_minima = np.empty(n_minima, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] y_minima = np.empty(n_minima, dtype=np.float64)
    
    for i in range(n_minima):
        x_minima[i] = x_dec[minima[i]]
        y_minima[i] = y_dec[minima[i]]
    
    return x_minima, y_minima


cpdef np.ndarray[DTYPE_t, ndim=1] windowed_cross_similarity(np.ndarray[DTYPE_t, ndim=2] embeddings, int window_size):
    """
    Computes the average similarity in a window of size `window_size` for a given embedding matrix
    of shape n×d, skipping diagonal elements in the cross similarity matrix.

    :param embeddings: n×d matrix of embeddings (n lines, d-dimensional embeddings)
    :param window_size: Size of the sliding window for averaging. Must be odd and >= 3.
    :return: A 1D array of size n representing the average similarity over a sliding window, excluding diagonal elements
    """

    cdef int n = embeddings.shape[0]
    
    # Ensure the window size is odd and >= 3
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("Window size must be odd and >= 3")
    
    cdef int half_window = window_size // 2

    # Pre-allocate the output array for storing windowed averages
    cdef np.ndarray[DTYPE_t, ndim=1] averaged = np.zeros(n, dtype=np.float32)

    # Iterate through each row of the embeddings matrix
    cdef int i, j
    cdef DTYPE_t similarity_sum
    cdef int window_count

    # Compute the cross similarity using the sliding window, skipping diagonal elements
    for i in range(n):
        similarity_sum = 0.0
        window_count = 0

        # Sliding window: only compute the dot product for entries within the window, excluding the diagonal
        for j in range(max(0, i - half_window), min(n, i + half_window + 1)):
            for k in range(j, min(n, i + half_window + 1)):
                if i == j and i == k:
                    continue  # Skip the diagonal element

                similarity_sum += np.dot(embeddings[j], embeddings[k])
                window_count += 1

        # Compute the average for the window
        averaged[i] = similarity_sum / window_count if window_count > 0 else 0.0

    return averaged
