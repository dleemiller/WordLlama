# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport abs as c_abs
from libcpp.vector cimport vector
from math import factorial

ctypedef np.float32_t DTYPE_t

np.import_array()


cdef np.ndarray[DTYPE_t, ndim=2] compute_savitzky_golay_coeffs(int window_size, int poly_order):
    """
    Compute Savitzky-Golay filter coefficients.

    Parameters:
        window_size (int): The length of the window. Must be an odd integer.
        poly_order (int): The order of the polynomial used to fit the samples.

    Returns:
        np.ndarray: 2D array of filter coefficients for 0th, 1st, and 2nd derivatives.
    """
    cdef int half_window = (window_size - 1) // 2
    cdef np.ndarray[DTYPE_t, ndim=2] A = np.zeros((window_size, poly_order + 1), dtype=np.float32)
    cdef np.ndarray[DTYPE_t, ndim=2] coeffs = np.zeros((3, window_size), dtype=np.float32)
    cdef int i, j

    for i in range(window_size):
        for j in range(poly_order + 1):
            A[i, j] = pow(i - half_window, j)

    # Compute the pseudoinverse of A
    cdef np.ndarray[DTYPE_t, ndim=2] pinv_A = np.linalg.pinv(A).astype(np.float32)

    # Compute coefficients for 0th, 1st, and 2nd derivatives
    coeffs[0] = pinv_A[0]
    coeffs[1] = pinv_A[1] * 1.0
    coeffs[2] = pinv_A[2] * 2.0

    return coeffs

cdef np.ndarray[DTYPE_t, ndim=1] apply_savitzky_golay_filter(np.ndarray[DTYPE_t, ndim=2] coeffs, DTYPE_t[:] y, int deriv):
    """
    Apply Savitzky-Golay filter to the input data.

    Parameters:
        coeffs (np.ndarray): Precomputed Savitzky-Golay filter coefficients.
        y (DTYPE_t[:]): Input data.
        deriv (int): Derivative order (0, 1, or 2).

    Returns:
        np.ndarray: Filtered data.
    """
    cdef int n = y.shape[0]
    cdef int window_size = coeffs.shape[1]
    cdef int half_window = (window_size - 1) // 2
    cdef np.ndarray[DTYPE_t, ndim=1] y_padded = np.pad(y, (half_window, half_window), mode='edge')
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float32)

    result = np.convolve(y_padded, coeffs[deriv][::-1], mode='valid')

    return result

cpdef tuple find_local_minima(np.ndarray[DTYPE_t, ndim=1] y, int window_size=11, int poly_order=2):
    """
    Public function to find local minima in the input data.

    This function validates input data and then calls an internal function to find minima
    using Savitzky-Golay filtering.

    Parameters:
        y (np.ndarray): Dependent variable (float32 or convertible).
        window_size (int): Window size for the Savitzky-Golay filter.
        poly_order (int): Polynomial order for the Savitzky-Golay filter.

    Returns:
        tuple: Two numpy arrays, one for the indices of the local minima, and one for the values.
    """
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError("Window size must be an odd, positive integer.")
    if poly_order >= window_size:
        raise ValueError("Polynomial order cannot be larger than window size.")
    elif poly_order < 1:
        raise ValueError("Polynomial order must be larger than 1.")

    cdef np.ndarray[DTYPE_t, ndim=1] y_float

    # Ensure that y is a float32 array
    y_float = np.asarray(y, dtype=np.float32)
    
    return _find_local_minima_impl(y_float, window_size, poly_order)

cdef tuple _find_local_minima_impl(DTYPE_t[:] y, int window_size, int poly_order):
    """
    Internal function to find local minima.

    This function applies the Savitzky-Golay filter to compute first and second
    derivatives of the input data and then identifies minima where the first derivative
    changes sign (negative to positive) and the second derivative is positive.

    Parameters:
        y (DTYPE_t[:]): Dependent variable (float32).
        window_size (int): Window size for Savitzky-Golay filtering.
        poly_order (int): Polynomial order for Savitzky-Golay filtering.

    Returns:
        tuple: Two numpy arrays, one for the indices of the local minima, and one for the values.
    """
    cdef int n = y.shape[0]

    # Precompute Savitzky-Golay coefficients
    cdef np.ndarray[DTYPE_t, ndim=2] coeffs = compute_savitzky_golay_coeffs(window_size, poly_order)
    
    # Apply the filter for the first and second derivatives
    cdef np.ndarray[DTYPE_t, ndim=1] dy = apply_savitzky_golay_filter(coeffs, y, deriv=1)
    cdef np.ndarray[DTYPE_t, ndim=1] ddy = apply_savitzky_golay_filter(coeffs, y, deriv=2)

    cdef list minima_indices = []
    cdef list minima_values = []
    cdef DTYPE_t[:] dy_view = dy
    cdef DTYPE_t[:] ddy_view = ddy

    cdef int i
    cdef DTYPE_t interp_weight

    # Identify minima by checking first derivative change and positive second derivative
    for i in range(1, n - 1):
        if dy_view[i] < 0 < dy_view[i + 1] and ddy_view[i] > 0:
            # Calculate the weight of the zero crossing between i and i+1
            interp_weight = -dy_view[i] / (dy_view[i + 1] - dy_view[i])

            # Determine if the zero crossing is closer to i or i+1
            if interp_weight < 0.5:
                minima_indices.append(i)
                minima_values.append(y[i])
            else:
                minima_indices.append(i + 1)
                minima_values.append(y[i + 1])

    # Convert the lists to NumPy arrays
    cdef np.ndarray[np.int32_t, ndim=1] minima_idx_np = np.array(minima_indices, dtype=np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] minima_values_np = np.array(minima_values, dtype=np.float32)

    return minima_idx_np, minima_values_np

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
                if j == k: # skip same
                    continue

                similarity_sum += np.dot(embeddings[j], embeddings[k])
                window_count += 1

        # Compute the average for the window
        averaged[i] = similarity_sum / window_count if window_count > 0 else 0.0

    return averaged
