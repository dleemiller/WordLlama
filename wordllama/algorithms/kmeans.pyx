# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, fastmath=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport sqrtf, INFINITY, fabs
from libc.float cimport FLT_MAX

cdef extern from "time.h":
    ctypedef long long clock_t
    clock_t clock()
    double CLOCKS_PER_SEC

np.import_array()

ctypedef np.float32_t FLOAT_t
ctypedef np.int64_t INT_t

cdef kmeans_plusplus_initialization(np.ndarray[FLOAT_t, ndim=2] X, int k, object random_state):
    cdef Py_ssize_t n_samples = X.shape[0]
    cdef np.ndarray[FLOAT_t, ndim=2] centers = np.empty((k, X.shape[1]), dtype=np.float32)
    cdef np.ndarray[FLOAT_t, ndim=1] distances = np.empty(n_samples, dtype=np.float32)
    cdef int i, j

    # Choose the first center randomly
    first_center = random_state.randint(n_samples)
    centers[0] = X[first_center]

    for i in range(1, k):
        # Compute distances to the nearest center for each point
        distances = np.min(((X[:, np.newaxis, :] - centers[np.newaxis, :i, :]) ** 2).sum(axis=2), axis=1)
        
        # Choose the next center with probability proportional to distance squared
        probabilities = distances / distances.sum()
        next_center = random_state.choice(n_samples, p=probabilities)
        centers[i] = X[next_center]

    return centers

cdef _kmeans_single(np.ndarray[FLOAT_t, ndim=2] X, int k, int min_iterations, int max_iterations, float tolerance, object random_state):
    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]
    cdef np.ndarray[FLOAT_t, ndim=2] centers
    cdef np.ndarray[FLOAT_t, ndim=2] new_centers
    cdef np.ndarray[INT_t, ndim=1] labels = np.empty(n_samples, dtype=np.int64)
    cdef np.ndarray[FLOAT_t, ndim=2] distances
    cdef float inertia = INFINITY, old_inertia = INFINITY
    cdef int iteration = 0
    cdef int no_improvement_count = 0
    cdef clock_t start, end, total_start, total_end
    cdef double time_taken

    cdef double total_init_time = 0
    cdef double total_distance_time = 0
    cdef double total_assignment_time = 0
    cdef double total_update_time = 0
    cdef double total_convergence_time = 0

    total_start = clock()

    # Time the initialization
    start = clock()
    centers = kmeans_plusplus_initialization(X, k, random_state)
    end = clock()
    total_init_time = (end - start) / CLOCKS_PER_SEC

    for iteration in range(max_iterations):
        # Time distance computation
        start = clock()
        distances = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        end = clock()
        total_distance_time += (end - start) / CLOCKS_PER_SEC
        
        # Time assignment
        start = clock()
        labels = np.argmin(distances, axis=1)
        end = clock()
        total_assignment_time += (end - start) / CLOCKS_PER_SEC

        # Time centroid update
        start = clock()
        new_centers = np.zeros((k, n_features), dtype=np.float32)
        for i in range(k):
            mask = (labels == i)
            if np.any(mask):
                new_centers[i] = X[mask].mean(axis=0)
            else:
                new_centers[i] = X[random_state.randint(n_samples)]
        end = clock()
        total_update_time += (end - start) / CLOCKS_PER_SEC

        # Time convergence check
        start = clock()
        old_inertia = inertia
        inertia = np.sum(distances[np.arange(n_samples), labels])
        
        if iteration >= min_iterations - 1:
            if fabs(old_inertia - inertia) < tolerance * old_inertia:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            
            if no_improvement_count >= 3:  # Stop if no improvement for 3 consecutive iterations
                break
        
        centers = new_centers
        end = clock()
        total_convergence_time += (end - start) / CLOCKS_PER_SEC

    total_end = clock()
    total_time = (total_end - total_start) / CLOCKS_PER_SEC

    print(f"Total iterations: {iteration + 1}")
    print(f"Initialization time: {total_init_time:.6f} seconds")
    print(f"Total distance computation time: {total_distance_time:.6f} seconds")
    print(f"Total assignment time: {total_assignment_time:.6f} seconds")
    print(f"Total centroid update time: {total_update_time:.6f} seconds")
    print(f"Total convergence check time: {total_convergence_time:.6f} seconds")
    print(f"Total kmeans single: {total_time:.2f}")
    print(f"Unaccounted time: {total_time - (total_init_time + total_distance_time + total_assignment_time + total_update_time + total_convergence_time):.6f} seconds")

    return labels, centers, inertia

def kmeans_clustering(np.ndarray[FLOAT_t, ndim=2] X, int k, int n_init=10, int min_iterations=10, int max_iterations=300, float tolerance=1e-4, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    cdef np.ndarray[INT_t, ndim=1] best_labels
    cdef np.ndarray[FLOAT_t, ndim=2] best_centers
    cdef float best_inertia = INFINITY
    cdef int i
    cdef np.ndarray[INT_t, ndim=1] labels
    cdef np.ndarray[FLOAT_t, ndim=2] centers
    cdef float inertia
    cdef clock_t start, end
    cdef double total_time = 0

    start = clock()
    for i in range(n_init):
        labels, centers, inertia = _kmeans_single(X, k, min_iterations, max_iterations, tolerance, random_state)
        if inertia < best_inertia:
            best_labels = labels
            best_centers = centers
            best_inertia = inertia
    end = clock()
    total_time = (end - start) / CLOCKS_PER_SEC

    print(f"Total kmeans clustering time: {total_time:.2f} seconds")

    return best_labels, best_inertia
