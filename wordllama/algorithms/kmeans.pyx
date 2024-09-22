# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, fastmath=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
from numpy.random import RandomState
cimport numpy as np
from libc.math cimport sqrtf, INFINITY, fabsf
from libc.float cimport FLT_MAX
import logging

cdef extern from "time.h":
    ctypedef long long clock_t
    clock_t clock()
    double CLOCKS_PER_SEC

np.import_array()

ctypedef np.float32_t FLOAT_t
ctypedef np.int64_t INT_t

logger = logging.getLogger('kmeans_logger')

cdef inline float squared_euclidean_distance(const float[:] vec1, const float[:] vec2, Py_ssize_t dim) nogil:
    cdef Py_ssize_t i
    cdef float dist = 0.0
    for i in range(dim):
        dist += (vec1[i] - vec2[i]) ** 2
    return dist

cdef void compute_distances(const float[:, :] embeddings, const float[:, :] centroids, float[:, :] distances):
    cdef Py_ssize_t num_points = embeddings.shape[0]
    cdef Py_ssize_t num_centroids = centroids.shape[0]
    cdef Py_ssize_t dim = embeddings.shape[1]
    cdef Py_ssize_t i, j
    cdef float dist

    for i in range(num_points):
        for j in range(num_centroids):
            dist = squared_euclidean_distance(embeddings[i], centroids[j], dim)
            distances[i, j] = sqrtf(dist)

cdef update_centroids(const float[:, :] embeddings, const INT_t[:] labels, Py_ssize_t num_clusters):
    cdef Py_ssize_t num_points = embeddings.shape[0]
    cdef Py_ssize_t dim = embeddings.shape[1]
    cdef float[:, :] new_centroids = np.zeros((num_clusters, dim), dtype=np.float32)
    cdef INT_t[:] count = np.zeros(num_clusters, dtype=np.int64)
    cdef Py_ssize_t i, j
    cdef INT_t label

    for i in range(num_points):
        label = labels[i]
        for j in range(dim):
            new_centroids[label, j] += embeddings[i, j]
        count[label] += 1

    for i in range(num_clusters):
        if count[i] > 0:
            for j in range(dim):
                new_centroids[i, j] /= count[i]

    return np.asarray(new_centroids)

cdef kmeans_plusplus_initialization(float[:, :] embeddings, int k, object random_state):
    cdef Py_ssize_t n_samples = embeddings.shape[0]
    cdef Py_ssize_t n_features = embeddings.shape[1]
    cdef np.ndarray[FLOAT_t, ndim=2] centroids = np.empty((k, n_features), dtype=np.float32)
    cdef np.ndarray[FLOAT_t, ndim=1] distances = np.empty(n_samples, dtype=np.float32)
    cdef np.ndarray[FLOAT_t, ndim=1] probabilities
    cdef np.ndarray[FLOAT_t, ndim=1] cumulative_probabilities
    cdef float[:, :] centroids_view = centroids
    cdef float[:] distances_view = distances
    cdef Py_ssize_t i, j, index
    cdef float r, dist_sum

    index = random_state.randint(n_samples)
    for j in range(n_features):
        centroids_view[0, j] = embeddings[index, j]

    for i in range(n_samples):
        distances_view[i] = squared_euclidean_distance(embeddings[i], centroids_view[0], n_features)

    for i in range(1, k):
        probabilities = distances / np.sum(distances)
        cumulative_probabilities = np.cumsum(probabilities)
        r = random_state.rand()
        index = np.searchsorted(cumulative_probabilities, r)
        
        for j in range(n_features):
            centroids_view[i, j] = embeddings[index, j]

        for j in range(n_samples):
            new_dist = squared_euclidean_distance(embeddings[j], centroids_view[i], n_features)
            if new_dist < distances_view[j]:
                distances_view[j] = new_dist

    return centroids

cdef _kmeans_single(np.ndarray[FLOAT_t, ndim=2] X, int k, int min_iterations, int max_iterations, float tolerance, object random_state):
    cdef Py_ssize_t n_samples = X.shape[0]
    cdef np.ndarray[FLOAT_t, ndim=2] centers
    cdef np.ndarray[FLOAT_t, ndim=2] new_centers
    cdef np.ndarray[INT_t, ndim=1] labels = np.empty(n_samples, dtype=np.int64)
    cdef np.ndarray[FLOAT_t, ndim=2] distances = np.empty((n_samples, k), dtype=np.float32)
    cdef float inertia = INFINITY, prev_inertia = INFINITY
    cdef int iteration = 0
    cdef clock_t start, end
    cdef double time_taken

    start = clock()
    centers = kmeans_plusplus_initialization(X, k, random_state)

    for iteration in range(max_iterations):
        compute_distances(X, centers, distances)
        labels = np.argmin(distances, axis=1)
        new_centers = update_centroids(X, labels, k)

        prev_inertia = inertia
        inertia = np.sum(np.min(distances, axis=1) ** 2)
        
        if iteration >= min_iterations - 1 and abs(prev_inertia - inertia) < tolerance:
            break
        
        centers = new_centers

    end = clock()
    time_taken = (end - start) / CLOCKS_PER_SEC

    return labels, centers, inertia, iteration + 1, time_taken

def kmeans_clustering(np.ndarray[FLOAT_t, ndim=2] X, int k, int n_init=10, int min_iterations=10, int max_iterations=300, float tolerance=1e-4, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    elif random_state and isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)


    cdef np.ndarray[INT_t, ndim=1] best_labels
    cdef np.ndarray[FLOAT_t, ndim=2] best_centers
    cdef float best_inertia = INFINITY
    cdef int i
    cdef np.ndarray[INT_t, ndim=1] labels
    cdef np.ndarray[FLOAT_t, ndim=2] centers
    cdef float inertia
    cdef int n_iterations
    cdef double time_taken
    cdef clock_t start, end
    cdef double total_time = 0

    start = clock()
    for i in range(n_init):
        labels, centers, inertia, n_iterations, time_taken = _kmeans_single(X, k, min_iterations, max_iterations, tolerance, random_state)
        logger.info(f"Initialization {i + 1}/{n_init}: Inertia = {inertia:.2f}, Iterations = {n_iterations}, Time = {time_taken:.2f} seconds")
        
        if inertia < best_inertia:
            best_labels = labels
            best_centers = centers
            best_inertia = inertia
            logger.info(f"New best inertia: {best_inertia:.2f}")

    end = clock()
    total_time = (end - start) / CLOCKS_PER_SEC

    logger.info(f"KMeans clustering complete. Best inertia: {best_inertia:.2f}")
    logger.info(f"Total kmeans clustering time: {total_time:.2f} seconds")

    return best_labels.tolist(), best_inertia

