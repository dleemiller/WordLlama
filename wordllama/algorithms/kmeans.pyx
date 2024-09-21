# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, fastmath=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport sqrtf
from libc.float cimport FLT_MAX

np.import_array()

ctypedef np.npy_intp DTYPE_t
ctypedef np.float32_t FLOAT_t

cdef inline float squared_euclidean_distance(const float[:] vec1, const float[:] vec2, Py_ssize_t dim) nogil:
    cdef Py_ssize_t i
    cdef float dist = 0.0
    for i in range(dim):
        dist += (vec1[i] - vec2[i]) ** 2
    return dist

def compute_distances(const float[:, :] embeddings, const float[:, :] centroids):
    cdef Py_ssize_t num_points = embeddings.shape[0]
    cdef Py_ssize_t num_centroids = centroids.shape[0]
    cdef Py_ssize_t dim = embeddings.shape[1]
    cdef float[:, :] distances = np.empty((num_points, num_centroids), dtype=np.float32)
    cdef Py_ssize_t i, j

    for i in range(num_points):
        for j in range(num_centroids):
            distances[i, j] = sqrtf(squared_euclidean_distance(embeddings[i], centroids[j], dim))

    return np.asarray(distances)

def update_centroids(const float[:, :] embeddings, const DTYPE_t[:] labels, Py_ssize_t num_clusters):
    cdef Py_ssize_t num_points = embeddings.shape[0]
    cdef Py_ssize_t dim = embeddings.shape[1]
    cdef float[:, :] new_centroids = np.zeros((num_clusters, dim), dtype=np.float32)
    cdef DTYPE_t[:] count = np.zeros(num_clusters, dtype=np.intp)
    cdef Py_ssize_t i, j
    cdef DTYPE_t label

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

def kmeans_plusplus_initialization(float[:, :] embeddings, int k, object random_state):
    cdef Py_ssize_t n_samples = embeddings.shape[0]
    cdef Py_ssize_t n_features = embeddings.shape[1]
    cdef float[:, :] centroids = np.empty((k, n_features), dtype=np.float32)
    cdef float[:] distances = np.empty(n_samples, dtype=np.float32)
    cdef float[:] weights = np.empty(n_samples, dtype=np.float32)
    cdef Py_ssize_t i, j
    cdef int centroid_idx
    cdef float total_weight

    # Choose the first centroid randomly
    centroid_idx = random_state.randint(n_samples)
    centroids[0] = embeddings[centroid_idx]

    # Choose the remaining centroids
    for i in range(1, k):
        # Compute distances to the nearest centroid for each point
        for j in range(n_samples):
            distances[j] = FLT_MAX
            for centroid_idx in range(i):
                distances[j] = min(distances[j], squared_euclidean_distance(embeddings[j], centroids[centroid_idx], n_features))

        # Compute weights (squared distances)
        total_weight = 0.0
        for j in range(n_samples):
            weights[j] = distances[j]
            total_weight += weights[j]

        # Choose the next centroid with probability proportional to distance squared
        centroid_idx = random_state.choice(n_samples, p=np.asarray(weights) / total_weight)
        centroids[i] = embeddings[centroid_idx]

    return np.asarray(centroids)

def kmeans_clustering(float[:, :] embeddings, int k, int n_init=10, int min_iterations=10, int max_iterations=100, float tolerance=1e-4, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    cdef Py_ssize_t n_samples = embeddings.shape[0]
    cdef Py_ssize_t n_features = embeddings.shape[1]
    cdef float[:, :] best_centroids
    cdef DTYPE_t[:] best_labels
    cdef float best_inertia = np.inf
    cdef int i

    for i in range(n_init):
        print(i)
        labels, centroids = _kmeans_single(embeddings, k, min_iterations, max_iterations, tolerance, random_state)
        print(labels, centroids)
        inertia = _compute_inertia(embeddings, labels, centroids)
        print(i, inertia)
        
        if inertia < best_inertia:
            best_centroids = centroids
            best_labels = labels
            best_inertia = inertia

    return np.asarray(best_labels), best_inertia

cdef _kmeans_single(float[:, :] embeddings, int k, int min_iterations, int max_iterations, float tolerance, object random_state):
    cdef Py_ssize_t n_samples = embeddings.shape[0]
    cdef Py_ssize_t n_features = embeddings.shape[1]
    cdef float[:, :] centroids = kmeans_plusplus_initialization(embeddings, k, random_state)
    cdef float[:, :] new_centroids
    cdef DTYPE_t[:] labels = np.empty(n_samples, dtype=np.intp)
    cdef float[:, :] distances
    cdef float max_centroid_shift
    cdef int iteration

    for iteration in range(max_iterations):
        # Assign points to nearest centroids
        distances = compute_distances(embeddings, centroids)
        for i in range(n_samples):
            labels[i] = np.argmin(distances[i])

        # Update centroids
        new_centroids = update_centroids(embeddings, labels, k)

        # Check for convergence
        max_centroid_shift = 0.0
        for i in range(k):
            max_centroid_shift = max(max_centroid_shift, squared_euclidean_distance(centroids[i], new_centroids[i], n_features))

        if iteration >= min_iterations - 1 and max_centroid_shift < tolerance * tolerance:
            break

        centroids = new_centroids

    return labels, centroids

cdef float _compute_inertia(float[:, :] embeddings, DTYPE_t[:] labels, float[:, :] centroids):
    cdef Py_ssize_t n_samples = embeddings.shape[0]
    cdef Py_ssize_t n_features = embeddings.shape[1]
    cdef float inertia = 0.0
    cdef Py_ssize_t i

    for i in range(n_samples):
        inertia += squared_euclidean_distance(embeddings[i], centroids[labels[i]], n_features)

    return inertia

