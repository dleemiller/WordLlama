# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

ctypedef np.npy_intp DTYPE_t

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
            distances[i, j] = sqrt(squared_euclidean_distance(embeddings[i], centroids[j], dim))

    return np.asarray(distances)

def update_centroids(const float[:, :] embeddings, const DTYPE_t[:] labels, Py_ssize_t num_clusters, Py_ssize_t dim):
    cdef float[:, :] new_centroids = np.zeros((num_clusters, dim), dtype=np.float32)
    cdef DTYPE_t[:] count = np.zeros(num_clusters, dtype=np.intp)
    cdef Py_ssize_t i, j
    cdef DTYPE_t label

    # Accumulate sums and counts for each cluster
    for i in range(labels.shape[0]):
        label = labels[i]
        for j in range(dim):
            new_centroids[label, j] += embeddings[i, j]
        count[label] += 1

    # Calculate the mean for each cluster
    for i in range(num_clusters):
        if count[i] > 0:
            for j in range(dim):
                new_centroids[i, j] /= count[i]

    return np.asarray(new_centroids)

