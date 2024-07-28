import numpy as np
from typing import List, Tuple
from .kmeans_helpers import compute_distances, update_centroids


def kmeans_plusplus_initialization(
    embeddings: np.ndarray, k: int, random_state: np.random.RandomState
) -> np.ndarray:
    """
    Initialize centroids using the K-Means++ algorithm.

    Parameters:
    embeddings (np.ndarray): The input data points (embeddings) to cluster.
    k (int): The number of clusters.
    random_state (np.random.RandomState): Random state for reproducibility.

    Returns:
    np.ndarray: The initialized centroids.
    """
    n_samples, n_features = embeddings.shape
    centroids = np.empty((k, n_features), dtype=embeddings.dtype)

    # Step 1a: Choose the first centroid randomly from the data points
    centroids[0] = embeddings[random_state.randint(n_samples)]
    distances = np.linalg.norm(embeddings - centroids[0], axis=1)

    for i in range(1, k):
        # Step 1b: Compute the probability distribution based on squared distances
        probabilities = distances**2
        probabilities /= probabilities.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = random_state.rand()
        index = np.searchsorted(cumulative_probabilities, r)
        centroids[i] = embeddings[index]

        # Update distances to the nearest centroid for the next iteration
        new_distances = np.linalg.norm(embeddings - centroids[i], axis=1)
        distances = np.minimum(distances, new_distances)

    return centroids


def kmeans_clustering(
    embeddings: np.ndarray,
    k: int,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    n_init: int = 10,
    min_iterations: int = 5,
    random_state=None,
) -> Tuple[List[int], List[float]]:
    """
    Perform K-Means clustering on the provided embeddings.

    Parameters:
    embeddings (np.ndarray): The input data points (embeddings) to cluster.
    k (int): The number of clusters.
    max_iterations (int, optional): The maximum number of iterations to run the algorithm. Defaults to 100.
    tolerance (float, optional): The tolerance to declare convergence. Defaults to 1e-4.
    n_init (int, optional): Number of times the algorithm will be run with different centroid seeds. The final result will be the best output in terms of loss. Defaults to 10.
    min_iterations (int, optional): Minimum number of iterations before checking for convergence. Defaults to 5.
    random_state (int or np.random.RandomState, optional): Random state for reproducibility.

    Returns:
    Tuple[List[int], List[float]]: A tuple containing the cluster labels and the list of loss values for each iteration.
    """

    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    best_labels = None
    best_inertia = float("inf")
    best_losses = None

    for init_run in range(n_init):
        centroids = kmeans_plusplus_initialization(embeddings, k, random_state)
        prev_inertia = float("inf")
        losses = []

        for iteration in range(max_iterations):
            # Step 2: Assign each point to the nearest centroid using the Cython optimized function
            distances = compute_distances(embeddings, centroids)
            labels = np.argmin(distances, axis=1)

            # Calculate inertia using distances directly
            inertia = np.sum(np.min(distances, axis=1) ** 2)
            losses.append(inertia)

            # Check for convergence based on inertia
            if iteration >= min_iterations and abs(prev_inertia - inertia) < tolerance:
                break

            prev_inertia = inertia

            # Step 3: Update centroids using the Cython optimized function
            centroids = update_centroids(embeddings, labels, k, embeddings.shape[1])

            # Check for convergence based on centroids
            if iteration >= min_iterations and np.allclose(
                centroids,
                update_centroids(embeddings, labels, k, embeddings.shape[1]),
                atol=tolerance,
            ):
                break

        # Check if this initialization run has the best result
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_losses = losses

    return best_labels.tolist(), best_losses
