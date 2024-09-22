import unittest
import numpy as np

from wordllama.algorithms.kmeans import (
    #    kmeans_plusplus_initialization,
    kmeans_clustering,
)


class TestKMeansClustering(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(42)
        self.embeddings = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.1, 0.3],
                [0.8, 0.7, 0.6],
                [0.9, 0.8, 0.7],
                [0.4, 0.5, 0.6],
                [0.5, 0.4, 0.7],
            ],
            dtype=np.float32,
        )

    # def test_kmeans_plusplus_initialization(self):
    #     k = 2
    #     centroids = kmeans_plusplus_initialization(
    #         self.embeddings, k, self.random_state
    #     )

    #     self.assertEqual(centroids.shape[0], k)
    #     self.assertEqual(centroids.shape[1], self.embeddings.shape[1])

    #     # Check that centroids are among the original points
    #     for centroid in centroids:
    #         self.assertTrue(
    #             any(np.allclose(centroid, point) for point in self.embeddings)
    #         )

    def test_kmeans_clustering_convergence(self):
        k = 2
        labels, inertia = kmeans_clustering(
            self.embeddings, k, random_state=self.random_state
        )

        self.assertEqual(len(labels), self.embeddings.shape[0])
        self.assertGreater(inertia, 0)

    def test_kmeans_clustering_labels(self):
        k = 2
        labels, _ = kmeans_clustering(
            self.embeddings, k, random_state=self.random_state
        )

        # Check that labels are within the valid range
        for label in labels:
            self.assertIn(label, range(k))

    def test_kmeans_clustering_different_k(self):
        k = 3
        labels, _ = kmeans_clustering(
            self.embeddings, k, random_state=self.random_state
        )

        self.assertEqual(len(labels), self.embeddings.shape[0])

        # Check that labels are within the valid range
        for label in labels:
            self.assertIn(label, range(k))

    def test_kmeans_clustering_random_state(self):
        k = 2
        labels1, losses1 = kmeans_clustering(self.embeddings, k, random_state=42)
        labels2, losses2 = kmeans_clustering(self.embeddings, k, random_state=42)

        self.assertEqual(labels1, labels2)
        self.assertEqual(losses1, losses2)

    def test_kmeans_clustering_different_initializations(self):
        k = 2
        labels1, inertia1 = kmeans_clustering(
            self.embeddings, k, random_state=42, n_init=1
        )
        labels2, inertia2 = kmeans_clustering(
            self.embeddings, k, random_state=42, n_init=10
        )

        self.assertGreater(inertia1, inertia2)


if __name__ == "__main__":
    unittest.main()
