import numpy as np
import pytest

from wordllama.algorithms.kmeans import kmeans_clustering


class TestKMeansClustering:
    @pytest.fixture(autouse=True)
    def setup(self):
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

    def test_kmeans_clustering_convergence(self):
        k = 2
        labels, inertia = kmeans_clustering(self.embeddings, k, random_state=self.random_state)

        assert len(labels) == self.embeddings.shape[0]
        assert inertia > 0

    def test_kmeans_clustering_labels(self):
        k = 2
        labels, _ = kmeans_clustering(self.embeddings, k, random_state=self.random_state)

        # Check that labels are within the valid range
        for label in labels:
            assert label in range(k)

    def test_kmeans_clustering_different_k(self):
        k = 3
        labels, _ = kmeans_clustering(self.embeddings, k, random_state=self.random_state)

        assert len(labels) == self.embeddings.shape[0]

        # Check that labels are within the valid range
        for label in labels:
            assert label in range(k)

    def test_kmeans_clustering_random_state(self):
        k = 2
        labels1, losses1 = kmeans_clustering(self.embeddings, k, random_state=42)
        labels2, losses2 = kmeans_clustering(self.embeddings, k, random_state=42)

        assert labels1 == labels2
        assert losses1 == losses2

    def test_kmeans_clustering_different_initializations(self):
        k = 2
        labels1, inertia1 = kmeans_clustering(self.embeddings, k, random_state=42, n_init=1)
        labels2, inertia2 = kmeans_clustering(self.embeddings, k, random_state=42, n_init=10)

        assert inertia1 > inertia2


@pytest.mark.parametrize(
    "layout",
    ["contiguous", "row-strided", "column-strided", "both-strided", "fortran", "offset"],
)
def test_kmeans_clustering_float32_layout_equivalence(layout):
    values = np.array(
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
    rows, columns = values.shape
    if layout == "contiguous":
        matrix = values.copy()
    elif layout == "row-strided":
        storage = np.empty((rows * 2, columns), dtype=np.float32)
        storage[::2] = values
        matrix = storage[::2]
    elif layout == "column-strided":
        storage = np.empty((rows, columns * 2), dtype=np.float32)
        storage[:, ::2] = values
        matrix = storage[:, ::2]
    elif layout == "both-strided":
        storage = np.empty((rows * 2, columns * 2), dtype=np.float32)
        storage[::2, ::2] = values
        matrix = storage[::2, ::2]
    elif layout == "fortran":
        matrix = np.asfortranarray(values)
    else:
        storage = np.empty((rows + 2, columns + 2), dtype=np.float32)
        storage[1:-1, 1:-1] = values
        matrix = storage[1:-1, 1:-1]

    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix, values)
    expected_rng = np.random.RandomState(42)
    actual_rng = np.random.RandomState(42)
    expected_labels, expected_inertia = kmeans_clustering(
        values.copy(), 2, n_init=10, random_state=expected_rng
    )
    actual_labels, actual_inertia = kmeans_clustering(matrix, 2, n_init=10, random_state=actual_rng)

    assert actual_labels == expected_labels
    assert actual_inertia == expected_inertia
    expected_state = expected_rng.get_state()
    actual_state = actual_rng.get_state()
    assert actual_state[0] == expected_state[0]
    np.testing.assert_array_equal(actual_state[1], expected_state[1])
    assert actual_state[2:] == expected_state[2:]
