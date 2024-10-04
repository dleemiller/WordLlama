import unittest
import numpy as np
from wordllama.algorithms.find_local_minima import (
    find_local_minima,
    windowed_cross_similarity,
)


class TestSavitzkyGolay(unittest.TestCase):
    def setUp(self):
        self.x1 = np.linspace(0, 2 * np.pi, 100, dtype=np.float32)
        self.x = np.arange(100)
        self.y = np.sin(self.x1)

    def test_find_local_minima(self):
        # Testing the local minima detection for a simple sine wave
        x_minima, y_minima = find_local_minima(self.y, window_size=3, poly_order=2)

        # Known minima for sin(x) in the given range [0, 2*pi]
        expected_x_minima = np.array([3 * np.pi / 2], dtype=np.float32)
        expected_y_minima = np.array([-1.0], dtype=np.float32)

        # Check if the found minima are correct (allow small numerical tolerance)
        np.testing.assert_array_almost_equal(
            self.y[x_minima], expected_y_minima, decimal=2
        )
        np.testing.assert_array_almost_equal(y_minima, expected_y_minima, decimal=2)

    def test_find_local_minima_invalid_window_size(self):
        # Test that the function raises a ValueError for an invalid window size
        with self.assertRaises(ValueError):
            find_local_minima(self.y, window_size=2, poly_order=2)

    def test_find_local_minima_invalid_polynomial_order(self):
        # Test that the function raises a ValueError for an invalid polynomial order
        with self.assertRaises(ValueError):
            find_local_minima(self.y, window_size=11, poly_order=11)


class TestWindowedCrossSimilarity(unittest.TestCase):
    def setUp(self):
        # Example embedding matrix (5 vectors of 3 dimensions each)
        self.embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

    def test_windowed_cross_similarity(self):
        # Test windowed cross similarity with valid input
        result = windowed_cross_similarity(self.embeddings, window_size=3)

        # Expected results (off-diagonal dot products within a window)
        expected_result = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Check if the result is as expected (zero cross similarity for orthogonal vectors)
        np.testing.assert_array_almost_equal(result, expected_result, decimal=2)

    def test_windowed_cross_similarity_invalid_window(self):
        # Test invalid window size (even window size should raise ValueError)
        with self.assertRaises(ValueError):
            windowed_cross_similarity(self.embeddings, window_size=4)

        # Test invalid window size (window size < 3)
        with self.assertRaises(ValueError):
            windowed_cross_similarity(self.embeddings, window_size=2)

    def test_windowed_cross_similarity_small_window(self):
        # Test windowed cross similarity with a small window (size 3)
        result = windowed_cross_similarity(self.embeddings, window_size=3)
        self.assertEqual(result.shape[0], self.embeddings.shape[0])


if __name__ == "__main__":
    unittest.main()
