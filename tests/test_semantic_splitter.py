import unittest
import numpy as np
from wordllama.algorithms.semantic_splitter import SemanticSplitter


class TestSemanticSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = SemanticSplitter()

    def test_flatten(self):
        nested_list = [[1, 2], [3, 4], [5, 6]]
        flattened = self.splitter.flatten(nested_list)
        self.assertEqual(flattened, [1, 2, 3, 4, 5, 6])

    def test_constrained_split(self):
        text = "This is a test sentence. Another sentence here. And one more. " * 10
        chunks = self.splitter.constrained_split(text, target_size=50)
        self.assertTrue(all(len(chunk) <= 50 for chunk in chunks))
        self.assertEqual(" ".join(chunks), text.strip())

    def test_split(self):
        text = "Short sentence.\n\nTwo sentences. Without a line break.\n\nAnother short one."
        chunks = self.splitter.split(
            text, target_size=30, cleanup_size=10, intermediate_size=20
        )
        self.assertTrue(all(len(chunk) <= 30 for chunk in chunks))
        self.assertTrue(all(len(chunk) >= 10 for chunk in chunks))

    def test_reconstruct(self):
        lines = ["Short text.", "Another short text.", "A bit longer text here."]
        embeddings = np.random.rand(3, 16).astype(np.float32)

        reconstructed = self.splitter.reconstruct(
            lines,
            embeddings,
            target_size=30,
            window_size=3,
            poly_order=2,
            savgol_window=3,
        )

        self.assertIsInstance(reconstructed, list)
        self.assertTrue(all(isinstance(chunk, str) for chunk in reconstructed))

    def test_reconstruct_return_minima(self):
        lines = ["Short text.", "Another short text.", "A bit longer text here."]
        embeddings = np.random.rand(3, 16).astype(
            np.float32
        )  # 3 texts, 16-dimensional embeddings

        result = self.splitter.reconstruct(
            lines,
            embeddings,
            target_size=30,
            window_size=3,
            poly_order=2,
            savgol_window=3,
            return_minima=True,
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        roots, y, sim_avg = result
        self.assertIsInstance(roots, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(sim_avg, np.ndarray)


if __name__ == "__main__":
    unittest.main()
