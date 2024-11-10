import unittest
import numpy as np

from wordllama.algorithms import vector_similarity, binarize_and_packbits


class TestVectorSimilarity(unittest.TestCase):
    def test_binarization_and_packing(self):
        vec = np.zeros((1, 64))
        vec[0][7] = 1
        binary_output = binarize_and_packbits(vec)
        self.assertIsInstance(binary_output, np.ndarray)
        self.assertEqual(binary_output.dtype, np.uint64)
        self.assertEqual(binary_output, 1)

    def test_cosine_similarity_direct(self):
        vec1 = np.random.rand(1, 64)
        vec2 = np.random.rand(1, 64)
        result = vector_similarity(vec1, vec2, binary=False)
        self.assertIsInstance(result.item(), float)

    def test_hamming_similarity_direct(self):
        vec1 = np.expand_dims(np.random.randint(2, size=64, dtype=np.uint64), axis=0)
        vec2 = np.expand_dims(np.random.randint(2, size=64, dtype=np.uint64), axis=0)
        result = vector_similarity(vec1, vec2, binary=True)
        self.assertIsInstance(result.item(), float)


if __name__ == "__main__":
    unittest.main()
