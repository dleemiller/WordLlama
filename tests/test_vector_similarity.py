import numpy as np

from wordllama.algorithms import binarize_and_packbits, vector_similarity


class TestVectorSimilarity:
    def test_binarization_and_packing(self):
        vec = np.zeros((1, 64))
        vec[0][7] = 1
        binary_output = binarize_and_packbits(vec)
        assert isinstance(binary_output, np.ndarray)
        assert binary_output.dtype == np.uint64
        assert binary_output == 1

    def test_cosine_similarity_direct(self):
        vec1 = np.random.rand(1, 64)
        vec2 = np.random.rand(1, 64)
        result = vector_similarity(vec1, vec2, binary=False)
        assert isinstance(result.item(), float)

    def test_hamming_similarity_direct(self):
        vec1 = np.expand_dims(np.random.randint(2, size=64, dtype=np.uint64), axis=0)
        vec2 = np.expand_dims(np.random.randint(2, size=64, dtype=np.uint64), axis=0)
        result = vector_similarity(vec1, vec2, binary=True)
        assert isinstance(result.item(), float)
