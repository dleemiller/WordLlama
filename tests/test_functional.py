import unittest
from wordllama import WordLlama


class TestFunctional(unittest.TestCase):
    def test_function_clustering(self):
        wl = WordLlama.load()
        wl.cluster(["a", "b"], k=2)

    def test_function_similarity(self):
        wl = WordLlama.load()
        wl.similarity("a", "b")

    def test_function_similarity_binary(self):
        wl = WordLlama.load()
        wl.binary = True
        wl.similarity("a", "b")
