import unittest
from wordllama import WordLlama


class TestFunctional(unittest.TestCase):

    def test_function_clustering(self):
        wl = WordLlama.load()
        wl.cluster(["a", "b"], k=2)
