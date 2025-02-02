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

    def test_function_sorted(self):
        wl = WordLlama.load()
        query = "test query"
        candidates = ["example A", "example B", "example C"]

        sim_key = wl.key(query)
        sorted_candidates = sorted(candidates, key=sim_key, reverse=True)

        self.assertIsInstance(sorted_candidates, list)
        self.assertEqual(len(sorted_candidates), len(candidates))

    def test_function_max(self):
        wl = WordLlama.load()
        query = "test query"
        candidates = ["example A", "example B", "example C"]

        sim_key = wl.key(query)
        best_candidate = max(candidates, key=sim_key)

        self.assertIn(best_candidate, candidates)
        self.assertEqual(
            best_candidate, max(candidates, key=lambda x: wl.similarity(query, x))
        )
