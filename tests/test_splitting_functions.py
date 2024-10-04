import unittest
from wordllama.algorithms.splitter import (
    constrained_batches,
    split_sentences,
    constrained_coalesce,
    reverse_merge,
)
import string


class TestSplitter(unittest.TestCase):
    def test_constrained_batches(self):
        # Basic batching
        data = ["a", "bb", "ccc", "dddd", "eeeee"]
        batches = list(constrained_batches(data, max_size=5))
        expected = [("a", "bb"), ("ccc",), ("dddd",), ("eeeee",)]
        self.assertEqual(batches, expected)

        # Batching with max_count
        data = ["a", "bb", "ccc", "dddd", "eeeee"]
        batches = list(constrained_batches(data, max_size=10, max_count=2))
        self.assertEqual(batches, [("a", "bb"), ("ccc", "dddd"), ("eeeee",)])

        # Batching with get_len
        data = ["a", "bb", "ccc", "dddd", "eeeee"]
        batches = list(constrained_batches(data, max_size=5, get_len=lambda x: 1))
        self.assertEqual(batches, [("a", "bb", "ccc", "dddd", "eeeee")])

        # Non-strict mode
        data = ["aaaaaa", "b", "c"]
        batches = list(constrained_batches(data, max_size=5, strict=False))
        self.assertEqual(batches, [("aaaaaa",), ("b", "c")])

        # Strict mode with item exceeding max_size
        data = ["aaaaaa", "b", "c"]
        with self.assertRaises(ValueError):
            batches = list(constrained_batches(data, max_size=5))

    def test_split_sentences(self):
        # Basic test
        text = "This is a sentence. This is another sentence! And another one?"
        sentences = split_sentences(text)
        expected = [
            "This is a sentence.",
            "This is another sentence!",
            "And another one?",
        ]
        self.assertEqual(sentences, expected)

        # Test with no punctuation
        text = "This is a text without punctuation"
        sentences = split_sentences(text)
        expected = ["This is a text without punctuation"]
        self.assertEqual(sentences, expected)

        # Test with custom punctuation
        text = "Sentence one# Sentence two# Sentence three"
        sentences = split_sentences(text, punct_chars={"#"})
        expected = ["Sentence one#", "Sentence two#", "Sentence three"]
        self.assertEqual(sentences, expected)

        # Test with text ending without punctuation
        text = "This is a sentence. This is another sentence"
        sentences = split_sentences(text)
        expected = ["This is a sentence.", "This is another sentence"]
        self.assertEqual(sentences, expected)

    def test_constrained_coalesce(self):
        letters = list(string.ascii_lowercase)
        # Using the example from the documentation
        result = constrained_coalesce(letters, max_size=5, separator="")
        expected = ["abcd", "efgh", "ijkl", "mnop", "qrst", "uvwx", "yz"]
        self.assertEqual(result, expected)

        # Test with max_iterations=1
        result = constrained_coalesce(
            letters, max_size=5, separator="", max_iterations=1
        )
        expected_one_pass = [
            "ab",
            "cd",
            "ef",
            "gh",
            "ij",
            "kl",
            "mn",
            "op",
            "qr",
            "st",
            "uv",
            "wx",
            "yz",
        ]
        self.assertEqual(result, expected_one_pass)

        # Test with data that cannot be combined
        data = ["a"] * 100
        result = constrained_coalesce(data, max_size=1, max_iterations=5)
        self.assertEqual(result, ["a"] * 100)

    def test_reverse_merge(self):
        # Basic merging test
        data = ["long enough", "short", "tiny", "adequate", "s"]
        result = reverse_merge(data, n=6, separator=" ")
        expected = ["long enough short tiny", "adequate s"]
        self.assertEqual(result, expected)

        # Test with empty list
        data = []
        result = reverse_merge(data, n=5)
        expected = []
        self.assertEqual(result, expected)

        # All strings longer than n
        data = ["string1", "string2", "string3"]
        result = reverse_merge(data, n=5)
        expected = ["string1", "string2", "string3"]
        self.assertEqual(result, expected)

        # All strings shorter than n
        data = ["a", "bb", "ccc"]
        result = reverse_merge(data, n=5, separator=" ")
        expected = ["a bb ccc"]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
