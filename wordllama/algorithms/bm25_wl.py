import numpy as np
import pandas as pd
import tqdm
from wordllama import WordLlama
from wordllama.algorithms import vector_similarity



def softmax(logits, temperature=0.09):
    """
    Compute the softmax of each row of the input array in a numerically stable and efficient manner.

    Parameters:
    - logits (np.ndarray): 2D array of logits (shape: [batch_size, num_classes]).
    - temperature (float): Temperature parameter to scale the logits.

    Returns:
    - np.ndarray: Softmax probabilities (same shape as logits).
    """
    logits = logits.astype(np.float32, copy=False)
    
    # Subtract the maximum value in each row for numerical stability
    # This prevents large exponentials which can cause overflow
    logits_max = np.max(logits, axis=1, keepdims=True)
    logits -= logits_max
    
    # Scale the logits by the temperature
    logits /= temperature
    
    # Compute exponentials in-place to save memory
    np.exp(logits, out=logits)
    sum_exp = np.sum(logits, axis=1, keepdims=True)
    logits /= sum_exp
    
    return logits

def batch_iterator(iterable, batch_size):
    """
    Generator that yields batches of data from the iterable.

    Parameters:
    - iterable (list): The list of items to batch.
    - batch_size (int): The number of items per batch.

    Yields:
    - list: A batch of items.
    """
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

class BM25Searcher:
    def __init__(self, wl, tokenized_texts, similarity_matrix, idf_vector, doc_length):
        """
        Initializes the BM25Searcher with tokenized texts, similarity matrix, IDF vector, and average document length.

        Parameters:
        - tokenized_texts (list): List of tokenized documents.
        - similarity_matrix (np.ndarray): Precomputed similarity matrix.
        - idf_vector (np.ndarray): Precomputed IDF vector.
        - avg_doc_len (float): Average document length in the corpus.
        """
        self.wl = wl
        self.tokenized_texts = tokenized_texts
        self.similarity_matrix = similarity_matrix
        self.idf_vector = idf_vector
        self.doc_length = doc_length
        self.avg_doc_len = np.mean(doc_length)
        self.k1 = 1.2
        self.b = 0.8
        #self.m = 1  # Parameter for combining fW and fD

    @classmethod
    def from_corpus(cls, dataframe, sample_size=10000, temperature=0.09, split=False):
        """
        Creates an instance of BM25Searcher from a given corpus.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame containing a 'text' column.
        - sample_size (int): Number of samples to process from the corpus.
        - temperature (float): Temperature parameter for softmax.

        Returns:
        - BM25Searcher: An instance of the BM25Searcher class.
        """
        # Load WordLlama
        wl = WordLlama.load()

        # Sample the corpus
        sample = dataframe.sample(n=sample_size)
        texts = []
        if split:
            for text in tqdm.tqdm(sample.text, desc="Splitting texts"):
                split_text = wl.split(text)
                texts.extend(split_text)
            print(f"Split to {len(texts)} chunks")
        else:
            texts = sample.text.tolist()

        # Tokenize texts
        tokenized_texts = []
        for batch in tqdm.tqdm(batch_iterator(texts, batch_size=32), desc="Tokenizing texts"):
            tokenized_batch = wl.tokenize(batch)
            tokenized_texts.extend(tokenized_batch)

        # Calculate average document length
        doc_length = np.array([sum(x.attention_mask) for x in tokenized_texts])
        avg_doc_len = np.mean(doc_length)
        print(f"Average document length: {avg_doc_len}")

        # Compute norm / idf (magnitudes)
        norm = np.linalg.norm(wl.embedding, axis=1, keepdims=True)
        idf = norm.squeeze()
        print(f"IDF shape: {idf.shape} Min: {np.min(idf):.2f} Max: {np.max(idf):.2f} Median: {np.median(idf):.2f}")

        # Compute similarity matrix
        v = wl.embedding / norm
        print(f"Pre-computing similarities...")
        s = vector_similarity(v, v, binary=False)

        # Apply softmax to similarity scores
        print(f"Computing softmax...")
        f = softmax(s, temperature=temperature)

        return cls(wl, tokenized_texts, f, idf, doc_length)

    def search(self, query_str, top_k=10):
        """
        Searches the corpus for the top_k documents relevant to the query_str.

        Parameters:
        - query_str (str): The query string.
        - top_k (int): Number of top results to return.

        Returns:
        - List of tuples: Each tuple contains (score, document index, document text).
        """
        query = self.wl.tokenize(query_str)[0]
        scores = np.zeros(len(self.tokenized_texts))

        avg_doc_len = self.avg_doc_len
        alpha = self.k1 + 1
        query_idx = np.array(query.ids)
        query_rows = np.ascontiguousarray(self.similarity_matrix[query_idx, :])
        alpha = self.idf_vector[query_idx] * alpha 

        for i, doc in enumerate(tqdm.tqdm(self.tokenized_texts, desc="Scoring documents")):
            doc_length = self.doc_length[i]
            beta = self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_len))

            d = doc.ids
            if 0 in d:
                doc_idx = d[0:d.index(0)]
            else:
                doc_idx = d

            # Get simlarities for query and document tokens
            fq = query_rows[:, doc_idx].sum(axis=1)

            # Compute BM25 score
            score = (alpha * fq) / (fq + beta)
            scores[i] = score.sum()

        # Retrieve top_k scores and their indices
        top_indices = scores.argsort()[-top_k:][::-1]
        top_scores = scores[top_indices]
        results = [(top_scores[j], top_indices[j], self.get_text(top_indices[j])) for j in range(top_k)]
        return results

    def get_text(self, index):
        """
        Retrieves the original text for a given document index.

        Parameters:
        - index (int): Document index.

        Returns:
        - str: The text of the document.
        """
        doc = self.tokenized_texts[index]
        return self.wl.tokenizer.decode(doc.ids)

