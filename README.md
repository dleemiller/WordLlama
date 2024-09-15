
# WordLlama

**WordLlama** is a fast, lightweight NLP toolkit that handles tasks like fuzzy-deduplication, similarity and ranking with minimal inference-time dependencies and optimized for CPU hardware.

<p align="center">
  <img src="wordllama.png" alt="Word Llama" width="50%">
</p>


## Table of Contents
- [Quick Start](#quick-start)
- [What is it?](#what-is-it)
- [MTEB Results](#mteb-results-l2_supercat)
- [Embed Text](#embed-text)
- [Training Notes](#training-notes)
- [Roadmap](#roadmap)
- [Extracting Token Embeddings](#extracting-token-embeddings)
- [Citations](#citations)
- [License](#license)

## Quick Start

Install:
```bash
pip install wordllama
```

Load the 256-dim model.
```python
from wordllama import WordLlama

# Load the default WordLlama model
wl = WordLlama.load()

# Calculate similarity between two sentences
similarity_score = wl.similarity("i went to the car", "i went to the pawn shop")
print(similarity_score)  # Output: 0.06641249096796882

# Rank documents based on their similarity to a query
query = "i went to the car"
candidates = ["i went to the park", "i went to the shop", "i went to the truck", "i went to the vehicle"]
ranked_docs = wl.rank(query, candidates)
print(ranked_docs)
# Output:
# [
#   ('i went to the vehicle', 0.7441646856486314),
#   ('i went to the truck', 0.2832691551894259),
#   ('i went to the shop', 0.19732814982305436),
#   ('i went to the park', 0.15101404519322253)
# ]

# additional inference methods
wl.deduplicate(candidates, threshold=0.8) # fuzzy deduplication
wl.cluster(docs, k=5, max_iterations=100, tolerance=1e-4) # labels using kmeans/kmeans++ init
wl.filter(query, candidates, threshold=0.3) # filter candidates based on query
wl.topk(query, candidates, k=3) # return topk strings based on query
```

## What is it?

WordLlama is a utility for NLP and word embedding model that recycles components from large language models (LLMs) to create efficient and compact word representations (such as GloVe, Word2Vec or FastText).
WordLlama begins by extracting the token embedding codebook from a state-of-the-art LLM (e.g., LLama3 70B), and training a small context-less model in a general purpose embedding framework.

WordLlama improves on all MTEB benchmarks above word models like GloVe 300d, while being substantially smaller in size (**16MB default model** @ 256-dim vs >2GB).

Features of WordLlama include:

1. **Matryoshka Representations**: Truncate embedding dimension as needed.
2. **Low Resource Requirements**: A simple token lookup with average pooling, enables this to operate fast on CPU.
3. **Binarization**: Models trained using the straight through estimator can be packed to small integer arrays for even faster hamming distance calculations. (coming soon)
4. **Numpy-only inference**: Lightweight and simple.

For flexibility, WordLlama employs the Matryoshka representation learning training technique. The largest model (1024-dim) can be truncated to 64, 128, 256 or 512.
For binary embedding models, we implement straight-through estimators during training. For dense embeddings, 256 dimensions sufficiently captures most of the performance, while for binary embeddings validation accuracy is close to saturation at 512-dimensions (64 bytes packed).

The final weights are saved *after* weighting, projection and truncation of the entire tokenizer vocabulary. Thus, WordLlama becomes a single embedding matrix (nn.Embedding) that is considerably smaller than the gigabyte-sized llm codebooks we start with. The original tokenizer is still used to preprocess the text into tokens, and the reduced size token embeddings are average pooled. There is very little computation required, and the resulting model sizes range from 16mb to 250mb for the 128k llama3 vocabulary.

It's good option for some nlp-lite tasks. You can train sklearn classifiers on it, perform basic semantic matching, fuzzy deduplication, ranking and clustering.
I think it should work well for creating LLM output evaluators, or other preparatory tasks involved in multi-hop or agentic workflows.
You can perform your own llm surgery and train your own model on consumer GPUs in a few hours.
Because of its fast and portable size, it makes a good "Swiss-Army Knife" utility for exploratory analysis and utility applications.

## MTEB Results (l2_supercat)

| Metric                 | WL64        | WL128        | WL256 (X)    | WL512        | WL1024        | GloVe 300d | Komninos | all-MiniLM-L6-v2 |
|------------------------|-------------|--------------|--------------|--------------|---------------|------------|----------|------------------|
| Clustering             | 30.27       | 32.20        | 33.25        | 33.40        | 33.62         | 27.73      | 26.57    | 42.35            |
| Reranking              | 50.38       | 51.52        | 52.03        | 52.32        | 52.39         | 43.29      | 44.75    | 58.04            |
| Classification         | 53.14       | 56.25        | 58.21        | 59.13        | 59.50         | 57.29      | 57.65    | 63.05            |
| Pair Classification    | 75.80       | 77.59        | 78.22        | 78.50        | 78.60         | 70.92      | 72.94    | 82.37            |
| STS                    | 66.24       | 67.53        | 67.91        | 68.22        | 68.27         | 61.85      | 62.46    | 78.90            |
| CQA DupStack           | 18.76       | 22.54        | 24.12        | 24.59        | 24.83         | 15.47      | 16.79    | 41.32            |
| SummEval               | 30.79       | 29.99        | 30.99        | 29.56        | 29.39         | 28.87      | 30.49    | 30.81            |

The [l2_supercat](https://huggingface.co/dleemiller/word-llama-l2-supercat) is a Llama2-vocabulary model. To train this model, I concatenated codebooks from several models, including Llama2 70B and phi3 medium (after removing additional special tokens).
Because several models have used the Llama2 tokenizer, their codebooks can be concatenated and trained together. Performance of the resulting model is comparable to training the Llama3 70B codebook, while being 4x smaller (32k vs 128k vocabulary).

### Other Models
[Results](wordllama/RESULTS.md)

Llama3-based: [l3_supercat](https://huggingface.co/dleemiller/wordllama-l3-supercat)

## Embed Text

Here’s how you can load pre-trained embeddings and use them to embed text:

```python
from wordllama import WordLlama

# Load pre-trained embeddings
# truncate dimension to 64
wl = WordLlama.load(trunc_dim=64)

# Embed text
embeddings = wl.embed(["the quick brown fox jumps over the lazy dog", "and all that jazz"])
print(embeddings.shape)  # (2, 64)
```

Binary embedding models can be used like this:

```python
# Binary embeddings are packed into uint64
# 64-dims => array of 1x uint64 
wl = WordLlama.load(trunc_dim=64, binary=True)  # this will download the binary model from huggingface
wl.embed("I went to the car") # Output: array([[3029168427562626]], dtype=uint64)

# load binary trained model trained with straight through estimator
wl = WordLlama.load(dim=1024, binary=True)

# Uses the hamming similarity to binarize
similarity_score = wl.similarity("i went to the car", "i went to the pawn shop")
print(similarity_score)  # Output: 0.57421875

ranked_docs = wl.rank("i went to the car", ["van", "truck"])

wl.binary = False # turn off hamming and use cosine

# load a different model class
wl = WordLlama.load(config="l3_supercat", dim=1024) # downloads model from HF
```

## Training Notes

Binary embedding models showed more pronounced improvement at higher dimensions, and either 512 or 1024 is recommended for binary embedding.

L2 Supercat was trained using a batch size of 512 on a single A100 for 12 hours.

## Roadmap

- Working on adding inference features:
  - Semantic text splitting
- Add example notebooks
  - DSPy evaluators
  - RAG pipelines

## Extracting Token Embeddings

To extract token embeddings from a model, ensure you have agreed to the user agreement and logged in using the Hugging Face CLI (for llama3 models). You can then use the following snippet:

```python
from wordllama.extract import extract_safetensors

# Extract embeddings for the specified configuration
extract_safetensors("llama3_70B", "path/to/saved/model-0001-of-00XX.safetensors")
```

HINT: Embeddings are usually in the first safetensors file, but not always. Sometimes there is a manifest, sometimes you have to snoop around and figure it out.

For training, use the scripts in the github repo. You have to add a configuration file (copy/modify an existing one into the folder).
```
$ pip install wordllama[train]
$ python train.py train --config your_new_config
(training stuff happens)
$ python train.py save --config your_new_config --checkpoint ... --outdir /path/to/weights/
(saves 1 model per matryoshka dim)
```

## Citations

If you use WordLlama in your research or project, please consider citing it as follows:

```bibtex
@software{miller2024wordllama,
  author = {Miller, D. Lee},
  title = {WordLlama: Recycled Token Embeddings from Large Language Models},
  year = {2024},
  url = {https://github.com/dleemiller/wordllama},
  version = {0.2.6}
}
```

## License

This project is licensed under the MIT License.

