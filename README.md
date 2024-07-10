
# Word Llama

The power of 13 trillion tokens of training, extracted, flogged and minimized into a cute little package for word embedding.

<p align="center">
  <img src="wordllama.png" alt="Word Llama" width="60%">
</p>


## Table of Contents
- [Quick Start](#quick-start)
- [What is it?](#what-is-it)
- [MTEB Results (standard models)](#mteb-results-standard-models)
- [Embed Text](#embed-text)
- [Training Notes](#training-notes)
- [Roadmap](#roadmap)
- [Features](#features)
- [Installation](#installation)
- [Extracting Token Embeddings](#extracting-token-embeddings)
- [Citations](#citations)
- [License](#license)

## Quick Start

Install:
```bash
git clone git@github.com:dleemiller/wordllama.git
pip install .
```

Load the 64-dim model.
```python
from wordllama import load

# Load the WordLlama model
wl = load()

# Calculate similarity between two sentences
similarity_score = wl.similarity("i went to the car", "i went to the pawn shop")
print(similarity_score)  # Output: 0.4026190060777956

# Rank documents based on their similarity to a query
ranked_docs = wl.rank("i went to the car", ["i went to the park", "i went to the shop", "i went to the truck", "i went to the vehicle"], use_hamming=False)
print(ranked_docs)
# Output:
# [
#   ('i went to the vehicle', 0.8765271774778441),
#   ('i went to the truck', 0.5792372791755765),
#   ('i went to the shop', 0.45162150518177724),
#   ('i went to the park', 0.36642963613509194)
# ]
```

## What is it?

WordLlama is a word embedding model that recycles components from large language models (LLMs) to create efficient and compact word representations (such as GloVe or Word2Vec).
WordLlama begins by extracting the token embedding codebook from a state-of-the-art LLM (e.g., LLama3 70B).

The key features of WordLlama include:

1. Dimension Reduction: We train a projection to reduce the embedding dimension, making it more manageable for various applications.
2. Low Resource Requirements: A simple token lookup with average pooling, enables this to operate fast on CPU.
3. Binarization: Models trained using the straight through estimator can be packed to small integer arrays for even faster hamming distance calculations.
4. Numpy-only inference: Keep it lightweight and simple.

For flexibility, WordLlama employs the Matryoshka representation learning training technique. The largest model (1024-dim) can be truncated to 64, 128, 256 or 512.
For binary embedding models, we implement straight-through estimators during training. For dense embeddings, 256 dimensions sufficiently captures most of the performance, while for binary embeddings validation accuracy is close to saturation at 512-dimensions (64 bytes packed).

The final weights are saved after weighting, projection and truncation of the entire tokenizer vocabulary. Thus, WordLlama becomes a single embedding matrix (nn.Embedding) that is considerably smaller than the gigabyte-sized llm codebooks we start with. The original tokenizer is still used to preprocess the text into tokens, and the reduced size token embeddings are average pooled. There is very little computation required, and the resulting model sizes range from 16mb to 250mb for the 128k llama3 vocabulary.

It's good option for some nlp-lite tasks. You can train sklearn classifiers on it, perform basic semantic matching, fuzzy matching, ranking and clustering. You can perform your own llm surgery and train your own model on consumer GPUs in a few hours.

## MTEB Results (standard models)

| Metric                 | WL64        | WL128        | WL256        | WL512        | WL1024        | GloVe 300d | Komninos | all-MiniLM-L6-v2 |
|------------------------|-------------|--------------|--------------|--------------|---------------|------------|----------|------------------|
| Clustering             | 32.23       | 34.20        | 35.11        | 35.27        | 35.34         | 27.73      | 26.57    | 42.35            |
| Reranking              | 50.33       | 51.52        | 52.03        | 52.20        | 52.37         | 43.29      | 44.75    | 58.04            |
| Classification         | 53.56       | 56.93        | 58.89        | 59.76        | 60.18         | 57.29      | 57.65    | 63.05            |
| Pair Classification    | 75.71       | 77.30        | 77.94        | 78.13        | 78.14         | 70.92      | 72.94    | 82.37            |
| STS                    | 65.48       | 66.53        | 66.84        | 66.85        | 66.89         | 61.85      | 62.46    | 78.90            |
| CQA DupStack           | 17.54       | 21.62        | 23.13        | 23.78        | 23.96         | 15.47      | 16.79    | 41.32            |
| SummEval               | 30.31       | 30.65        | 31.08        | 30.30        | 30.54         | 28.87      | 30.49    | 30.81            |

## Embed Text

Here’s how you can load pre-trained embeddings and use them to embed text:

```python
from wordllama import load

# Load pre-trained embeddings
wl = load(dim=64)

# Embed text
embeddings = wl.embed(["the quick brown fox jumps over the lazy dog", "and all that jazz"])
print(embeddings.shape)  # (2, 64)

# Binary embeddings are packed into uint32
# 64-dims => array of 2x uint32 
wl = load(dim=64, binary=True)
wl.embed("I went to the car", binarize=True, pack=True) # Output: array([[3029168104, 2427562626]], dtype=uint32)

# load large binary trained model
wl = load(dim=1024, binary=True)

# Use the use_hamming flag to binarize
similarity_score = wl.similarity("i went to the car", "i went to the pawn shop", use_hamming=True)
print(similarity_score)  # Output: 0.57421875

ranked_docs = wl.rank("i went to the car", ["van", "truck"], use_hamming=False)

# load a different model class
wl = load(config_name="mixtral")
```

## Training Notes

Binary embedding models showed more pronounced improvement at higher dimensions, and either 512 or 1024 is recommended for binary embedding.

The current MTEB results use the embedding codebook from Llama3 70B, and anticipate further improvement when the 405B model is released.

Deberta v3 Large has a similar vocab size, but only 1024 hidden dimension, and does not train as well as Llama3 8B (4096)+ or 70B (8192)++.

Command R+ have given me the best results so far, slightly edging out Llama3 70B. The codebook is also 5.9GB, which is the largest I have seen.
My ~observation is that models train better with larger hidden dimensions, larger vocabs and more pretraining tokens.

Hidden dimensions are projected down, but the vocab size is fixed, so 256k vocabs are double the final size as compared llama3. DBRX is also trianing quite well and has 100k vocab, and gemma2 27B seems to train well, with a large vocab and smaller hidden dimension. Mixtral 8x22B did not train as well, but would be significantly more compact with only 32k vocab, while still retaining most of the performance. With the exception of Command R+, which I rented an L40S for, I could reasonably train all of these on an A4500 with 20GB of VRAM and different batch sizes and truncation lengths. Batch size 256 seems to train as well or maybe slightly better than 512.

## Roadmap

- Test distillation training from a larger embedding model
- Test concatenation with Llama guard 2
- Retrain on llama3 405B (waiting on release...)
- Select and figure out hosting for v1 weights
- Write a requirements minimized package with C python bindings for basic operations


## Extracting Token Embeddings

To extract token embeddings from a model, ensure you have agreed to the user agreement and logged in using the Hugging Face CLI (for llama3 models). You can then use the following snippet:

```python
from wordllama import Config
from wordllama.extract import extract_safetensors

# Extract embeddings for the specified configuration
extract_safetensors("llama3_70B", "path/to/saved/model-0001-of-00XX.safetensors")
```

HINT: Embeddings are usually in the first safetensors file, but not always. Sometimes you have to snoop around and figure it out.

## Citations

If you use WordLlama in your research or project, please consider citing it as follows:

```bibtex
@software{miller2024wordllama,
  author = {Miller, D. Lee},
  title = {WordLlama: Recycled Token Embeddings from Large Language Models},
  year = {2024},
  url = {https://github.com/dleemiller/wordllama},
  version = {0.0.0}
}
```

## License

This project is licensed under the MIT License.

