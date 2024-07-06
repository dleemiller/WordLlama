<p align="center">
  <img src="wordllama.png" alt="Word Llama" width="60%">
</p>

# Word Llama

The power of 13 trillion tokens of training, extracted, flogged and minimized into a cute little package for word embedding.

## What is it?

WordLlama is a word embedding model that recycles components from large language models (LLMs) to create efficient and compact word representations (such as GloVe or Word2Vec).
WordLlama begins by extracting the token embedding codebook from a state-of-the-art LLM (e.g., LLama3 70B).

The key features of WordLlama include:

1. Dimension Reduction: We train a projection to reduce the embedding dimension, making it more manageable for various applications.
2. Token Weighting: An additional learnable token weighting factor is incorporated, with stopwords initially assigned lower weights.
3. Training Approach: The model is trained using sentence-transformers with MultipleNegativesRankingLoss, similar to generalized embedding models.
4. Embedding Generation: Final embeddings are created by average pooling the projected token embeddings.

To optimize for performance, WordLlama employs the Matryoshka training technique, allowing for flexible truncation of the embedding dimension.
For even greater efficiency, we implement straight-through estimators during training to produce binary embeddings.
This approach enables us to create ultra-compact representations, with the smallest model producing 64-bit embeddings that can leverage rapid hamming distance calculations.

The final weights are saved after weighting, projection and truncation of the entire tokenizer vocabulary. Thus, WordLlama becomes a single embedding matrix (nn.Embedding). The original
tokenizer is still used to preprocess the text into tokens, and the reduced size token embeddings are averaged. There is very little computation required, and the
resulting model sizes range from 16mb to 250mb for the 128k llama3 vocabulary.

## Next steps

- Test distillation training from a larger embedding model
- Test other LLM token embeddings: small vocab like phi-3, xl vocab like gemma 2
- Test concatenation with Llama guard 2
- Retrain on llama3 405B (waiting on release...)
- Figure out hosting for final v1 weights
- Add some convenience methods for similarity
- Write a requirements minimized package with C python bindings for basic operations

## MTEB Results (preliminary)

(Running Evaluators)

## Features

- **Load and Embed**: Quickly load pre-trained embeddings and use them to embed texts.
- **Extract Embeddings**: Extract token embeddings from transformer-based models and save them for later use.
- **Matryoshka Training**: Truncate to size with Matryoshka embedding training
- **Binariz-abe**: Even smaller and faster by training with straight through estimators for binarization.

Note: binary embeddings have a greater performance loss at smaller dimensions that dense embeddings.

Included is the 64-dim binary trained model, because it's small enough for github. I'll update weights once the 405B parameter model is released.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/dleemiller/word_llama.git
cd word_llama
pip install -r requirements.txt
```

## Quick Start

Here’s how you can load pre-trained embeddings and use them to embed text:

```python
from word_llama import load

# Load pre-trained embeddings
wl = load("weights/wordllama_64_binary.safetensors") # binary trained, truncated to 64-dims

# Embed text
embeddings = wl.embed(["the quick brown fox jumps over the lazy dog", "and all that jazz"])
print(embeddings)
```

## Demonstration

```python
from word_llama import load, Config

# Load Word Llama with configuration
wl = load("weights/wordllama_64_binary.safetensors", Config.wordllama_64)

# Embed texts
a = wl.embed("I went to the car")
b = wl.embed("I went to the sedan")
c = wl.embed("I went to the park")

# Calculate cosine similarity
print(wl.cosine_similarity(a, b))  # Output: 0.68713075
print(wl.cosine_similarity(a, c))  # Output: 0.28954816

# Embed texts with binarization
a = wl.embed("I went to the car", binarize=True)
print(a)  # Output: array([232, 109, 141, 180, 130, 170, 177, 144], dtype=uint8)
b = wl.embed("I went to the sedan", binarize=True)

# Calculate Hamming similarity
print(wl.hamming_similarity(a, b))  # Output: 0.78125
```

## Extracting Token Embeddings

To extract token embeddings from a model, ensure you have agreed to the user agreement and logged in using the Hugging Face CLI (for llama3 models). You can then use the following snippet:

```python
from word_llama import Config
from word_llama.extract import extract_hf

# Extract embeddings for the specified configuration
extract_hf(Config.llama3_8B, "path/to/save/llama3_8B_embeddings.safetensors")
```

## Citations

If you use WordLlama in your research or project, please consider citing it as follows:

```bibtex
@software{miller2024wordllama,
  author = {Miller, D. Lee},
  title = {WordLlama: Recycled Token Embeddings from Large Language Models},
  year = {2024},
  url = {https://github.com/dleemiller/word_llama},
  version = {0.0.0}
}
```

## License

This project is licensed under the MIT License.

