![Word Llama](wordllama.png)

# Word Llama

The power of 13 trillion tokens of training, extracted, bastardized and minimized into a little package for word embedding.

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

## Advanced Usage

(Provide advanced usage examples here)

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

## License

This project is licensed under the MIT License.

