# Word Llama

Word Llama is designed to facilitate the extraction and utilization of token embeddings from large language models. It supports operations such as embedding texts directly and extracting token embeddings to save as separate files for more efficient reuse. The goal of this library is utilize the high quality token embeddings learned in LLM training for a variety of lightweight NLP tasks, by training adapters and classifiers.

## Features

- **Load and Embed**: Quickly load pre-trained embeddings and use them to embed texts.
- **Extract Embeddings**: Extract token embeddings from transformer-based models and save them for later use.
- **Customizable**: Configure extraction and embedding processes with easy-to-use interfaces.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-repository/word_llama.git
cd word_llama
pip install -r requirements.txt
```

## Quick Start

Here’s how you can load pre-trained embeddings and use them to embed text:

```python
from word_llama import load

# Load pre-trained embeddings
wl = load("path/to/embeddings")

# Embed text
embeddings = wl.embed(["the quick brown fox jumps over the lazy dog", "and all that jazz"])
print(embeddings)
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
### Training Classifiers

Example

```bash
python train.py linear_svc path/to/llama3_embedding.safetensors --C 0.5 --device cuda
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

## License

This project is licensed under the MIT License.


