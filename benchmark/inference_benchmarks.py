import time
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from wordllama import WordLlama

N = 8192

# Set environment variables to limit CPU usage
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"  # MKL threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS threads

# If you're using CUDA, you can also limit GPU usage
torch.set_num_threads(1)

# Step 1: Load the Dataset
# We'll use the AG News dataset and select the first N texts.
dataset = load_dataset("fancyzhx/ag_news", split=f"train[:{N}]")
texts = dataset["text"]

# Step 2: Define Embedding Functions


def embed_texts_wordllama(texts):
    """Embed texts using WordLlama."""
    wl = WordLlama.load()
    start_time = time.perf_counter()
    embeddings = wl.embed(texts)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    return embeddings, total_time


def embed_texts_sentence_transformers(model_name, texts, device, batch_size=32):
    """Embed texts using SentenceTransformer models."""
    model = SentenceTransformer(model_name, device=device)
    start_time = time.perf_counter()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    return embeddings, total_time


# Step 3: Define Benchmarks

benchmarks = [
    {"model_name": "WordLlama", "device": "cpu"},
    {"model_name": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"},
    {"model_name": "sentence-transformers/all-MiniLM-L6-v2", "device": "cuda"},
    {"model_name": "intfloat/e5-base", "device": "cpu"},
    {"model_name": "intfloat/e5-base", "device": "cuda"},
]

# Step 4: Run Benchmarks

results = []

for benchmark in benchmarks:
    model_name = benchmark["model_name"]
    device = benchmark["device"]
    print(f"Running benchmark for {model_name} on {device.upper()}")
    if model_name == "WordLlama":
        embeddings, total_time = embed_texts_wordllama(texts)
    else:
        embeddings, total_time = embed_texts_sentence_transformers(
            model_name, texts, device
        )
    results.append({"model_name": model_name, "device": device, "time": total_time})

# Step 5: Prepare Data for Plotting

labels = []
times = []
for result in results:
    label = f"{result['model_name'].split('/')[-1]} ({result['device'].upper()})"
    labels.append(label)
    times.append(result["time"])

# Calculate speedups relative to WordLlama
wordllama_time = next(
    result["time"] for result in results if result["model_name"] == "WordLlama"
)
speedups = [time / wordllama_time for time in times]
colors = ["skyblue" if "CPU" in label else "lightgreen" for label in labels]
print(max(times))

# Step 6: Plot the Results in XKCD Style

import matplotlib.pyplot as plt

with plt.xkcd():
    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos, times, align="center", color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"Embedding Time Comparison for {N} Texts")
    max_time = max(times)
    ax.set_ylim(0.5, max_time * 5)
    ax.set_yscale("log")

    # Annotate bars with time and speedup
    for i, (time_val, speedup) in enumerate(zip(times, speedups)):
        if i == 0:
            ax.text(i, time_val, f"{time_val:.2f}s", ha="center", va="bottom")
        else:
            ax.text(
                i,
                time_val,
                f"{time_val:.2f}s\n({speedup:.1f}x slower)",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    plt.show()
