from .embedding import Config
from .embedding.word_llama_embedding import WordLlamaEmbedding


def load(filepath):
    return WordLlamaEmbedding.build(filepath, Config.llama3_8B)
