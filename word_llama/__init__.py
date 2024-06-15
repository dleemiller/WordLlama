from .embedding import Config
from .embedding.word_llama_embedding import WordLlamaEmbedding


def load(filepath, config=Config.llama3_8B):
    return WordLlamaEmbedding.build(filepath, config)
