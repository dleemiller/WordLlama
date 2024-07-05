import torch
import safetensors.torch

from torch import nn
from transformers import AutoTokenizer
from typing import Union


class WordLlamaEmbedding(nn.Module):
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "return_attention_mask": False,
        "max_length": 1024,
        "padding": "max_length",
        "truncation": True,
        "add_special_tokens": False, # don't need without context
    }

    def __init__(self, config, tokenizer_kwargs=None):
        super().__init__()
        self.config = config
        model = config.model
        self.embedding = nn.Embedding(model.n_vocab, model.dim)
        if tokenizer_kwargs:
            self.tokenizer_kwargs = tokenizer_kwargs

        # turn off gradients
        for param in self.embedding.parameters():
            param.requires_grad = False

        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model.hf_model_id)
        self.tokenizer.pad_token_id = self.tokenizer.vocab["<|end_of_text|>"] # for llama3 models

    @classmethod
    def build(cls, filepath, config):
        word_llama = cls(config)
        safetensors.torch.load_model(word_llama, filepath)
        return word_llama

    def forward(self, tensors: dict[torch.Tensor]):
        return {
            "token_ids": tensors["input_ids"],
            "token_embeddings": self.embedding(tensors["input_ids"]),
            "attention_mask": tensors["attention_mask"],
        }

    def tokenize(self, *args, **kwargs):
        texts = list(args).pop(0)
        texts = [texts] if isinstance(texts, str) else texts
        return self.tokenizer(texts, **self.tokenizer_kwargs)

    @torch.inference_mode()
    def embed(self, texts: Union[str, list[str]], norm:bool=False, binarize=None):
        tensors = self.tokenize(texts)
        with torch.no_grad():
            x = self.embedding(tensors["input_ids"].to(self.embedding.weight.device))
            x = AvgPool.avg_pool(x, tensors["attention_mask"], norm=norm)

        if binarize:
            return x.sign().bool()
        return x

    def save(self, *args, **kwargs):
        pass
