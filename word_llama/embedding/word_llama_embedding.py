import numpy as np
import torch
from torch import nn
import safetensors.torch

from transformers import AutoTokenizer
from typing import Union

from ..adapters import AvgPool


class WordLlamaEmbedding(nn.Module):
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "return_attention_mask": True,
        "max_length": 1024,
        "padding": "max_length",
        "truncation": True,
        "add_special_tokens": False,  # don't need without context
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
        self.tokenizer.pad_token_id = self.tokenizer.vocab[
            "<|end_of_text|>"
        ]  # for llama3 models

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
    def embed(
        self,
        texts: Union[str, list[str]],
        norm: bool = False,
        binarize: bool = False,
        pack: bool = True,
        return_pt: bool = False
    ) -> np.array:
        """Tokenize and embed a string or list of strings"""
        single_text = False
        if isinstance(texts, str):
            single_text = True
            self.tokenizer_kwargs["return_attention_mask"] = False
            self.tokenizer_kwargs["padding"] = "do_not_pad"
        else:
            self.tokenizer_kwargs["return_attention_mask"] = True
            self.tokenizer_kwargs["padding"] = "max_length"

        # tokenize
        tensors = self.tokenize(texts)

        # create embedding
        with torch.no_grad():
            x = self.embedding(tensors["input_ids"].to(self.embedding.weight.device))
            x = AvgPool.avg_pool(x, tensors.get("attention_mask"), norm=norm)
            if not return_pt:
                x = x.cpu().numpy()

        if binarize:
            x = x > 0
            if pack:
                x = np.packbits(x, axis=-1)

        if single_text:
            x = x[0]
        return x

    @staticmethod
    def hamming_similarity(a, b):
        assert a.shape == b.shape
        assert a.ndim == 1
        assert a.dtype == np.uint8
        assert b.dtype == np.uint8

        max_dist = a.size * 8

        # calculate distance
        dist = sum(bin(x).count("1") for x in a ^ b)

        return 1.0 - dist / max_dist

    @staticmethod
    def cosine_similarity(a, b):
        assert a.shape == b.shape
        assert a.dtype in (np.float16, np.float32)
        assert b.dtype in (np.float16, np.float32)

        # Compute cosine similarity
        if a.ndim > 1:
            a_norm = np.linalg.norm(a, axis=1)
            b_norm = np.linalg.norm(b, axis=1)
            return np.sum(a * b, axis=1) / (a_norm * b_norm)
        else:
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            return  np.sum(a * b) / (a_norm * b_norm)

    def save(self, *args, **kwargs):
        pass
