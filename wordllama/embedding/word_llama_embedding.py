import numpy as np
import torch
from torch import nn
import safetensors.torch

from transformers import AutoTokenizer
from typing import Union, List, Dict

from ..adapters import AvgPool
import warnings


class WordLlamaEmbedding(nn.Module):
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "return_attention_mask": True,
        "max_length": 1024,
        "padding": "max_length",
        "truncation": True,
        "add_special_tokens": False,  # don't need without context
    }

    def __init__(self, config, tokenizer_kwargs=None, dims=None):
        super().__init__()
        self.config = config
        model = config.model
        self.embedding = nn.Embedding(
            model.n_vocab, model.dim if dims is None else dims
        )
        if tokenizer_kwargs:
            self.tokenizer_kwargs = tokenizer_kwargs

        # turn off gradients
        for param in self.embedding.parameters():
            param.requires_grad = False

        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model.hf_model_id)
        self.tokenizer.pad_token_id = self.tokenizer.vocab[model.pad_token]

    @classmethod
    def build(cls, filepath, config, dims=None):
        word_llama = cls(config, dims=dims)
        safetensors.torch.load_model(word_llama, filepath)
        return word_llama

    def forward(self, tensors: Dict[str, torch.Tensor]):
        return {
            "token_ids": tensors["input_ids"],
            "token_embeddings": self.embedding(tensors["input_ids"]),
            "attention_mask": tensors["attention_mask"],
        }

    def tokenize(self, *args, **kwargs):
        texts = list(args).pop(0)
        texts = [texts] if isinstance(texts, str) else texts
        if "inference" in self.tokenizer_kwargs:
            del self.tokenizer_kwargs["inference"]
        return self.tokenizer(texts, **self.tokenizer_kwargs)

    @torch.inference_mode()
    def embed(
        self,
        texts: Union[str, List[str]],
        norm: bool = False,
        binarize: bool = False,
        pack: bool = True,
        return_pt: bool = False,
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

        # Clamp out-of-bounds input_ids
        if tensors["input_ids"].max() >= self.embedding.num_embeddings:
            warnings.warn("Some input_ids are out of bounds. Clamping to valid range.")
            tensors["input_ids"] = tensors["input_ids"].clamp(
                0, self.embedding.num_embeddings - 1
            )

        # Check for NaNs in input_ids and replace with 0
        if torch.isnan(tensors["input_ids"]).any():
            warnings.warn("NaN values found in input_ids. Replacing NaNs with 0.")
            tensors["input_ids"][torch.isnan(tensors["input_ids"])] = 0

        # Ensure at least one non-zero value in the attention mask
        if tensors.get("attention_mask") is not None:
            attention_mask = tensors["attention_mask"]
            if (attention_mask.sum(dim=1) == 0).any():
                warnings.warn(
                    "Some attention masks are all zeros. Setting the first token to 1 for these cases."
                )
                attention_mask[attention_mask.sum(dim=1) == 0, 0] = 1

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

    def save(self, *args, **kwargs):
        pass
