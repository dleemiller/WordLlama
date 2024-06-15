import torch
import safetensors.torch

from torch import nn
from transformers import AutoTokenizer
from typing import Union


class WordLlamaEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model = config.model
        self.embedding = nn.Embedding(model.n_vocab, model.dim)

        # turn off gradients
        for param in self.embedding.parameters():
            param.requires_grad = False

        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model.hf_model_id)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.pad_token_id = 0

    @classmethod
    def build(cls, filepath, config):
        word_llama = cls(config)
        safetensors.torch.load_model(word_llama, filepath)
        return word_llama

    def forward(self, tensors: dict[torch.Tensor]):
        return {
            "token_embeddings": self.embedding(tensors["input_ids"]),
            "attention_mask": tensors["attention_mask"],
        }

    def tokenize(self, *args, **kwargs):
        texts = list(args).pop(0)
        return self.tokenizer(
            [texts] if isinstance(texts, str) else texts,
            return_tensors="pt",
            return_attention_mask=True,
            max_length=kwargs.get("max_length", 96),
            padding="max_length",
            truncation=True,
            add_special_tokens=False,
        )

    @torch.inference_mode()
    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.embedding(input_ids)
        return self.avg_pool(x, attention_mask)

    @torch.inference_mode()
    def avg_pool(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Mask
        mask = attention_mask.unsqueeze(dim=-1)

        # Average pool with mask
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        return x
