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

        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model.hf_model_id)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.pad_token_id = 0

    @classmethod
    def build(cls, filepath, config):
        word_llama = cls(config)
        safetensors.torch.load_model(word_llama, filepath)
        return word_llama

    def forward(self, *args, **kwargs):
        pass

    #        return self.embed(input_ids)

    @torch.inference_mode()
    def embed(self, texts: Union[str, list[str]], max_length: int = 128):
        assert isinstance(texts, str) or isinstance(texts, list)
        input_ids = self.tokenizer(
            [texts] if isinstance(texts, str) else texts,
            return_tensors="pt",
            return_attention_mask=False,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=False,
        )["input_ids"].to(self.embedding.weight.device)

        x = self.embedding(input_ids)
        return self.avg_pool(x, input_ids)

    @torch.inference_mode()
    def avg_pool(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # Mask
        mask = (input_ids != 0).unsqueeze(dim=-1)

        # Average pool with mask
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        return x
