import safetensors.torch
from transformers import AutoModelForCausalLM, AutoModel

from ..config import WordLlamaConfig
from ..embedding.word_llama_embedding import WordLlamaEmbedding


def extract_from_hf(config: WordLlamaConfig, name: str):
    # load the model in transformers
    model_id = config.model.hf_model_id
    if config.model.is_encoder:
        model = AutoModel.from_pretrained(model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)

    # key is embed_tokens, might be different for other models
    if "deberta" in model_id.lower():
        embed = model.embeddings.word_embeddings
    else:
        embed = model.model.embed_tokens
    state_dict = embed.state_dict()

    # load up the word mistral model
    wl = WordLlamaEmbedding(config)
    wl.load_state_dict({"embedding.weight": state_dict["weight"]})
    safetensors.torch.save_model(wl.half(), f"{name}.safetensors")
