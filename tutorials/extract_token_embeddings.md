# Extracting Token Embeddings from an LLM

Pulling out the token embeddings is a simple process once you understand their format.
In this example, we'll extract the token embeddings from `Gemma2 2B`.

[Huggingface Repository](https://huggingface.co/google/gemma-2-2b-it/tree/main)

## Safetensors

There's a few formats out there. `Safetensors` is popular because it doesn't use `pickle` to serialize the tensors (which is unsafe because it permits code execution).

Safetensors also permits serializing to multiple files. Most larger models will serialize to multiple files, numerically indexed.
You will need to identify the file that contains the token embeddings. Most often, because it is nearest to the inputs of the model,
it will be contained in the first file. But there's no reason why it can't be in *any* of the other files.

## Index Files

Sometimes, there will be an index file. If there is, then you are in luck! You can just reference the index to determine the location.

[model.safetensors.index.json](https://huggingface.co/google/gemma-2-2b-it/blob/main/model.safetensors.index.json)
```
{
  "metadata": {
    "total_size": 5228683776
  },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.mlp.up_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.post_attention_layernorm.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.post_feedforward_layernorm.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.pre_feedforward_layernorm.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
    ...
```

Here we can see that right at the beginning is `model.embed_tokens.weight`.
These are the token embeddings that were learned during training the model. You can see that after that is the first transformer layer.

Download `model-00001-of-00002.safetensors`, because it contains the token embeddings we want.

## Model Surgery

Install the dependencies for training.
`pip install wordllama[train]`

You can load it directly with safetensors.
```
In [1]: from safetensors import safe_open

In [2]: with safe_open("/home/lee/Downloads/model-00001-of-00002.safetensors", framework="pt") as f:
   ...:     weights = f.get_tensor("model.embed_tokens.weight")
   ...: 

In [3]: weights.shape
Out[3]: torch.Size([256000, 2304])
```

The first dimension is the vocabulary, and the second is the size of the embedding vector.
If you wanted, you could also concatenate it with the Gemma2 9B and 27B token embeddings, because it uses the same tokenizer.
This would yield a tensor `torch.Size([256000, 10496])`.

Before training the projections, I have found it helpful to concatenate to larger dimensions,
especially for models that have been trained on different corpus.

## Configurations

Clone the repository. Copy an existing toml in `wordllama/config`, and edit it for your configuration.
For Gemma2 2B, we can copy the existing config for 27B, and edit the `dim` to match the model.

```
[model]
dim = 2304
n_vocab = 256000
hf_model_id = "google/gemma-2-2b-it"
pad_token = "<eos>"
```

It doesn't really matter what you use for a pad token, as long is it is an actual special token from the tokenizer.
Check the config: [tokenizer_config.json](https://huggingface.co/google/gemma-2-2b/blob/main/tokenizer_config.json)

The average pooling in WordLlama ignores the pad tokens.

## Saving

```
In [1]: from wordllama.extract.extract_safetensors import extract_safetensors

In [2]: extract_safetensors(
   ...:     config_name="gemma2_2B",
   ...:     tensor_path="/path/to/model-00001-of-00002.safetensors",
   ...:     key="model.embed_tokens.weight"
   ...: )
```

This will save the embedding weights as half precision, in a format ready for training.
