# ModelServer

Lightweight inference wrapper that serves a paired LLM + embedding stack via the vLLM
inference layer.

## Models

- LLM: Qwen2.5-7B-Instruct-AWQ
- Embeddings: multilingual-e5-large-instruct

## Hardware

- Designed to execute on a single GPU with ~16 GB VRAM.

## Repo layout

- `model_inference/model_inference.py`: main entrypoint for running the LLM and
  embedding model together.
- `model_inference/llm_client.py`: LLM client wrapper.
- `model_inference/embedding_client.py`: embedding client wrapper.

## Notes

- The LLM and embedding model run together through the vLLM inference layer.
