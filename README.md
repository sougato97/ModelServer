# ModelServer

Docker-first inference stack for chat + embeddings, with one API gateway and multiple model backends.

## What is included

- Unified FastAPI gateway: `model_inference/model_inference.py`
- Backend clients:
  - `model_inference/llm_client.py`
  - `model_inference/embedding_client.py`
  - `model_inference/models.py`
- Docker build files:
  - `deployment/model_inference/Dockerfile.api`
  - `deployment/model_inference/Dockerfile.vllm.cuda`
  - `deployment/model_inference/Dockerfile.vllm.rocm`
  - `deployment/model_inference/Dockerfile.llama.cuda`
  - `deployment/model_inference/Dockerfile.llama.rocm`
- Compose orchestration: `docker-compose.yml`

## Current status

- vLLM routing with both CUDA and ROCm profiles is supported.
- CUDA memory constraints/defaults are tuned for an RTX 4090 Laptop GPU (16 GB VRAM).
- ROCm compose defaults are optimized for WSL2 passthrough (`/dev/dxg` + WSL-mounted ROCm bridge libs).
- For native Linux ROCm, passthrough must be adjusted to `/dev/kfd` and `/dev/dri`.
- `llama.cpp` needs further hardening/fixes (especially CUDA build/link behavior) before relying on it in production.

## Models

- Chat LLM: `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Embeddings:
  - `intfloat/multilingual-e5-large-instruct`
  - `Qwen/Qwen3-Embedding-4B`

## Run

Prerequisites:
- Docker + Docker Compose
- NVIDIA setup for `cuda` profile, or ROCm setup for `rocm` profile
- ROCm note: current vLLM compose mappings are WSL2-first. For native Linux, switch device passthrough in `docker-compose.yml` as annotated in comments.

Start API + CUDA vLLM services:
```bash
docker compose --profile cuda up --build -d \
  model_inference \
  vllm_llm_cuda_qwen2-5_7b_instruct_awq \
  vllm_embed_cuda_intfloat_multilingual_e5_large_instruct
```

Optional CUDA Qwen embedding service:
```bash
docker compose --profile cuda up --build -d vllm_embed_cuda_qwen3_embedding_4b
```

Start API + ROCm vLLM services:
```bash
docker compose --profile rocm up --build -d \
  model_inference \
  vllm_llm_rocm_qwen2-5_7b_instruct_awq \
  vllm_embed_rocm_intfloat_multilingual_e5_large_instruct
```

## API endpoints

- `GET /health`
- `POST /chat`
- `POST /chat/stream`
- `POST /embeddings`
- `POST /mcp/call`
- `GET /events`

API host port: `http://localhost:8080`

## Sample API usage

Health:
```bash
curl -s http://localhost:8080/health
```

Chat (explicit model + route):
```bash
curl -s -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "what is the capital of oregon usa?",
    "session_id": null,
    "system": null,
    "temperature": 0.2,
    "max_tokens": 512,
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "backend": "vllm",
    "framework": "cuda"
  }'
```

Embeddings:
```bash
curl -s -X POST http://localhost:8080/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["hello world"],
    "model": "Qwen/Qwen3-Embedding-4B",
    "normalize": false,
    "prefix": "query",
    "backend": "vllm",
    "framework": "rocm"
  }'
```

## Notes

- `model` in `/chat` is optional. If omitted on vLLM route, it defaults to `Qwen/Qwen2.5-7B-Instruct-AWQ`.
- Qwen embedding requests are routed to the `7001` embedding endpoint when available.
- For llama services, place GGUF files in `deployment/model_inference/compatible_weights/`.
- Keep the ROCm comments in `docker-compose.yml` as-is: they document WSL2 defaults and the native Linux passthrough alternative.
