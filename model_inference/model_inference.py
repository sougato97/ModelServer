import asyncio
import os
import subprocess
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from embedding_client import EmbeddingClient, EmbeddingConfig
from llm_client import LLMClient, LLMConfig
from models import LlamaServerClient, LlamaServerConfig


# Parse a boolean from environment with a default.
def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


SUPPORTED_BACKENDS = ("vllm", "llama")
SUPPORTED_FRAMEWORKS = ("cuda", "rocm")

# Defaults used when request does not specify routing.
BACKEND = os.getenv("BACKEND", "vllm").strip().lower()
FRAMEWORK = os.getenv("FRAMEWORK", "cuda").strip().lower()
SPAWN_MODELS = _env_bool("SPAWN_MODELS", True)
VLLM_READINESS_MODE = os.getenv("VLLM_READINESS_MODE", "reachable").strip().lower()
LLAMA_READINESS_MODE = os.getenv("LLAMA_READINESS_MODE", "reachable").strip().lower()

DEFAULT_SYSTEM = "You are a concise assistant."

VLLM_LLM_MODEL = os.getenv("VLLM_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
VLLM_EMBED_MODEL = os.getenv("VLLM_EMBED_MODEL", "intfloat/multilingual-e5-large-instruct")
QWEN_EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"

LLAMA_CPP_MODEL_PATH = os.getenv(
    "LLAMA_CPP_MODEL_PATH",
    "deployment/model_inference/compatible_weights/Qwen2.5-7B-Instruct.Q4_K_M.gguf",
)

VLLM_SERVE_COMMANDS = [
    [
        "vllm",
        "serve",
        VLLM_LLM_MODEL,
        "--port",
        "8000",
        "--gpu-memory-utilization",
        "0.65",
        "--max-model-len",
        "16384",
    ],
    [
        "vllm",
        "serve",
        VLLM_EMBED_MODEL,
        "--port",
        "7000",
        "--task",
        "embedding",
        "--gpu-memory-utilization",
        "0.20",
    ],
]
VLLM_MODEL_NAMES = [VLLM_LLM_MODEL, VLLM_EMBED_MODEL]
LLAMA_CPP_SERVE_COMMANDS = [
    [
        "llama-server",
        "-m",
        LLAMA_CPP_MODEL_PATH,
        "-ngl",
        "999",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
]


def _default_llm_url(backend: str, framework: str) -> str:
    return f"http://{backend}_llm_{framework}:8000/v1"


def _default_embed_e5_url(backend: str, framework: str) -> str:
    return f"http://{backend}_embed_{framework}:7000/v1"


def _default_embed_qwen_url(backend: str, framework: str) -> str:
    return f"http://vllm_embed_qwen_{framework}:7001/v1"


def _target_env_prefix(backend: str, framework: str) -> str:
    return f"{backend.upper()}_{framework.upper()}"


def _build_target_urls() -> Dict[Tuple[str, str], Dict[str, str]]:
    """Resolve endpoint URLs for all backend/framework combinations."""
    result: Dict[Tuple[str, str], Dict[str, str]] = {}
    for backend in SUPPORTED_BACKENDS:
        for framework in SUPPORTED_FRAMEWORKS:
            prefix = _target_env_prefix(backend, framework)
            llm_default = _default_llm_url(backend, framework)
            embed_e5_default = _default_embed_e5_url(backend, framework)
            embed_qwen_default = _default_embed_qwen_url(backend, framework)

            is_default_target = backend == BACKEND and framework == FRAMEWORK

            if is_default_target:
                llm_base_url = os.getenv("LLM_BASE_URL", os.getenv(f"{prefix}_LLM_BASE_URL", llm_default))
                embed_e5_base_url = os.getenv(
                    "EMBED_E5_BASE_URL",
                    os.getenv("EMBED_BASE_URL", os.getenv(f"{prefix}_EMBED_E5_BASE_URL", embed_e5_default)),
                )
                embed_qwen_base_url = os.getenv(
                    "EMBED_QWEN_BASE_URL",
                    os.getenv(f"{prefix}_EMBED_QWEN_BASE_URL", embed_qwen_default),
                )
            else:
                llm_base_url = os.getenv(f"{prefix}_LLM_BASE_URL", llm_default)
                embed_e5_base_url = os.getenv(f"{prefix}_EMBED_E5_BASE_URL", embed_e5_default)
                embed_qwen_base_url = os.getenv(f"{prefix}_EMBED_QWEN_BASE_URL", embed_qwen_default)

            if backend == "llama":
                # llama does not have a dedicated qwen embedding endpoint by default.
                embed_qwen_base_url = embed_e5_base_url

            result[(backend, framework)] = {
                "llm": llm_base_url,
                "embed_e5": embed_e5_base_url,
                "embed_qwen": embed_qwen_base_url,
            }
    return result


def _is_qwen_embed_model(model_name: Optional[str]) -> bool:
    return model_name in {"Qwen/Qwen3-Embedding-4B", "Qwen3-Embedding-4B"}


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message or follow-up question")
    session_id: Optional[str] = Field(None, description="Existing session ID for follow-ups")
    system: Optional[str] = Field(None, description="Optional system prompt for new sessions")
    temperature: float = 0.2
    max_tokens: int = 512
    model: Optional[str] = Field(
        None,
        description=(
            "Optional chat model override. For vllm, this should match a model served by the "
            "target endpoint. For llama, only 'local' is supported."
        ),
    )
    backend: Optional[str] = Field(None, description="Inference backend: vllm or llama")
    framework: Optional[str] = Field(None, description="Inference framework: cuda or rocm")


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class MCPCall(BaseModel):
    jsonrpc: str = "2.0"
    id: str
    method: str
    params: Dict[str, object]


class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="Input strings to embed")
    model: Optional[str] = Field(None, description="Embedding model name")
    normalize: bool = False
    prefix: Optional[str] = "query"
    backend: Optional[str] = Field(None, description="Inference backend: vllm or llama")
    framework: Optional[str] = Field(None, description="Inference framework: cuda or rocm")


class EmbeddingResponse(BaseModel):
    model: str
    vectors: List[List[float]]


class ModelInferenceAPI:
    # Initialize backend clients, state, and FastAPI app.
    def __init__(self, backend: str = BACKEND, framework: str = FRAMEWORK):
        self.default_backend = backend
        self.default_framework = framework
        if self.default_backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unknown BACKEND: {self.default_backend}")
        if self.default_framework not in SUPPORTED_FRAMEWORKS:
            raise ValueError(f"Unknown FRAMEWORK: {self.default_framework}")
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self.model_processes: List[subprocess.Popen] = []
        self.models_ready = False
        self.models_ready_event = asyncio.Event()
        self.models_ready_task: Optional[asyncio.Task] = None
        self._endpoint_state: Dict[str, Dict[str, float]] = {}

        self.target_urls = _build_target_urls()

        # Lazy client pools keyed by (backend, framework)
        self._vllm_llm_clients: Dict[Tuple[str, str, str], LLMClient] = {}
        self._vllm_embedding_clients: Dict[Tuple[str, str], EmbeddingClient] = {}
        self._llama_clients: Dict[Tuple[str, str], LlamaServerClient] = {}

        self.app = FastAPI(
            title="ModelServer Inference API",
            version="0.2.0",
            lifespan=self.lifespan,
        )
        self._register_routes()

    @asynccontextmanager
    # Start/stop model processes and readiness tasks during app lifespan.
    async def lifespan(self, _: FastAPI):
        if SPAWN_MODELS:
            if self.default_backend == "vllm":
                for cmd in VLLM_SERVE_COMMANDS:
                    self.model_processes.append(
                        subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT,
                        )
                    )
            elif self.default_backend == "llama":
                for cmd in LLAMA_CPP_SERVE_COMMANDS:
                    self.model_processes.append(
                        subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT,
                        )
                    )

        self.models_ready_task = asyncio.create_task(self._wait_for_default_ready())

        try:
            yield
        finally:
            if self.models_ready_task:
                self.models_ready_task.cancel()
            for proc in self.model_processes:
                proc.terminate()

    # Register all API routes.
    def _register_routes(self) -> None:
        self.app.add_api_route("/chat", self.chat, methods=["POST"], response_model=ChatResponse)
        self.app.add_api_route("/chat/stream", self.chat_stream, methods=["POST"])
        self.app.add_api_route("/embeddings", self.embeddings, methods=["POST"], response_model=EmbeddingResponse)
        self.app.add_api_route("/mcp/call", self.mcp_call, methods=["POST"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/events", self.events, methods=["GET"])

    # Normalize and validate backend/framework routing selection.
    def _resolve_route(self, backend: Optional[str], framework: Optional[str]) -> Tuple[str, str]:
        b = (backend or self.default_backend).strip().lower()
        f = (framework or self.default_framework).strip().lower()

        if b not in SUPPORTED_BACKENDS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid backend '{b}'. Allowed: {', '.join(SUPPORTED_BACKENDS)}",
            )
        if f not in SUPPORTED_FRAMEWORKS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid framework '{f}'. Allowed: {', '.join(SUPPORTED_FRAMEWORKS)}",
            )
        return b, f

    # Resolve endpoint URLs for a backend/framework pair.
    def _urls_for(self, backend: str, framework: str) -> Dict[str, str]:
        return self.target_urls[(backend, framework)]

    # Create/reuse a vLLM chat client for a target.
    def _resolve_chat_model(self, backend: str, requested_model: Optional[str]) -> str:
        if backend == "vllm":
            return requested_model.strip() if requested_model and requested_model.strip() else VLLM_LLM_MODEL

        if requested_model and requested_model.strip().lower() not in {"local"}:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Custom 'model' is not supported for llama backend. "
                    "Use backend=vllm for per-request model overrides."
                ),
            )
        return "local"

    def _get_vllm_llm_client(
        self,
        backend: str,
        framework: str,
        model: str,
    ) -> LLMClient:
        key = (backend, framework, model)
        client = self._vllm_llm_clients.get(key)
        if client is None:
            urls = self._urls_for(backend, framework)
            client = LLMClient(LLMConfig(
                base_url=urls["llm"],
                api_key="local",
                model=model,
            ))
            self._vllm_llm_clients[key] = client
        return client

    # Create/reuse a vLLM embeddings client for a target.
    def _get_vllm_embedding_client(self, backend: str, framework: str) -> EmbeddingClient:
        key = (backend, framework)
        client = self._vllm_embedding_clients.get(key)
        if client is None:
            urls = self._urls_for(backend, framework)
            client = EmbeddingClient(EmbeddingConfig(
                base_url=urls["embed_e5"],
                api_key="local",
                model=VLLM_EMBED_MODEL,
                model_base_urls={
                    QWEN_EMBED_MODEL: urls["embed_qwen"],
                },
            ))
            self._vllm_embedding_clients[key] = client
        return client

    # Create/reuse a llama.cpp client for a target.
    def _get_llama_client(self, backend: str, framework: str) -> LlamaServerClient:
        key = (backend, framework)
        client = self._llama_clients.get(key)
        if client is None:
            urls = self._urls_for(backend, framework)
            client = LlamaServerClient(LlamaServerConfig(
                llm_base_url=urls["llm"],
                embedding_base_url=urls["embed_e5"],
            ))
            self._llama_clients[key] = client
        return client

    """
    Usage: internal session lookup/creation for chat endpoints.
    API URL: N/A (internal helper)
    Usage JSON: N/A
    """
    # Create or fetch a session message list.
    def _get_or_create_session(self, session_id: Optional[str], system: Optional[str]):
        if session_id:
            messages = self.sessions.get(session_id)
            if messages is None:
                raise HTTPException(status_code=404, detail="Unknown session_id")
            return session_id, messages

        new_id = uuid.uuid4().hex
        messages = [{"role": "system", "content": system or DEFAULT_SYSTEM}]
        self.sessions[new_id] = messages
        return new_id, messages

    """
    Usage: internal text prefixing for embeddings.
    API URL: N/A (internal helper)
    Usage JSON: N/A
    """
    # Apply optional prefix to embedding inputs.
    def _apply_prefix(self, texts: List[str], prefix: Optional[str]) -> List[str]:
        if not prefix:
            return texts
        return [f"{prefix}: {t}" for t in texts]

    # Check whether an OpenAI-compatible endpoint is reachable and ready.
    def _endpoint_status(self, url: str) -> tuple[bool, bool]:
        try:
            with urllib.request.urlopen(f"{url}/models", timeout=2) as resp:
                return True, 200 <= resp.status < 300
        except Exception:
            return False, False

    # Determine which embedding endpoint should be used.
    def _embedding_url_for(self, backend: str, framework: str, model: Optional[str]) -> str:
        urls = self._urls_for(backend, framework)
        if backend == "vllm" and _is_qwen_embed_model(model):
            return urls["embed_qwen"]
        return urls["embed_e5"]

    # Build status object for the selected target.
    def _target_status(self, backend: str, framework: str) -> Dict[str, Dict[str, object]]:
        urls = self._urls_for(backend, framework)
        status: Dict[str, Dict[str, object]] = {}

        if backend == "vllm":
            candidates = {
                "llm": urls["llm"],
                "embed_e5": urls["embed_e5"],
                "embed_qwen": urls["embed_qwen"],
            }
        else:
            candidates = {
                "llm": urls["llm"],
                "embed_e5": urls["embed_e5"],
            }

        for key, url in candidates.items():
            reachable, ready = self._endpoint_status(url)
            status[key] = {"url": url, "reachable": reachable, "ready": ready}
        return status

    # Compute target readiness based on configured mode.
    def _target_live_ready(self, backend: str, framework: str) -> bool:
        status = self._target_status(backend, framework)
        states = list(status.values())
        reachable_ready = [bool(s["ready"]) for s in states if s["reachable"]]

        if backend == "vllm":
            if VLLM_READINESS_MODE == "all":
                return all(bool(s["reachable"]) and bool(s["ready"]) for s in states)
            return bool(reachable_ready) and all(reachable_ready)

        # llama readiness mode
        if LLAMA_READINESS_MODE == "all":
            return all(bool(s["reachable"]) and bool(s["ready"]) for s in states)
        return bool(reachable_ready) and all(reachable_ready)

    # Background readiness loop for default route.
    async def _wait_for_default_ready(self, timeout_s: int = 300, interval_s: float = 1.0) -> None:
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            if self._target_live_ready(self.default_backend, self.default_framework):
                self.models_ready = True
                self.models_ready_event.set()
                return
            await asyncio.sleep(interval_s)

    """
    Usage: internal chat request to the selected backend.
    API URL: N/A (internal helper)
    Usage JSON: N/A
    """
    # Run a non-streaming chat call against the selected backend/framework.
    def _chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        model: Optional[str],
        backend: str,
        framework: str,
    ) -> str:
        if backend == "vllm":
            selected_model = self._resolve_chat_model(backend, model)
            llm_client = self._get_vllm_llm_client(backend, framework, selected_model)
            return llm_client.chat(messages, temperature=temperature, max_tokens=max_tokens)

        self._resolve_chat_model(backend, model)
        llama_client = self._get_llama_client(backend, framework)
        return llama_client.chat(messages, temperature=temperature, max_tokens=max_tokens)

    """
    Usage: internal streaming chat request to the selected backend.
    API URL: N/A (internal helper)
    Usage JSON: N/A
    """
    # Run a streaming chat call against the selected backend/framework.
    def _stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        model: Optional[str],
        backend: str,
        framework: str,
    ) -> Iterator[str]:
        if backend == "vllm":
            selected_model = self._resolve_chat_model(backend, model)
            llm_client = self._get_vllm_llm_client(backend, framework, selected_model)
            return llm_client.stream_chat(messages, temperature=temperature, max_tokens=max_tokens)

        self._resolve_chat_model(backend, model)
        llama_client = self._get_llama_client(backend, framework)
        return llama_client.stream_chat(messages, temperature=temperature, max_tokens=max_tokens)

    """
    Usage: internal embeddings request to the selected backend.
    API URL: N/A (internal helper)
    Usage JSON: N/A
    """
    # Run an embedding request against the selected backend/framework.
    def _embed(
        self,
        texts: List[str],
        model: Optional[str],
        normalize: bool,
        prefix: Optional[str],
        backend: str,
        framework: str,
    ) -> EmbeddingResponse:
        if backend == "vllm":
            embedding_client = self._get_vllm_embedding_client(backend, framework)
            vectors = embedding_client(
                texts,
                model_name=model,
                normalize=normalize,
                prefix=prefix,
            )
            resolved_model = embedding_client._resolve_model(model)
            return EmbeddingResponse(model=resolved_model, vectors=vectors)

        llama_client = self._get_llama_client(backend, framework)
        prefixed = self._apply_prefix(texts, prefix)
        vectors = llama_client.embed(prefixed, normalize=normalize)
        return EmbeddingResponse(model="local", vectors=vectors)

    # Ensure chat endpoint for selected route is ready.
    def _ensure_chat_ready(self, backend: str, framework: str) -> None:
        url = self._urls_for(backend, framework)["llm"]
        reachable, ready = self._endpoint_status(url)
        if not (reachable and ready):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model endpoint is still loading or unavailable. "
                    f"backend={backend}, framework={framework}, endpoint={url}"
                ),
            )

    # Ensure embedding endpoint for selected route/model is ready.
    def _ensure_embed_ready(self, backend: str, framework: str, model: Optional[str]) -> None:
        url = self._embedding_url_for(backend, framework, model)
        reachable, ready = self._endpoint_status(url)
        if not (reachable and ready):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Embedding endpoint is still loading or unavailable. "
                    f"backend={backend}, framework={framework}, endpoint={url}"
                ),
            )

    """
    Usage: POST a message to create or continue a chat session.
    API URL: /chat
    Usage JSON: {"message":"Hello","session_id":null,"system":null,"temperature":0.2,"max_tokens":512,"model":"Qwen/Qwen2.5-7B-Instruct-AWQ","backend":"vllm","framework":"cuda"}
    """
    # Handle synchronous chat requests.
    def chat(self, req: ChatRequest):
        backend, framework = self._resolve_route(req.backend, req.framework)
        self._ensure_chat_ready(backend, framework)

        sess_id, messages = self._get_or_create_session(req.session_id, req.system)
        messages.append({"role": "user", "content": req.message})
        reply = self._chat(
            messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            model=req.model,
            backend=backend,
            framework=framework,
        )
        messages.append({"role": "assistant", "content": reply})
        return ChatResponse(session_id=sess_id, reply=reply)

    """
    Usage: POST a message and receive a streamed response (SSE).
    API URL: /chat/stream
    Usage JSON: {"message":"Hello","session_id":null,"system":null,"temperature":0.2,"max_tokens":512,"model":"Qwen/Qwen2.5-7B-Instruct-AWQ","backend":"vllm","framework":"cuda"}
    """
    # Handle streaming chat requests.
    def chat_stream(self, req: ChatRequest):
        backend, framework = self._resolve_route(req.backend, req.framework)
        self._ensure_chat_ready(backend, framework)

        sess_id, messages = self._get_or_create_session(req.session_id, req.system)
        messages.append({"role": "user", "content": req.message})

        # Emit SSE chunks and persist the full assistant reply.
        def event_stream():
            collected = []
            for chunk in self._stream_chat(
                messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                model=req.model,
                backend=backend,
                framework=framework,
            ):
                collected.append(chunk)
                yield f"data: {chunk}\n\n"
            if collected:
                messages.append({"role": "assistant", "content": "".join(collected)})

        headers = {"X-Session-Id": sess_id}
        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

    """
    Usage: POST text list to generate embeddings.
    API URL: /embeddings
    Usage JSON: {"texts":["hello","world"],"model":null,"normalize":false,"prefix":"query","backend":"vllm","framework":"rocm"}
    """
    # Handle embedding requests.
    def embeddings(self, req: EmbeddingRequest):
        backend, framework = self._resolve_route(req.backend, req.framework)
        self._ensure_embed_ready(backend, framework, req.model)
        return self._embed(
            req.texts,
            req.model,
            req.normalize,
            req.prefix,
            backend=backend,
            framework=framework,
        )

    """
    Usage: MCP JSON-RPC bridge for the chat tool.
    API URL: /mcp/call
    Usage JSON: {"jsonrpc":"2.0","id":"1","method":"call_tool","params":{"name":"chat","arguments":{"message":"Hi"}}}
    """
    # Bridge MCP JSON-RPC calls to the chat endpoint.
    def mcp_call(self, call: MCPCall):
        if call.method != "call_tool":
            raise HTTPException(status_code=400, detail="Unsupported method")

        name = call.params.get("name")
        if name != "chat":
            raise HTTPException(status_code=404, detail="Unknown tool")

        arguments = call.params.get("arguments") or {}
        req = ChatRequest(**arguments)
        resp = self.chat(req)

        return {
            "jsonrpc": "2.0",
            "id": call.id,
            "result": {
                "content": [{"type": "text", "text": resp.reply}],
                "session_id": resp.session_id,
            },
        }

    # Update endpoint timing state and build model health record.
    def _model_health_record(self, now: float, name: str, model_type: str, endpoint: str) -> Dict[str, object]:
        reachable, ready = self._endpoint_status(endpoint)
        if reachable:
            self._endpoint_state.setdefault(endpoint, {}).setdefault("first_up", now)
        if ready:
            self._endpoint_state.setdefault(endpoint, {}).setdefault("first_ready", now)

        times = self._endpoint_state.get(endpoint, {})
        first_up = times.get("first_up")
        first_ready = times.get("first_ready")
        load_time = (first_ready - first_up) if first_up and first_ready else None

        return {
            "name": name,
            "type": model_type,
            "endpoint": endpoint,
            "loaded": bool(ready),
            "load_time_seconds": load_time,
            "ready": bool(ready),
        }

    """
    Usage: Health check for the service.
    API URL: /health
    Usage JSON: {}
    Query params: backend={vllm|llama}, framework={cuda|rocm}
    """
    # Report live readiness and endpoint status.
    def health(self, backend: Optional[str] = None, framework: Optional[str] = None):
        now = time.monotonic()

        backends = [backend.strip().lower()] if backend else list(SUPPORTED_BACKENDS)
        frameworks = [framework.strip().lower()] if framework else list(SUPPORTED_FRAMEWORKS)

        for b in backends:
            if b not in SUPPORTED_BACKENDS:
                raise HTTPException(status_code=400, detail=f"Invalid backend '{b}'")
        for f in frameworks:
            if f not in SUPPORTED_FRAMEWORKS:
                raise HTTPException(status_code=400, detail=f"Invalid framework '{f}'")

        result = []
        for b in backends:
            for f in frameworks:
                urls = self._urls_for(b, f)

                if b == "vllm":
                    models = [
                        self._model_health_record(now, VLLM_LLM_MODEL, "llm", urls["llm"]),
                        self._model_health_record(now, VLLM_EMBED_MODEL, "embedding", urls["embed_e5"]),
                        self._model_health_record(now, QWEN_EMBED_MODEL, "embedding", urls["embed_qwen"]),
                    ]
                else:
                    models = [
                        self._model_health_record(now, "local", "llm", urls["llm"]),
                        self._model_health_record(now, "local-embedding", "embedding", urls["embed_e5"]),
                    ]

                result.append({
                    "backend": b,
                    "framework": f,
                    "models": models,
                })

        return result

    """
    Usage: Server-sent events that emit when default route models are loaded.
    API URL: /events
    Usage JSON: {}
    """
    # Emit periodic readiness status over SSE.
    async def events(self):
        async def event_stream():
            while True:
                ready = self._target_live_ready(self.default_backend, self.default_framework)
                payload = {
                    "event": "model_status",
                    "backend": self.default_backend,
                    "framework": self.default_framework,
                    "models_loaded": ready,
                    "models": VLLM_MODEL_NAMES if (ready and self.default_backend == "vllm") else [],
                }
                yield f"data: {payload}\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(event_stream(), media_type="text/event-stream")


api = ModelInferenceAPI()
app = api.app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
