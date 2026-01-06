import asyncio
import subprocess
import urllib.request
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Iterator, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from embedding_client import EmbeddingClient, EmbeddingConfig
from llm_client import LLMClient, LLMConfig
from models import LlamaServerClient, LlamaServerConfig


# Switch between backends: "vllm" or "llama"
BACKEND = "llama"

DEFAULT_SYSTEM = "You are a concise assistant."

LLM_BASE_URL = "http://127.0.0.1:8000/v1"
EMBED_BASE_URL = "http://127.0.0.1:7000/v1"
VLLM_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
VLLM_EMBED_MODEL = "intfloat/multilingual-e5-large-instruct"

LLAMA_CPP_MODEL_PATH = "/home/sougato97/documents/startup_work/model_inference/compatible_weights/Qwen2.5-7B-Instruct.Q4_K_M.gguf"

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


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message or follow-up question")
    session_id: Optional[str] = Field(None, description="Existing session ID for follow-ups")
    system: Optional[str] = Field(None, description="Optional system prompt for new sessions")
    temperature: float = 0.2
    max_tokens: int = 512


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


class EmbeddingResponse(BaseModel):
    model: str
    vectors: List[List[float]]


class ModelInferenceAPI:
    def __init__(self, backend: str = BACKEND):
        self.backend = backend
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self.model_processes: List[subprocess.Popen] = []
        self.models_ready = False
        self.models_ready_event = asyncio.Event()
        self.models_ready_task: Optional[asyncio.Task] = None

        self.llm_client = None
        self.embedding_client = None
        self.llama_client = None

        if self.backend == "vllm":
            self.llm_client = LLMClient(LLMConfig(
                base_url=LLM_BASE_URL,
                api_key="local",
                model=VLLM_LLM_MODEL,
            ))
            self.embedding_client = EmbeddingClient(EmbeddingConfig(
                base_url=EMBED_BASE_URL,
                api_key="local",
            ))
        elif self.backend == "llama":
            self.llama_client = LlamaServerClient(LlamaServerConfig(
                llm_base_url=LLM_BASE_URL,
                embedding_base_url=EMBED_BASE_URL,
            ))
        else:
            raise ValueError(f"Unknown BACKEND: {self.backend}")

        self.app = FastAPI(
            title="Seekers Story Model Inference API",
            version="0.1.0",
            lifespan=self.lifespan,
        )
        self._register_routes()

    @asynccontextmanager
    async def lifespan(self, _: FastAPI):
        if self.backend == "vllm":
            for cmd in VLLM_SERVE_COMMANDS:
                self.model_processes.append(
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                    )
                )
            self.models_ready_task = asyncio.create_task(self._wait_for_vllm_ready())
        elif self.backend == "llama":
            for cmd in LLAMA_CPP_SERVE_COMMANDS:
                self.model_processes.append(
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                    )
                )
            self.models_ready_task = asyncio.create_task(self._wait_for_llama_ready())
        else:
            self.models_ready_event.set()
        try:
            yield
        finally:
            if self.models_ready_task:
                self.models_ready_task.cancel()
            for proc in self.model_processes:
                proc.terminate()

    def _register_routes(self) -> None:
        self.app.add_api_route("/chat", self.chat, methods=["POST"], response_model=ChatResponse)
        self.app.add_api_route("/chat/stream", self.chat_stream, methods=["POST"])
        self.app.add_api_route("/embeddings", self.embeddings, methods=["POST"], response_model=EmbeddingResponse)
        self.app.add_api_route("/mcp/call", self.mcp_call, methods=["POST"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/events", self.events, methods=["GET"])

    """
    Usage: internal session lookup/creation for chat endpoints.
    API URL: N/A (internal helper)
    Usage JSON: N/A
    """
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
    def _apply_prefix(self, texts: List[str], prefix: Optional[str]) -> List[str]:
        if not prefix:
            return texts
        return [f"{prefix}: {t}" for t in texts]

    """
    Usage: internal chat request to the selected backend.
    API URL: N/A (internal helper)
    Usage JSON: N/A
    """
    def _chat(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        if self.backend == "vllm":
            return self.llm_client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return self.llama_client.chat(messages, temperature=temperature, max_tokens=max_tokens)

    """
    Usage: internal streaming chat request to the selected backend.
    API URL: N/A (internal helper)
    Usage JSON: N/A
    """
    def _stream_chat(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> Iterator[str]:
        if self.backend == "vllm":
            return self.llm_client.stream_chat(messages, temperature=temperature, max_tokens=max_tokens)
        return self.llama_client.stream_chat(messages, temperature=temperature, max_tokens=max_tokens)

    """
    Usage: internal embeddings request to the selected backend.
    API URL: N/A (internal helper)
    Usage JSON: N/A
    """
    def _embed(self, texts: List[str], model: Optional[str], normalize: bool, prefix: Optional[str]) -> EmbeddingResponse:
        if self.backend == "vllm":
            vectors = self.embedding_client(
                texts,
                model_name=model,
                normalize=normalize,
                prefix=prefix,
            )
            resolved_model = self.embedding_client._resolve_model(model)
            return EmbeddingResponse(model=resolved_model, vectors=vectors)

        prefixed = self._apply_prefix(texts, prefix)
        vectors = self.llama_client.embed(prefixed, normalize=normalize)
        return EmbeddingResponse(model="local", vectors=vectors)

    def _vllm_server_ready(self, url: str) -> bool:
        try:
            with urllib.request.urlopen(f"{url}/models", timeout=2) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    async def _wait_for_vllm_ready(self, timeout_s: int = 300, interval_s: float = 1.0) -> None:
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            llm_ok = await asyncio.to_thread(self._vllm_server_ready, LLM_BASE_URL)
            embed_ok = await asyncio.to_thread(self._vllm_server_ready, EMBED_BASE_URL)
            if llm_ok and embed_ok:
                self.models_ready = True
                self.models_ready_event.set()
                return
            await asyncio.sleep(interval_s)

    async def _wait_for_llama_ready(self, timeout_s: int = 300, interval_s: float = 1.0) -> None:
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            llm_ok = await asyncio.to_thread(self._vllm_server_ready, LLM_BASE_URL)
            if llm_ok:
                self.models_ready = True
                self.models_ready_event.set()
                return
            await asyncio.sleep(interval_s)

    """
    Usage: POST a message to create or continue a chat session.
    API URL: /chat
    Usage JSON: {"message":"Hello","session_id":null,"system":null,"temperature":0.2,"max_tokens":512}
    """
    def chat(self, req: ChatRequest):
        sess_id, messages = self._get_or_create_session(req.session_id, req.system)
        messages.append({"role": "user", "content": req.message})
        reply = self._chat(messages, temperature=req.temperature, max_tokens=req.max_tokens)
        messages.append({"role": "assistant", "content": reply})
        return ChatResponse(session_id=sess_id, reply=reply)

    """
    Usage: POST a message and receive a streamed response (SSE).
    API URL: /chat/stream
    Usage JSON: {"message":"Hello","session_id":null,"system":null,"temperature":0.2,"max_tokens":512}
    """
    def chat_stream(self, req: ChatRequest):
        sess_id, messages = self._get_or_create_session(req.session_id, req.system)
        messages.append({"role": "user", "content": req.message})

        def event_stream():
            collected = []
            for chunk in self._stream_chat(messages, temperature=req.temperature, max_tokens=req.max_tokens):
                collected.append(chunk)
                yield f"data: {chunk}\n\n"
            if collected:
                messages.append({"role": "assistant", "content": "".join(collected)})

        headers = {"X-Session-Id": sess_id}
        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

    """
    Usage: POST text list to generate embeddings.
    API URL: /embeddings
    Usage JSON: {"texts":["hello","world"],"model":null,"normalize":false,"prefix":"query"}
    """
    def embeddings(self, req: EmbeddingRequest):
        return self._embed(req.texts, req.model, req.normalize, req.prefix)

    """
    Usage: MCP JSON-RPC bridge for the chat tool.
    API URL: /mcp/call
    Usage JSON: {"jsonrpc":"2.0","id":"1","method":"call_tool","params":{"name":"chat","arguments":{"message":"Hi"}}}
    """
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

    """
    Usage: Health check for the service.
    API URL: /health
    Usage JSON: {}
    """
    def health(self):
        if self.backend == "vllm":
            return {
                "ok": True,
                "backend": self.backend,
                "models_loaded": self.models_ready,
                "models": VLLM_MODEL_NAMES if self.models_ready else [],
            }
        return {"ok": True, "backend": self.backend}

    """
    Usage: Server-sent events that emit when models are loaded.
    API URL: /events
    Usage JSON: {}
    """
    async def events(self):
        async def event_stream():
            while True:
                if self.backend == "vllm":
                    payload = {
                        "event": "model_status",
                        "backend": self.backend,
                        "models_loaded": self.models_ready_event.is_set(),
                        "models": VLLM_MODEL_NAMES if self.models_ready_event.is_set() else [],
                    }
                elif self.backend == "llama":
                    payload = {
                        "event": "model_status",
                        "backend": self.backend,
                        "models_loaded": self.models_ready_event.is_set(),
                        "model_path": LLAMA_CPP_MODEL_PATH if self.models_ready_event.is_set() else "",
                    }
                else:
                    payload = {
                        "event": "model_status",
                        "backend": self.backend,
                        "models_loaded": False,
                    }
                yield f"data: {payload}\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(event_stream(), media_type="text/event-stream")


api = ModelInferenceAPI()
app = api.app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
