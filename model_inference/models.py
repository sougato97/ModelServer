from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, Iterator, List, Sequence

import requests


@dataclass
class LlamaServerConfig:
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    embedding_base_url: str = "http://127.0.0.1:7000/v1"
    llm_model: str = "local"
    embedding_model: str = "local"
    timeout_s: float = 60.0


class LlamaServerClient:
    """
    Thin client for llama.cpp OpenAI-compatible server endpoints.
    """

    def __init__(self, cfg: LlamaServerConfig | None = None):
        self.cfg = cfg or LlamaServerConfig()

    def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 256) -> str:
        url = f"{self.cfg.llm_base_url}/chat/completions"
        payload = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, json=payload, timeout=self.cfg.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ) -> Iterator[str]:
        url = f"{self.cfg.llm_base_url}/chat/completions"
        payload = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        with requests.post(url, json=payload, stream=True, timeout=self.cfg.timeout_s) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[len("data: "):]
                if line.strip() == "[DONE]":
                    break
                try:
                    chunk = line if isinstance(line, str) else line.decode("utf-8")
                    data = json.loads(chunk)
                except Exception:
                    continue
                delta = data["choices"][0].get("delta") or {}
                text = delta.get("content")
                if text:
                    yield text

    def embed(self, texts: Sequence[str], *, normalize: bool = False) -> List[List[float]]:
        url = f"{self.cfg.embedding_base_url}/embeddings"
        payload = {
            "model": self.cfg.embedding_model,
            "input": list(texts),
        }
        resp = requests.post(url, json=payload, timeout=self.cfg.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        vectors = [item["embedding"] for item in data["data"]]

        if normalize:
            vectors = [self._l2_normalize(vec) for vec in vectors]

        return vectors

    @staticmethod
    def _l2_normalize(vec: List[float]) -> List[float]:
        denom = sum(x * x for x in vec) ** 0.5
        if denom == 0:
            return vec
        return [x / denom for x in vec]
