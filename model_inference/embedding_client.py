from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import math

from openai import OpenAI


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding generation.
    """

    base_url: str = "http://127.0.0.1:7000/v1"
    api_key: str = "local"
    # vLLM expects the served model name; using the HF repo name is typical.
    model: str = "intfloat/multilingual-e5-large-instruct"
    timeout_s: float = 30.0


class EmbeddingClient:
    """
    Simple client for embedding generation (defaults to Multilingual-E5-large-instruct).
    """

    def __init__(self, cfg: EmbeddingConfig | None = None):
        self.cfg = cfg or EmbeddingConfig()
        self.allowed_models = {
            "intfloat/multilingual-e5-large-instruct",
            "Qwen/Qwen3-Embedding-4B",
            "Qwen3-Embedding-4B",
        }
        self.client = OpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            timeout=self.cfg.timeout_s,
        )

    def __call__(
        self,
        texts: Sequence[str],
        *,
        model_name: str | None = None,
        normalize: bool = True,
        prefix: str | None = "query",
    ) -> List[List[float]]:
        return self.embed(texts, model_name=model_name, normalize=normalize, prefix=prefix)

    def embed(
        self,
        texts: Sequence[str],
        *,
        model_name: str | None = None,
        normalize: bool = True,
        prefix: str | None = "query",
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        """
        if isinstance(texts, str):
            texts = [texts]

        if prefix:
            texts = [f"{prefix}: {t}" for t in texts]

        model = self._resolve_model(model_name)
        resp = self.client.embeddings.create(
            model=model,
            input=list(texts),
        )

        vectors = [item.embedding for item in resp.data]

        if normalize:
            vectors = [self._l2_normalize(vec) for vec in vectors]

        return vectors

    def _resolve_model(self, model_name: str | None) -> str:
        if model_name in self.allowed_models:
            if model_name == "Qwen3-Embedding-4B":
                return "Qwen/Qwen3-Embedding-4B"
            return model_name
        return self.cfg.model

    @staticmethod
    def _l2_normalize(vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]
