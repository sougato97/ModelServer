# llm_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union
import json
import time

from openai import OpenAI


Message = Dict[str, str]  # {"role": "...", "content": "..."}


@dataclass
class LLMConfig:
    base_url: str                 # e.g. "http://127.0.0.1:8000/v1" (vLLM) or "https://api.openai.com/v1"
    api_key: str                  # "local" for vLLM, real key for OpenAI/provider
    model: str                    # e.g. "Qwen/Qwen2.5-7B-Instruct" or backend-specific name
    timeout_s: float = 60.0
    max_retries: int = 2
    retry_backoff_s: float = 0.8  # base backoff; will be multiplied


class LLMClient:
    """
    A small wrapper around an OpenAI-compatible Chat Completions API.

    Goals:
    - portable across vLLM/TGI/managed endpoints
    - single place for retry/timeouts
    - optional streaming
    - optional tool-calling normalization (simple baseline)
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout_s,
        )

    def new_session(
        self,
        *,
        system: Optional[str] = None,
        initial_messages: Optional[List[Message]] = None,
    ) -> "ChatSession":
        """
        Create a chat session that tracks history for follow-up questions.
        """
        messages: List[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        if initial_messages:
            messages.extend(initial_messages)
        return ChatSession(self, messages)

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Non-streaming. Returns the assistant text.
        """
        resp = self._with_retries(
            lambda: self.client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            )
        )

        msg = resp.choices[0].message
        # Some backends return tool calls instead of content.
        if getattr(msg, "tool_calls", None):
            return self._normalize_tool_calls(msg.tool_calls)
        return msg.content or ""

    def stream_chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Iterator[str]:
        """
        Streaming. Yields text chunks.
        """
        stream = self._with_retries(
            lambda: self.client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
                stream=True,
            )
        )

        # Collect tool-call deltas if present
        tool_call_buf: List[Dict[str, Any]] = []

        for chunk in stream:
            delta = chunk.choices[0].delta

            if getattr(delta, "content", None):
                yield delta.content

            # Tool call streaming varies a bit; this is a conservative collector.
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    tool_call_buf.append(tc)

        # If only tool calls were produced, emit a normalized JSON line at end.
        if tool_call_buf:
            yield "\n" + self._normalize_tool_calls(tool_call_buf)

    def _with_retries(self, fn):
        last_err = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_err = e
                if attempt >= self.cfg.max_retries:
                    raise
                # simple exponential backoff
                time.sleep(self.cfg.retry_backoff_s * (2 ** attempt))
        raise last_err  # should never reach

    @staticmethod
    def _normalize_tool_calls(tool_calls: Iterable[Any]) -> str:
        """
        Normalize tool calls into a stable JSON string you can parse uniformly.
        Different backends format tool calls slightly differently; we keep a common subset.
        """
        normalized = []
        for tc in tool_calls:
            # tc may be a dict-like or an object; handle both.
            name = None
            args = None

            # vLLM/OpenAI style: tc.function.name, tc.function.arguments
            func = getattr(tc, "function", None)
            if func is not None:
                name = getattr(func, "name", None)
                args = getattr(func, "arguments", None)

            # dict-like fallback
            if name is None and isinstance(tc, dict):
                func = tc.get("function") or {}
                name = func.get("name")
                args = func.get("arguments")

            # arguments may be a JSON string; try to parse for stability
            parsed_args = None
            if isinstance(args, str):
                try:
                    parsed_args = json.loads(args)
                except Exception:
                    parsed_args = {"_raw": args}
            else:
                parsed_args = args

            normalized.append({"name": name, "arguments": parsed_args})

        return json.dumps({"tool_calls": normalized}, ensure_ascii=False)


class ChatSession:
    """
    Keeps a running message history so follow-up questions include prior context.
    """

    def __init__(self, llm: LLMClient, messages: Optional[List[Message]] = None):
        self.llm = llm
        self.messages: List[Message] = messages or []

    def ask(
        self,
        user_content: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Append a user turn, get the model reply, and append it to history.
        """
        self.messages.append({"role": "user", "content": user_content})
        reply = self.llm.chat(
            self.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )
        self.messages.append({"role": "assistant", "content": reply})
        return reply
