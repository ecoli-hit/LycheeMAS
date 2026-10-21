"""Synchronous llama-server ``/completion`` backend with per-call seeds.

This uses only the Python standard library. The caller renders the prompt, so
chat templates and their exact bytes remain under the experiment's control.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LlamaCppCompletionResult:
    text: str
    n_prompt_pos: int
    n_gen_tokens: int
    latency_s: float
    stop_type: str | None
    server_seed: int | None
    stopped: bool | None
    stopping_word: str | None
    raw_response: dict[str, Any]


class LlamaCppBackend:
    """Call one already-running llama-server without changing its lifecycle."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 900.0):
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP URL")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(
        self,
        prompt: str,
        *,
        seed: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: list[str] | None = None,
        ignore_eos: bool = False,
        top_k: int = 0,
        min_p: float = 0.0,
        typical_p: float = 1.0,
        repeat_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        mirostat: int = 0,
        cache_prompt: bool = False,
    ) -> LlamaCppCompletionResult:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(seed, int) or not 1 <= seed <= 2**31 - 1:
            raise ValueError("seed must be a positive 31-bit integer")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if stop is not None and not all(isinstance(item, str) for item in stop):
            raise TypeError("stop must contain strings")
        payload = {
            "prompt": prompt,
            "seed": seed,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "typical_p": typical_p,
            "repeat_penalty": repeat_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "mirostat": mirostat,
            "cache_prompt": cache_prompt,
            "stop": list(stop or []),
            "ignore_eos": ignore_eos,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            reply = json.loads(response.read())
        latency_s = time.perf_counter() - start
        if not isinstance(reply, dict) or not isinstance(reply.get("content"), str):
            raise ValueError("llama-server returned no text content")
        for name in ("tokens_evaluated", "tokens_predicted"):
            if not isinstance(reply.get(name), int):
                raise ValueError(f"llama-server returned no integer {name}")
        settings = reply.get("generation_settings") or {}
        if not isinstance(settings, dict):
            raise ValueError("llama-server returned invalid generation_settings")
        return LlamaCppCompletionResult(
            text=reply["content"],
            n_prompt_pos=reply["tokens_evaluated"],
            n_gen_tokens=reply["tokens_predicted"],
            latency_s=latency_s,
            stop_type=reply.get("stop_type"),
            server_seed=settings.get("seed"),
            stopped=reply.get("stop"),
            stopping_word=reply.get("stopping_word"),
            raw_response=reply,
        )
