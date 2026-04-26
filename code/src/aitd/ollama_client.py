from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ollama
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class UnparseableResponse(ValueError):
    pass


@dataclass
class GenerationResult:
    text: str
    tokens: int | None = None
    raw: dict[str, Any] | None = None
    logprobs: dict[str, float] | None = None


def _extract_first_token_top_logprobs(resp: Any) -> dict[str, float] | None:
    lps = resp.get("logprobs") if hasattr(resp, "get") else None
    if not lps:
        return None
    first = lps[0]
    candidates = first.get("top_logprobs") if hasattr(first, "get") else getattr(first, "top_logprobs", None)
    if not candidates:
        return None
    out: dict[str, float] = {}
    for c in candidates:
        tok = c.get("token") if hasattr(c, "get") else getattr(c, "token", None)
        lp = c.get("logprob") if hasattr(c, "get") else getattr(c, "logprob", None)
        if tok is None or lp is None:
            continue
        if tok not in out:
            out[tok] = float(lp)
    return out or None


class OllamaClient:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        max_attempts: int = 3,
        wait_min: float = 1.0,
        wait_max: float = 10.0,
    ):
        self.model = model
        self.client = ollama.Client(host=host)
        self.max_attempts = max_attempts
        self._wait_min = wait_min
        self._wait_max = wait_max
        self._generate = self._build_generate()

    def _build_generate(self):
        @retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=1, min=self._wait_min, max=self._wait_max),
            retry=retry_if_exception_type((UnparseableResponse, ConnectionError, TimeoutError)),
            reraise=True,
        )
        def _call(
            prompt: str,
            num_predict: int,
            temperature: float,
            think: bool,
            system: str | None,
            return_logprobs: bool,
            top_logprobs_k: int,
            keep_alive: str | int | None,
            num_ctx: int | None,
            num_batch: int | None,
        ) -> GenerationResult:
            options: dict[str, Any] = {
                "num_predict": num_predict,
                "temperature": temperature,
            }
            if num_ctx is not None:
                options["num_ctx"] = num_ctx
            if num_batch is not None:
                options["num_batch"] = num_batch
            kwargs: dict[str, Any] = dict(
                model=self.model,
                prompt=prompt,
                options=options,
                stream=False,
            )
            if keep_alive is not None:
                kwargs["keep_alive"] = keep_alive
            if system is not None:
                kwargs["system"] = system
            if return_logprobs:
                kwargs["logprobs"] = True
                # Ollama 0.21.1 caps top_logprobs at 20. Clamp silently to keep callers portable.
                kwargs["top_logprobs"] = min(top_logprobs_k, 20)
            try:
                resp = self.client.generate(think=think, **kwargs)
            except TypeError:
                resp = self.client.generate(**kwargs)
            text = (resp.get("response") or "").strip()
            if not text:
                raise UnparseableResponse("Empty response from Ollama")
            raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            logprobs = _extract_first_token_top_logprobs(raw) if return_logprobs else None
            return GenerationResult(
                text=text,
                tokens=raw.get("eval_count"),
                raw=raw,
                logprobs=logprobs,
            )

        return _call

    def generate(
        self,
        prompt: str,
        num_predict: int = 16,
        temperature: float = 0.0,
        think: bool = False,
        system: str | None = None,
        return_logprobs: bool = False,
        top_logprobs_k: int = 10,
        keep_alive: str | int | None = "1h",
        num_ctx: int | None = None,
        num_batch: int | None = None,
    ) -> GenerationResult:
        return self._generate(
            prompt,
            num_predict,
            temperature,
            think,
            system,
            return_logprobs,
            top_logprobs_k,
            keep_alive,
            num_ctx,
            num_batch,
        )

    def health_check(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False
