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
        def _call(prompt: str, num_predict: int, temperature: float, think: bool) -> GenerationResult:
            kwargs = dict(
                model=self.model,
                prompt=prompt,
                options={
                    "num_predict": num_predict,
                    "temperature": temperature,
                },
                stream=False,
            )
            try:
                resp = self.client.generate(think=think, **kwargs)
            except TypeError:
                resp = self.client.generate(**kwargs)
            text = (resp.get("response") or "").strip()
            if not text:
                raise UnparseableResponse("Empty response from Ollama")
            return GenerationResult(text=text, tokens=resp.get("eval_count"), raw=dict(resp))

        return _call

    def generate(
        self,
        prompt: str,
        num_predict: int = 16,
        temperature: float = 0.0,
        think: bool = False,
    ) -> GenerationResult:
        return self._generate(prompt, num_predict, temperature, think)

    def health_check(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False
