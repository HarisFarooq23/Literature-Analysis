"""
ollama_client.py — single point of contact with the Ollama server.

Handles:
  - structured-output requests (passes the Pydantic JSON schema as `format`),
    with a safe fallback to plain `format="json"` for older Ollama servers
  - markdown-fence stripping (some models wrap JSON in ```json fences despite
    being told not to)
  - retries with backoff on transport errors or invalid JSON
  - validation against the Pydantic schema before anything downstream ever
    sees the result — a paper either yields a *valid* ExtractionFields
    instance or an explicit, logged failure. There is no silent partial state.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional, Type, TypeVar

import ollama
from loguru import logger
from pydantic import BaseModel, ValidationError

from .config import ModelsConfig

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class CallResult:
    model: str
    ok: bool
    parsed: Optional[BaseModel]
    raw_response: str
    error: Optional[str]
    attempts: int
    elapsed_seconds: float


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _try_parse(raw_text: str, response_model: Type[T]) -> T:
    cleaned = _strip_fences(raw_text)
    data = json.loads(cleaned)
    return response_model.model_validate(data)


def check_ollama_ready(models_config: ModelsConfig, required_model_names: list[str]) -> None:
    """Fail fast and clearly, before any paper is processed, if either:
      (a) the Ollama server at models_config.host isn't reachable, or
      (b) any of required_model_names hasn't actually been pulled.

    Deliberately NOT called from config.load_config() -- that function is
    pure config validation with no network dependency, and some CLI commands
    (e.g. 'build-dataset', which only re-runs normalization on already-
    extracted records) don't need a live Ollama server at all. Call this
    explicitly from commands that do.
    """
    client = ollama.Client(host=models_config.host)
    try:
        response = client.list()
    except ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {models_config.host}. Is the server running? "
            f"(start it with `ollama serve`). Underlying error: {exc}"
        ) from exc

    available = {m.model for m in response.models if m.model}
    missing = [m for m in required_model_names if m and m not in available]
    if missing:
        raise RuntimeError(
            f"Configured model(s) not found on the Ollama server at "
            f"{models_config.host}: {missing}. Available models: "
            f"{sorted(available) or '(none pulled)'}. "
            f"Pull a missing model with `ollama pull <name>`."
        )


class OllamaExtractor:
    """Thin, defensive wrapper around the Ollama server for one schema.

    Takes the whole ModelsConfig (not a flat options dict) so each call can
    resolve per-model option overrides — some models in a tiered setup may
    need a different num_ctx, temperature, etc. than the rest.
    """

    def __init__(
        self,
        host: str,
        models_config: ModelsConfig,
        max_retries: int = 3,
        retry_wait_seconds: int = 4,
    ) -> None:
        self.client = ollama.Client(host=host)
        self.models_config = models_config
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds

    def extract(
        self,
        model: str,
        prompt: str,
        response_model: Type[T],
    ) -> CallResult:
        start = time.monotonic()
        last_error = ""
        last_raw = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_text = self._call(model, prompt, response_model)
                last_raw = raw_text
                parsed = _try_parse(raw_text, response_model)
                return CallResult(
                    model=model,
                    ok=True,
                    parsed=parsed,
                    raw_response=raw_text,
                    error=None,
                    attempts=attempt,
                    elapsed_seconds=time.monotonic() - start,
                )
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    f"[{model}] attempt {attempt}/{self.max_retries} "
                    f"produced invalid output: {last_error}"
                )
            except Exception as exc:  # transport/server errors
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    f"[{model}] attempt {attempt}/{self.max_retries} "
                    f"failed: {last_error}"
                )

            if attempt < self.max_retries:
                time.sleep(self.retry_wait_seconds)

        return CallResult(
            model=model,
            ok=False,
            parsed=None,
            raw_response=last_raw,
            error=last_error,
            attempts=self.max_retries,
            elapsed_seconds=time.monotonic() - start,
        )

    def _call(self, model: str, prompt: str, response_model: Type[BaseModel]) -> str:
        """Try structured-output mode first (format=<json schema>); fall back
        to format='json' when the server either (a) actively rejects the
        schema format, or (b) returns a 200 OK with an empty/whitespace-only
        response body.

        The ollama python client (verified against v0.6.2's source) wraps a
        refused connection as a plain ConnectionError and an HTTP error
        response as ollama.ResponseError(status_code=...). A ConnectionError
        means the server is unreachable, and retrying the same request with a
        different `format` won't change that, so it's propagated immediately;
        the outer retry loop in extract() already handles backoff once per
        attempt.

        Case (b) is a distinct, observed failure mode for some models (e.g.
        nemotron-3-nano:30b in current Ollama builds): the server accepts the
        schema-constrained request and returns 200 OK, but generation produces
        no tokens at all -- no exception is raised, so the original
        except-ollama.ResponseError fallback never triggers, and the caller
        gets an empty string that fails JSON parsing on every retry
        regardless of how many attempts are configured. Detecting this and
        falling back to the unconstrained format='json' (which most models
        handle even when schema-constrained grammar generation fails them)
        recovers cleanly instead of burning every retry on a deterministic
        empty response.
        """
        options = self.models_config.resolved_options(model)
        try:
            response = self.client.generate(
                model=model,
                prompt=prompt,
                format=response_model.model_json_schema(),
                options=options,
            )
            text = response["response"]
            if text and text.strip():
                return text
            logger.debug(
                f"[{model}] schema-constrained generation returned an empty "
                "response body (no server error raised); falling back to "
                "format='json'"
            )
        except ConnectionError:
            raise
        except ollama.ResponseError as exc:
            logger.debug(
                f"[{model}] server rejected schema-constrained generation "
                f"(status {getattr(exc, 'status_code', '?')}: {exc}); "
                "falling back to format='json'"
            )

        response = self.client.generate(
            model=model,
            prompt=prompt,
            format="json",
            options=options,
        )
        return response["response"]
