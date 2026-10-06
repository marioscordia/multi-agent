"""LLM access.

Only two agents reason in natural language (ContextAgent and VerifierAgent), and
both of them need strict JSON back. This module gives them one entry point that

* reads the key from the environment (never from source);
* asks for JSON and repairs the common "prose around the JSON" failure;
* retries transient API errors with backoff;
* logs tokens and latency for the load-balance report;
* falls back to a deterministic mock so the whole system is demonstrable with no
  key and no network.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .config import Settings
from .runtime import Context, SafeSchoolError

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LLMError(SafeSchoolError):
    recoverable = True


def _extract_json(text: str) -> dict[str, Any]:
    """Models sometimes wrap JSON in prose or a code fence. Recover it."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise LLMError(
                "model did not return JSON; first 200 chars: " + text[:200]
            ) from None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError(f"model returned malformed JSON: {exc}") from exc


class LLMClient:
    """Thin wrapper over the Anthropic Messages API with a mock fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        self.provider = "anthropic" if settings.llm_available else "mock"
        if self.provider == "anthropic":
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            except Exception:
                # never let a client-construction problem take the run down
                self.provider = "mock"

    @property
    def model_name(self) -> str:
        return self.settings.model if self.provider == "anthropic" else "mock-llm"

    def complete_json(
        self,
        *,
        agent: str,
        ctx: Context,
        system: str,
        prompt: str,
        schema_hint: str,
        mock: dict[str, Any],
        purpose: str = "reason",
    ) -> dict[str, Any]:
        """Ask for one JSON object. ``mock`` is returned verbatim in mock mode, so
        every LLM-dependent code path stays exercised offline."""
        if self.provider == "mock":
            ctx.logger.log(
                case_id=ctx.case.case_id,
                agent=agent,
                event="llm",
                name=f"{purpose}@mock-llm",
                duration_ms=0.0,
                detail={"mode": "mock", "reason": "no ANTHROPIC_API_KEY configured"},
            )
            return dict(mock)

        full_prompt = (
            f"{prompt}\n\n"
            f"Return a single JSON object and nothing else, matching:\n{schema_hint}"
        )
        attempts = self.settings.max_attempts
        for attempt in range(1, attempts + 1):
            start = time.perf_counter()
            try:
                response = self._client.messages.create(  # type: ignore[union-attr]
                    model=self.settings.model,
                    max_tokens=self.settings.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                parsed = _extract_json(text)
            except Exception as exc:  # noqa: BLE001 - retry layer
                ctx.logger.log(
                    case_id=ctx.case.case_id,
                    agent=agent,
                    event="llm",
                    name=f"{purpose}@{self.settings.model}",
                    ok=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    attempt=attempt,
                    detail={"error": f"{type(exc).__name__}: {exc}"},
                )
                if attempt == attempts:
                    raise LLMError(
                        f"LLM call failed after {attempts} attempts: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                time.sleep(self.settings.retry_backoff_seconds * (2 ** (attempt - 1)))
                continue

            usage = getattr(response, "usage", None)
            ctx.logger.log(
                case_id=ctx.case.case_id,
                agent=agent,
                event="llm",
                name=f"{purpose}@{self.settings.model}",
                duration_ms=(time.perf_counter() - start) * 1000,
                attempt=attempt,
                tokens_in=getattr(usage, "input_tokens", 0) or 0,
                tokens_out=getattr(usage, "output_tokens", 0) or 0,
                detail={"keys": sorted(parsed)[:10]},
            )
            return parsed

        raise LLMError("unreachable")
