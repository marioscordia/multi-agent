"""LLM access.

Three agents use the language model, each for a different job: ContextAgent
classifies the incident, VerifierAgent audits that classification, and
PrivacyReportAgent writes the plain-language message for staff. All three need
strict JSON back. This module gives them one entry point that

* reads the key from the environment (never from source);
* constrains the answer to a JSON schema (structured outputs), so a reply that
  does not parse cannot happen; the "prose around the JSON" repair stays as a
  safety net for calls made without a schema;
* retries transient API errors and cut-off answers with backoff;
* on models that support it, asks the API to re-run a refused request on a
  fallback model, and if the whole chain refuses, falls back to the policy
  heuristic and says so in the log rather than failing the case;
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

#: models that accept the server-side refusal fallback in its "default" form
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


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
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ask for one JSON object. ``mock`` is returned verbatim in mock mode, so
        every LLM-dependent code path stays exercised offline.

        With ``json_schema`` the API is constrained to that schema. If the model
        refuses, ``mock`` (the policy heuristic) is returned with an
        ``_llm_fallback`` key so the caller can record that no model judged it."""
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
            prompt if json_schema else
            f"{prompt}\n\n"
            f"Return a single JSON object and nothing else, matching:\n{schema_hint}"
        )
        request: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": full_prompt}],
        }
        if json_schema:
            request["output_config"] = {
                "format": {"type": "json_schema", "schema": json_schema}
            }
        use_fallback = self.settings.model in _FALLBACK_MODELS
        if use_fallback:
            request["betas"] = [_FALLBACK_BETA]
            request["fallbacks"] = "default"

        attempts = self.settings.max_attempts
        for attempt in range(1, attempts + 1):
            start = time.perf_counter()
            try:
                api = self._client.beta.messages if use_fallback else self._client.messages  # type: ignore[union-attr]
                response = api.create(**request)
                stop_reason = getattr(response, "stop_reason", None)
                if stop_reason == "refusal":
                    return self._refused(agent, ctx, purpose, response, start, attempt, mock)
                if stop_reason == "max_tokens":
                    raise LLMError(
                        f"answer was cut off at max_tokens={self.settings.max_tokens}"
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
                detail={"keys": sorted(parsed)[:10],
                        "served_by": getattr(response, "model", self.settings.model)},
            )
            return parsed

        raise LLMError("unreachable")

    def _refused(self, agent, ctx, purpose, response, start, attempt, mock) -> dict[str, Any]:
        """The model (and any fallback) declined. Retrying the same request would
        decline again, so use the policy heuristic and make that visible."""
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        ctx.logger.log(
            case_id=ctx.case.case_id,
            agent=agent,
            event="llm",
            name=f"{purpose}@{self.settings.model}",
            ok=False,
            duration_ms=(time.perf_counter() - start) * 1000,
            attempt=attempt,
            detail={"refusal": category or "unspecified",
                    "action": "used the policy heuristic instead"},
        )
        ctx.case.note(
            f"{agent}: the model declined ({category or 'no category'}); "
            f"the policy heuristic decided this stage"
        )
        result = dict(mock)
        result["_llm_fallback"] = "refusal"
        return result
