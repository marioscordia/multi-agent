"""Agent runtime: the machinery every agent inherits.

Reliability requirements from the assignment specification are implemented here
once, rather than being re-implemented (and forgotten) per agent:

* retries with exponential backoff and a readable final error;
* a global step budget and wall-clock deadline;
* loop protection — the same agent seeing the same input twice aborts the run;
* logging of every agent, tool and LLM call.

An agent subclass only implements :meth:`Agent.handle`, declares its ``tools``,
and states its ``done_criterion``.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from pydantic import ValidationError

from .config import Settings
from .logging_bus import CallLogger
from .messages import AgentMessage, MessageType, Payload
from .state import CaseFile, StageStatus

T = TypeVar("T", bound=Payload)


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #

class SafeSchoolError(Exception):
    """Base error. Carries a message meant to be shown to an operator."""

    recoverable = True


class BudgetExceeded(SafeSchoolError):
    recoverable = False


class LoopDetected(SafeSchoolError):
    recoverable = False


class StageFailed(SafeSchoolError):
    def __init__(self, stage: str, agent: str, cause: BaseException, attempts: int) -> None:
        super().__init__(
            f"stage '{stage}' ({agent}) failed after {attempts} attempt(s): "
            f"{type(cause).__name__}: {cause}"
        )
        self.stage = stage
        self.agent = agent
        self.cause = cause
        self.attempts = attempts


# --------------------------------------------------------------------------- #
# budget + loop guard
# --------------------------------------------------------------------------- #

@dataclass
class Budget:
    """Step and time limits for one case."""

    max_steps: int
    deadline_seconds: float
    started_at: float = field(default_factory=time.monotonic)
    steps_used: int = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_seconds - self.elapsed)

    def spend(self, agent: str) -> None:
        self.steps_used += 1
        if self.steps_used > self.max_steps:
            raise BudgetExceeded(
                f"step budget exhausted ({self.steps_used}/{self.max_steps}) "
                f"while running {agent}; aborting to avoid an unbounded run"
            )
        if self.elapsed > self.deadline_seconds:
            raise BudgetExceeded(
                f"time budget exhausted ({self.elapsed:.1f}s > {self.deadline_seconds}s) "
                f"while running {agent}; aborting"
            )


class LoopGuard:
    """Aborts when an (agent, input) pair repeats more often than allowed.

    Retries of a *failed* attempt are counted separately by the retry logic; this
    guard catches the different pathology of two agents handing each other the
    same payload forever.
    """

    def __init__(self, max_repeats: int = 2) -> None:
        self.max_repeats = max_repeats
        self._seen: dict[tuple[str, str], int] = {}

    @staticmethod
    def fingerprint(payload: Payload | dict[str, Any]) -> str:
        raw = payload.model_dump(mode="json") if isinstance(payload, Payload) else payload
        blob = json.dumps(raw, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def check(self, agent: str, payload: Payload | dict[str, Any]) -> None:
        key = (agent, self.fingerprint(payload))
        self._seen[key] = self._seen.get(key, 0) + 1
        if self._seen[key] > self.max_repeats:
            raise LoopDetected(
                f"loop detected: {agent} received an identical input "
                f"{self._seen[key]} times (limit {self.max_repeats}); aborting"
            )


# --------------------------------------------------------------------------- #
# execution context
# --------------------------------------------------------------------------- #

@dataclass
class Context:
    """Everything an agent is allowed to touch outside its own inputs."""

    settings: Settings
    logger: CallLogger
    budget: Budget
    loop_guard: LoopGuard
    case: CaseFile

    def tool(self, agent: str, name: str, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Run a tool call with timing + logging. All tool use goes through here,
        which is what makes the per-agent load numbers in the logs trustworthy."""
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self.logger.log(
                case_id=self.case.case_id,
                agent=agent,
                event="tool",
                name=name,
                ok=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                detail={"error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        self.logger.log(
            case_id=self.case.case_id,
            agent=agent,
            event="tool",
            name=name,
            duration_ms=(time.perf_counter() - start) * 1000,
            detail=_summarize(result),
        )
        return result


def _summarize(value: Any) -> Any:
    """Compact description of a tool result for the log."""
    if isinstance(value, Payload):
        return {"type": type(value).__name__}
    if isinstance(value, (list, tuple)):
        return {"type": "list", "len": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(value)[:8]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__}


# --------------------------------------------------------------------------- #
# base agent
# --------------------------------------------------------------------------- #

class Agent(ABC):
    """One role. Subclasses declare their identity and implement ``handle``."""

    name: str = "agent"
    stage: str = "stage"
    role: str = ""
    tools: tuple[str, ...] = ()
    done_criterion: str = ""

    #: input/output message types, used for the generated specification tables
    input_type: MessageType | None = None
    output_type: MessageType | None = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # the specification caps an agent at five tools; enforce it at import time
        if len(cls.tools) > 5:
            raise TypeError(
                f"{cls.__name__} declares {len(cls.tools)} tools; the limit is 5"
            )

    @abstractmethod
    def handle(self, message: AgentMessage, ctx: Context) -> Payload:
        """Do the agent's work and return its output payload."""

    # -- orchestration entry point -------------------------------------- #

    def execute(self, message: AgentMessage, ctx: Context) -> Payload:
        """Validate, run with retries, log, and record the result in the case file."""
        ctx.budget.spend(self.name)
        ctx.loop_guard.check(self.name, message.payload)
        ctx.case.mark(self.stage, StageStatus.RUNNING)
        ctx.case.record_message(message)

        attempts = ctx.settings.max_attempts
        last_exc: BaseException | None = None

        for attempt in range(1, attempts + 1):
            start = time.perf_counter()
            try:
                result = self.handle(message, ctx)
            except Exception as exc:  # noqa: BLE001 - deliberately broad: retry layer
                last_exc = exc
                duration = (time.perf_counter() - start) * 1000
                fatal = isinstance(exc, (BudgetExceeded, LoopDetected)) or (
                    isinstance(exc, SafeSchoolError) and not exc.recoverable
                )
                ctx.logger.log(
                    case_id=ctx.case.case_id,
                    agent=self.name,
                    event="error",
                    name=f"{self.stage}.attempt",
                    ok=False,
                    duration_ms=duration,
                    attempt=attempt,
                    detail={"error": f"{type(exc).__name__}: {exc}", "fatal": fatal},
                )
                if fatal:
                    ctx.case.mark(self.stage, StageStatus.FAILED)
                    raise
                if attempt < attempts:
                    backoff = ctx.settings.retry_backoff_seconds * (2 ** (attempt - 1))
                    time.sleep(min(backoff, max(0.0, ctx.budget.remaining_seconds)))
                    continue
                ctx.case.mark(self.stage, StageStatus.FAILED)
                raise StageFailed(self.stage, self.name, exc, attempt) from exc
            else:
                duration = (time.perf_counter() - start) * 1000
                ctx.logger.log(
                    case_id=ctx.case.case_id,
                    agent=self.name,
                    event="agent",
                    name=self.stage,
                    duration_ms=duration,
                    attempt=attempt,
                    detail={"output": type(result).__name__},
                )
                ctx.case.put(self.stage, result)
                return result

        # unreachable, but keeps the type checker and the reader honest
        raise StageFailed(self.stage, self.name, last_exc or RuntimeError("unknown"), attempts)

    # -- helpers for subclasses ----------------------------------------- #

    def parse_input(self, message: AgentMessage, model: type[T]) -> T:
        try:
            return model.model_validate(message.payload)
        except ValidationError as exc:
            raise SafeSchoolError(
                f"{self.name} received a payload that is not a valid "
                f"{model.__name__}: {exc.error_count()} validation error(s)"
            ) from exc

    def spec(self) -> dict[str, Any]:
        """Machine-readable role specification (feeds `safeschool agents`)."""
        return {
            "name": self.name,
            "stage": self.stage,
            "role": self.role,
            "inputs": self.input_type.value if self.input_type else None,
            "outputs": self.output_type.value if self.output_type else None,
            "tools": list(self.tools),
            "done_criterion": self.done_criterion,
        }
