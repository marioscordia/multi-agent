"""Reliability machinery: budgets, loop protection, retries, tool limits."""

import pytest

from safeschool.messages import AgentMessage, CaseRequest, MessageType, Payload
from safeschool.runtime import (
    Agent,
    Budget,
    BudgetExceeded,
    Context,
    LoopDetected,
    LoopGuard,
    SafeSchoolError,
    StageFailed,
)
from safeschool.state import CaseFile


class Echo(Payload):
    value: int = 0


def make_ctx(settings, logger, *, max_steps=10, deadline=60.0):
    return Context(
        settings=settings, logger=logger,
        budget=Budget(max_steps=max_steps, deadline_seconds=deadline),
        loop_guard=LoopGuard(max_repeats=2), case=CaseFile("case_test"),
    )


def envelope(payload):
    return AgentMessage.wrap(
        case_id="case_test", sender="test", recipient="x",
        type=MessageType.CASE_REQUEST, payload=payload,
    )


def test_step_budget_stops_runaway():
    budget = Budget(max_steps=2, deadline_seconds=60)
    budget.spend("a")
    budget.spend("a")
    with pytest.raises(BudgetExceeded, match="step budget"):
        budget.spend("a")


def test_time_budget_stops_runaway():
    budget = Budget(max_steps=100, deadline_seconds=-1.0)
    with pytest.raises(BudgetExceeded, match="time budget"):
        budget.spend("a")


def test_loop_guard_trips_on_repeated_input():
    guard = LoopGuard(max_repeats=2)
    payload = CaseRequest(video_path="a.mp4")
    guard.check("A", payload.model_dump(mode="json"))
    guard.check("A", payload.model_dump(mode="json"))
    with pytest.raises(LoopDetected, match="loop detected"):
        guard.check("A", payload.model_dump(mode="json"))


def test_loop_guard_allows_different_inputs():
    guard = LoopGuard(max_repeats=2)
    for i in range(6):
        guard.check("A", {"value": i})


def test_retry_then_succeed(settings, logger):
    calls = {"n": 0}

    class Flaky(Agent):
        name, stage = "Flaky", "ingest"

        def handle(self, message, ctx):
            calls["n"] += 1
            if calls["n"] < 3:
                raise SafeSchoolError("transient")
            return Echo(value=calls["n"])

    settings = settings.model_copy(update={"retry_backoff_seconds": 0.0})
    ctx = make_ctx(settings, logger)
    result = Flaky().execute(envelope(CaseRequest(video_path="a.mp4")), ctx)
    assert result.value == 3
    assert calls["n"] == 3


def test_retry_exhausted_raises_stage_failed(settings, logger):
    class Broken(Agent):
        name, stage = "Broken", "ingest"

        def handle(self, message, ctx):
            raise SafeSchoolError("always down")

    settings = settings.model_copy(update={"retry_backoff_seconds": 0.0, "max_attempts": 2})
    ctx = make_ctx(settings, logger)
    with pytest.raises(StageFailed) as exc:
        Broken().execute(envelope(CaseRequest(video_path="a.mp4")), ctx)
    assert "after 2 attempt" in str(exc.value)
    assert ctx.case.failed_stages == ["ingest"]


def test_fatal_error_is_not_retried(settings, logger):
    calls = {"n": 0}

    class Fatal(Agent):
        name, stage = "Fatal", "ingest"

        def handle(self, message, ctx):
            calls["n"] += 1
            raise LoopDetected("stop now")

    ctx = make_ctx(settings, logger)
    with pytest.raises(LoopDetected):
        Fatal().execute(envelope(CaseRequest(video_path="a.mp4")), ctx)
    assert calls["n"] == 1, "a non-recoverable error must not be retried"


def test_tool_limit_is_enforced_at_import_time():
    with pytest.raises(TypeError, match="the limit is 5"):
        class TooManyTools(Agent):
            name, stage = "Greedy", "x"
            tools = ("a", "b", "c", "d", "e", "f")

            def handle(self, message, ctx):
                return Echo()


def test_every_tool_call_is_logged(settings, logger):
    ctx = make_ctx(settings, logger)
    ctx.tool("A", "demo.tool", lambda x: x * 2, 21)
    records = [r for r in logger.records if r["event"] == "tool"]
    assert len(records) == 1
    assert records[0]["name"] == "demo.tool" and records[0]["ok"] is True


def test_failed_tool_call_is_logged_then_reraised(settings, logger):
    ctx = make_ctx(settings, logger)
    with pytest.raises(ValueError):
        ctx.tool("A", "bad.tool", lambda: (_ for _ in ()).throw(ValueError("boom")))
    record = [r for r in logger.records if r["event"] == "tool"][0]
    assert record["ok"] is False and "boom" in record["detail"]["error"]


def test_secrets_never_reach_the_log(settings, logger):
    ctx = make_ctx(settings, logger)
    ctx.logger.log(
        case_id="c", agent="A", event="system", name="x",
        detail={"anthropic_api_key": "sk-ant-secret", "nested": {"token": "abc"}},
    )
    blob = logger.log_path.read_text()
    assert "sk-ant-secret" not in blob and "abc" not in blob and "***" in blob
