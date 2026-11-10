"""The Anthropic path, exercised with a stub client.

Without a key the system runs on the mock, so these tests stand in for the live
API: they check the JSON recovery, the retry behaviour, token accounting, and
that a model answer outside the allowed category set cannot reach a report.
"""

import pytest

from safeschool.llm import LLMClient, LLMError, _extract_json
from safeschool.messages import AgentMessage, CaseRequest, MessageType
from safeschool.runtime import Budget, Context, LoopGuard
from safeschool.state import CaseFile


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Usage:
    input_tokens, output_tokens = 1234, 56


class _Response:
    def __init__(self, text, stop_reason="end_turn"):
        self.content, self.usage = [_Block(text)], _Usage()
        self.stop_reason, self.stop_details = stop_reason, None


class StubMessages:
    """Replays scripted responses: a string is returned, an Exception is raised,
    a _Response is returned as-is."""

    def __init__(self, script):
        self.script, self.calls, self.last_kwargs = list(script), 0, {}

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, _Response) else _Response(item)


class StubAnthropic:
    def __init__(self, script):
        self.messages = StubMessages(script)
        # the refusal-fallback path goes through client.beta.messages
        self.beta = type("Beta", (), {"messages": self.messages})()


def make_ctx(settings, logger):
    return Context(
        settings=settings, logger=logger,
        budget=Budget(max_steps=50, deadline_seconds=60),
        loop_guard=LoopGuard(), case=CaseFile("case_llm"),
    )


def live_client(settings, script):
    client = LLMClient(settings.model_copy(update={"retry_backoff_seconds": 0.0}))
    client.provider = "anthropic"
    client._client = StubAnthropic(script)
    return client


# --- JSON recovery --------------------------------------------------------- #

@pytest.mark.parametrize("raw", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Certainly! Here is the result:\n{"a": 1}\nLet me know if you need more.',
])
def test_json_is_recovered_from_common_wrappings(raw):
    assert _extract_json(raw) == {"a": 1}


def test_non_json_answer_is_an_error():
    with pytest.raises(LLMError, match="did not return JSON"):
        _extract_json("I am afraid I cannot help with that.")


# --- client behaviour ------------------------------------------------------ #

def test_successful_call_records_tokens(settings, logger):
    client = live_client(settings, ['{"verdict": "accept"}'])
    ctx = make_ctx(settings, logger)
    result = client.complete_json(
        agent="VerifierAgent", ctx=ctx, system="s", prompt="p",
        schema_hint="{}", mock={"verdict": "reject"}, purpose="verify",
    )
    assert result == {"verdict": "accept"}
    record = [r for r in logger.records if r["event"] == "llm"][0]
    assert record["tokens_in"] == 1234 and record["tokens_out"] == 56
    assert record["ok"] is True


def test_transient_api_failure_is_retried(settings, logger):
    client = live_client(settings, [RuntimeError("503 overloaded"), '{"verdict": "accept"}'])
    ctx = make_ctx(settings, logger)
    result = client.complete_json(
        agent="VerifierAgent", ctx=ctx, system="s", prompt="p",
        schema_hint="{}", mock={}, purpose="verify",
    )
    assert result == {"verdict": "accept"}
    assert client._client.messages.calls == 2
    assert [r["ok"] for r in logger.records if r["event"] == "llm"] == [False, True]


def test_persistent_api_failure_raises_after_max_attempts(settings, logger):
    settings = settings.model_copy(update={"max_attempts": 3, "retry_backoff_seconds": 0.0})
    client = live_client(settings, [RuntimeError("down")] * 3)
    ctx = make_ctx(settings, logger)
    with pytest.raises(LLMError, match="failed after 3 attempts"):
        client.complete_json(
            agent="VerifierAgent", ctx=ctx, system="s", prompt="p",
            schema_hint="{}", mock={}, purpose="verify",
        )


def test_json_schema_is_sent_as_structured_output(settings, logger):
    client = live_client(settings, ['{"verdict": "accept"}'])
    ctx = make_ctx(settings, logger)
    schema = {"type": "object", "properties": {"verdict": {"type": "string"}},
              "required": ["verdict"], "additionalProperties": False}
    client.complete_json(
        agent="VerifierAgent", ctx=ctx, system="s", prompt="p",
        schema_hint="{}", json_schema=schema, mock={}, purpose="verify",
    )
    sent = client._client.messages.last_kwargs
    assert sent["output_config"]["format"] == {"type": "json_schema", "schema": schema}


def test_cut_off_answer_is_retried(settings, logger):
    client = live_client(settings, [
        _Response('{"verdict": "acc', stop_reason="max_tokens"),
        '{"verdict": "accept"}',
    ])
    ctx = make_ctx(settings, logger)
    result = client.complete_json(
        agent="VerifierAgent", ctx=ctx, system="s", prompt="p",
        schema_hint="{}", mock={}, purpose="verify",
    )
    assert result == {"verdict": "accept"}
    assert client._client.messages.calls == 2


def test_refusal_falls_back_to_the_heuristic_visibly(settings, logger):
    client = live_client(settings, [_Response("", stop_reason="refusal")])
    ctx = make_ctx(settings, logger)
    result = client.complete_json(
        agent="ContextAgent", ctx=ctx, system="s", prompt="p",
        schema_hint="{}", mock={"category": "physical_violence"}, purpose="classify",
    )
    assert result["category"] == "physical_violence"
    assert result["_llm_fallback"] == "refusal"
    assert client._client.messages.calls == 1          # a refusal is not retried
    record = [r for r in logger.records if r["event"] == "llm"][0]
    assert record["ok"] is False and "refusal" in record["detail"]
    assert any("declined" in n for n in ctx.case.notes)


def test_mock_mode_needs_no_key_and_is_labelled(settings, logger):
    client = LLMClient(settings)      # settings fixture has no key
    assert client.provider == "mock"
    ctx = make_ctx(settings, logger)
    result = client.complete_json(
        agent="ContextAgent", ctx=ctx, system="s", prompt="p",
        schema_hint="{}", mock={"category": "no_violation"}, purpose="classify",
    )
    assert result == {"category": "no_violation"}
    assert [r for r in logger.records if r["event"] == "llm"][0]["detail"]["mode"] == "mock"


# --- guarding against a bad model answer ----------------------------------- #

def test_category_outside_the_candidate_set_is_rejected(settings, logger, monkeypatch):
    """A model that invents a category must not be able to set the verdict."""
    from safeschool.agents.context import ContextAgent
    from safeschool.messages import ClipDetections, DetectionSet, MotionAssessment

    client = live_client(settings, [
        '{"category": "alien_invasion", "severity": 4, "is_child_context": true,'
        ' "policy_refs": ["XX-9"], "reasoning": "made up"}'
    ])
    agent = ContextAgent(client)
    ctx = make_ctx(settings, logger)

    motion = MotionAssessment(
        case_id="case_llm",
        per_clip=[{
            "clip_id": "c0", "violence_score": 0.9,
            "features": {}, "rationale": "r",
        }],
        top_clip_id="c0", top_score=0.9,
    )
    ctx.case.put("vision", DetectionSet(
        case_id="case_llm", model_name="stub",
        per_clip=[ClipDetections(clip_id="c0", detections=[], frames_analyzed=5,
                                 person_count_max=2)],
    ))
    message = AgentMessage.wrap(
        case_id="case_llm", sender="MotionAnalystAgent", recipient="ContextAgent",
        type=MessageType.MOTION_ASSESSMENT, payload=motion,
    )
    verdict = agent.handle(message, ctx)
    assert verdict.category != "alien_invasion"
    assert any("outside the candidate" in n for n in ctx.case.notes)
