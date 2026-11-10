"""The staff summary written by the PrivacyReportAgent's language model.

Three properties matter: the model's text is used, the model cannot change who
is alerted, and a model failure never costs the report.
"""

import json

from safeschool.agents.orchestrator import build_context
from safeschool.agents.report import PrivacyReportAgent
from safeschool.messages import AgentMessage, MessageType
from safeschool.pipeline import run_case

from test_llm import live_client


def _report_again(settings, logger, script):
    """Run a fight case in mock mode, then redo only the report stage with a
    stubbed live model. Returns (mock_report, new_report, stub, case)."""
    result, _ = run_case("fixture://corridor_fight", settings=settings)
    case = result.case
    client = live_client(settings, script)
    agent = PrivacyReportAgent(client)
    ctx = build_context(settings, logger, case)
    message = AgentMessage.wrap(
        case_id=case.case_id, sender="VerifierAgent", recipient=agent.name,
        type=MessageType.VERIFICATION, payload=case.require("verify"),
    )
    return result.report, agent.handle(message, ctx), client._client.messages, case


def test_staff_summary_is_written_by_the_model(settings, logger):
    answer = json.dumps({
        "summary": "Two students appear to be fighting in a corridor.",
        "staff_action": "Go to the corridor now. Escalate this to an emergency.",
    })
    mock_report, report, _, _ = _report_again(settings, logger, [answer])

    assert report.summary == "Two students appear to be fighting in a corridor."
    assert report.staff_action.startswith("Go to the corridor now")
    assert report.summary_source == "llm"
    # the model asked for an emergency in its text; the policy tier is unchanged
    assert report.escalation_tier == mock_report.escalation_tier == "notify_admin"


def test_no_image_or_path_reaches_the_model(settings, logger):
    answer = json.dumps({"summary": "s", "staff_action": "a"})
    _, _, stub, case = _report_again(settings, logger, [answer])

    prompt = stub.last_kwargs["messages"][0]["content"]
    for keyframe in case.require("ingest").clips[0].keyframes:
        assert keyframe.path not in prompt
    assert ".jpg" not in prompt and "fixtures/" not in prompt


def test_long_text_is_cut_at_a_sentence_end():
    from safeschool.agents.report import _clip_to_sentence

    text = "First sentence here. Second one is longer. Third goes past the limit"
    assert _clip_to_sentence(text, 45) == "First sentence here. Second one is longer."
    assert _clip_to_sentence("short", 45) == "short"


def test_model_failure_falls_back_to_the_template(settings, logger):
    settings = settings.model_copy(update={"max_attempts": 2, "retry_backoff_seconds": 0.0})
    _, report, _, case = _report_again(settings, logger, [RuntimeError("down")] * 2)

    assert report.summary_source == "template"
    assert "physical violence" in report.summary
    assert report.report_markdown_path            # the report still exists
    assert any("template used" in n for n in case.notes)
