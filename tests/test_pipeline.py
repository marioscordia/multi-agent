"""End-to-end behaviour of the agent team."""

import pytest

from safeschool.agents import WORKER_AGENTS
from safeschool.agents.orchestrator import OrchestratorAgent
from safeschool.llm import LLMClient
from safeschool.messages import MessageType
from safeschool.pipeline import run_case


def _run(settings, video):
    return run_case(video, settings=settings)


# --- the hand-off contract ------------------------------------------------- #

def test_two_agents_exchange_validated_messages(settings):
    """The assignment's core prototype requirement, checked explicitly."""
    result, _ = _run(settings, "fixture://corridor_fight")
    senders = [(m.sender, m.recipient, m.type) for m in result.case.messages]

    assert ("OrchestratorAgent", "IngestAgent", MessageType.CASE_REQUEST) in senders
    assert ("IngestAgent", "VisionDetectorAgent", MessageType.CLIP_SET) in senders
    assert ("VisionDetectorAgent", "MotionAnalystAgent", MessageType.DETECTION_SET) in senders
    assert ("MotionAnalystAgent", "ContextAgent", MessageType.MOTION_ASSESSMENT) in senders
    assert ("ContextAgent", "VerifierAgent", MessageType.CONTEXT_VERDICT) in senders
    assert ("VerifierAgent", "PrivacyReportAgent", MessageType.VERIFICATION) in senders


def test_all_six_stages_complete(settings):
    result, _ = _run(settings, "fixture://corridor_fight")
    assert result.ok
    assert all(v.value == "done" for v in result.case.status.values()), result.case.status


# --- outcomes -------------------------------------------------------------- #

def test_fight_fixture_escalates(settings):
    result, _ = _run(settings, "fixture://corridor_fight")
    assert result.report.category == "physical_violence"
    assert result.report.escalation_tier in {"notify_admin", "notify_staff"}


def test_weapon_fixture_reaches_emergency(settings):
    result, _ = _run(settings, "fixture://weapon_threat")
    assert result.report.category == "weapon_threat"
    assert result.report.severity == 4


def test_rough_play_does_not_escalate_as_violence(settings):
    """The dominant false-positive class must not produce a violence alert."""
    result, _ = _run(settings, "fixture://rough_play")
    assert result.report.category not in {"physical_violence", "weapon_threat"}
    assert result.report.escalation_tier not in {"emergency", "notify_admin"}


# --- error paths ----------------------------------------------------------- #

def test_missing_file_fails_cleanly(settings):
    result, _ = run_case(
        "does/not/exist.mp4", settings=settings, allow_fixture_fallback=False
    )
    assert not result.ok
    assert result.error.stage == "ingest"
    assert "not found" in result.error.message


def test_missing_file_falls_back_to_fixtures_when_allowed(settings):
    result, _ = run_case("does/not/exist.mp4", settings=settings, allow_fixture_fallback=True)
    assert result.ok
    assert any("fell back to fixtures" in n for n in result.case.notes)


def test_fixture_run_is_flagged_as_non_operational(settings):
    """A fixture-backed result must never be mistakable for a real detection."""
    result, _ = _run(settings, "fixture://corridor_fight")
    codes = {i.code for i in result.case.get("verify").issues}
    assert "fixture_backed_run" in codes
    assert result.case.get("vision").source == "fixture"


# --- privacy --------------------------------------------------------------- #

def test_report_only_references_redacted_frames(settings):
    result, _ = _run(settings, "fixture://corridor_fight")
    assert result.report.faces_redacted > 0
    for path in result.report.redacted_keyframes:
        assert "redacted" in path


def test_context_agent_prompt_contains_no_image_paths(settings):
    """Privacy rule PR-2: the reasoning stages never receive imagery."""
    from safeschool.agents.context import ContextAgent
    from safeschool.messages import ClipMotion, MotionFeatures

    motion = ClipMotion(
        clip_id="c0", violence_score=0.9, features=MotionFeatures(), rationale="r"
    )
    prompt = ContextAgent._build_prompt(
        top=motion, features=motion.features.model_dump(), candidates=[],
        weapon_present=False, person_count=2, scale_disparity=1.0, hits=[],
    )
    assert ".jpg" not in prompt and "keyframe" not in prompt.lower()


# --- load balance ---------------------------------------------------------- #

@pytest.mark.parametrize("video", ["fixture://corridor_fight", "fixture://rough_play"])
def test_no_agent_exceeds_forty_percent_of_calls(settings, video):
    """The specification's hard limit, measured from the run log."""
    _, logger = _run(settings, video)
    shares = logger.load_shares()
    assert shares, "no calls were logged"
    worst_agent, worst_share = max(shares.items(), key=lambda kv: kv[1])
    assert worst_share <= 0.40, f"{worst_agent} holds {worst_share:.1%} of calls"


def test_orchestrator_does_no_domain_work(settings):
    """It plans and routes; it must not be making tool or LLM calls."""
    _, logger = _run(settings, "fixture://corridor_fight")
    orchestrator_calls = [
        r for r in logger.records
        if r["agent"] == "OrchestratorAgent" and r["event"] in {"tool", "llm"}
    ]
    assert orchestrator_calls == []


# --- specification --------------------------------------------------------- #

def test_agent_team_meets_the_specification(settings):
    """4+ agents required, 5+ for full marks; <=5 tools each; distinct roles."""
    llm = LLMClient(settings)
    agents = [OrchestratorAgent(llm)] + [
        cls(llm) if cls.__name__ in {"ContextAgent", "VerifierAgent"} else cls()
        for cls in WORKER_AGENTS
    ]
    assert len(agents) >= 5
    names = [a.name for a in agents]
    stages = [a.stage for a in agents]
    assert len(set(names)) == len(names), "agent names must be unique"
    assert len(set(stages)) == len(stages), "roles must not duplicate"
    for agent in agents:
        assert len(agent.tools) <= 5, f"{agent.name} declares {len(agent.tools)} tools"
        assert agent.role and agent.done_criterion, f"{agent.name} is under-specified"
