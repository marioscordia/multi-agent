"""The Verifier's re-run path, and the guarantee that it cannot loop.

A blocking inconsistency should cost one re-run at a lower detection threshold.
If the re-run does not resolve it, the system must reject rather than keep
cycling — that is the loop protection the specification asks for.
"""

import json

import pytest

from safeschool.agents.context import ContextAgent
from safeschool.messages import ContextVerdict
from safeschool.pipeline import run_case


@pytest.fixture
def one_person_fixture(settings, tmp_path):
    """A fixture with a single person in frame, so an interaction verdict is
    provably incoherent and the Verifier's blocking rule must fire."""
    source = json.loads(
        (settings.fixtures_dir / "corridor_fight.json").read_text(encoding="utf-8")
    )
    clip = source["detection_set"]["per_clip"][0]
    clip["detections"] = [d for d in clip["detections"] if d.get("track_id") == 1]
    clip["person_count_max"] = 1

    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    (fixtures_dir / "lone.json").write_text(json.dumps(source), encoding="utf-8")
    return settings.model_copy(update={"fixtures_dir": fixtures_dir})


@pytest.fixture
def force_violence_verdict(monkeypatch):
    """Force the context stage to assert an interaction regardless of evidence,
    which is exactly the error the Verifier exists to catch."""
    def fake_handle(self, message, ctx):
        return ContextVerdict(
            case_id=ctx.case.case_id, category="physical_violence", severity=3,
            is_child_context=True, policy_refs=["SSP-1.1"],
            reasoning="forced verdict for the re-run test", citations=[], llm_used=False,
        )

    monkeypatch.setattr(ContextAgent, "handle", fake_handle)


def test_blocking_inconsistency_triggers_one_rerun_then_rejects(
    one_person_fixture, force_violence_verdict
):
    settings = one_person_fixture
    result, logger = run_case("fixture://lone", settings=settings)

    # exactly one re-run was spent
    assert result.case.rerun_count == 1, "the verifier should have requested one re-run"
    reruns = [r for r in logger.records if r["name"] == "rerun_requested"]
    assert len(reruns) == 1
    assert reruns[0]["detail"]["parameter"] == "confidence_threshold"

    verification = result.case.get("verify")
    codes = {i.code for i in verification.issues}
    assert "interaction_requires_two" in codes
    # the re-run did not fix the contradiction, so the chain is rejected, not looped
    assert verification.verdict == "reject"
    assert "rerun_budget_spent" in codes


def test_rejected_chain_is_not_escalated(one_person_fixture, force_violence_verdict):
    """A rejected verdict must not page anybody."""
    result, _ = run_case("fixture://lone", settings=one_person_fixture)
    assert result.ok, "a rejection is still a completed run, not a crash"
    assert result.report.escalation_tier == "none"
    assert result.report.notified_channel is None


def test_vision_stage_actually_reran_at_a_lower_threshold(
    one_person_fixture, force_violence_verdict
):
    result, logger = run_case("fixture://lone", settings=one_person_fixture)
    vision_runs = [
        r for r in logger.records if r["event"] == "agent" and r["agent"] == "VisionDetectorAgent"
    ]
    assert len(vision_runs) == 2, "vision should have run twice: original plus re-run"


def test_run_stays_within_its_step_budget(one_person_fixture, force_violence_verdict):
    """Even with a re-run, the whole case fits the budget."""
    settings = one_person_fixture.model_copy(update={"max_steps": 40})
    result, _ = run_case("fixture://lone", settings=settings)
    assert result.ok
