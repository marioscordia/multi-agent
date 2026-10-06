"""Test scenarios with expected outcomes.

The assignment asks for a scenario set with stated expectations (4+ at the next
stage, 6+ at the exam). Each scenario names the input, what it is probing, and
what the system must do with it — including the cases where the correct behaviour
is to *not* raise an alert, which is the failure mode the literature review found
to be systematically under-reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"


@dataclass
class Scenario:
    name: str
    video: str
    description: str
    #: acceptable categories for the context verdict
    expect_categories: tuple[str, ...] = ()
    #: acceptable escalation tiers
    expect_tiers: tuple[str, ...] = ()
    #: the run is expected to fail cleanly rather than produce a report
    expect_failure: bool = False
    #: verifier issue codes that must appear
    expect_issues: tuple[str, ...] = ()
    allow_fixtures: bool = True
    notes: str = ""

    def expectation(self) -> str:
        if self.expect_failure:
            parts = ["clean failure"]
        else:
            parts = []
            if self.expect_categories:
                parts.append("category in {" + ",".join(self.expect_categories) + "}")
            if self.expect_tiers:
                parts.append("tier in {" + ",".join(self.expect_tiers) + "}")
        if self.expect_issues:
            parts.append("issues " + ",".join(self.expect_issues))
        return "; ".join(parts) or "completes"

    def check(self, result: Any) -> tuple[bool, str]:
        """Compare a RunResult against this scenario's expectation."""
        if self.expect_failure:
            if result.error is not None:
                return True, f"failed cleanly ({result.error.error_type})"
            return False, f"unexpectedly succeeded (tier={result.report.escalation_tier})"

        if result.error is not None:
            return False, f"failed: {result.error.error_type}: {result.error.message[:60]}"

        report = result.report
        actual = f"{report.category}/{report.escalation_tier}"
        ok = True
        if self.expect_categories and report.category not in self.expect_categories:
            ok = False
        if self.expect_tiers and report.escalation_tier not in self.expect_tiers:
            ok = False
        if self.expect_issues:
            verification = result.case.get("verify")
            codes = {i.code for i in verification.issues} if verification else set()
            missing = set(self.expect_issues) - codes
            if missing:
                ok = False
                actual += f" (missing issues: {','.join(sorted(missing))})"
        return ok, actual


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="corridor_walking",
        video=str(RAW / "corridor_people.mp4"),
        description="real CCTV of people walking a corridor — must NOT raise an alert",
        expect_categories=("no_violation", "rough_play_benign", "bullying_nonphysical"),
        expect_tiers=("log", "none"),
        notes="the true-negative baseline; a false alarm here is the costliest error",
    ),
    Scenario(
        name="crowded_aisle",
        video=str(RAW / "store_aisle.mp4"),
        description="several people moving in a confined aisle — crowding must suppress, not amplify",
        expect_categories=("no_violation", "rough_play_benign", "bullying_nonphysical"),
        expect_tiers=("log", "none", "notify_staff"),
        notes="probes the negative crowding weight in the scorer",
    ),
    Scenario(
        name="single_person",
        video=str(RAW / "one_person.mp4"),
        description="one person only — an interaction verdict here is incoherent",
        expect_categories=("no_violation", "rough_play_benign"),
        expect_tiers=("log", "none"),
        notes="if the chain ever claims physical_violence, the verifier must block it",
    ),
    Scenario(
        name="empty_scene",
        video=str(RAW / "empty_corridor.mp4"),
        description="no people in frame — empty-detection path",
        expect_categories=("no_violation",),
        expect_tiers=("log", "none"),
        notes="exercises the zero-detection branch end to end",
    ),
    Scenario(
        name="corrupt_file",
        video=str(RAW / "corrupt_clip.mp4"),
        description="unreadable container — error path, must fail with a readable message",
        expect_failure=True,
        allow_fixtures=False,
        notes="--no-fixtures, so the decode failure is not masked by the fallback",
    ),
    Scenario(
        name="missing_file",
        video=str(RAW / "does_not_exist.mp4"),
        description="nonexistent input — error path",
        expect_failure=True,
        allow_fixtures=False,
        notes="the most common operator mistake",
    ),
    Scenario(
        name="fixture_fight",
        video="fixture://corridor_fight",
        description="recorded two-person fight detections — the positive path, offline",
        expect_categories=("physical_violence",),
        expect_tiers=("notify_admin", "notify_staff"),
        expect_issues=("fixture_backed_run",),
        notes="demonstrates the alerting path without redistributing benchmark footage",
    ),
    Scenario(
        name="fixture_weapon",
        video="fixture://weapon_threat",
        description="recorded detections including a weapon-like object — emergency path",
        expect_categories=("weapon_threat",),
        expect_tiers=("emergency", "notify_admin"),
        expect_issues=("fixture_backed_run",),
        notes="severity 4 escalation, and the weapon_without_detection rule's happy path",
    ),
    Scenario(
        name="fixture_adult_child",
        video="fixture://adult_child",
        description="recorded detections with a large body-scale disparity — safeguarding path",
        expect_categories=("adult_on_child_abuse", "physical_violence"),
        expect_tiers=("emergency", "notify_admin"),
        notes="the thesis's core case: an adult acting physically on a child",
    ),
    Scenario(
        name="fixture_rough_play",
        video="fixture://rough_play",
        description="energetic but mutual play — must not escalate as violence",
        expect_categories=("rough_play_benign", "no_violation", "bullying_nonphysical"),
        expect_tiers=("log", "none", "notify_staff"),
        notes=(
            "the dominant false-positive class identified by the literature review. "
            "The requirement is specifically that this does NOT come out as "
            "physical_violence or weapon_threat and does NOT reach notify_admin or "
            "emergency: a low-severity, non-violent outcome is a pass, because "
            "pastoral follow-up on ambiguous group behaviour is proportionate "
            "while an emergency response is not."
        ),
    ),
)


def by_name(name: str) -> Scenario | None:
    return next((s for s in SCENARIOS if s.name == name), None)
