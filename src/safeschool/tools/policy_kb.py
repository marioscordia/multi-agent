"""School-safety policy knowledge base.

A small, explicit rule set the ContextAgent matches against, so a verdict can
name the clause it rests on instead of being an unexplained model output. The
clauses encode the safeguarding categories and the dominant false-positive class
("rough play") identified by the stage-1 review.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..runtime import SafeSchoolError

POLICY_FILE = "school_safety_policy.json"


class PolicyUnavailable(SafeSchoolError):
    recoverable = False


@lru_cache(maxsize=4)
def load_policy(policy_dir: Path | str) -> dict[str, Any]:
    path = Path(policy_dir) / POLICY_FILE
    if not path.exists():
        raise PolicyUnavailable(
            f"policy file missing: {path}. The ContextAgent cannot classify a "
            f"violation without a policy to classify it against."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PolicyUnavailable(f"policy file {path.name} is not valid JSON: {exc}") from exc


def clauses(policy_dir: Path | str) -> list[dict[str, Any]]:
    return load_policy(policy_dir)["clauses"]


def privacy_rules(policy_dir: Path | str) -> list[dict[str, Any]]:
    return load_policy(policy_dir)["privacy_rules"]


def clause_for_category(policy_dir: Path | str, category: str) -> dict[str, Any] | None:
    for clause in clauses(policy_dir):
        if clause["category"] == category:
            return clause
    return None


def escalation_tier(policy_dir: Path | str, severity: int) -> str:
    tiers = load_policy(policy_dir)["escalation_tiers"]
    return tiers.get(str(severity), "log")


def candidate_clauses(
    policy_dir: Path | str,
    *,
    weapon_present: bool,
    violence_score: float,
    person_count: int,
    scale_disparity: float,
    threshold: float,
) -> list[dict[str, Any]]:
    """Pre-filter the clauses worth considering, so the LLM chooses among a few
    policy-grounded options rather than inventing a category."""
    all_clauses = clauses(policy_dir)
    by_category = {c["category"]: c for c in all_clauses}
    picked: list[dict[str, Any]] = []

    if weapon_present:
        picked.append(by_category["weapon_threat"])
    if violence_score >= threshold:
        picked.append(by_category["physical_violence"])
        picked.append(by_category["rough_play_benign"])
        # a large body-scale difference is the signal that distinguishes an
        # adult-on-child incident from a peer altercation
        if scale_disparity >= 1.45:
            picked.append(by_category["adult_on_child_abuse"])
    if person_count >= 3 and violence_score < threshold:
        picked.append(by_category["bullying_nonphysical"])
    if not picked or violence_score < threshold * 0.6:
        picked.append(by_category["no_violation"])

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for clause in picked:
        if clause["id"] not in seen:
            seen.add(clause["id"])
            unique.append(clause)
    return unique
