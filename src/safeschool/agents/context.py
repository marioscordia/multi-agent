"""Agent 4 — ContextAgent.

Role: decide what the motion *means* in a school. A high motion score is not a
safety violation by itself — a corridor scuffle, a wrestling lesson and an adult
dragging a child produce overlapping motion signatures and call for completely
different responses.

Grounding comes from two sources, both local:

* the school-safety policy knowledge base, which supplies the candidate
  categories and their severities, so the agent chooses among policy clauses
  rather than inventing a label;
* the thesis literature corpus (50 reviewed papers), retrieved with BM25, so the
  verdict cites the published evidence behind its interpretation.

Privacy rule PR-2: this agent receives aggregated numbers and text only. No
frame, crop or keyframe path is ever put into the prompt.
"""

from __future__ import annotations

import math
from typing import Any

from ..messages import (
    AgentMessage,
    Citation,
    ContextVerdict,
    MessageType,
    MotionAssessment,
)
from ..runtime import Agent, Context, SafeSchoolError
from ..tools import corpus_retriever, policy_kb

_SYSTEM = (
    "You are a child-safeguarding analyst for an automated school video "
    "monitoring system. You classify an incident using ONLY the policy clauses "
    "you are given and the motion features measured by the vision pipeline. "
    "You never see images. You are conservative: when the evidence is weak or "
    "ambiguous, prefer the benign category, because a false alarm in a school "
    "erodes staff trust and desensitizes responders."
)

#: Maximum difference in bounding-box bottom edge (normalized) for two people to
#: be treated as standing at the same depth, and therefore size-comparable.
FOOT_LEVEL_TOLERANCE = 0.08

#: how close to the frame border a box must be to count as truncated
EDGE_MARGIN = 0.02


def _touches_frame_edge(bbox: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = bbox
    return (
        x1 <= EDGE_MARGIN
        or y1 <= EDGE_MARGIN
        or x2 >= 1.0 - EDGE_MARGIN
        or y2 >= 1.0 - EDGE_MARGIN
    )

_SCHEMA = """{
  "category": "<one of the candidate categories>",
  "severity": <integer 1-4>,
  "is_child_context": <true|false>,
  "policy_refs": ["<clause id>"],
  "reasoning": "<2-3 sentences citing the specific feature values>"
}"""


class ContextAgent(Agent):
    name = "ContextAgent"
    stage = "context"
    role = "Interpret the motion assessment against school policy and the reviewed literature"
    tools = (
        "policy_kb.candidate_clauses",
        "corpus_retriever.search",
        "llm.complete_json",
    )
    done_criterion = "a verdict with a policy category, severity 1-4 and at least one citation"
    input_type = MessageType.MOTION_ASSESSMENT
    output_type = MessageType.CONTEXT_VERDICT

    def __init__(self, llm) -> None:
        self.llm = llm

    def handle(self, message: AgentMessage, ctx: Context) -> ContextVerdict:
        motion = self.parse_input(message, MotionAssessment)
        detections = ctx.case.get("vision")
        if detections is None:
            raise SafeSchoolError("context stage ran before the vision stage produced detections")

        top = max(motion.per_clip, key=lambda c: c.violence_score)
        clip_detections = next(
            (c for c in detections.per_clip if c.clip_id == top.clip_id),  # type: ignore[attr-defined]
            None,
        )
        weapon_present = bool(clip_detections and clip_detections.weapon_present)
        person_count = int(clip_detections.person_count_max) if clip_detections else 0
        scale_disparity = self._scale_disparity(clip_detections)

        candidates = ctx.tool(
            self.name, "policy_kb.candidate_clauses",
            policy_kb.candidate_clauses, ctx.settings.policy_dir,
            weapon_present=weapon_present,
            violence_score=top.violence_score,
            person_count=person_count,
            scale_disparity=scale_disparity,
            threshold=ctx.settings.violence_threshold,
        )

        query = self._retrieval_query(top, weapon_present, scale_disparity, person_count)
        index = corpus_retriever.get_index(ctx.settings.corpus_dir, ctx.settings.matrix_csv)
        hits = ctx.tool(
            self.name, "corpus_retriever.search", index.search, query, top_k=3
        )
        citations = [
            Citation(
                source=hit.document.source,
                reference=hit.document.reference,
                snippet=hit.snippet,
            )
            for hit in hits
        ]

        features = top.features.model_dump()
        prompt = self._build_prompt(
            top=top,
            features=features,
            candidates=candidates,
            weapon_present=weapon_present,
            person_count=person_count,
            scale_disparity=scale_disparity,
            hits=hits,
        )
        fallback = self._heuristic_verdict(
            candidates, top.violence_score, weapon_present, scale_disparity,
            ctx.settings.violence_threshold,
        )

        result = self.llm.complete_json(
            agent=self.name, ctx=ctx, system=_SYSTEM, prompt=prompt,
            schema_hint=_SCHEMA, mock=fallback, purpose="classify",
        )

        valid_categories = {c["category"] for c in candidates}
        category = result.get("category")
        if category not in valid_categories:
            ctx.case.note(
                f"context: model proposed category '{category}' outside the candidate "
                f"set; falling back to the policy heuristic"
            )
            category = fallback["category"]
            severity = int(fallback["severity"])
        else:
            severity = int(result.get("severity", fallback["severity"]))
        severity = max(1, min(4, severity))

        clause = policy_kb.clause_for_category(ctx.settings.policy_dir, category)
        policy_refs = result.get("policy_refs") or ([clause["id"]] if clause else [])

        if not citations:
            # retrieval returning nothing is a soft failure: note it, keep going
            ctx.case.note("context: literature retrieval returned no hits for this query")

        return ContextVerdict(
            case_id=motion.case_id,
            category=category,
            severity=severity,
            is_child_context=bool(result.get("is_child_context", True)),
            policy_refs=[str(r) for r in policy_refs][:4],
            reasoning=str(result.get("reasoning", fallback["reasoning"]))[:1200],
            citations=citations,
            llm_used=self.llm.provider != "mock",
        )

    # -- helpers --------------------------------------------------------- #

    @staticmethod
    def _scale_disparity(clip_detections) -> float:
        """Body-height ratio of the closest *interacting* pair, at equal depth.

        A naive max/min over every box in the frame measures camera perspective,
        not age: a shopper near the lens is twice the height of one at the end of
        the aisle. Two constraints make the ratio mean what it is supposed to:

        * only the closest pair in each frame is compared, since a size
          difference is only evidence of an adult-child interaction if the two
          are actually interacting;
        * their feet must be at a similar image height. On a ground plane, equal
          foot height means equal depth, so a remaining size difference is a real
          difference in stature rather than distance from the camera.
        """
        if clip_detections is None:
            return 1.0

        by_frame: dict[int, list] = {}
        for d in clip_detections.detections:
            if d.cls != "person" or d.bbox[3] <= d.bbox[1]:
                continue
            # a box clipped by the frame edge has an unknown true height, so it
            # cannot be evidence about anybody's stature
            if _touches_frame_edge(d.bbox):
                continue
            by_frame.setdefault(d.frame_index, []).append(d)

        best = 1.0
        for persons in by_frame.values():
            if len(persons) < 2:
                continue
            closest: tuple[float, Any, Any] | None = None
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    a, b = persons[i], persons[j]
                    ax = (a.bbox[0] + a.bbox[2]) / 2
                    bx = (b.bbox[0] + b.bbox[2]) / 2
                    ay = (a.bbox[1] + a.bbox[3]) / 2
                    by = (b.bbox[1] + b.bbox[3]) / 2
                    distance = math.hypot(ax - bx, ay - by)
                    if closest is None or distance < closest[0]:
                        closest = (distance, a, b)
            if closest is None:
                continue
            _, a, b = closest
            # feet at a similar image height => similar depth => comparable size
            if abs(a.bbox[3] - b.bbox[3]) > FOOT_LEVEL_TOLERANCE:
                continue
            ha, hb = a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1]
            best = max(best, max(ha, hb) / max(min(ha, hb), 1e-6))
        return round(best, 3)

    @staticmethod
    def _retrieval_query(top, weapon_present: bool, scale_disparity: float, person_count: int) -> str:
        parts = ["school campus violence detection surveillance children"]
        if weapon_present:
            parts.append("weapon knife firearm detection")
        if scale_disparity >= 1.45:
            parts.append("child abuse adult child aggression recognition")
        if person_count >= 3:
            parts.append("crowd bullying group behaviour analysis")
        if top.features.contact_frames_ratio > 0.4:
            parts.append("physical fight interaction proximity skeleton pose")
        else:
            parts.append("false positive rough play normal activity false alarm")
        return " ".join(parts)

    @staticmethod
    def _build_prompt(*, top, features, candidates, weapon_present, person_count,
                      scale_disparity, hits) -> str:
        feature_lines = "\n".join(f"  - {k}: {v:.3f}" for k, v in features.items())
        candidate_lines = "\n".join(
            f"  - {c['category']} (clause {c['id']}, severity {c['severity']}): "
            f"{c['title']}. Indicators: {'; '.join(c['indicators'])}"
            for c in candidates
        )
        evidence_lines = "\n".join(
            f"  - {h.document.reference[:160]}: {h.snippet[:220]}" for h in hits
        ) or "  (no literature hits)"
        return f"""A school surveillance clip has been analyzed by the vision pipeline.

MEASURED MOTION FEATURES (normalized 0-1 unless noted):
{feature_lines}

PIPELINE SUMMARY:
  - violence score: {top.violence_score:.3f}
  - scorer rationale: {top.rationale}
  - max persons in frame: {person_count}
  - weapon-like object detected: {weapon_present}
  - person body-scale disparity (max/min bbox height): {scale_disparity}

CANDIDATE POLICY CLAUSES (choose exactly one category from this list):
{candidate_lines}

RELEVANT PUBLISHED EVIDENCE (from the project's literature review):
{evidence_lines}

Classify the incident. Justify your choice from the feature values above, and
state explicitly if the evidence is too weak to support an alert."""

    @staticmethod
    def _heuristic_verdict(candidates, score: float, weapon_present: bool,
                           scale_disparity: float, threshold: float) -> dict:
        """Policy-only verdict. Used as the mock-LLM response and as the fallback
        when the model proposes a category outside the candidate set."""
        by_category = {c["category"]: c for c in candidates}
        if weapon_present and "weapon_threat" in by_category:
            category = "weapon_threat"
        elif score >= threshold and scale_disparity >= 1.45 and "adult_on_child_abuse" in by_category:
            category = "adult_on_child_abuse"
        elif score >= threshold and "physical_violence" in by_category:
            category = "physical_violence"
        elif "bullying_nonphysical" in by_category and score >= threshold * 0.6:
            category = "bullying_nonphysical"
        else:
            category = "no_violation" if "no_violation" in by_category else candidates[0]["category"]
        clause = by_category[category]
        return {
            "category": category,
            "severity": int(clause["severity"]),
            "is_child_context": True,
            "policy_refs": [clause["id"]],
            "reasoning": (
                f"Policy heuristic: violence score {score:.2f} against threshold "
                f"{threshold:.2f}, weapon={weapon_present}, scale disparity "
                f"{scale_disparity:.2f} matches clause {clause['id']} ({clause['title']})."
            ),
        }
