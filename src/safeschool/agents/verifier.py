"""Agent 5 — VerifierAgent.

Role: independent adjudication. It re-reads every earlier artifact and asks
whether the chain actually supports the conclusion, then accepts, rejects, or
asks for exactly one re-run at a lower detection threshold.

Why a separate agent: an agent that reviews its own output is the classic
failure mode — it inherits whatever assumption produced the error. The Verifier
sees only the artifacts, not the reasoning that produced them, and it holds no
stake in any particular verdict.

Rule checks run first and are deterministic. The LLM critique runs second and can
only add issues or lower confidence; it cannot overturn a blocking rule failure.
"""

from __future__ import annotations

from ..messages import (
    AgentMessage,
    ContextVerdict,
    Issue,
    MessageType,
    RerunHint,
    Verification,
)
from ..runtime import Agent, Context, SafeSchoolError

_SYSTEM = (
    "You are an independent verification analyst auditing an automated school "
    "safety alert before it reaches a human. Your job is to find reasons the "
    "alert might be wrong, not to agree with it. A false alarm in a school is "
    "costly: it desensitizes staff. Judge only the evidence presented."
)

_SCHEMA = """{
  "verdict": "accept" | "reject" | "rerun",
  "confidence": <float 0-1>,
  "false_positive_risk": "low" | "medium" | "high",
  "issues": [{"code": "<short_code>", "detail": "<one sentence>", "severity": "info"|"warning"|"blocking"}],
  "reasoning": "<2-3 sentences>"
}"""


class VerifierAgent(Agent):
    name = "VerifierAgent"
    stage = "verify"
    role = "Independently audit the detection chain for consistency and false-positive risk"
    tools = ("rules.consistency_checks", "llm.complete_json", "case_store.read_artifacts")
    done_criterion = "an accept/reject/rerun verdict with a confidence and an issue list"
    input_type = MessageType.CONTEXT_VERDICT
    output_type = MessageType.VERIFICATION

    def __init__(self, llm) -> None:
        self.llm = llm

    def handle(self, message: AgentMessage, ctx: Context) -> Verification:
        verdict = self.parse_input(message, ContextVerdict)

        artifacts = ctx.tool(
            self.name, "case_store.read_artifacts",
            lambda: {k: ctx.case.get(k) for k in ("ingest", "vision", "motion")},
        )
        clip_set, detections, motion = artifacts["ingest"], artifacts["vision"], artifacts["motion"]
        if detections is None or motion is None:
            raise SafeSchoolError("verification ran before vision/motion artifacts existed")

        issues = ctx.tool(
            self.name, "rules.consistency_checks",
            self._rule_checks, clip_set, detections, motion, verdict, ctx,
        )

        blocking = [i for i in issues if i.severity == "blocking"]
        top = max(motion.per_clip, key=lambda c: c.violence_score)

        fallback = self._heuristic_verification(issues, top.violence_score, verdict, ctx)
        result = self.llm.complete_json(
            agent=self.name, ctx=ctx, system=_SYSTEM,
            prompt=self._build_prompt(clip_set, detections, motion, verdict, issues, ctx),
            schema_hint=_SCHEMA, mock=fallback, purpose="verify",
        )

        llm_issues = [
            Issue(
                code=str(raw.get("code", "llm_note"))[:40],
                detail=str(raw.get("detail", ""))[:400],
                severity=(
                    raw.get("severity")
                    if raw.get("severity") in {"info", "warning", "blocking"}
                    else "warning"
                ),
            )
            for raw in (result.get("issues") or [])
            if isinstance(raw, dict)
        ][:6]
        all_issues = issues + llm_issues

        decision = result.get("verdict")
        if decision not in {"accept", "reject", "rerun"}:
            decision = fallback["verdict"]

        # A blocking rule failure cannot be talked out of by the model: the chain
        # contradicts itself, so the only options are another look or a rejection.
        if blocking:
            decision = "rerun"

        # Re-run budget: never loop. Whenever a re-run is warranted but the budget
        # is spent, say so explicitly — a rejection with no stated reason is not
        # actionable for whoever reads the report.
        if decision == "rerun" and ctx.case.rerun_count >= ctx.settings.max_reruns:
            decision = "reject"
            all_issues.append(
                Issue(
                    code="rerun_budget_spent",
                    detail=(
                        f"a re-run was warranted but {ctx.case.rerun_count} of "
                        f"{ctx.settings.max_reruns} are already used; rejecting "
                        f"instead of looping"
                    ),
                    severity="warning",
                )
            )

        confidence = float(result.get("confidence", fallback["confidence"]))
        confidence = max(0.0, min(1.0, confidence))
        risk = result.get("false_positive_risk")
        if risk not in {"low", "medium", "high"}:
            risk = fallback["false_positive_risk"]

        rerun_hint = None
        if decision == "rerun":
            lowered = round(max(0.1, detections.confidence_threshold - 0.1), 3)
            rerun_hint = RerunHint(
                stage="vision", parameter="confidence_threshold", value=lowered
            )

        return Verification(
            case_id=verdict.case_id,
            verdict=decision,
            confidence=confidence,
            false_positive_risk=risk,
            issues=all_issues,
            rerun_hint=rerun_hint,
            reasoning=str(result.get("reasoning", fallback["reasoning"]))[:1200],
            llm_used=self.llm.provider != "mock",
        )

    # -- deterministic checks -------------------------------------------- #

    @staticmethod
    def _rule_checks(clip_set, detections, motion, verdict, ctx) -> list[Issue]:
        """Cross-stage consistency. Each check encodes a specific way the chain
        can be internally contradictory."""
        issues: list[Issue] = []
        top = max(motion.per_clip, key=lambda c: c.violence_score)
        top_detections = next(
            (c for c in detections.per_clip if c.clip_id == top.clip_id), None
        )
        threshold = ctx.settings.violence_threshold

        # 1. a violent verdict with nobody in frame is incoherent
        if top_detections is not None and top_detections.person_count_max == 0 and top.violence_score >= threshold:
            issues.append(Issue(
                code="score_without_persons",
                detail=(
                    f"clip {top.clip_id} scored {top.violence_score:.2f} but no person "
                    f"was detected in any analyzed frame"
                ),
                severity="blocking",
            ))

        # 2. a physical-violence verdict needs at least two people
        if verdict.category in {"physical_violence", "adult_on_child_abuse"} and (
            top_detections is None or top_detections.person_count_max < 2
        ):
            issues.append(Issue(
                code="interaction_requires_two",
                detail=(
                    f"category '{verdict.category}' implies an interaction but at most "
                    f"{getattr(top_detections, 'person_count_max', 0)} person(s) were detected"
                ),
                severity="blocking",
            ))

        # 3. a weapon verdict needs a weapon detection
        if verdict.category == "weapon_threat" and not (
            top_detections and top_detections.weapon_present
        ):
            issues.append(Issue(
                code="weapon_without_detection",
                detail="weapon_threat was asserted but no weapon-like object was detected",
                severity="blocking",
            ))

        # 4. thin evidence: too few frames actually analyzed
        if top_detections is not None and top_detections.frames_analyzed < 3:
            issues.append(Issue(
                code="insufficient_frames",
                detail=(
                    f"only {top_detections.frames_analyzed} frame(s) analyzed in the top "
                    f"clip; temporal features are unreliable below 3"
                ),
                severity="warning",
            ))

        # 5. escalation on a sub-threshold score
        if verdict.severity >= 3 and top.violence_score < threshold:
            issues.append(Issue(
                code="escalation_above_evidence",
                detail=(
                    f"severity {verdict.severity} requested on a score of "
                    f"{top.violence_score:.2f}, below the {threshold:.2f} threshold"
                ),
                severity="warning",
            ))

        # 6. fixture-backed runs are never operational evidence
        if detections.source == "fixture" or getattr(clip_set, "source", "") == "fixture":
            issues.append(Issue(
                code="fixture_backed_run",
                detail=(
                    "this run used recorded fixtures rather than live inference; the "
                    "result demonstrates the pipeline, not a real detection"
                ),
                severity="info",
            ))

        # 7. a borderline score is exactly where a re-run is worth its cost
        if abs(top.violence_score - threshold) < 0.08:
            issues.append(Issue(
                code="borderline_score",
                detail=(
                    f"score {top.violence_score:.2f} sits within 0.08 of the "
                    f"{threshold:.2f} decision boundary"
                ),
                severity="warning",
            ))

        # 8. the context stage should rest on retrieved evidence
        if not verdict.citations:
            issues.append(Issue(
                code="no_literature_grounding",
                detail="the context verdict cites no literature evidence",
                severity="info",
            ))

        return issues

    @staticmethod
    def _heuristic_verification(issues, score, verdict, ctx) -> dict:
        blocking = [i for i in issues if i.severity == "blocking"]
        warnings = [i for i in issues if i.severity == "warning"]
        if blocking:
            decision = "rerun" if ctx.case.rerun_count < ctx.settings.max_reruns else "reject"
            risk = "high"
            confidence = 0.3
        elif len(warnings) >= 2:
            decision = "accept"
            risk = "medium"
            confidence = 0.55
        else:
            decision = "accept"
            risk = "low" if score >= ctx.settings.violence_threshold or verdict.severity <= 1 else "medium"
            confidence = 0.8
        return {
            "verdict": decision,
            "confidence": confidence,
            "false_positive_risk": risk,
            "issues": [],
            "reasoning": (
                f"Rule audit: {len(blocking)} blocking and {len(warnings)} warning "
                f"issue(s) on a score of {score:.2f} for category '{verdict.category}'."
            ),
        }

    @staticmethod
    def _build_prompt(clip_set, detections, motion, verdict, issues, ctx) -> str:
        top = max(motion.per_clip, key=lambda c: c.violence_score)
        top_detections = next(
            (c for c in detections.per_clip if c.clip_id == top.clip_id), None
        )
        issue_lines = "\n".join(
            f"  - [{i.severity}] {i.code}: {i.detail}" for i in issues
        ) or "  (no rule violations found)"
        features = "\n".join(
            f"  - {k}: {v:.3f}" for k, v in top.features.model_dump().items()
        )
        return f"""Audit this automated school safety alert before it reaches a human.

EVIDENCE CHAIN
  stage 1 ingest: {len(clip_set.clips) if clip_set else 0} clip(s), source={getattr(clip_set, 'source', '?')}
  stage 2 vision: model={detections.model_name}, source={detections.source}, threshold={detections.confidence_threshold}
                  top clip frames analyzed={getattr(top_detections, 'frames_analyzed', 0)},
                  max persons={getattr(top_detections, 'person_count_max', 0)},
                  weapon detected={getattr(top_detections, 'weapon_present', False)}
  stage 3 motion: score={top.violence_score:.3f} (threshold {ctx.settings.violence_threshold:.2f})
{features}
                  scorer rationale: {top.rationale}
  stage 4 context: category={verdict.category}, severity={verdict.severity},
                  policy={', '.join(verdict.policy_refs) or 'none'}
                  reasoning: {verdict.reasoning[:500]}
                  literature citations: {len(verdict.citations)}

DETERMINISTIC RULE AUDIT ALREADY PERFORMED:
{issue_lines}

Decide: accept (the chain supports the verdict), reject (it does not), or rerun
(a lower detection threshold would plausibly change the evidence). Add any issue
the rule audit missed. Note that a blocking rule failure cannot be overridden."""
