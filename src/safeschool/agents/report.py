"""Agent 6 — PrivacyReportAgent.

Role: the only component permitted to touch identifiable imagery, and the only
one that emits anything outward. It redacts faces, writes the incident report,
and escalates according to the policy tier.

Concentrating both the redaction and the outbound side effect in one agent is the
point: there is exactly one place to audit for a privacy leak. The stage-1 review
found that 2 of 50 published systems addressed privacy at all, which is the gap
this agent exists to close.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..messages import (
    AgentMessage,
    ContextVerdict,
    IncidentReport,
    MessageType,
    Verification,
)
from ..runtime import Agent, Context
from ..tools import blur, notifier, policy_kb


class PrivacyReportAgent(Agent):
    name = "PrivacyReportAgent"
    stage = "report"
    role = "Redact faces, write the incident report, escalate per the policy tier"
    tools = ("blur.redact_frame", "policy_kb.escalation_tier", "report.render", "notifier.notify")
    done_criterion = (
        "a markdown and a JSON report exist and reference only redacted keyframes"
    )
    input_type = MessageType.VERIFICATION
    output_type = MessageType.INCIDENT_REPORT

    def handle(self, message: AgentMessage, ctx: Context) -> IncidentReport:
        verification = self.parse_input(message, Verification)
        verdict: ContextVerdict = ctx.case.require("context")  # type: ignore[assignment]
        motion = ctx.case.require("motion")
        clip_set = ctx.case.require("ingest")
        detections = ctx.case.require("vision")

        case_dir = ctx.settings.cases_dir / verification.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # --- escalation tier ------------------------------------------- #
        if verification.verdict == "reject" or verdict.category in {
            "no_violation", "rough_play_benign"
        }:
            tier = "none" if verification.verdict == "reject" else "log"
        else:
            tier = ctx.tool(
                self.name, "policy_kb.escalation_tier",
                policy_kb.escalation_tier, ctx.settings.policy_dir, verdict.severity,
            )
            # an unreliable chain is downgraded rather than suppressed: a human
            # still sees it, but it does not trigger an emergency response
            if verification.false_positive_risk == "high" and tier == "emergency":
                tier = "notify_admin"

        # --- privacy redaction (PR-1) ---------------------------------- #
        redacted, faces = self._redact(clip_set, detections, motion, case_dir, ctx)

        # --- reports ---------------------------------------------------- #
        top = max(motion.per_clip, key=lambda c: c.violence_score)  # type: ignore[attr-defined]
        summary = (
            f"{verdict.category.replace('_', ' ')} (severity {verdict.severity}) on "
            f"clip {top.clip_id} at t={top.t_peak if top.t_peak is not None else top.clip_id}s, "
            f"motion score {top.violence_score:.2f}, verification {verification.verdict} "
            f"with {verification.false_positive_risk} false-positive risk"
        )

        markdown = ctx.tool(
            self.name, "report.render", self._render_markdown,
            case_id=verification.case_id, clip_set=clip_set, detections=detections,
            motion=motion, verdict=verdict, verification=verification,
            tier=tier, redacted=redacted, faces=faces, ctx=ctx,
        )
        md_path = case_dir / "report.md"
        md_path.write_text(markdown, encoding="utf-8")

        report = IncidentReport(
            case_id=verification.case_id,
            escalation_tier=tier,
            category=verdict.category,
            severity=verdict.severity,
            confidence=verification.confidence,
            summary=summary,
            redacted_keyframes=redacted,
            faces_redacted=faces,
            report_markdown_path=str(md_path),
        )

        json_path = case_dir / "report.json"
        json_path.write_text(
            json.dumps(
                {
                    "report": report.model_dump(mode="json"),
                    "context_verdict": verdict.model_dump(mode="json"),
                    "verification": verification.model_dump(mode="json"),
                },
                indent=2, ensure_ascii=False, default=str,
            ),
            encoding="utf-8",
        )
        report.report_json_path = str(json_path)

        # --- escalation ------------------------------------------------- #
        if tier in {"notify_staff", "notify_admin", "emergency"}:
            report.notified_channel = ctx.tool(
                self.name, "notifier.notify", notifier.notify,
                channel_dir=ctx.settings.data_dir / "channel",
                case_id=verification.case_id, tier=tier,
                category=verdict.category, severity=verdict.severity, summary=summary,
            )
        return report

    # -- privacy --------------------------------------------------------- #

    def _redact(self, clip_set, detections, motion, case_dir: Path, ctx: Context):
        """Blur faces on the top clip's keyframes. On any failure the frame is
        withheld entirely — an unredacted frame is never emitted."""
        top = max(motion.per_clip, key=lambda c: c.violence_score)
        clip = next((c for c in clip_set.clips if c.clip_id == top.clip_id), None)
        clip_detections = next(
            (c for c in detections.per_clip if c.clip_id == top.clip_id), None
        )
        if clip is None or not clip.keyframes:
            ctx.case.note("report: no keyframes available to redact")
            return [], 0

        all_detections = list(clip_detections.detections) if clip_detections else []
        out_dir = case_dir / "redacted"
        written: list[str] = []
        faces = 0

        for keyframe in clip.keyframes:
            src = Path(keyframe.path)
            if not src.is_absolute():
                # fixture keyframes are stored repo-relative so they stay portable
                src = (Path(ctx.settings.fixtures_dir).parent / keyframe.path).resolve()
            frame_detections = self._detections_for_frame(all_detections, keyframe.frame_index)
            dest = out_dir / f"redacted_{src.name}"
            try:
                count = ctx.tool(
                    self.name, "blur.redact_frame", blur.redact_frame,
                    src,
                    detections=[d.model_dump(mode="json") for d in frame_detections],
                    out_path=dest,
                )
            except Exception as exc:  # noqa: BLE001
                # privacy failure: withhold the frame, record why, do not re-raise
                ctx.case.note(
                    f"report: withheld keyframe {src.name} because redaction failed "
                    f"({type(exc).__name__}: {exc})"
                )
                continue
            faces += count
            written.append(str(dest))
        return written, faces

    @staticmethod
    def _detections_for_frame(detections, frame_index: int, *, tolerance: int = 4):
        """Detections made on this keyframe's frame, or the nearest sampled frame.

        Detection runs on a strided subset of frames, so an exact index match is
        not guaranteed; the nearest sampled frame within ``tolerance`` is the
        right pairing, and anything further away is treated as no data (which
        makes the caller blur nothing rather than blurring the wrong faces).
        """
        exact = [d for d in detections if d.frame_index == frame_index]
        if exact:
            return exact
        indices = {d.frame_index for d in detections}
        if not indices:
            return []
        nearest = min(indices, key=lambda i: abs(i - frame_index))
        if abs(nearest - frame_index) > tolerance:
            return []
        return [d for d in detections if d.frame_index == nearest]

    # -- rendering ------------------------------------------------------- #

    @staticmethod
    def _render_markdown(*, case_id, clip_set, detections, motion, verdict,
                         verification, tier, redacted, faces, ctx) -> str:
        top = max(motion.per_clip, key=lambda c: c.violence_score)
        privacy = policy_kb.privacy_rules(ctx.settings.policy_dir)

        clip_rows = "\n".join(
            f"| {c.clip_id} | {c.violence_score:.3f} | "
            f"{c.features.contact_frames_ratio:.2f} | {c.features.peak_speed:.2f} | "
            f"{c.features.limb_extension_peak:.2f} |"
            for c in motion.per_clip
        )
        issue_rows = "\n".join(
            f"| {i.severity} | `{i.code}` | {i.detail} |" for i in verification.issues
        ) or "| info | `none` | no issues raised |"
        citation_lines = "\n".join(
            f"{n}. {c.reference}\n   > {c.snippet[:260]}"
            for n, c in enumerate(verdict.citations, start=1)
        ) or "_no literature citations retrieved_"
        frame_lines = "\n".join(f"- `{p}`" for p in redacted) or "_none emitted_"

        return f"""# Incident report — {case_id}

| field | value |
|---|---|
| escalation tier | **{tier}** |
| category | **{verdict.category}** |
| severity | {verdict.severity} / 4 |
| verification | **{verification.verdict}** (confidence {verification.confidence:.2f}) |
| false-positive risk | **{verification.false_positive_risk}** |
| source | `{clip_set.source_path}` |
| evidence basis | ingest=`{clip_set.source}`, vision=`{detections.source}`, scorer=`{motion.scorer}` |
| detector | `{detections.model_name}` at conf {detections.confidence_threshold} |
| re-runs used | {ctx.case.rerun_count} / {ctx.settings.max_reruns} |

## What was detected

{top.rationale}

Peak clip `{top.clip_id}`, t_peak = {top.t_peak}s, over {len(clip_set.clips)} clip(s)
spanning {clip_set.duration_seconds:.1f}s at {clip_set.fps:.1f} fps.

| clip | score | contact ratio | peak speed | limb extension |
|---|---|---|---|---|
{clip_rows}

## Policy interpretation (ContextAgent)

Clauses applied: {', '.join(f'`{r}`' for r in verdict.policy_refs) or '_none_'}
· child context: {verdict.is_child_context}
· LLM used: {verdict.llm_used}

{verdict.reasoning}

## Independent verification (VerifierAgent)

{verification.reasoning}

| severity | code | detail |
|---|---|---|
{issue_rows}

## Privacy handling

Faces redacted: **{faces}** region(s) across {len(redacted)} emitted keyframe(s).
Rules enforced: {', '.join(r['id'] for r in privacy)}.

{frame_lines}

## Literature grounding

{citation_lines}

---
_Generated by the SafeSchool multi-agent system. Automated output; requires human
review before any action is taken._
"""
