"""Agent 6 — PrivacyReportAgent.

Role: the only component permitted to touch identifiable imagery, and the only
one that emits anything outward. It redacts faces, writes the incident report,
and escalates according to the policy tier.

Concentrating both the redaction and the outbound side effect in one agent is the
point: there is exactly one place to audit for a privacy leak. The stage-1 review
found that 2 of 50 published systems addressed privacy at all, which is the gap
this agent exists to close.

The plain-language summary for staff is written by the language model, because
turning a technical verdict into a clear message for a teacher is a language
task. Two limits keep that safe:

* the model only writes text. The escalation tier is decided by the policy
  before the model is called, and the model cannot change who is alerted;
* the model receives the verdict, scores and policy text only. Frames, crops
  and file paths never leave this agent (privacy rule PR-2).

If the model fails or declines, a fixed template summary is used instead, so a
report is never lost because the summary could not be written.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..llm import LLMError
from ..messages import (
    AgentMessage,
    ContextVerdict,
    IncidentReport,
    MessageType,
    Verification,
)
from ..runtime import Agent, Context
from ..tools import blur, notifier, policy_kb

_SYSTEM = (
    "You write the message a school duty officer reads when an automated video "
    "monitoring system flags or dismisses an incident. Write in plain language "
    "for a teacher, not an engineer: no feature names, scores or jargon. Use only "
    "the facts you are given and never invent names, ages, places or details. "
    "Say how confident the system is. Mention the evidence source only when it "
    "is recorded test data. Be brief: the summary is at most 3 short sentences "
    "(under 70 words), and the action is one sentence (under 35 words). When an "
    "alert is sent, the action says what to do and that a person must check the "
    "footage first; when no alert is sent, the action says no action is needed. "
    "You do not decide the alert level; it is given to you and you must not "
    "suggest a different one."
)

_SCHEMA = """{
  "summary": "<at most 3 short plain sentences: what the system saw, how sure it is>",
  "staff_action": "<one sentence: what the reader should do now>"
}"""

#: hard caps, far above what the prompt asks for; text is cut at a sentence end
MAX_SUMMARY_CHARS = 900
MAX_ACTION_CHARS = 400


def _clip_to_sentence(text: str, limit: int) -> str:
    """Shorten to at most ``limit`` characters without cutting a sentence."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > 0 else cut.rstrip() + "…"

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "staff_action": {"type": "string"},
    },
    "required": ["summary", "staff_action"],
    "additionalProperties": False,
}

#: what each tier means for the reader, so the action sentence matches it
_TIER_MEANING = {
    "none": "no alert is sent; the case is closed after an automated check rejected the alert",
    "log": "no alert is sent; the case is only recorded",
    "notify_staff": "the class teacher or pastoral lead is asked to follow up",
    "notify_admin": "the duty administrator is notified to attend",
    "emergency": "an emergency response is triggered",
}


class PrivacyReportAgent(Agent):
    name = "PrivacyReportAgent"
    stage = "report"
    role = "Redact faces, write the incident report, escalate per the policy tier"
    tools = (
        "blur.redact_frame",
        "policy_kb.escalation_tier",
        "llm.complete_json",
        "report.render",
        "notifier.notify",
    )
    done_criterion = (
        "a markdown and a JSON report exist and reference only redacted keyframes"
    )
    input_type = MessageType.VERIFICATION
    output_type = MessageType.INCIDENT_REPORT

    def __init__(self, llm) -> None:
        self.llm = llm

    def handle(self, message: AgentMessage, ctx: Context) -> IncidentReport:
        verification = self.parse_input(message, Verification)
        verdict: ContextVerdict = ctx.case.require("context")  # type: ignore[assignment]
        motion = ctx.case.require("motion")
        clip_set = ctx.case.require("ingest")
        detections = ctx.case.require("vision")

        case_dir = ctx.settings.cases_dir / verification.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # --- escalation tier ------------------------------------------- #
        benign = verdict.category in {"no_violation", "rough_play_benign"}
        if verification.verdict == "reject" and benign:
            # the Verifier disputes a dismissal: a person should look, but it is
            # not confirmed enough to trigger an emergency response
            tier = "notify_staff"
        elif verification.verdict == "reject":
            tier = "none"
        elif benign:
            tier = "log"
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

        # --- staff summary (written by the model, tier already fixed) --- #
        top = max(motion.per_clip, key=lambda c: c.violence_score)  # type: ignore[attr-defined]
        summary, staff_action, summary_source = self._write_summary(
            verdict=verdict, verification=verification, top=top, tier=tier,
            clip_set=clip_set, detections=detections, faces=faces, ctx=ctx,
        )

        # --- reports ---------------------------------------------------- #
        markdown = ctx.tool(
            self.name, "report.render", self._render_markdown,
            case_id=verification.case_id, clip_set=clip_set, detections=detections,
            motion=motion, verdict=verdict, verification=verification,
            tier=tier, redacted=redacted, faces=faces, ctx=ctx,
            summary=summary, staff_action=staff_action, summary_source=summary_source,
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
            staff_action=staff_action,
            summary_source=summary_source,
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
                category=verdict.category, severity=verdict.severity,
                summary=f"{summary} Action: {staff_action}" if staff_action else summary,
            )
        return report

    # -- staff summary --------------------------------------------------- #

    def _write_summary(self, *, verdict, verification, top, tier, clip_set,
                       detections, faces, ctx: Context) -> tuple[str, str, str]:
        """Ask the model for the staff-facing text. Returns (summary, action, source).

        Only text and numbers go into the prompt: no frame, crop or path."""
        clause = policy_kb.clause_for_category(ctx.settings.policy_dir, verdict.category)
        fallback = self._template_summary(verdict, verification, top, tier, clause)

        issues = "\n".join(
            f"  - {i.detail}" for i in verification.issues
            if i.severity in {"warning", "blocking"}
        ) or "  (none)"
        from_test_data = "fixture" in {getattr(clip_set, "source", ""), detections.source}
        prompt = f"""Write the staff message for this case.

ALERT LEVEL (already decided by school policy, do not change it): {tier}
  meaning: {_TIER_MEANING.get(tier, tier)}
WHAT THE SYSTEM CONCLUDED: {verdict.category.replace('_', ' ')}, severity {verdict.severity} of 4
  policy clause: {clause['title'] if clause else 'none'}
  policy response for this clause: {clause['response'] if clause else 'none'}
WHY (analysis by the classification agent):
  {verdict.reasoning}
INDEPENDENT CHECK: {verification.verdict}, confidence {verification.confidence:.2f},
  false-alarm risk {verification.false_positive_risk}
  {verification.reasoning}
CONCERNS RAISED BY THE CHECK:
{issues}
WHEN: around {top.t_peak if top.t_peak is not None else 'an unknown time'} s into the clip
PRIVACY: {faces} face region(s) were blurred before this report was written
EVIDENCE SOURCE: {'recorded test data, not a live camera' if from_test_data else 'live camera footage'}

Remember: at most 3 short sentences for the summary, one sentence for the action."""

        try:
            result = self.llm.complete_json(
                agent=self.name, ctx=ctx, system=_SYSTEM, prompt=prompt,
                schema_hint=_SCHEMA, json_schema=_JSON_SCHEMA,
                mock=fallback, purpose="summarize",
            )
        except LLMError as exc:
            # a summary that could not be written must never cost the report
            ctx.case.note(f"report: summary model failed, template used ({exc})")
            return fallback["summary"], fallback["staff_action"], "template"

        written_by_model = self.llm.provider != "mock" and not result.get("_llm_fallback")
        summary = _clip_to_sentence(
            str(result.get("summary") or fallback["summary"]), MAX_SUMMARY_CHARS
        )
        action = _clip_to_sentence(
            str(result.get("staff_action") or fallback["staff_action"]), MAX_ACTION_CHARS
        )
        return summary, action, "llm" if written_by_model else "template"

    @staticmethod
    def _template_summary(verdict, verification, top, tier, clause) -> dict[str, str]:
        """The fixed-format summary: the mock answer and the failure fallback."""
        summary = (
            f"{verdict.category.replace('_', ' ')} (severity {verdict.severity}) on "
            f"clip {top.clip_id} at t={top.t_peak if top.t_peak is not None else top.clip_id}s, "
            f"motion score {top.violence_score:.2f}, verification {verification.verdict} "
            f"with {verification.false_positive_risk} false-positive risk"
        )
        if tier in {"notify_staff", "notify_admin", "emergency"} and clause:
            action = clause["response"] + " Check the footage before acting."
        else:
            action = "No action needed; the case is recorded for audit."
        return {"summary": summary, "staff_action": action}

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
                         verification, tier, redacted, faces, ctx,
                         summary="", staff_action="", summary_source="template") -> str:
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

## Summary for staff

{summary}

**What to do:** {staff_action or '_no action stated_'}

_Written by: {'the language model (PrivacyReportAgent)' if summary_source == 'llm' else 'fixed template'}._

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
