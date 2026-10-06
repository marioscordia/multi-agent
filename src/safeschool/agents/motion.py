"""Agent 3 — MotionAnalystAgent.

Role: aggregate per-frame detections into clip-level motion features and a
violence score. Numeric work only, no LLM: the output is a feature vector plus a
score, which the ContextAgent and the VerifierAgent both consume.

The score comes from an interpretable weighted model
(:mod:`safeschool.tools.features`) rather than a trained black box, so the
rationale string can name the features that drove it.
"""

from __future__ import annotations

from ..messages import (
    AgentMessage,
    ClipMotion,
    DetectionSet,
    MessageType,
    MotionAssessment,
    MotionFeatures,
)
from ..runtime import Agent, Context, SafeSchoolError
from ..tools import features as feat
from ..tools import tracking


class MotionAnalystAgent(Agent):
    name = "MotionAnalystAgent"
    stage = "motion"
    role = (
        "Associate detections into tracks, then compute interpretable motion "
        "features and a violence score per clip"
    )
    tools = ("tracking.assign_ids", "features.extract_features", "features.score")
    done_criterion = "every clip carries a score in [0,1] with its full feature vector"
    input_type = MessageType.DETECTION_SET
    output_type = MessageType.MOTION_ASSESSMENT

    def handle(self, message: AgentMessage, ctx: Context) -> MotionAssessment:
        detection_set = self.parse_input(message, DetectionSet)
        if not detection_set.per_clip:
            raise SafeSchoolError("motion analysis received an empty detection set")

        fps_hint = 25.0
        clip_motions: list[ClipMotion] = []

        for clip in detection_set.per_clip:
            raw = [d.model_dump(mode="json") for d in clip.detections]

            # The detector reports boxes without identities. Associating them
            # into tracks is temporal reasoning, so it belongs to this stage, not
            # to perception — and without it an identity switch between frames
            # reads as a person teleporting, i.e. as violent motion.
            if any(d.get("track_id") is None for d in raw if d["cls"] == "person"):
                by_frame: dict[int, list[dict]] = {}
                for d in raw:
                    by_frame.setdefault(d["frame_index"], []).append(d)
                ordered = [by_frame[k] for k in sorted(by_frame)]
                ctx.tool(self.name, "tracking.assign_ids", tracking.assign_ids, ordered)

            values = ctx.tool(
                self.name, "features.extract_features",
                feat.extract_features, raw, fps=fps_hint,
            )
            violence_score, rationale = ctx.tool(
                self.name, "features.score", feat.score, values
            )
            t_peak = None
            persons = [d for d in clip.detections if d.cls == "person"]
            if persons:
                # report the densest moment as the peak, a useful operator hint
                by_frame: dict[int, list[float]] = {}
                for d in persons:
                    by_frame.setdefault(d.frame_index, []).append(d.t)
                busiest = max(by_frame.items(), key=lambda kv: len(kv[1]))
                t_peak = round(busiest[1][0], 3)

            clip_motions.append(
                ClipMotion(
                    clip_id=clip.clip_id,
                    violence_score=violence_score,
                    features=MotionFeatures(**values),
                    rationale=rationale,
                    t_peak=t_peak,
                )
            )

        top = max(clip_motions, key=lambda c: c.violence_score)
        return MotionAssessment(
            case_id=detection_set.case_id,
            per_clip=clip_motions,
            top_clip_id=top.clip_id,
            top_score=top.violence_score,
            scorer=f"interpretable-v1({detection_set.source})",
        )
