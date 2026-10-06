"""Agent 2 — VisionDetectorAgent.

Role: per-frame perception. Runs the pose model (persons + 17 keypoints) and the
object detector (weapon-like objects) over each clip and emits normalized
detections. It makes no judgement about whether anything is violent — that is the
MotionAnalystAgent's job, and keeping the split means a perception failure is
distinguishable in the logs from a reasoning failure.
"""

from __future__ import annotations

from pathlib import Path

from ..messages import (
    AgentMessage,
    ClipDetections,
    Detection,
    DetectionSet,
    ClipSet,
    MessageType,
)
from ..runtime import Agent, Context, SafeSchoolError
from ..tools import fixtures, video, yolo


class VisionDetectorAgent(Agent):
    name = "VisionDetectorAgent"
    stage = "vision"
    role = "Detect persons, poses and weapon-like objects in every clip frame"
    tools = ("yolo.load_model", "yolo.detect_poses", "yolo.detect_objects", "fixtures.load")
    done_criterion = "every clip has a detection record (an empty one is a valid result)"
    input_type = MessageType.CLIP_SET
    output_type = MessageType.DETECTION_SET

    def handle(self, message: AgentMessage, ctx: Context) -> DetectionSet:
        clip_set = self.parse_input(message, ClipSet)

        # a re-run requested by the Verifier arrives with a lowered threshold
        threshold = float(
            message.overrides.get("confidence_threshold")
            or ctx.settings.confidence_threshold
        )

        if clip_set.source == "fixture":
            return self._from_fixture(clip_set, ctx, reason="ingest produced fixture clips")

        try:
            pose_model = ctx.tool(
                self.name,
                "yolo.load_model(pose)",
                yolo.load_model,
                ctx.settings.pose_weights,
                ctx.settings.weights_dir,
            )
            object_model = ctx.tool(
                self.name,
                "yolo.load_model(detect)",
                yolo.load_model,
                ctx.settings.detector_weights,
                ctx.settings.weights_dir,
            )
        except SafeSchoolError as exc:
            ctx.case.note(f"vision fell back to fixtures: {exc}")
            return self._from_fixture(clip_set, ctx, reason=str(exc))

        source = Path(clip_set.source_path)
        per_clip: list[ClipDetections] = []

        for clip in clip_set.clips:
            detections: list[Detection] = []
            per_frame_person_counts: list[int] = []
            weapon_present = False

            sampled = list(
                video.iter_frames(
                    source,
                    t_start=clip.t_start,
                    t_end=clip.t_end,
                    stride=ctx.settings.frame_stride,
                )
            )
            frames = [frame for _, _, frame in sampled]

            # one batched inference call per model per clip, rather than one per
            # frame: faster, and it keeps this agent's call count proportionate
            # to the work it actually represents
            pose_batches = ctx.tool(
                self.name, "yolo.detect_poses", yolo.detect_poses,
                pose_model, frames, conf=threshold,
            )
            object_batches = ctx.tool(
                self.name, "yolo.detect_objects", yolo.detect_objects,
                object_model, frames, conf=threshold,
            )

            for (frame_index, t, _frame), poses, objects in zip(
                sampled, pose_batches, object_batches
            ):
                per_frame_person_counts.append(len(poses))
                for p in poses:
                    detections.append(
                        Detection(
                            frame_index=frame_index, t=round(t, 3), cls="person",
                            confidence=p["confidence"], bbox=p["bbox"],
                            track_id=p.get("track_id"), keypoints=p.get("keypoints", []),
                        )
                    )
                for o in objects:
                    if o["cls"] == yolo.PERSON_CLASS:
                        continue  # persons come from the pose model, with keypoints
                    if o["cls"] not in yolo.WEAPON_LIKE:
                        continue
                    weapon_present = True
                    detections.append(
                        Detection(
                            frame_index=frame_index, t=round(t, 3), cls=o["cls"],
                            confidence=o["confidence"], bbox=o["bbox"],
                            track_id=o.get("track_id"),
                        )
                    )

            per_clip.append(
                ClipDetections(
                    clip_id=clip.clip_id,
                    detections=detections,
                    frames_analyzed=len(sampled),
                    person_count_max=max(per_frame_person_counts or [0]),
                    weapon_present=weapon_present,
                )
            )

        if not per_clip:
            raise SafeSchoolError("no clip produced a detection record")

        return DetectionSet(
            case_id=clip_set.case_id,
            per_clip=per_clip,
            model_name=f"{ctx.settings.pose_weights}+{ctx.settings.detector_weights}",
            source="yolo",
            confidence_threshold=threshold,
        )

    def _from_fixture(self, clip_set: ClipSet, ctx: Context, *, reason: str) -> DetectionSet:
        name = fixtures.resolve_name(ctx.settings.fixtures_dir, clip_set.source_path)
        data = ctx.tool(
            self.name, "fixtures.load", fixtures.load, ctx.settings.fixtures_dir, name
        )
        detection_set = DetectionSet.model_validate(data["detection_set"])
        detection_set.case_id = clip_set.case_id
        detection_set.source = "fixture"
        # rebind fixture clip ids onto this case's clips, in order
        for fixture_clip, real_clip in zip(detection_set.per_clip, clip_set.clips):
            fixture_clip.clip_id = real_clip.clip_id
        ctx.logger.log(
            case_id=clip_set.case_id, agent=self.name, event="system",
            name="fixture_fallback", detail={"fixture": name, "reason": reason[:200]},
        )
        return detection_set
