"""Agent 1 — IngestAgent.

Role: turn one video file into a set of analyzable clips with keyframes. It owns
all decoding; no other agent touches a video container.

Deliberately contains no LLM call: segmenting a file is deterministic work, and
spending tokens on it would both waste money and make a reproducible step
non-reproducible.
"""

from __future__ import annotations

from pathlib import Path

from ..messages import AgentMessage, CaseRequest, Clip, ClipSet, Keyframe, MessageType
from ..runtime import Agent, Context, SafeSchoolError
from ..tools import fixtures, video


class IngestAgent(Agent):
    name = "IngestAgent"
    stage = "ingest"
    role = "Decode the source video, segment it into fixed-length clips, extract keyframes"
    tools = ("video.probe", "video.extract_keyframes", "fixtures.load")
    done_criterion = "at least one clip with a valid time interval is registered in the case file"
    input_type = MessageType.CASE_REQUEST
    output_type = MessageType.CLIP_SET

    def handle(self, message: AgentMessage, ctx: Context) -> ClipSet:
        request = self.parse_input(message, CaseRequest)
        path = Path(request.video_path)

        try:
            meta = ctx.tool(self.name, "video.probe", video.probe, path)
        except SafeSchoolError as exc:
            if not request.allow_fixture_fallback:
                raise
            ctx.case.note(f"ingest fell back to fixtures: {exc}")
            return self._from_fixture(request, ctx, reason=str(exc))

        clip_seconds = ctx.settings.clip_seconds
        duration = float(meta["duration_seconds"])
        keyframe_dir = ctx.settings.cases_dir / request.case_id / "keyframes"

        clips: list[Clip] = []
        index = 0
        t = 0.0
        while t < duration and index < ctx.settings.max_clips:
            t_end = min(t + clip_seconds, duration)
            if t_end - t < 0.3:  # trailing sliver, not worth a clip
                break
            keyframes = ctx.tool(
                self.name,
                "video.extract_keyframes",
                video.extract_keyframes,
                path,
                t_start=t,
                t_end=t_end,
                out_dir=keyframe_dir,
            )
            clips.append(
                Clip(
                    clip_id=f"{request.case_id}-c{index:02d}",
                    index=index,
                    t_start=round(t, 3),
                    t_end=round(t_end, 3),
                    frame_count=int((t_end - t) * float(meta["fps"])),
                    keyframes=[Keyframe(**kf) for kf in keyframes],
                )
            )
            index += 1
            t = t_end

        if not clips:
            raise SafeSchoolError(
                f"{path.name} decoded to {duration:.2f}s, which yields no analyzable clip"
            )

        return ClipSet(
            case_id=request.case_id,
            source_path=str(path),
            fps=float(meta["fps"]),
            duration_seconds=round(duration, 3),
            width=int(meta["width"]),
            height=int(meta["height"]),
            clips=clips,
            source="decoded",
        )

    def _from_fixture(self, request: CaseRequest, ctx: Context, *, reason: str) -> ClipSet:
        name = fixtures.resolve_name(ctx.settings.fixtures_dir, request.video_path)
        data = ctx.tool(
            self.name, "fixtures.load", fixtures.load, ctx.settings.fixtures_dir, name
        )
        clip_set = ClipSet.model_validate(data["clip_set"])
        # the fixture carries its own ids; rebind them to this case
        clip_set.case_id = request.case_id
        clip_set.source = "fixture"
        for i, clip in enumerate(clip_set.clips):
            clip.clip_id = f"{request.case_id}-c{i:02d}"
        ctx.logger.log(
            case_id=request.case_id,
            agent=self.name,
            event="system",
            name="fixture_fallback",
            detail={"fixture": name, "reason": reason[:200]},
        )
        return clip_set
