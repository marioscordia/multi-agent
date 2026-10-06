"""The inter-agent contract: payloads must survive a JSON round trip and
must reject malformed input at the boundary rather than downstream."""

import pytest
from pydantic import ValidationError

from safeschool.messages import (
    AgentMessage,
    CaseRequest,
    Clip,
    ClipSet,
    ContextVerdict,
    Detection,
    Keyframe,
    MessageType,
)


def test_envelope_round_trip():
    request = CaseRequest(video_path="a.mp4", camera_id="cam-1")
    envelope = AgentMessage.wrap(
        case_id=request.case_id, sender="OrchestratorAgent", recipient="IngestAgent",
        type=MessageType.CASE_REQUEST, payload=request,
    )
    restored = AgentMessage.model_validate_json(envelope.model_dump_json())
    assert restored.payload["video_path"] == "a.mp4"
    assert CaseRequest.model_validate(restored.payload) == request


def test_clip_set_round_trip():
    clip_set = ClipSet(
        case_id="c1", source_path="x.mp4", fps=25.0, duration_seconds=4.0,
        width=640, height=360,
        clips=[Clip(clip_id="c1-c00", index=0, t_start=0.0, t_end=2.0, frame_count=50,
                    keyframes=[Keyframe(path="a.jpg", frame_index=12, t=0.5)])],
    )
    assert ClipSet.model_validate(clip_set.model_dump(mode="json")) == clip_set


def test_unknown_field_is_rejected():
    """Payloads are strict: a typo'd field fails at the boundary, loudly."""
    with pytest.raises(ValidationError):
        CaseRequest(video_path="a.mp4", camrea_id="typo")


def test_detection_confidence_is_bounded():
    with pytest.raises(ValidationError):
        Detection(frame_index=0, t=0.0, cls="person", confidence=1.4, bbox=(0, 0, 1, 1))


def test_severity_is_bounded():
    with pytest.raises(ValidationError):
        ContextVerdict(
            case_id="c1", category="physical_violence", severity=9,
            is_child_context=True, reasoning="x",
        )
