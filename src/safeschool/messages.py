"""Structured messages exchanged between agents.

Every inter-agent payload is a pydantic model. Agents never pass raw dicts or
positional tuples around: the receiving agent validates its input against the
schema declared here, so a contract violation fails loudly at the boundary
instead of silently corrupting a later stage.

All payloads travel inside an :class:`AgentMessage` envelope, which is what gets
persisted and logged as JSON.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Payload(BaseModel):
    """Base for every payload: strict, so unexpected fields are rejected."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# envelope
# --------------------------------------------------------------------------- #

class MessageType(str, Enum):
    CASE_REQUEST = "case_request"
    PLAN = "plan"
    CLIP_SET = "clip_set"
    DETECTION_SET = "detection_set"
    MOTION_ASSESSMENT = "motion_assessment"
    CONTEXT_VERDICT = "context_verdict"
    VERIFICATION = "verification"
    INCIDENT_REPORT = "incident_report"
    ERROR = "error"


class AgentMessage(BaseModel):
    """Transport envelope. This is the only thing that crosses an agent boundary."""

    model_config = ConfigDict(extra="forbid")

    msg_id: str = Field(default_factory=lambda: _new_id("msg"))
    case_id: str
    sender: str
    recipient: str
    type: MessageType
    payload: dict[str, Any]
    created_at: datetime = Field(default_factory=_utc_now)
    # set when this message is a retry/re-run of an earlier attempt
    attempt: int = 1
    causation_id: str | None = None
    #: Control metadata for the receiving agent (e.g. a re-run's lowered
    #: threshold). Kept on the envelope rather than in the payload so that domain
    #: payloads stay exactly what their schema says they are — they are strict,
    #: and smuggling a control field inside one fails validation at the boundary.
    overrides: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def wrap(
        cls,
        *,
        case_id: str,
        sender: str,
        recipient: str,
        type: MessageType,
        payload: Payload,
        attempt: int = 1,
        causation_id: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "AgentMessage":
        return cls(
            case_id=case_id,
            sender=sender,
            recipient=recipient,
            type=type,
            payload=payload.model_dump(mode="json"),
            attempt=attempt,
            causation_id=causation_id,
            overrides=dict(overrides or {}),
        )


# --------------------------------------------------------------------------- #
# stage 0 — orchestrator
# --------------------------------------------------------------------------- #

class CaseRequest(Payload):
    """What the operator asks the system to look at."""

    case_id: str = Field(default_factory=lambda: _new_id("case"))
    video_path: str
    camera_id: str = "cam-unknown"
    location: str = "unspecified"
    # operator hint; the system must not trust it as ground truth
    notes: str | None = None
    # fixture mode is forced when CV deps / weights / video are unavailable
    allow_fixture_fallback: bool = True


class PlanStep(Payload):
    stage: str
    agent: str
    depends_on: list[str] = Field(default_factory=list)
    reason: str


class Plan(Payload):
    case_id: str
    steps: list[PlanStep]
    max_steps: int
    deadline_seconds: int


# --------------------------------------------------------------------------- #
# stage 1 — ingest
# --------------------------------------------------------------------------- #

class Keyframe(Payload):
    """One extracted still, tagged with where it came from.

    The frame index matters: the redaction stage must blur the faces that are in
    *this* frame, not every face seen anywhere in the clip.
    """

    path: str
    frame_index: int
    t: float


class Clip(Payload):
    clip_id: str
    index: int
    t_start: float
    t_end: float
    frame_count: int
    keyframes: list[Keyframe] = Field(default_factory=list)


class ClipSet(Payload):
    case_id: str
    source_path: str
    fps: float
    duration_seconds: float
    width: int
    height: int
    clips: list[Clip]
    source: Literal["decoded", "fixture"] = "decoded"


# --------------------------------------------------------------------------- #
# stage 2 — vision
# --------------------------------------------------------------------------- #

class Detection(Payload):
    """One detected object in one frame."""

    frame_index: int
    t: float
    cls: str                       # person | knife | gun | ...
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]   # xyxy, normalized 0..1
    track_id: int | None = None
    # 17 COCO keypoints as (x, y, conf), normalized; empty when pose is unavailable
    keypoints: list[tuple[float, float, float]] = Field(default_factory=list)


class ClipDetections(Payload):
    clip_id: str
    detections: list[Detection]
    frames_analyzed: int
    person_count_max: int = 0
    weapon_present: bool = False


class DetectionSet(Payload):
    case_id: str
    per_clip: list[ClipDetections]
    model_name: str
    source: Literal["yolo", "fixture"] = "yolo"
    confidence_threshold: float = 0.25


# --------------------------------------------------------------------------- #
# stage 3 — motion
# --------------------------------------------------------------------------- #

class MotionFeatures(Payload):
    """Interpretable features — deliberately not a black box, so the Verifier
    and the written report can both reason about *why* a score is what it is."""

    mean_speed: float = 0.0
    peak_speed: float = 0.0
    speed_variance: float = 0.0
    min_pair_distance: float = 1.0
    median_pair_distance: float = 1.0
    contact_frames_ratio: float = 0.0
    limb_extension_peak: float = 0.0
    pose_asymmetry: float = 0.0
    crowding: float = 0.0


class ClipMotion(Payload):
    clip_id: str
    violence_score: float = Field(ge=0.0, le=1.0)
    features: MotionFeatures
    rationale: str
    t_peak: float | None = None


class MotionAssessment(Payload):
    case_id: str
    per_clip: list[ClipMotion]
    top_clip_id: str | None = None
    top_score: float = 0.0
    scorer: str = "interpretable-v1"


# --------------------------------------------------------------------------- #
# stage 4 — context
# --------------------------------------------------------------------------- #

class Citation(Payload):
    source: str            # file or CSV row the claim came from
    reference: str         # human-readable citation
    snippet: str


class ContextVerdict(Payload):
    case_id: str
    category: Literal[
        "physical_violence",
        "weapon_threat",
        "adult_on_child_abuse",
        "bullying_nonphysical",
        "rough_play_benign",
        "no_violation",
    ]
    severity: int = Field(ge=1, le=4)   # 1 log-only .. 4 emergency
    is_child_context: bool
    policy_refs: list[str] = Field(default_factory=list)
    reasoning: str
    citations: list[Citation] = Field(default_factory=list)
    llm_used: bool = True


# --------------------------------------------------------------------------- #
# stage 5 — verification
# --------------------------------------------------------------------------- #

class Issue(Payload):
    code: str
    detail: str
    severity: Literal["info", "warning", "blocking"] = "warning"


class RerunHint(Payload):
    stage: str
    parameter: str
    value: float


class Verification(Payload):
    case_id: str
    verdict: Literal["accept", "reject", "rerun"]
    confidence: float = Field(ge=0.0, le=1.0)
    false_positive_risk: Literal["low", "medium", "high"]
    issues: list[Issue] = Field(default_factory=list)
    rerun_hint: RerunHint | None = None
    reasoning: str = ""
    llm_used: bool = True


# --------------------------------------------------------------------------- #
# stage 6 — privacy + report
# --------------------------------------------------------------------------- #

class IncidentReport(Payload):
    case_id: str
    created_at: datetime = Field(default_factory=_utc_now)
    escalation_tier: Literal["none", "log", "notify_staff", "notify_admin", "emergency"]
    category: str
    severity: int
    confidence: float
    summary: str
    redacted_keyframes: list[str] = Field(default_factory=list)
    faces_redacted: int = 0
    report_markdown_path: str | None = None
    report_json_path: str | None = None
    notified_channel: str | None = None


class ErrorPayload(Payload):
    case_id: str
    stage: str
    agent: str
    error_type: str
    message: str
    attempts: int = 1
    recoverable: bool = True
