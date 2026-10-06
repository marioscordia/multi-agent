"""YOLO inference tool (ultralytics). Used only by the VisionDetectorAgent.

Weights are downloaded by ultralytics on first use into ``data/weights``. When
ultralytics, the weights or the video are unavailable, the agent falls back to
recorded fixtures instead of failing the run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..runtime import SafeSchoolError

# COCO classes that matter for school safety. "person" drives the motion stage;
# the rest are treated as weapon-like objects that escalate severity.
PERSON_CLASS = "person"
WEAPON_LIKE = {"knife", "scissors", "baseball bat", "bottle"}

_MODEL_CACHE: dict[str, Any] = {}


class DetectorUnavailable(SafeSchoolError):
    """Raised when inference cannot run; the agent may fall back to fixtures."""


def ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return True


def load_model(weights: str, weights_dir: Path):
    """Load (and memoize) a YOLO model. Weights are cached on disk."""
    if weights in _MODEL_CACHE:
        return _MODEL_CACHE[weights]
    if not ultralytics_available():
        raise DetectorUnavailable(
            "ultralytics is not installed (pip install -r requirements-cv.txt)"
        )
    weights_dir.mkdir(parents=True, exist_ok=True)
    local = weights_dir / weights
    try:
        from ultralytics import YOLO

        # keep ultralytics' auto-download inside our data dir
        os.environ.setdefault("YOLO_CONFIG_DIR", str(weights_dir))
        model = YOLO(str(local) if local.exists() else weights)
    except Exception as exc:
        raise DetectorUnavailable(
            f"could not load YOLO weights '{weights}': {type(exc).__name__}: {exc}"
        ) from exc
    _MODEL_CACHE[weights] = model
    return model


def detect_objects(model, frames, *, conf: float) -> list[list[dict[str, Any]]]:
    """Run the object detector over a batch of frames.

    Batching is not only faster than frame-by-frame prediction, it is also what
    keeps this agent's share of tool calls proportionate: one inference call per
    clip rather than one per frame.

    Returns one list of normalized detections per input frame.
    """
    frames = list(frames)
    if not frames:
        return []
    results = model.predict(frames, conf=conf, verbose=False)
    height, width = frames[0].shape[:2]
    batched: list[list[dict[str, Any]]] = []
    for res in results:
        out: list[dict[str, Any]] = []
        names = res.names
        boxes = getattr(res, "boxes", None)
        if boxes is not None:
            for box in boxes:
                cls_name = names[int(box.cls.item())]
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                out.append(
                    {
                        "cls": cls_name,
                        "confidence": float(box.conf.item()),
                        "bbox": (x1 / width, y1 / height, x2 / width, y2 / height),
                        "track_id": int(box.id.item()) if getattr(box, "id", None) is not None else None,
                    }
                )
        batched.append(out)
    return batched


def detect_poses(model, frames, *, conf: float) -> list[list[dict[str, Any]]]:
    """Run the pose model over a batch of frames.

    Returns one list of persons (with normalized 17-keypoint skeletons) per
    input frame.
    """
    frames = list(frames)
    if not frames:
        return []
    results = model.predict(frames, conf=conf, verbose=False)
    height, width = frames[0].shape[:2]
    batched: list[list[dict[str, Any]]] = []
    for res in results:
        out: list[dict[str, Any]] = []
        kps = getattr(res, "keypoints", None)
        boxes = getattr(res, "boxes", None)
        if kps is not None and boxes is not None:
            data = kps.data  # (n, 17, 3)
            for i, box in enumerate(boxes):
                if i >= len(data):
                    break
                points = [
                    (float(x) / width, float(y) / height, float(c))
                    for x, y, c in data[i].tolist()
                ]
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                out.append(
                    {
                        "cls": PERSON_CLASS,
                        "confidence": float(box.conf.item()),
                        "bbox": (x1 / width, y1 / height, x2 / width, y2 / height),
                        "track_id": int(box.id.item()) if getattr(box, "id", None) is not None else None,
                        "keypoints": points,
                    }
                )
        batched.append(out)
    return batched
