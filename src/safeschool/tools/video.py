"""Video decoding tool (OpenCV). Used only by the IngestAgent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..runtime import SafeSchoolError


class VideoUnavailable(SafeSchoolError):
    """Raised when the file cannot be decoded; the agent may fall back to fixtures."""


def cv2_available() -> bool:
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return True


def probe(path: Path) -> dict[str, Any]:
    """Read container metadata without decoding the whole file."""
    if not cv2_available():
        raise VideoUnavailable("OpenCV is not installed (pip install -r requirements-cv.txt)")
    import cv2

    path = Path(path)
    if not path.exists():
        raise VideoUnavailable(f"video file not found: {path}")
    if path.stat().st_size == 0:
        raise VideoUnavailable(f"video file is empty: {path.name}")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoUnavailable(
                f"cannot decode {path.name}: unsupported codec or corrupt file"
            )
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frames <= 0 or width <= 0 or height <= 0:
            raise VideoUnavailable(
                f"{path.name} reports {frames} frames at {width}x{height}; nothing to analyze"
            )
        if fps <= 0:
            fps = 25.0  # some containers omit fps; assume a common surveillance rate
        return {
            "fps": fps,
            "frame_count": frames,
            "width": width,
            "height": height,
            "duration_seconds": frames / fps,
        }
    finally:
        cap.release()


def extract_keyframes(
    path: Path, *, t_start: float, t_end: float, out_dir: Path, max_frames: int = 3
) -> list[dict[str, Any]]:
    """Write up to ``max_frames`` evenly spaced JPEGs from the interval.

    Each entry carries the frame index and timestamp it was taken at, so a later
    stage can pair the still with the detections made on that same frame.
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    written: list[dict[str, Any]] = []
    try:
        if not cap.isOpened():
            raise VideoUnavailable(f"cannot open {Path(path).name} for keyframe extraction")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        span = max(t_end - t_start, 1e-6)
        for i in range(max_frames):
            t = t_start + span * (i + 0.5) / max_frames
            frame_index = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            dest = out_dir / f"kf_{frame_index:06d}.jpg"
            cv2.imwrite(str(dest), frame)
            written.append(
                {"path": str(dest), "frame_index": frame_index, "t": round(t, 3)}
            )
    finally:
        cap.release()
    return written


def iter_frames(path: Path, *, t_start: float, t_end: float, stride: int = 3):
    """Yield ``(frame_index, timestamp, frame)`` for the interval."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoUnavailable(f"cannot open {Path(path).name}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        start_idx = int(t_start * fps)
        end_idx = int(t_end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        idx = start_idx
        while idx < end_idx:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start_idx) % stride == 0:
                yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()
