"""Face redaction. Used only by the PrivacyReportAgent.

Privacy rule PR-1: no frame leaves the pipeline with an identifiable face. Two
redaction strategies, in order of preference:

1. pose-driven — blur a disc around the head keypoints the pose model already
   produced (no extra model, and it works on profiles and partial occlusion);
2. detector-driven — blur the top third of every person bounding box.

Strategy 2 is the fallback, and is intentionally conservative: over-blurring is
an acceptable failure, under-blurring is not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

# COCO head keypoints: nose, eyes, ears
HEAD_KEYPOINTS = (0, 1, 2, 3, 4)


def redact_frame(
    image_path: Path,
    *,
    detections: Sequence[dict[str, Any]],
    out_path: Path,
    mosaic_blocks: int = 4,
) -> int:
    """Blur faces in one keyframe. Returns the number of regions redacted.

    Raises ImportError if OpenCV is absent — the caller treats that as a hard
    privacy failure and withholds the frame rather than emitting it unredacted.
    """
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"cannot read keyframe {image_path}")
    height, width = image.shape[:2]
    regions = 0

    for det in detections:
        if det.get("cls") != "person":
            continue
        keypoints = det.get("keypoints") or []
        box = det.get("bbox")
        head_pts = [
            (kx * width, ky * height)
            for i, (kx, ky, kc) in enumerate(keypoints)
            if i in HEAD_KEYPOINTS and kc > 0.25
        ]
        if head_pts:
            cx = sum(p[0] for p in head_pts) / len(head_pts)
            cy = sum(p[1] for p in head_pts) / len(head_pts)
            if box:
                radius = max((box[3] - box[1]) * height * 0.22, 14.0)
            else:
                radius = max(height * 0.07, 14.0)
            x1, y1 = int(max(cx - radius, 0)), int(max(cy - radius, 0))
            x2, y2 = int(min(cx + radius, width)), int(min(cy + radius, height))
        elif box:
            bx1, by1, bx2, by2 = box
            x1 = int(max(bx1 * width, 0))
            x2 = int(min(bx2 * width, width))
            y1 = int(max(by1 * height, 0))
            # top third of the body box is where the head is
            y2 = int(min((by1 + (by2 - by1) * 0.33) * height, height))
        else:
            continue

        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = image[y1:y2, x1:x2]
        # mosaic first, then blur the mosaic: downsampling to a handful of blocks
        # destroys the facial detail outright, and the blur removes the block
        # edges that would otherwise still carry a little structure
        small = cv2.resize(
            roi, (mosaic_blocks, mosaic_blocks), interpolation=cv2.INTER_AREA
        )
        mosaic = cv2.resize(
            small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        k = max(5, int(min(roi.shape[:2]) * 0.30) | 1)
        image[y1:y2, x1:x2] = cv2.GaussianBlur(mosaic, (k, k), 0)
        regions += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)
    return regions
