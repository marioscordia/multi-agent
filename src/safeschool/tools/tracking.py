"""Lightweight identity association across sampled frames.

The detector reports boxes per frame with no identity. Without identities, the
motion stage has to guess which box in frame N is which box in frame N+1, and
with more than two people in view it guesses wrong: an identity switch looks like
a person teleporting across the frame, which registers as enormous speed and
speed variance. That is a false-alarm generator, and it is exactly the failure
the literature review flagged as under-reported.

A greedy IoU association over consecutive sampled frames is enough here. Clips
are short, the sampling stride is small, and a full tracker (ByteTrack, SORT)
would have to run per frame — which would both slow the pipeline and skew this
agent's share of tool calls. Boxes that cannot be matched start a new identity.
"""

from __future__ import annotations

from typing import Any

#: minimum IoU for two boxes in consecutive sampled frames to be the same person
IOU_THRESHOLD = 0.25


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def assign_ids(frames: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """Attach a ``track_id`` to each person detection, in place-safe fashion.

    ``frames`` is the per-frame list of detections, in temporal order. Returns the
    same structure with ``track_id`` populated for every person.
    """
    next_id = 1
    previous: list[tuple[int, tuple[float, ...]]] = []   # (track_id, bbox)

    for detections in frames:
        persons = [d for d in detections if d.get("cls") == "person"]
        current: list[tuple[int, tuple[float, ...]]] = []

        # score every (previous, current) pair, then take them greedily best-first
        candidates = sorted(
            (
                (_iou(prev_box, tuple(d["bbox"])), prev_id, index)
                for prev_id, prev_box in previous
                for index, d in enumerate(persons)
            ),
            key=lambda triple: -triple[0],
        )
        claimed_detections: set[int] = set()
        claimed_tracks: set[int] = set()
        for iou, prev_id, index in candidates:
            if iou < IOU_THRESHOLD:
                break
            if index in claimed_detections or prev_id in claimed_tracks:
                continue
            persons[index]["track_id"] = prev_id
            claimed_detections.add(index)
            claimed_tracks.add(prev_id)

        for index, d in enumerate(persons):
            if index not in claimed_detections:
                d["track_id"] = next_id
                next_id += 1
            current.append((d["track_id"], tuple(d["bbox"])))

        previous = current
    return frames
