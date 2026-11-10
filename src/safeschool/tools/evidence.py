"""Evidence summaries for the two reasoning agents (Context and Verifier).

The clip-level features say *that* there was fast contact; they do not say who
did what to whom. A reviewer needs that to tell a one-sided attack from mutual
play, and to judge whether a detected knife is actually in somebody's hand.
This module turns raw detections into those facts, as plain numbers and text,
so no image ever reaches a language model (privacy rule PR-2).
"""

from __future__ import annotations

import math
from typing import Any

from .features import _limb_extension

#: a weapon box this close to a wrist (normalized frame units) counts as held
HELD_DISTANCE = 0.06

_WRISTS = (9, 10)   # COCO-17 left_wrist, right_wrist

#: How to read the clip features, measured on the bundled real CCTV clips.
#: Given to both agents so they judge the numbers against a known baseline
#: instead of guessing what "0.2" means.
FEATURE_GUIDE = """HOW TO READ THE MOTION FEATURES
  - mean_speed / peak_speed: body-lengths per second divided by 2 (1.0 = 2 body-lengths/s or more).
    Baseline from real CCTV: people walking a corridor measure mean ~0.20, peak ~0.35;
    a person standing still ~0.01. Values well above walking with close contact mean
    sudden lunges, shoves or blows.
  - speed_variance: how erratic the motion is. Walking ~0.10-0.13.
  - contact_frames_ratio: share of frames with two people within contact distance.
  - limb_extension_peak: 0 = arms at rest, 1 = an arm thrown fully out from the body (strike or grab).
  - pose_asymmetry: left/right imbalance WITHIN each body, averaged over everyone. It is
    NOT a measure of which person has the advantage; use the per-person summary for that.
  - crowding: 0 = two people or fewer; higher means a dense crowd."""


def _wrist_points(det: Any) -> list[tuple[float, float]]:
    kps = det.keypoints or []
    if len(kps) < 17:
        return []
    return [(kps[i][0], kps[i][1]) for i in _WRISTS if kps[i][2] > 0.3]


def summarize(clip_detections: Any, *, fps: float) -> dict[str, Any]:
    """Per-person behaviour and weapon facts for one clip."""
    if clip_detections is None:
        return {"persons": [], "weapon": None}

    persons = [d for d in clip_detections.detections if d.cls == "person"]
    objects = [d for d in clip_detections.detections if d.cls != "person"]

    # --- per person (per track) ------------------------------------------- #
    by_track: dict[Any, list[Any]] = {}
    for d in persons:
        if d.track_id is not None:
            by_track.setdefault(d.track_id, []).append(d)

    summaries = []
    for track_id, dets in sorted(by_track.items(), key=lambda kv: kv[0]):
        dets = sorted(dets, key=lambda d: d.frame_index)
        heights = [d.bbox[3] - d.bbox[1] for d in dets]
        extensions = [_limb_extension(list(d.keypoints)) for d in dets]
        speeds = []
        for prev, cur in zip(dets, dets[1:]):
            dt = max(cur.t - prev.t, 1.0 / max(fps, 1.0))
            px, py = (prev.bbox[0] + prev.bbox[2]) / 2, (prev.bbox[1] + prev.bbox[3]) / 2
            cx, cy = (cur.bbox[0] + cur.bbox[2]) / 2, (cur.bbox[1] + cur.bbox[3]) / 2
            diag = max(math.hypot(cur.bbox[2] - cur.bbox[0], cur.bbox[3] - cur.bbox[1]), 1e-6)
            speeds.append(min(1.0, math.hypot(cx - px, cy - py) / diag / dt / 2.0))
        first, last = dets[0], dets[-1]
        summaries.append({
            "person": track_id,
            "frames": len(dets),
            "relative_height": round(sum(heights) / len(heights), 3),
            "mean_speed": round(sum(speeds) / len(speeds), 3) if speeds else 0.0,
            "limb_extension_peak": round(max(extensions), 3) if extensions else 0.0,
            "frames_with_arm_extended": sum(1 for e in extensions if e >= 0.6),
            "net_horizontal_move": round(
                (last.bbox[0] + last.bbox[2]) / 2 - (first.bbox[0] + first.bbox[2]) / 2, 3
            ),
        })

    # --- weapon ------------------------------------------------------------ #
    weapon = None
    if objects:
        frames_with_object = {d.frame_index for d in objects}
        held = 0
        for obj in objects:
            ox, oy = (obj.bbox[0] + obj.bbox[2]) / 2, (obj.bbox[1] + obj.bbox[3]) / 2
            wrists = [
                w for p in persons if p.frame_index == obj.frame_index
                for w in _wrist_points(p)
            ]
            if any(math.hypot(ox - wx, oy - wy) <= HELD_DISTANCE for wx, wy in wrists):
                held += 1
        weapon = {
            "classes": sorted({d.cls for d in objects}),
            "max_confidence": round(max(d.confidence for d in objects), 2),
            "mean_confidence": round(sum(d.confidence for d in objects) / len(objects), 2),
            "frames_detected": len(frames_with_object),
            "frames_analyzed": clip_detections.frames_analyzed,
            "detections_in_a_hand": held,
            "total_detections": len(objects),
        }

    return {"persons": summaries, "weapon": weapon}


def describe(summary: dict[str, Any]) -> str:
    """Render a summary as prompt text."""
    lines = ["PER-PERSON SUMMARY (who did what):"]
    if not summary["persons"]:
        lines.append("  (no tracked persons)")
    for p in summary["persons"]:
        lines.append(
            f"  - person {p['person']}: relative height {p['relative_height']}, "
            f"mean speed {p['mean_speed']}, limb extension peak {p['limb_extension_peak']} "
            f"(arm extended in {p['frames_with_arm_extended']} of {p['frames']} frames), "
            f"net horizontal move {p['net_horizontal_move']:+}"
        )
    weapon = summary["weapon"]
    lines.append("WEAPON-LIKE OBJECT:")
    if weapon is None:
        lines.append("  none detected")
    else:
        lines.append(
            f"  - class {', '.join(weapon['classes'])}; confidence max {weapon['max_confidence']}, "
            f"mean {weapon['mean_confidence']}; seen in {weapon['frames_detected']} of "
            f"{weapon['frames_analyzed']} analyzed frames; "
            f"{weapon['detections_in_a_hand']} of {weapon['total_detections']} detections "
            f"are at a person's wrist (held in a hand)"
        )
    return "\n".join(lines)
