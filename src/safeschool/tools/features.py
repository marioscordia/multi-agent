"""Motion-feature extraction and the interpretable violence scorer.

Used only by the MotionAnalystAgent. The review stage of this thesis found that
published work reports accuracy while ignoring false-alarm behaviour, so the
scorer here is deliberately *interpretable*: a small set of named features
combined with fixed, documented weights. Every score can be explained in one
sentence, which is what the VerifierAgent and the incident report need.

A trained temporal classifier replaces this scorer in the next stage; the
interface (features in, score + rationale out) stays the same.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

# COCO-17 keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16

#: Feature weights for the baseline scorer. Positive weights push toward
#: "violent"; the negative one is a crowding prior (a dense corridor crowd is
#: normal movement, not a fight) that suppresses the common false alarm.
#:
#: The weighting is deliberately dominated by *sustained contact* and *limb
#: extension* rather than raw speed. Speed alone does not separate a fight from
#: brisk walking — an early calibration that leaned on peak speed raised a
#: physical_violence alert on CCTV of a supermarket aisle, which is precisely the
#: false-alarm behaviour the literature review found to be under-reported.
WEIGHTS: dict[str, float] = {
    "contact_frames_ratio": 2.6,
    "limb_extension_peak": 1.8,
    "speed_variance": 1.4,
    "proximity": 1.0,       # derived: 1 - median_pair_distance
    "peak_speed": 0.9,
    "pose_asymmetry": 0.6,
    "crowding": -0.9,
}
BIAS = -2.9

#: two person centroids closer than this (normalized frame units) count as contact
CONTACT_DISTANCE = 0.18


def _centroid(bbox: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_diag(bbox: Iterable[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(math.hypot(x2 - x1, y2 - y1), 1e-6)


def _limb_extension(keypoints: list[tuple[float, float, float]]) -> float:
    """How far a limb is thrown away from the body: the strike/kick signal.

    Measured as wrist distance from the torso centre in torso-lengths, with a
    rest baseline subtracted. Ankles are deliberately *excluded* from the arm
    term — legs are long, so including them pins the feature near its maximum for
    every upright person and it stops discriminating anything at all. A kick is
    captured by a separate term that only fires when an ankle rises above hip
    level, which standing and walking never do.

    Returns 0 when the pose is too occluded to judge.
    """
    if len(keypoints) < 17:
        return 0.0

    def pt(i: int):
        x, y, c = keypoints[i]
        return (x, y) if c > 0.3 else None

    ls, rs, lh, rh = pt(L_SHOULDER), pt(R_SHOULDER), pt(L_HIP), pt(R_HIP)
    anchors = [p for p in (ls, rs, lh, rh) if p]
    if len(anchors) < 2:
        return 0.0
    cx = sum(p[0] for p in anchors) / len(anchors)
    cy = sum(p[1] for p in anchors) / len(anchors)

    # torso scale: shoulder-to-hip distance, falling back to shoulder width
    torso = 0.0
    if ls and lh:
        torso = max(torso, math.hypot(ls[0] - lh[0], ls[1] - lh[1]))
    if rs and rh:
        torso = max(torso, math.hypot(rs[0] - rh[0], rs[1] - rh[1]))
    if torso < 1e-3 and ls and rs:
        torso = math.hypot(ls[0] - rs[0], ls[1] - rs[1])
    if torso < 1e-3:
        return 0.0

    # --- arm term ------------------------------------------------------- #
    wrist_reaches = [
        math.hypot(p[0] - cx, p[1] - cy) / torso
        for p in (pt(L_WRIST), pt(R_WRIST))
        if p
    ]
    # a relaxed arm puts the wrist about 0.9 torso-lengths from the centre;
    # a committed strike roughly doubles that
    arm = max(0.0, (max(wrist_reaches) - 0.9) / 0.9) if wrist_reaches else 0.0

    # --- kick term ------------------------------------------------------- #
    hip_y = max((p[1] for p in (lh, rh) if p), default=None)
    kick = 0.0
    if hip_y is not None:
        for ankle in (pt(L_ANKLE), pt(R_ANKLE)):
            if ankle and ankle[1] < hip_y:   # image y grows downward: above the hip
                kick = max(kick, min(1.0, (hip_y - ankle[1]) / torso))

    return float(np.clip(max(arm, kick), 0.0, 1.0))


def _pose_asymmetry(keypoints: list[tuple[float, float, float]]) -> float:
    """Left/right imbalance of the upper body — high during strikes and grapples."""
    if len(keypoints) < 17:
        return 0.0
    lw, rw = keypoints[L_WRIST], keypoints[R_WRIST]
    ls, rs = keypoints[L_SHOULDER], keypoints[R_SHOULDER]
    if min(lw[2], rw[2], ls[2], rs[2]) < 0.3:
        return 0.0
    shoulder_w = max(math.hypot(ls[0] - rs[0], ls[1] - rs[1]), 1e-3)
    left = math.hypot(lw[0] - ls[0], lw[1] - ls[1]) / shoulder_w
    right = math.hypot(rw[0] - rs[0], rw[1] - rs[1]) / shoulder_w
    return float(np.clip(abs(left - right) / 2.0, 0.0, 1.0))


def extract_features(detections: list[dict[str, Any]], *, fps: float) -> dict[str, float]:
    """Aggregate per-frame person detections into clip-level motion features.

    ``detections`` are dicts with ``frame_index``, ``t``, ``cls``, ``bbox`` and
    optional ``track_id`` / ``keypoints`` — i.e. ``Detection`` dumped to JSON.
    """
    persons = [d for d in detections if d.get("cls") == "person"]
    if not persons:
        return {
            "mean_speed": 0.0, "peak_speed": 0.0, "speed_variance": 0.0,
            "min_pair_distance": 1.0, "median_pair_distance": 1.0,
            "contact_frames_ratio": 0.0, "limb_extension_peak": 0.0,
            "pose_asymmetry": 0.0, "crowding": 0.0,
        }

    # --- group by frame and by track ------------------------------------ #
    by_frame: dict[int, list[dict[str, Any]]] = {}
    by_track: dict[Any, list[dict[str, Any]]] = {}
    for d in persons:
        by_frame.setdefault(d["frame_index"], []).append(d)
        by_track.setdefault(d.get("track_id"), []).append(d)

    # --- speeds, per track where available ------------------------------- #
    speeds: list[float] = []
    for track_id, dets in by_track.items():
        if track_id is None:
            # no tracker: approximate with the nearest-centroid displacement
            # between consecutive frames, which is adequate for 2-3 subjects
            continue
        dets = sorted(dets, key=lambda d: d["frame_index"])
        for prev, cur in zip(dets, dets[1:]):
            dt = max(cur["t"] - prev["t"], 1.0 / max(fps, 1.0))
            px, py = _centroid(prev["bbox"])
            cx, cy = _centroid(cur["bbox"])
            scale = _bbox_diag(cur["bbox"])
            speeds.append(math.hypot(cx - px, cy - py) / scale / dt)

    if not speeds:
        frames = sorted(by_frame)
        for f_prev, f_cur in zip(frames, frames[1:]):
            prev_c = [_centroid(d["bbox"]) for d in by_frame[f_prev]]
            for d in by_frame[f_cur]:
                cx, cy = _centroid(d["bbox"])
                if not prev_c:
                    continue
                nearest = min(prev_c, key=lambda p: math.hypot(cx - p[0], cy - p[1]))
                dt = max(
                    by_frame[f_cur][0]["t"] - by_frame[f_prev][0]["t"],
                    1.0 / max(fps, 1.0),
                )
                speeds.append(
                    math.hypot(cx - nearest[0], cy - nearest[1]) / _bbox_diag(d["bbox"]) / dt
                )

    speed_arr = np.array(speeds) if speeds else np.array([0.0])
    # normalize: 2.0 body-lengths/second is already fast motion
    mean_speed = float(np.clip(speed_arr.mean() / 2.0, 0.0, 1.0))
    peak_speed = float(np.clip(np.percentile(speed_arr, 95) / 2.0, 0.0, 1.0))
    speed_variance = float(np.clip(speed_arr.std() / 1.5, 0.0, 1.0))

    # --- proximity / contact --------------------------------------------- #
    frame_minima: list[float] = []
    contact_frames = 0
    for frame_dets in by_frame.values():
        cents = [_centroid(d["bbox"]) for d in frame_dets]
        frame_min = 1.0
        for i in range(len(cents)):
            for j in range(i + 1, len(cents)):
                dist = math.hypot(cents[i][0] - cents[j][0], cents[i][1] - cents[j][1])
                frame_min = min(frame_min, dist)
        frame_minima.append(frame_min)
        if frame_min < CONTACT_DISTANCE:
            contact_frames += 1
    contact_ratio = contact_frames / max(len(by_frame), 1)

    # The *global* minimum is the single closest instant in the clip, which one
    # pass-by in a corridor drives to near zero. The median frame-wise minimum
    # describes how close the subjects stay, which is what distinguishes a
    # sustained altercation from two people walking past each other.
    min_pair = min(frame_minima) if frame_minima else 1.0
    median_pair = float(np.median(frame_minima)) if frame_minima else 1.0

    # --- pose-derived ----------------------------------------------------- #
    extensions = [_limb_extension(d.get("keypoints") or []) for d in persons]
    asymmetries = [_pose_asymmetry(d.get("keypoints") or []) for d in persons]

    # --- crowding --------------------------------------------------------- #
    counts = [len(v) for v in by_frame.values()]
    crowding = float(np.clip((np.mean(counts) - 2.0) / 6.0, 0.0, 1.0))

    return {
        "mean_speed": mean_speed,
        "peak_speed": peak_speed,
        "speed_variance": speed_variance,
        "min_pair_distance": float(min_pair),
        "median_pair_distance": float(median_pair),
        "contact_frames_ratio": float(contact_ratio),
        "limb_extension_peak": float(max(extensions) if extensions else 0.0),
        "pose_asymmetry": float(np.mean(asymmetries) if asymmetries else 0.0),
        "crowding": crowding,
    }


def score(features: dict[str, float]) -> tuple[float, str]:
    """Combine features into a [0,1] violence score plus a one-sentence rationale."""
    terms = dict(features)
    terms["proximity"] = 1.0 - features.get(
        "median_pair_distance", features.get("min_pair_distance", 1.0)
    )

    contributions = {
        key: weight * terms.get(key, 0.0) for key, weight in WEIGHTS.items()
    }
    logit = BIAS + sum(contributions.values())
    probability = 1.0 / (1.0 + math.exp(-logit))

    ranked = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:3]
    drivers = ", ".join(
        f"{name}={terms.get(name, 0.0):.2f} ({value:+.2f})" for name, value in ranked
    )
    rationale = (
        f"score {probability:.2f} from logit {logit:+.2f}; "
        f"dominant features: {drivers}"
    )
    return float(np.clip(probability, 0.0, 1.0)), rationale
