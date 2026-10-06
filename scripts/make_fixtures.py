#!/usr/bin/env python3
"""Generate the recorded scenario fixtures in ``fixtures/``.

A fixture is a ``ClipSet`` + ``DetectionSet`` pair: exactly what the IngestAgent
and VisionDetectorAgent would have produced, recorded so that the four reasoning
stages downstream can be demonstrated with no weights, no GPU and no API key.

The detections here are **synthesized**, not recorded from real footage, because
the violence benchmarks cannot be redistributed. They are built from an explicit
skeleton model so the motion features they produce are the ones a real detection
of that scenario would produce. Every fixture-backed run is marked
``source="fixture"`` in the payload and raises the ``fixture_backed_run`` issue in
verification, so it can never be mistaken for a real detection.

Small synthetic keyframes are written alongside, so the face-redaction stage runs
on actual images rather than being skipped.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "fixtures"
FRAMES = FIXTURES / "frames"

W, H = 640, 360
FPS = 15.0
STRIDE = 3

# COCO-17 order
KP_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def skeleton(
    *, cx: float, cy: float, scale: float, reach: float = 0.0, facing: float = 1.0
) -> list[tuple[float, float, float]]:
    """Build 17 normalized keypoints for one person.

    ``scale`` is body height in normalized units, ``reach`` in [0,1] extends the
    leading arm outward (0 = at rest, 1 = fully extended strike), ``facing`` is
    +1 (facing right) or -1.
    """
    s = scale
    head_y = cy - s * 0.42
    sh_y = cy - s * 0.26
    hip_y = cy + s * 0.04
    knee_y = cy + s * 0.26
    ankle_y = cy + s * 0.46
    sh_dx = s * 0.11

    # the leading wrist travels outward and upward as reach grows
    lead_x = cx + facing * (sh_dx + reach * s * 0.46)
    lead_y = sh_y + (1.0 - reach) * s * 0.20 - reach * s * 0.03
    rear_x = cx - facing * sh_dx * 0.7
    rear_y = sh_y + s * 0.22

    points = {
        "nose": (cx + facing * s * 0.03, head_y),
        "left_eye": (cx - s * 0.03, head_y - s * 0.02),
        "right_eye": (cx + s * 0.03, head_y - s * 0.02),
        "left_ear": (cx - s * 0.06, head_y),
        "right_ear": (cx + s * 0.06, head_y),
        "left_shoulder": (cx - sh_dx, sh_y),
        "right_shoulder": (cx + sh_dx, sh_y),
        "left_elbow": (cx - sh_dx * 1.3, sh_y + s * 0.12),
        "right_elbow": (cx + sh_dx * 1.3, sh_y + s * 0.12),
        "left_wrist": (lead_x, lead_y) if facing < 0 else (rear_x, rear_y),
        "right_wrist": (lead_x, lead_y) if facing > 0 else (rear_x, rear_y),
        "left_hip": (cx - s * 0.07, hip_y),
        "right_hip": (cx + s * 0.07, hip_y),
        "left_knee": (cx - s * 0.07, knee_y),
        "right_knee": (cx + s * 0.07, knee_y),
        "left_ankle": (cx - s * 0.07, ankle_y),
        "right_ankle": (cx + s * 0.07, ankle_y),
    }
    return [(round(points[n][0], 4), round(points[n][1], 4), 0.9) for n in KP_NAMES]


def bbox_from(cx: float, cy: float, scale: float) -> tuple[float, float, float, float]:
    half_w = scale * 0.17
    return (
        round(max(cx - half_w, 0.0), 4), round(max(cy - scale * 0.5, 0.0), 4),
        round(min(cx + half_w, 1.0), 4), round(min(cy + scale * 0.5, 1.0), 4),
    )


def person(frame_index: int, t: float, track_id: int, cx: float, cy: float,
           scale: float, reach: float, facing: float) -> dict:
    return {
        "frame_index": frame_index,
        "t": round(t, 3),
        "cls": "person",
        "confidence": 0.88,
        "bbox": bbox_from(cx, cy, scale),
        "track_id": track_id,
        "keypoints": skeleton(cx=cx, cy=cy, scale=scale, reach=reach, facing=facing),
    }


def weapon(frame_index: int, t: float, cx: float, cy: float) -> dict:
    return {
        "frame_index": frame_index,
        "t": round(t, 3),
        "cls": "knife",
        "confidence": 0.61,
        "bbox": (round(cx - 0.02, 4), round(cy - 0.015, 4), round(cx + 0.02, 4), round(cy + 0.015, 4)),
        "track_id": None,
        "keypoints": [],
    }


# --------------------------------------------------------------------------- #
# scenario builders
# --------------------------------------------------------------------------- #

def build_fight(*, frames: int = 10) -> list[dict]:
    """Two peers, close together, high reciprocal reach, rapid oscillation."""
    out: list[dict] = []
    for i in range(frames):
        fi = i * STRIDE
        t = fi / FPS
        phase = math.sin(i * 1.5)
        reach_a = max(0.0, phase)
        reach_b = max(0.0, -phase)
        cx_a = 0.455 + 0.016 * phase
        cx_b = 0.545 - 0.016 * phase
        out.append(person(fi, t, 1, cx_a, 0.56, 0.44, reach_a, +1.0))
        out.append(person(fi, t, 2, cx_b, 0.56, 0.43, reach_b, -1.0))
    return out


def build_weapon(*, frames: int = 10) -> list[dict]:
    """A fight, plus a weapon-like object detected in the leading hand."""
    out = build_fight(frames=frames)
    for i in range(frames):
        fi = i * STRIDE
        if i % 2 == 0:
            out.append(weapon(fi, fi / FPS, 0.51, 0.47))
    return out


def build_adult_child(*, frames: int = 10) -> list[dict]:
    """One large and one small body in contact: the safeguarding case.

    The scale ratio (0.60 / 0.33 = 1.8) is what distinguishes clause SSP-2.1
    from a peer altercation, and the child's horizontal motion opposes its own
    facing direction — being moved rather than moving.
    """
    out: list[dict] = []
    for i in range(frames):
        fi = i * STRIDE
        t = fi / FPS
        reach = 0.55 + 0.3 * max(0.0, math.sin(i * 1.2))
        adult_x = 0.46 + 0.004 * i
        child_x = 0.555 + 0.010 * i      # dragged along
        out.append(person(fi, t, 1, adult_x, 0.52, 0.60, reach, +1.0))
        out.append(person(fi, t, 2, child_x, 0.62, 0.33, 0.1, -1.0))
    return out


def build_rough_play(*, frames: int = 10) -> list[dict]:
    """Energetic but benign: the two bodies swing apart and back together, so
    contact is intermittent rather than sustained, limbs stay down, and a third
    onlooker stands by.

    This is the class the literature review identified as the dominant source of
    false alarms, so the geometry is set deliberately: median separation well
    above the contact distance, and arms at rest.
    """
    out: list[dict] = []
    for i in range(frames):
        fi = i * STRIDE
        t = fi / FPS
        swing = math.sin(i * 0.9)
        out.append(person(fi, t, 1, 0.38 + 0.06 * swing, 0.58, 0.42, 0.12, +1.0))
        out.append(person(fi, t, 2, 0.64 - 0.06 * swing, 0.58, 0.41, 0.10, -1.0))
        out.append(person(fi, t, 3, 0.84, 0.60, 0.40, 0.02, -1.0))
    return out


SCENARIOS = {
    "corridor_fight": (build_fight, "two peers exchanging strikes in a corridor"),
    "weapon_threat": (build_weapon, "a fight with a weapon-like object detected"),
    "adult_child": (build_adult_child, "an adult physically handling a child"),
    "rough_play": (build_rough_play, "energetic but mutual play with an onlooker"),
}


# --------------------------------------------------------------------------- #
# keyframe rendering
# --------------------------------------------------------------------------- #

SKELETON_EDGES = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def render_keyframes(name: str, detections: list[dict]) -> list[dict]:
    """Draw the fixture's figures into small JPEGs so redaction has real pixels."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print(f"  (OpenCV missing: no keyframes for {name})")
        return []

    FRAMES.mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, list[dict]] = {}
    for d in detections:
        if d["cls"] == "person":
            by_frame.setdefault(d["frame_index"], []).append(d)
    chosen = sorted(by_frame)[:: max(1, len(by_frame) // 3)][:3]

    paths: list[dict] = []
    for n, frame_index in enumerate(chosen):
        img = np.full((H, W, 3), 54, dtype=np.uint8)
        cv2.rectangle(img, (0, int(H * 0.72)), (W, H), (78, 78, 84), -1)
        cv2.putText(img, f"SYNTHETIC FIXTURE  {name}  f{frame_index}", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 155), 1, cv2.LINE_AA)

        for d in by_frame[frame_index]:
            kps = [(int(x * W), int(y * H)) for x, y, _ in d["keypoints"]]
            for a, b in SKELETON_EDGES:
                cv2.line(img, kps[a], kps[b], (200, 200, 205), 2, cv2.LINE_AA)
            # a filled head disc: this is what the redaction stage must blur
            head = kps[0]
            radius = max(8, int((d["bbox"][3] - d["bbox"][1]) * H * 0.10))
            cv2.circle(img, head, radius, (205, 180, 160), -1, cv2.LINE_AA)
            cv2.circle(img, (head[0] - radius // 3, head[1] - radius // 4), 2, (40, 40, 40), -1)
            cv2.circle(img, (head[0] + radius // 3, head[1] - radius // 4), 2, (40, 40, 40), -1)
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(img, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)),
                          (90, 170, 220), 1)

        dest = FRAMES / f"{name}_kf{n}.jpg"
        cv2.imwrite(str(dest), img)
        paths.append(
            {
                "path": str(dest.relative_to(REPO_ROOT)),
                "frame_index": frame_index,
                "t": round(frame_index / FPS, 3),
            }
        )
    return paths


# --------------------------------------------------------------------------- #

def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, (builder, description) in SCENARIOS.items():
        detections = builder()
        keyframes = render_keyframes(name, detections)
        frames_analyzed = len({d["frame_index"] for d in detections})
        persons_per_frame: dict[int, int] = {}
        for d in detections:
            if d["cls"] == "person":
                persons_per_frame[d["frame_index"]] = persons_per_frame.get(d["frame_index"], 0) + 1

        duration = round(max(d["t"] for d in detections) + 1.0 / FPS, 3)
        payload = {
            "_meta": {
                "name": name,
                "description": description,
                "synthetic": True,
                "note": (
                    "Synthesized from an explicit skeleton model, not recorded from "
                    "real footage. Used only for offline demonstration of the "
                    "reasoning stages; always marked source=fixture."
                ),
            },
            "clip_set": {
                "case_id": f"fixture-{name}",
                "source_path": f"fixture://{name}",
                "fps": FPS,
                "duration_seconds": duration,
                "width": W,
                "height": H,
                "source": "fixture",
                "clips": [
                    {
                        "clip_id": f"fixture-{name}-c00",
                        "index": 0,
                        "t_start": 0.0,
                        "t_end": duration,
                        "frame_count": int(duration * FPS),
                        "keyframes": keyframes,
                    }
                ],
            },
            "detection_set": {
                "case_id": f"fixture-{name}",
                "model_name": "fixture/synthetic-skeleton-v1",
                "source": "fixture",
                "confidence_threshold": 0.25,
                "per_clip": [
                    {
                        "clip_id": f"fixture-{name}-c00",
                        "detections": detections,
                        "frames_analyzed": frames_analyzed,
                        "person_count_max": max(persons_per_frame.values(), default=0),
                        "weapon_present": any(d["cls"] != "person" for d in detections),
                    }
                ],
            },
        }
        dest = FIXTURES / f"{name}.json"
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {dest.relative_to(REPO_ROOT)} "
              f"({len(detections)} detections, {len(keyframes)} keyframes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
