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


def weapon(frame_index: int, t: float, cx: float, cy: float, confidence: float) -> dict:
    return {
        "frame_index": frame_index,
        "t": round(t, 3),
        "cls": "knife",
        "confidence": round(confidence, 2),
        "bbox": (round(cx - 0.02, 4), round(cy - 0.015, 4), round(cx + 0.02, 4), round(cy + 0.015, 4)),
        "track_id": None,
        "keypoints": [],
    }


def lead_wrist(det: dict, facing: float) -> tuple[float, float]:
    """Where the leading (striking) hand of a generated skeleton is."""
    index = KP_NAMES.index("right_wrist" if facing > 0 else "left_wrist")
    x, y, _ = det["keypoints"][index]
    return x, y


# --------------------------------------------------------------------------- #
# scenario builders
#
# Speeds are calibrated against the real CCTV clips: people walking a corridor
# measure mean_speed ~0.20 and peak_speed ~0.35. A fight must be clearly faster
# and more erratic than that, or a careful reviewer will rightly call it play.
# --------------------------------------------------------------------------- #

# per-step offsets (one step = STRIDE frames = 0.2 s): sudden lunges and recoils
_LUNGE = (0.00, 0.07, 0.01, 0.08, -0.01, 0.07, 0.02, 0.09, 0.00, 0.07)
_STRIKE = (0.3, 1.0, 0.4, 1.0, 0.3, 1.0, 0.5, 1.0, 0.3, 1.0)


def build_fight(*, frames: int = 10) -> list[dict]:
    """A one-sided altercation: person 1 lunges and strikes repeatedly, person 2
    is knocked back and ducks. Fast, jerky, asymmetric — not mutual play."""
    out: list[dict] = []
    for i in range(frames):
        fi = i * STRIDE
        t = fi / FPS
        cx_a = 0.40 + _LUNGE[i]
        cy_a = 0.56 + (-0.03 if _STRIKE[i] > 0.9 else 0.02)
        # the target is pushed back on each blow and stumbles forward between them
        cx_b = cx_a + 0.08 + (0.04 if _STRIKE[i] > 0.9 else -0.01)
        cy_b = 0.57 + (0.04 if _STRIKE[i] > 0.9 else -0.01)
        out.append(person(fi, t, 1, cx_a, cy_a, 0.44, _STRIKE[i], +1.0))
        out.append(person(fi, t, 2, cx_b, cy_b, 0.43, 0.15, -1.0))
    return out


def build_weapon(*, frames: int = 10) -> list[dict]:
    """Person 1 advances with a knife held in the leading hand and thrusts it at
    person 2, who backs away. The knife is detected in the hand on every frame."""
    out: list[dict] = []
    for i in range(frames):
        fi = i * STRIDE
        t = fi / FPS
        thrust = _STRIKE[i]
        cx_a = 0.30 + 0.035 * i + (0.06 if thrust > 0.9 else -0.02)
        cy_a = 0.56 + (-0.03 if thrust > 0.9 else 0.02)
        cx_b = 0.30 + 0.035 * i + 0.16 + (0.06 if thrust > 0.9 else 0.0)
        cy_b = 0.57 + (0.04 if thrust > 0.9 else -0.01)
        attacker = person(fi, t, 1, cx_a, cy_a, 0.44, thrust, +1.0)
        out.append(attacker)
        out.append(person(fi, t, 2, cx_b, cy_b, 0.43, 0.2, -1.0))
        wx, wy = lead_wrist(attacker, +1.0)
        out.append(weapon(fi, t, wx, wy, 0.78 + 0.06 * math.sin(i)))
    return out


def build_adult_child(*, frames: int = 10) -> list[dict]:
    """An adult grabs a child by the arm and drags them across the frame.

    The scale ratio (0.60 / 0.33 = 1.8) is what distinguishes clause SSP-2.1
    from a peer altercation. The child faces away (+1, trying to pull free) but
    is yanked the other way in jerks — being moved rather than moving.
    """
    yank = (0.00, 0.10, -0.02, 0.11, -0.03, 0.10, -0.01, 0.12, -0.02, 0.10)
    out: list[dict] = []
    adult_x = 0.62
    child_x = 0.71
    for i in range(frames):
        fi = i * STRIDE
        t = fi / FPS
        adult_x -= 0.03                       # the adult walks off to the left
        child_x = adult_x + 0.09 - 0.5 * yank[i] + (0.04 if yank[i] < 0 else 0.0)
        child_y = 0.62 + (-0.015 if yank[i] > 0.05 else 0.01)
        reach = 0.95 if yank[i] > 0.05 else 0.7
        out.append(person(fi, t, 1, adult_x, 0.52, 0.60, reach, +1.0))
        out.append(person(fi, t, 2, child_x, child_y, 0.33, 0.6, +1.0))
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
