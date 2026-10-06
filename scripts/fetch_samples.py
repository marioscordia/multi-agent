#!/usr/bin/env python3
"""Fetch sample surveillance clips into ``data/raw/`` (gitignored).

Two kinds of test input:

1. **Real surveillance footage** — public CCTV-style clips from Intel's
   ``sample-videos`` repository. Real people, real cameras, so YOLO person and
   pose detection genuinely exercises. None of them contains violence, which
   makes them the right material for the true-negative and benign scenarios.

2. **Locally synthesized edge cases** — an empty scene and a deliberately
   corrupt file, for the error and empty-detection paths.

Violence benchmarks (Hockey Fights, RWF-2000, UCF-Crime) are not downloaded
automatically: all three require accepting dataset terms on the provider's site.
Run this script, then see the printed instructions for adding them manually.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

BASE = "https://github.com/intel-iot-devkit/sample-videos/raw/master"

#: local name -> (remote file, what it is used for)
SAMPLES: dict[str, tuple[str, str]] = {
    "corridor_people.mp4": (
        "people-detection.mp4",
        "people walking a corridor — the true-negative / no_violation scenario",
    ),
    "store_aisle.mp4": (
        "store-aisle-detection.mp4",
        "multiple people in an aisle — the crowding / bullying-candidate scenario",
    ),
    "one_person.mp4": (
        "head-pose-face-detection-male.mp4",
        "a single person — the 'interaction requires two' verifier check",
    ),
}

MANUAL_INSTRUCTIONS = """
Violence benchmarks must be downloaded manually (each requires accepting terms):

  Hockey Fights   https://academictorrents.com/  (search "HockeyFights")
                  or Kaggle: yassershrief/hockey-fight-vidoes
  RWF-2000        https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
  UCF-Crime       https://www.crcv.ucf.edu/projects/real-world/

Drop two or three clips into data/raw/ and name them so the intent is obvious:

  data/raw/fight_01.mp4          -> expected: physical_violence
  data/raw/rough_play_01.mp4     -> expected: rough_play_benign (false-positive test)

Then run:  safeschool run data/raw/fight_01.mp4
"""


def download(url: str, dest: Path, *, timeout: int = 90) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  = {dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    print(f"  > {dest.name} <- {url}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "safeschool-agents/0.2"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"    ! failed: {exc}", file=sys.stderr)
        return False
    dest.write_bytes(data)
    print(f"    ok ({len(data) / 1e6:.1f} MB)")
    return True


def make_empty_scene(dest: Path, *, seconds: int = 4, fps: int = 15) -> bool:
    """A static scene with no people: exercises the empty-detection path."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("    ! OpenCV not installed; skipping synthesized clips", file=sys.stderr)
        return False
    if dest.exists():
        print(f"  = {dest.name} already present")
        return True
    width, height = 640, 360
    writer = cv2.VideoWriter(
        str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    rng = np.random.default_rng(7)
    base = np.full((height, width, 3), 48, dtype=np.uint8)
    cv2.rectangle(base, (0, 250), (width, height), (72, 72, 78), -1)   # floor
    cv2.rectangle(base, (80, 90), (200, 250), (96, 92, 86), -1)        # a door
    for _ in range(seconds * fps):
        frame = base.copy()
        noise = rng.integers(-6, 7, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    print(f"  + {dest.name} (synthesized empty corridor)")
    return True


def make_corrupt(dest: Path) -> bool:
    """A file with a video extension and garbage inside: error-path scenario."""
    if dest.exists():
        print(f"  = {dest.name} already present")
        return True
    dest.write_bytes(b"\x00\x01\x02not a video container at all\xff" * 64)
    print(f"  + {dest.name} (synthesized corrupt file)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true", help="only synthesize local clips")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"target: {RAW_DIR}")

    failures = 0
    if not args.skip_download:
        print("\nreal surveillance clips:")
        for local, (remote, purpose) in SAMPLES.items():
            print(f"  ({purpose})")
            if not download(f"{BASE}/{remote}", RAW_DIR / local):
                failures += 1

    print("\nsynthesized edge cases:")
    make_empty_scene(RAW_DIR / "empty_corridor.mp4")
    make_corrupt(RAW_DIR / "corrupt_clip.mp4")

    print(MANUAL_INSTRUCTIONS)
    if failures:
        print(f"{failures} download(s) failed; the pipeline still runs on fixtures.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
