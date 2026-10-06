"""Recorded fixtures.

Every stage that depends on something external (a video file, model weights, an
API key) has a recorded fallback here. The point is that the system is always
demonstrable: a reviewer with no GPU, no weights and no API key still sees the
full six-agent pipeline exchange real, schema-valid messages.

Fixture use is always logged and always marked in the payload (``source="fixture"``),
so a fixture-backed run can never be mistaken for a real detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..runtime import SafeSchoolError


class FixtureMissing(SafeSchoolError):
    recoverable = False


def available(fixtures_dir: Path) -> list[str]:
    return sorted(p.stem for p in Path(fixtures_dir).glob("*.json"))


def load(fixtures_dir: Path, name: str) -> dict[str, Any]:
    """Load one scenario fixture by name (without the .json extension)."""
    path = Path(fixtures_dir) / f"{name}.json"
    if not path.exists():
        raise FixtureMissing(
            f"fixture '{name}' not found in {fixtures_dir}. "
            f"Available: {', '.join(available(fixtures_dir)) or '(none)'}"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FixtureMissing(f"fixture '{name}' is not valid JSON: {exc}") from exc


def resolve_name(fixtures_dir: Path, video_path: str, requested: str | None = None) -> str:
    """Pick the fixture for a run.

    Priority: an explicit request, then a fixture whose name matches the video's
    stem (so ``data/raw/fight_01.mp4`` uses ``fixtures/fight_01.json``), then the
    generic fallback scenario.
    """
    names = available(fixtures_dir)
    if requested and requested in names:
        return requested
    stem = Path(video_path).stem
    if stem in names:
        return stem
    for candidate in ("corridor_fight", "default"):
        if candidate in names:
            return candidate
    if names:
        return names[0]
    raise FixtureMissing(f"no fixtures present in {fixtures_dir}")
