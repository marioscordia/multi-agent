"""Escalation channel. Used only by the PrivacyReportAgent.

Real deployments would post to a duty-officer channel or a school's incident
system. Writing to an append-only local file keeps the demo self-contained and
auditable, and keeps outbound side effects out of a student project.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def notify(
    *,
    channel_dir: Path,
    case_id: str,
    tier: str,
    category: str,
    severity: int,
    summary: str,
) -> str:
    """Append one alert to the local channel; returns the channel path."""
    channel_dir.mkdir(parents=True, exist_ok=True)
    channel = channel_dir / "alerts.jsonl"
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_id": case_id,
        "tier": tier,
        "category": category,
        "severity": severity,
        "summary": summary,
    }
    with channel.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(channel)
