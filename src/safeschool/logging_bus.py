"""Call logging.

Every agent invocation, every tool call and every LLM call is appended as one
JSON object per line. Two things depend on this file:

* ``tools/load_report.py`` proves the "no agent above 40% of calls" requirement
  from the assignment specification;
* post-mortem debugging of a failed case.

Logging lives in the runtime (see :mod:`safeschool.runtime`), not in the agents,
so an agent cannot forget to log.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_LOCK = threading.Lock()

# keys whose values must never reach the log
_REDACT_KEYS = {"api_key", "anthropic_api_key", "authorization", "token", "secret"}


def _scrub(value: Any, depth: int = 0) -> Any:
    """Keep the log readable and secret-free."""
    if depth > 4:
        return "..."
    if isinstance(value, dict):
        return {
            k: ("***" if k.lower() in _REDACT_KEYS else _scrub(v, depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            return [_scrub(v, depth + 1) for v in value[:8]] + [f"...+{len(value) - 8} more"]
        return [_scrub(v, depth + 1) for v in value]
    if isinstance(value, str) and len(value) > 300:
        return value[:300] + "..."
    return value


class CallLogger:
    """Append-only JSONL logger for one process/run."""

    def __init__(self, log_path: Path, *, echo: bool = False) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._records: list[dict[str, Any]] = []

    # -- writing -------------------------------------------------------- #

    def log(
        self,
        *,
        case_id: str,
        agent: str,
        event: str,           # agent | tool | llm | error | system
        name: str,
        ok: bool = True,
        duration_ms: float = 0.0,
        attempt: int = 1,
        tokens_in: int = 0,
        tokens_out: int = 0,
        detail: Any = None,
    ) -> dict[str, Any]:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "pid": os.getpid(),
            "case_id": case_id,
            "agent": agent,
            "event": event,
            "name": name,
            "ok": ok,
            "duration_ms": round(duration_ms, 2),
            "attempt": attempt,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "detail": _scrub(detail),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _LOCK:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        self._records.append(record)
        if self.echo:
            flag = "ok " if ok else "ERR"
            print(f"  [{flag}] {record['event']:5s} {agent:18s} {name:28s} {record['duration_ms']:8.1f}ms")
        return record

    # -- reading -------------------------------------------------------- #

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def call_counts(self, *, events: tuple[str, ...] = ("tool", "llm")) -> dict[str, int]:
        """Per-agent count of billable calls (tool + LLM), which is what the
        40%-per-agent limit is measured against."""
        counts: dict[str, int] = {}
        for r in self._records:
            if r["event"] in events:
                counts[r["agent"]] = counts.get(r["agent"], 0) + 1
        return counts

    def load_shares(self) -> dict[str, float]:
        counts = self.call_counts()
        total = sum(counts.values()) or 1
        return {a: c / total for a, c in sorted(counts.items(), key=lambda kv: -kv[1])}


def read_log(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
