"""Explicit task state: the CaseFile blackboard.

The assignment specification requires task state to be stored explicitly rather
than living in a call stack. Every stage result is written here, so any agent can
read what earlier stages produced, the orchestrator can resume or re-run a single
stage, and a crashed run leaves an inspectable record in SQLite.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .messages import AgentMessage, Payload

STAGES = ("ingest", "vision", "motion", "context", "verify", "report")


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class CaseFile:
    """Shared memory for one case. Not thread-safe by design: the orchestrator
    drives stages sequentially, which keeps the audit trail linear."""

    def __init__(self, case_id: str, store: "CaseStore | None" = None) -> None:
        self.case_id = case_id
        self.created_at = datetime.now(timezone.utc)
        self.artifacts: dict[str, Payload] = {}
        self.status: dict[str, StageStatus] = {s: StageStatus.PENDING for s in STAGES}
        self.messages: list[AgentMessage] = []
        self.notes: list[str] = []
        self.rerun_count = 0
        self._store = store

    # -- artifacts ------------------------------------------------------ #

    def put(self, stage: str, payload: Payload) -> None:
        self.artifacts[stage] = payload
        self.status[stage] = StageStatus.DONE
        self._persist()

    def get(self, stage: str) -> Payload | None:
        return self.artifacts.get(stage)

    def require(self, stage: str) -> Payload:
        value = self.artifacts.get(stage)
        if value is None:
            raise KeyError(
                f"case {self.case_id}: stage '{stage}' has no artifact yet "
                f"(status={self.status.get(stage)})"
            )
        return value

    def mark(self, stage: str, status: StageStatus) -> None:
        self.status[stage] = status
        self._persist()

    def record_message(self, message: AgentMessage) -> None:
        self.messages.append(message)

    def note(self, text: str) -> None:
        self.notes.append(text)

    # -- views ---------------------------------------------------------- #

    @property
    def failed_stages(self) -> list[str]:
        return [s for s, st in self.status.items() if st is StageStatus.FAILED]

    def reset_stages(self, stages: Iterable[str]) -> None:
        """Used when the Verifier asks for a re-run of specific stages."""
        for stage in stages:
            self.artifacts.pop(stage, None)
            self.status[stage] = StageStatus.PENDING
        self.rerun_count += 1
        self._persist()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "created_at": self.created_at.isoformat(),
            "status": {k: v.value for k, v in self.status.items()},
            "rerun_count": self.rerun_count,
            "notes": self.notes,
            "artifacts": {
                k: v.model_dump(mode="json") for k, v in self.artifacts.items()
            },
            "message_count": len(self.messages),
        }

    def _persist(self) -> None:
        if self._store is not None:
            self._store.save(self)


class CaseStore:
    """SQLite-backed persistence for case files (one row per case)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id    TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    state_json TEXT NOT NULL
                )
                """
            )

    def new_case(self, case_id: str) -> CaseFile:
        case = CaseFile(case_id, store=self)
        self.save(case)
        return case

    def save(self, case: CaseFile) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(case.to_dict(), ensure_ascii=False, default=str)
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO cases (case_id, created_at, updated_at, state_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    state_json = excluded.state_json
                """,
                (case.case_id, case.created_at.isoformat(), now, payload),
            )

    def load_raw(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT state_json FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list_cases(self, limit: int = 20) -> list[tuple[str, str]]:
        with self._connect() as con:
            return con.execute(
                "SELECT case_id, updated_at FROM cases ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
