"""One entry point for running a case, shared by the CLI and the tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .agents.orchestrator import OrchestratorAgent, RunResult, build_context
from .config import Settings, get_settings
from .llm import LLMClient
from .logging_bus import CallLogger
from .messages import CaseRequest
from .state import CaseStore


def run_case(
    video_path: str,
    *,
    settings: Settings | None = None,
    camera_id: str = "cam-unknown",
    location: str = "unspecified",
    notes: str | None = None,
    allow_fixture_fallback: bool = True,
    log_path: Path | None = None,
    echo: bool = False,
) -> tuple[RunResult, CallLogger]:
    """Run the full six-agent pipeline on one video and return the result + log."""
    settings = settings or get_settings()
    settings.ensure_dirs()

    request = CaseRequest(
        video_path=video_path,
        camera_id=camera_id,
        location=location,
        notes=notes,
        allow_fixture_fallback=allow_fixture_fallback,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    logger = CallLogger(
        log_path or settings.log_dir / f"run-{stamp}-{request.case_id}.jsonl",
        echo=echo,
    )
    store = CaseStore(settings.store_path)
    case = store.new_case(request.case_id)

    llm = LLMClient(settings)
    orchestrator = OrchestratorAgent(llm)
    ctx = build_context(settings, logger, case)

    result = orchestrator.run_case(request, ctx)
    return result, logger
