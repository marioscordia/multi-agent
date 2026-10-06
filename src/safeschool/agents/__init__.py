"""The agent team.

Six worker agents, one per pipeline stage, plus a planning orchestrator that does
no domain work of its own.
"""

from .context import ContextAgent
from .ingest import IngestAgent
from .motion import MotionAnalystAgent
from .orchestrator import OrchestratorAgent
from .report import PrivacyReportAgent
from .verifier import VerifierAgent
from .vision import VisionDetectorAgent

WORKER_AGENTS = (
    IngestAgent,
    VisionDetectorAgent,
    MotionAnalystAgent,
    ContextAgent,
    VerifierAgent,
    PrivacyReportAgent,
)

__all__ = [
    "OrchestratorAgent",
    "IngestAgent",
    "VisionDetectorAgent",
    "MotionAnalystAgent",
    "ContextAgent",
    "VerifierAgent",
    "PrivacyReportAgent",
    "WORKER_AGENTS",
]
