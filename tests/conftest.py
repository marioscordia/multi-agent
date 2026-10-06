import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from safeschool.config import get_settings          # noqa: E402
from safeschool.logging_bus import CallLogger       # noqa: E402
from safeschool.state import CaseStore              # noqa: E402


@pytest.fixture
def settings(tmp_path):
    """Settings pointed at a temp directory, with the LLM forced to mock."""
    base = get_settings()
    return base.model_copy(
        update={
            "llm_provider": "mock",
            "anthropic_api_key": None,
            "data_dir": tmp_path / "data",
            "cases_dir": tmp_path / "data" / "cases",
            "weights_dir": tmp_path / "data" / "weights",
            "log_dir": tmp_path / "logs",
            "store_path": tmp_path / "data" / "cases.sqlite3",
        }
    )


@pytest.fixture
def logger(tmp_path):
    return CallLogger(tmp_path / "test.jsonl")


@pytest.fixture
def store(settings):
    settings.ensure_dirs()
    return CaseStore(settings.store_path)
