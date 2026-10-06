"""Configuration. Secrets come from the environment or a gitignored .env — never
from source. ``ANTHROPIC_API_KEY`` is the only secret the system needs.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SAFESCHOOL_",
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ------------------------------------------------------------- #
    llm_provider: str = "anthropic"          # anthropic | mock
    model: str = "claude-sonnet-5"
    max_tokens: int = 1200
    # read from the un-prefixed ANTHROPIC_API_KEY, the name the SDK expects
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # --- data sources ---------------------------------------------------- #
    corpus_dir: Path = REPO_ROOT.parent / "disser" / "prepared"
    matrix_csv: Path = REPO_ROOT.parent / "disser" / "matrix_literature.csv"
    policy_dir: Path = REPO_ROOT / "data" / "policy"
    fixtures_dir: Path = REPO_ROOT / "fixtures"

    # --- run artifacts --------------------------------------------------- #
    data_dir: Path = REPO_ROOT / "data"
    cases_dir: Path = REPO_ROOT / "data" / "cases"
    weights_dir: Path = REPO_ROOT / "data" / "weights"
    log_dir: Path = REPO_ROOT / "logs"
    store_path: Path = REPO_ROOT / "data" / "cases.sqlite3"

    # --- budgets / loop protection --------------------------------------- #
    max_steps: int = 40
    deadline_seconds: int = 300
    max_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    max_reruns: int = 1

    # --- vision ---------------------------------------------------------- #
    detector_weights: str = "yolov8n.pt"
    pose_weights: str = "yolov8n-pose.pt"
    confidence_threshold: float = 0.25
    clip_seconds: float = 2.0
    max_clips: int = 6
    frame_stride: int = 3

    # --- decision thresholds --------------------------------------------- #
    violence_threshold: float = 0.55

    @property
    def llm_available(self) -> bool:
        return self.llm_provider == "anthropic" and bool(self.anthropic_api_key)

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.cases_dir, self.weights_dir, self.log_dir, self.policy_dir):
            d.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings(**overrides) -> Settings:
    """Process-wide settings. ``overrides`` returns a fresh instance (used by tests)."""
    global _settings
    if overrides:
        return Settings(**overrides)
    if _settings is None:
        _settings = Settings()
    return _settings
