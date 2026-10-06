"""Command-line interface.

    safeschool run <video>     run the pipeline on one clip
    safeschool demo            run every bundled scenario
    safeschool agents          print the agent specification table
    safeschool load [log]      per-agent call-share report (the <=40% check)
    safeschool scenarios       list the test scenarios and expected outcomes
    safeschool cases           list stored cases
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import get_settings
from .logging_bus import read_log
from .pipeline import run_case
from .scenarios import SCENARIOS
from .state import CaseStore

app = typer.Typer(add_completion=False, help=__doc__)


def _banner(text: str) -> None:
    typer.echo(f"\n\033[1m{text}\033[0m")


def _print_result(result, logger, *, verbose: bool) -> None:
    case = result.case
    if result.error is not None:
        err = result.error
        typer.secho(
            f"\n  FAILED at stage '{err.stage}' ({err.agent})", fg=typer.colors.RED, bold=True
        )
        typer.echo(f"  {err.error_type}: {err.message}")
        typer.echo(f"  recoverable: {err.recoverable}")
    else:
        report = result.report
        colour = {
            "none": typer.colors.BRIGHT_BLACK,
            "log": typer.colors.BLUE,
            "notify_staff": typer.colors.YELLOW,
            "notify_admin": typer.colors.MAGENTA,
            "emergency": typer.colors.RED,
        }.get(report.escalation_tier, typer.colors.WHITE)
        typer.echo("")
        typer.secho(f"  tier      {report.escalation_tier}", fg=colour, bold=True)
        typer.echo(f"  category  {report.category} (severity {report.severity}/4)")
        typer.echo(f"  summary   {report.summary}")
        typer.echo(f"  privacy   {report.faces_redacted} face region(s) blurred "
                   f"across {len(report.redacted_keyframes)} frame(s)")
        typer.echo(f"  report    {report.report_markdown_path}")

    typer.echo(f"\n  case      {case.case_id}")
    typer.echo(f"  stages    " + "  ".join(f"{k}={v.value}" for k, v in case.status.items()))
    if case.notes:
        typer.echo("  notes:")
        for note in case.notes:
            typer.echo(f"    - {note}")

    shares = logger.load_shares()
    if shares:
        typer.echo("\n  per-agent share of tool+LLM calls:")
        for agent, share in shares.items():
            flag = "  <-- OVER 40%" if share > 0.40 else ""
            bar = "#" * max(1, int(share * 40))
            typer.echo(f"    {agent:22s} {share * 100:5.1f}%  {bar}{flag}")
    typer.echo(f"  log       {logger.log_path}")

    if verbose:
        typer.echo("\n  messages exchanged:")
        for msg in case.messages:
            typer.echo(f"    {msg.sender:20s} -> {msg.recipient:22s} [{msg.type.value}]")


@app.command()
def run(
    video: str = typer.Argument(..., help="path to the video file"),
    camera: str = typer.Option("cam-unknown", help="camera identifier for the report"),
    location: str = typer.Option("unspecified", help="where the camera is"),
    no_fixtures: bool = typer.Option(
        False, "--no-fixtures", help="fail instead of falling back to recorded fixtures"
    ),
    quiet: bool = typer.Option(False, "--quiet", help="do not stream per-call log lines"),
    verbose: bool = typer.Option(False, "--verbose", help="also list the messages exchanged"),
) -> None:
    """Run the pipeline on one video."""
    settings = get_settings()
    _banner(f"SafeSchool · {Path(video).name}")
    typer.echo(
        f"  llm={'anthropic:' + settings.model if settings.llm_available else 'mock (no ANTHROPIC_API_KEY)'}"
        f"  budget={settings.max_steps} steps / {settings.deadline_seconds}s"
    )
    typer.echo("")
    result, logger = run_case(
        video, settings=settings, camera_id=camera, location=location,
        allow_fixture_fallback=not no_fixtures, echo=not quiet,
    )
    _print_result(result, logger, verbose=verbose)
    raise typer.Exit(0 if result.ok else 1)


@app.command()
def demo(
    quiet: bool = typer.Option(True, "--quiet/--loud", help="stream per-call log lines"),
) -> None:
    """Run every bundled scenario and summarize pass/fail against expectations."""
    settings = get_settings()
    _banner("SafeSchool · scenario suite")
    typer.echo(
        f"  llm={'anthropic:' + settings.model if settings.llm_available else 'mock (no ANTHROPIC_API_KEY)'}"
    )

    rows: list[tuple[str, str, str, bool]] = []
    for scenario in SCENARIOS:
        path = Path(scenario.video)
        is_fixture_uri = scenario.video.startswith("fixture://")
        if not is_fixture_uri and not path.exists() and not scenario.expect_failure:
            typer.secho(
                f"\n  - {scenario.name}: SKIPPED (missing {path}; run scripts/fetch_samples.py)",
                fg=typer.colors.BRIGHT_BLACK,
            )
            rows.append((scenario.name, "skipped", "-", True))
            continue

        typer.echo(f"\n  - {scenario.name}: {scenario.description}")
        result, logger = run_case(
            scenario.video, settings=settings,
            allow_fixture_fallback=scenario.allow_fixtures, echo=not quiet,
        )
        ok, actual = scenario.check(result)
        colour = typer.colors.GREEN if ok else typer.colors.RED
        typer.secho(f"    expected {scenario.expectation()} | got {actual}", fg=colour)
        rows.append((scenario.name, actual, scenario.expectation(), ok))

    passed = sum(1 for *_, ok in rows if ok)
    _banner(f"{passed}/{len(rows)} scenarios met expectations")
    for name, actual, expected, ok in rows:
        typer.secho(
            f"  {'PASS' if ok else 'FAIL'}  {name:22s} {actual:28s} (expected {expected})",
            fg=typer.colors.GREEN if ok else typer.colors.RED,
        )
    raise typer.Exit(0 if passed == len(rows) else 1)


@app.command()
def agents() -> None:
    """Print the agent specification table."""
    from .agents import WORKER_AGENTS
    from .agents.orchestrator import OrchestratorAgent
    from .llm import LLMClient

    settings = get_settings()
    llm = LLMClient(settings)
    specs = [OrchestratorAgent(llm).spec()]
    for cls in WORKER_AGENTS:
        agent = cls(llm) if cls.__name__ in {"ContextAgent", "VerifierAgent"} else cls()
        specs.append(agent.spec())

    _banner("Agent specification")
    for spec in specs:
        typer.secho(f"\n  {spec['name']}  [stage: {spec['stage']}]", bold=True)
        typer.echo(f"    role      {spec['role']}")
        typer.echo(f"    in -> out {spec['inputs']} -> {spec['outputs']}")
        typer.echo(f"    tools     ({len(spec['tools'])}/5) {', '.join(spec['tools'])}")
        typer.echo(f"    done when {spec['done_criterion']}")


@app.command()
def load(
    log_file: str = typer.Argument(None, help="JSONL log; defaults to the newest run"),
) -> None:
    """Per-agent share of tool and LLM calls (verifies the <=40% requirement)."""
    settings = get_settings()
    if log_file:
        path = Path(log_file)
    else:
        candidates = sorted(settings.log_dir.glob("run-*.jsonl"))
        if not candidates:
            typer.secho("no run logs found; run `safeschool demo` first", fg=typer.colors.RED)
            raise typer.Exit(1)
        path = candidates[-1]

    counts: dict[str, dict[str, int]] = {}
    for record in read_log(path):
        bucket = counts.setdefault(record["agent"], {"tool": 0, "llm": 0, "agent": 0, "error": 0})
        if record["event"] in bucket:
            bucket[record["event"]] += 1

    billable = {a: c["tool"] + c["llm"] for a, c in counts.items()}
    total = sum(billable.values()) or 1
    _banner(f"Call distribution · {path.name}")
    typer.echo(f"  {'agent':24s} {'tool':>6s} {'llm':>5s} {'share':>8s}")
    worst = 0.0
    for agent, value in sorted(billable.items(), key=lambda kv: -kv[1]):
        share = value / total
        worst = max(worst, share)
        typer.echo(
            f"  {agent:24s} {counts[agent]['tool']:6d} {counts[agent]['llm']:5d} "
            f"{share * 100:7.1f}%  {'#' * max(0, int(share * 30))}"
        )
    typer.echo(f"  {'TOTAL':24s} {'':6s} {'':5s} {total:7d} calls")
    if worst > 0.40:
        typer.secho(f"\n  FAIL: busiest agent holds {worst * 100:.1f}% (limit 40%)", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho(f"\n  OK: busiest agent holds {worst * 100:.1f}% (limit 40%)", fg=typer.colors.GREEN)


@app.command()
def scenarios() -> None:
    """List the test scenarios and their expected outcomes."""
    _banner("Test scenarios")
    for scenario in SCENARIOS:
        exists = Path(scenario.video).exists()
        typer.echo(
            f"\n  {scenario.name}  {'' if exists else '(input missing)'}\n"
            f"    input     {scenario.video}\n"
            f"    purpose   {scenario.description}\n"
            f"    expected  {scenario.expectation()}"
        )


@app.command()
def cases(limit: int = typer.Option(10, help="how many to list")) -> None:
    """List stored cases from the SQLite case store."""
    settings = get_settings()
    store = CaseStore(settings.store_path)
    rows = store.list_cases(limit)
    if not rows:
        typer.echo("no cases stored yet")
        return
    _banner(f"Stored cases ({len(rows)})")
    for case_id, updated in rows:
        raw = store.load_raw(case_id) or {}
        status = raw.get("status", {})
        done = sum(1 for v in status.values() if v == "done")
        typer.echo(f"  {case_id}  {updated[:19]}  {done}/{len(status)} stages done")


@app.command()
def show(case_id: str = typer.Argument(..., help="case id to dump")) -> None:
    """Dump one stored case file as JSON."""
    settings = get_settings()
    raw = CaseStore(settings.store_path).load_raw(case_id)
    if raw is None:
        typer.secho(f"case {case_id} not found", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.echo(json.dumps(raw, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
