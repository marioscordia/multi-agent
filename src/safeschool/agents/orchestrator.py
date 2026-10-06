"""Agent 0 — OrchestratorAgent.

Role: planning and routing only. It builds the stage plan, hands each worker the
previous stage's output as a typed message, enforces the step and time budgets,
and acts on a re-run request from the Verifier.

It performs no domain work: it never inspects a frame, never computes a feature,
never decides a category. Its two tools are the case store and the logger. This
separation is a requirement of the assignment specification, and it is also what
keeps its share of tool/LLM calls near zero in the load report.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..messages import (
    AgentMessage,
    CaseRequest,
    ErrorPayload,
    IncidentReport,
    MessageType,
    Plan,
    PlanStep,
    Verification,
)
from ..runtime import (
    Agent,
    Budget,
    BudgetExceeded,
    Context,
    LoopDetected,
    LoopGuard,
    SafeSchoolError,
    StageFailed,
)
from ..state import CaseFile, StageStatus
from .context import ContextAgent
from .ingest import IngestAgent
from .motion import MotionAnalystAgent
from .report import PrivacyReportAgent
from .verifier import VerifierAgent
from .vision import VisionDetectorAgent

#: (stage, type of the message that stage receives). The message a stage
#: receives is by construction the previous stage's output, which is what makes
#: the hand-off contract checkable.
_PIPELINE = (
    ("ingest", MessageType.CASE_REQUEST),
    ("vision", MessageType.CLIP_SET),
    ("motion", MessageType.DETECTION_SET),
    ("context", MessageType.MOTION_ASSESSMENT),
    ("verify", MessageType.CONTEXT_VERDICT),
    ("report", MessageType.VERIFICATION),
)


class RunResult:
    """What a completed (or failed) run returns to the CLI."""

    def __init__(
        self,
        *,
        case: CaseFile,
        plan: Plan,
        report: IncidentReport | None = None,
        error: ErrorPayload | None = None,
    ) -> None:
        self.case = case
        self.plan = plan
        self.report = report
        self.error = error

    @property
    def ok(self) -> bool:
        return self.report is not None


class OrchestratorAgent(Agent):
    name = "OrchestratorAgent"
    stage = "orchestrate"
    role = "Plan the stage sequence, route typed messages, enforce budgets and re-runs"
    tools = ("case_store.save", "logger.log")
    done_criterion = "the plan ran to completion, or aborted with a stated reason"
    input_type = MessageType.CASE_REQUEST
    output_type = MessageType.PLAN

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.workers = {
            "ingest": IngestAgent(),
            "vision": VisionDetectorAgent(),
            "motion": MotionAnalystAgent(),
            "context": ContextAgent(llm),
            "verify": VerifierAgent(llm),
            "report": PrivacyReportAgent(),
        }

    # -- planning (the orchestrator's only actual output) ----------------- #

    def handle(self, message: AgentMessage, ctx: Context) -> Plan:
        request = self.parse_input(message, CaseRequest)
        return self.build_plan(request, ctx)

    def build_plan(self, request: CaseRequest, ctx: Context) -> Plan:
        reasons = {
            "ingest": "a video container must be decoded and segmented before anything else",
            "vision": "clips must be perceived before motion can be aggregated",
            "motion": "features and a score must exist before policy can be applied",
            "context": "a score is meaningless until interpreted against school policy",
            "verify": "the chain must be audited by an agent that did not produce it",
            "report": "only after verification may a redacted report be emitted",
        }
        steps = [
            PlanStep(
                stage=stage,
                agent=self.workers[stage].name,
                depends_on=([_PIPELINE[i - 1][0]] if i else []),
                reason=reasons[stage],
            )
            for i, (stage, _) in enumerate(_PIPELINE)
        ]
        return Plan(
            case_id=request.case_id,
            steps=steps,
            max_steps=ctx.settings.max_steps,
            deadline_seconds=ctx.settings.deadline_seconds,
        )

    # -- execution -------------------------------------------------------- #

    def run_case(self, request: CaseRequest, ctx: Context) -> RunResult:
        """Drive the whole pipeline for one case."""
        plan = self.build_plan(request, ctx)
        # the plan is logged rather than stored as a stage artifact: it is the
        # orchestrator's routing decision, not evidence about the incident
        ctx.logger.log(
            case_id=request.case_id, agent=self.name, event="system", name="plan",
            detail={"stages": [s.stage for s in plan.steps],
                    "max_steps": plan.max_steps,
                    "deadline_s": plan.deadline_seconds,
                    "llm": self.llm.model_name},
        )

        payload = request
        sender = self.name
        overrides: dict[str, float] = {}
        stage_index = 0

        while stage_index < len(_PIPELINE):
            stage, msg_type = _PIPELINE[stage_index]
            worker = self.workers[stage]

            # a Verifier re-run hint travels as a transport-level override, so
            # the worker's input schema stays untouched
            envelope = AgentMessage.wrap(
                case_id=request.case_id, sender=sender, recipient=worker.name,
                type=msg_type, payload=payload,
                attempt=ctx.case.rerun_count + 1,
                overrides=overrides if stage == "vision" else None,
            )

            try:
                result = worker.execute(envelope, ctx)
            except (BudgetExceeded, LoopDetected) as exc:
                return self._abort(ctx, request, stage, worker.name, exc, recoverable=False)
            except StageFailed as exc:
                return self._abort(ctx, request, stage, worker.name, exc, recoverable=True)
            except SafeSchoolError as exc:
                return self._abort(ctx, request, stage, worker.name, exc, recoverable=exc.recoverable)

            # --- the one control-flow decision the orchestrator makes ---- #
            if stage == "verify":
                verification: Verification = result  # type: ignore[assignment]
                if verification.verdict == "rerun" and verification.rerun_hint:
                    hint = verification.rerun_hint
                    ctx.logger.log(
                        case_id=request.case_id, agent=self.name, event="system",
                        name="rerun_requested",
                        detail={"from_stage": hint.stage, "parameter": hint.parameter,
                                "value": hint.value,
                                "reruns_used": ctx.case.rerun_count,
                                "limit": ctx.settings.max_reruns},
                    )
                    overrides[hint.parameter] = hint.value
                    ctx.case.reset_stages(["vision", "motion", "context", "verify"])
                    ctx.case.note(
                        f"verifier requested a re-run from stage '{hint.stage}' with "
                        f"{hint.parameter}={hint.value}"
                    )
                    payload = ctx.case.require("ingest")
                    sender = worker.name
                    stage_index = 1  # back to vision
                    continue

            payload = result
            sender = worker.name
            stage_index += 1

        report: IncidentReport = ctx.case.require("report")  # type: ignore[assignment]
        ctx.logger.log(
            case_id=request.case_id, agent=self.name, event="system", name="case_complete",
            detail={"tier": report.escalation_tier, "category": report.category,
                    "steps_used": ctx.budget.steps_used,
                    "elapsed_s": round(ctx.budget.elapsed, 2)},
        )
        return RunResult(case=ctx.case, plan=plan, report=report)

    def _abort(self, ctx: Context, request: CaseRequest, stage: str, agent: str,
               exc: BaseException, *, recoverable: bool) -> RunResult:
        error = ErrorPayload(
            case_id=request.case_id, stage=stage, agent=agent,
            error_type=type(exc).__name__, message=str(exc),
            attempts=ctx.settings.max_attempts if isinstance(exc, StageFailed) else 1,
            recoverable=recoverable,
        )
        ctx.case.mark(stage, StageStatus.FAILED)
        ctx.case.note(f"aborted at stage '{stage}': {error.message}")
        ctx.logger.log(
            case_id=request.case_id, agent=self.name, event="error", name="case_aborted",
            ok=False, detail=error.model_dump(mode="json"),
        )
        return RunResult(case=ctx.case, plan=self.build_plan(request, ctx), error=error)


def build_context(settings, logger, case: CaseFile) -> Context:
    """Assemble the execution context for one case."""
    return Context(
        settings=settings,
        logger=logger,
        budget=Budget(
            max_steps=settings.max_steps,
            deadline_seconds=float(settings.deadline_seconds),
        ),
        loop_guard=LoopGuard(max_repeats=2),
        case=case,
    )
