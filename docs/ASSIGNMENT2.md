# Assignment 2 — where each rubric item is satisfied

Чек-лист для проверки. Each row points at the file, command or test that
demonstrates the requirement.

## Scoring criteria (100 points)

| # | Criterion | Pts | Where it is satisfied |
|---|---|---:|---|
| 1 | Topic and justification of the multi-agent approach | 15 | [README — "Why a team of agents, not one"](../README.md#why-a-team-of-agents-not-one). The argument is structural, not efficiency-based: an agent that reviews its own work inherits the assumption that produced the error, and a single privacy boundary needs a single owner. |
| 2 | Architecture of 5+ agents, roles, interaction scheme | 30 | **7 agents** (1 orchestrator + 6 workers). Diagram in [README](../README.md#architecture) and [ARCHITECTURE.md](ARCHITECTURE.md). Scheme = pipeline + orchestrator + shared blackboard, declared explicitly. `safeschool agents` prints the live table from the code. |
| 3 | Agent specification — I/O, message formats, tools | 20 | [ARCHITECTURE.md § message schemas](ARCHITECTURE.md#message-schemas) and [AGENTS_AND_COMMUNICATION.md](AGENTS_AND_COMMUNICATION.md) (measured message flow). Every payload is a pydantic model in [`messages.py`](../src/safeschool/messages.py) with `extra="forbid"`. Each agent declares `role`, `input_type`, `output_type`, `tools`, `done_criterion`. |
| 4 | Role balance — no overloaded agent | 10 | `python tools/load_report.py --markdown 'logs/*.jsonl'`. Across all 10 scenarios the busiest agent holds **25.5%** of 239 tool+LLM calls; the orchestrator holds 0%. Enforced by `test_no_agent_exceeds_forty_percent_of_calls`. |
| 5 | Prototype — ≥2 agents exchange data | 20 | All **six** workers exchange validated messages end to end. `safeschool demo` runs 10 scenarios; `test_two_agents_exchange_validated_messages` asserts each hand-off explicitly. |
| 6 | Repository and README | 5 | [README.md](../README.md) — install, run, commands, a real transcript, results, and a stated list of limitations. |

## Hard requirements from §2 of the specification

| Requirement | Status | Evidence |
|---|---|---|
| ≥4 agents, one clear role each | 7 agents | `safeschool agents` |
| Roles do not duplicate | yes | distinct `stage` per agent, asserted in `test_agent_team_meets_the_specification` |
| No agent >40% of LLM/tool calls, proven by logs | 25.5% max | `tools/load_report.py` |
| ≤5 tools per agent | max 4 | enforced at class-definition time in `runtime.Agent.__init_subclass__` |
| Orchestrator only plans and distributes | yes | 0 tool/LLM calls, asserted in `test_orchestrator_does_no_domain_work` |
| Interaction scheme shown on a diagram | yes | mermaid diagram in README and ARCHITECTURE.md |
| Structured messages (JSON / typed objects) | yes | pydantic models in an `AgentMessage` envelope |
| ≥2 external tools or data sources | 5 | Anthropic API · YOLOv8 weights · 50-paper literature corpus · policy KB · SQLite |
| Task state stored explicitly | yes | `CaseFile` blackboard persisted to SQLite — `safeschool show <case_id>` |
| Step and time limits, loop protection | yes | `runtime.Budget`, `runtime.LoopGuard`; tests in `test_runtime.py` |
| Error handling — retries, readable failure | yes | 3 attempts with backoff; `corrupt_file` and `missing_file` scenarios |
| Logging of all agent and tool calls | yes | `logging_bus.py`, written by the runtime so no agent can skip it |
| API keys not stored in code | yes | env / gitignored `.env`; `test_secrets_never_reach_the_log` |

## Already in place for Assignment 4 and the exam

| Future requirement | Status now |
|---|---|
| All agents implemented and working together | done |
| Load ≤40% confirmed by logs | done |
| Tool and data integration | done — 5 sources |
| Error handling, limits, loop protection | done |
| Call logging | done |
| 5+ test scenarios passing (exam: 10+) | **10 scenarios, 10/10 passing**, plus 43 unit tests |

Remaining for Assignment 4: replace the heuristic scorer with a trained temporal
classifier, add real benchmark footage, and a simple web interface. For the exam:
the 5–10 page report and the slide deck.

## How to verify in five commands

```bash
.venv/bin/python -m safeschool.cli agents       # 7 agents, roles, I/O, tools
.venv/bin/python -m safeschool.cli demo         # 10/10 scenarios
.venv/bin/python tools/load_report.py --markdown 'logs/*.jsonl'   # ≤40% proof
.venv/bin/python -m pytest -q                   # 43 tests
.venv/bin/python -m safeschool.cli show <case>  # explicit stored state
```
