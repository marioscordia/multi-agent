# The agent team and how it communicates

A short report on the seven agents of the SafeSchool system, the messages they
exchange, and the measured behaviour of that exchange. Figures in this document
are taken from actual runs, not from the design intent — reproduce them with
`safeschool demo` and `python tools/load_report.py`.

---

## 1. The team

Seven agents: one orchestrator and six workers. Each owns exactly one stage, and
no two share a role.

| # | Agent | Owns | Reasons with an LLM? |
|---|---|---|---|
| 0 | `OrchestratorAgent` | planning, routing, budgets | no |
| 1 | `IngestAgent` | video decoding and segmentation | no |
| 2 | `VisionDetectorAgent` | per-frame perception | no |
| 3 | `MotionAnalystAgent` | temporal association and scoring | no |
| 4 | `ContextAgent` | interpretation against policy and literature | **yes** |
| 5 | `VerifierAgent` | independent audit | **yes** |
| 6 | `PrivacyReportAgent` | redaction, reporting, escalation | no |

Only two agents use the language model. The other five do deterministic or
numeric work, where an LLM would add cost, latency and non-reproducibility
without adding capability. This is a deliberate division: **language is used
where judgement is needed, not as the default mechanism for everything.**

---

## 2. How they are wired

The system combines three of the four interaction schemes named in the
assignment specification, because each solves a different problem:

- **Pipeline** — the stages have a genuine data dependency. Motion cannot be
  computed before detection exists; policy cannot be applied before there is a
  score. The sequence is the structure of the problem, not a convention.
- **Orchestrator** — something must own the budget, the retry policy and the one
  control-flow decision. Leaving that to a worker would mean a worker setting its
  own retry policy.
- **Shared blackboard** — the Verifier must see *every* prior artifact to audit
  the chain, and the report needs all of them. Threading six payloads through six
  signatures would couple every stage to every other.

Hierarchy is deliberately **not** used: with six stages in a linear dependency, a
tree of sub-agents would add indirection and nothing else.

---

## 3. The message envelope

Agents never pass raw dictionaries or positional tuples. Every payload is a
pydantic model with `extra="forbid"`, carried inside a transport envelope:

```python
class AgentMessage(BaseModel):
    msg_id: str
    case_id: str
    sender: str
    recipient: str
    type: MessageType          # one of nine
    payload: dict[str, Any]    # a Payload model, dumped to JSON
    created_at: datetime
    attempt: int = 1
    causation_id: str | None = None
    overrides: dict[str, Any] = {}   # control metadata — never inside the payload
```

Two properties follow from this, and both matter in practice:

**A contract violation fails at the boundary.** The receiving agent validates the
payload against its declared input schema before doing any work. A misspelled
field is an error in the hand-off, not a mystery three stages later.

**Control data is separated from domain data.** When the Verifier asks for a
re-run at a lower detection threshold, that threshold describes the *process*,
not the incident. It travels in `overrides`. An earlier version put it inside the
`ClipSet` payload, which the strict schema correctly rejected — the re-run path
did not work at all until this was separated.

---

## 4. The normal exchange

One complete case, measured (`fixture://adult_child`):

| # | From | To | Message | Payload |
|---:|---|---|---|---:|
| 1 | `OrchestratorAgent` | `IngestAgent` | `case_request` | 173 B |
| 2 | `IngestAgent` | `VisionDetectorAgent` | `clip_set` | 521 B |
| 3 | `VisionDetectorAgent` | `MotionAnalystAgent` | `detection_set` | 10,479 B |
| 4 | `MotionAnalystAgent` | `ContextAgent` | `motion_assessment` | 741 B |
| 5 | `ContextAgent` | `VerifierAgent` | `context_verdict` | 1,431 B |
| 6 | `VerifierAgent` | `PrivacyReportAgent` | `verification` | 455 B |

Six messages, each the previous stage's output. The size profile is the
interesting part: the `detection_set` is twenty times larger than anything else,
because it carries every bounding box and all 17 keypoints per person per frame.
The MotionAnalystAgent's job is precisely to **collapse that volume into
meaning** — 10.5 KB of geometry becomes a 741-byte assessment with eight named
features and a score. Everything downstream reasons over the small object.

This also explains why the two LLM agents are cheap to run. They never see the
10 KB payload; they see the summary.

---

## 5. The exception: a re-run

The Verifier is the only agent that can alter the flow, and it can do so once.
The same case with the context stage forced into a verdict the evidence does not
support (one person in frame, `physical_violence` asserted):

| # | From | To | Message | attempt | overrides |
|---:|---|---|---|---:|---|
| 1 | `OrchestratorAgent` | `IngestAgent` | `case_request` | 1 | — |
| 2 | `IngestAgent` | `VisionDetectorAgent` | `clip_set` | 1 | — |
| 3 | `VisionDetectorAgent` | `MotionAnalystAgent` | `detection_set` | 1 | — |
| 4 | `MotionAnalystAgent` | `ContextAgent` | `motion_assessment` | 1 | — |
| 5 | `ContextAgent` | `VerifierAgent` | `context_verdict` | 1 | — |
| **6** | **`VerifierAgent`** | **`VisionDetectorAgent`** | **`clip_set`** | **2** | **`confidence_threshold: 0.15`** |
| 7 | `VisionDetectorAgent` | `MotionAnalystAgent` | `detection_set` | 2 | — |
| 8 | `MotionAnalystAgent` | `ContextAgent` | `motion_assessment` | 2 | — |
| 9 | `ContextAgent` | `VerifierAgent` | `context_verdict` | 2 | — |
| 10 | `VerifierAgent` | `PrivacyReportAgent` | `verification` | 2 | — |

Message 6 is the loop back. Note the sender: the Verifier is recorded as the
originator, because it requested the re-run — the orchestrator executes it but
does not decide it.

Final verdict: **reject**, with issues
`interaction_requires_two`, `escalation_above_evidence`, `fixture_backed_run`,
`no_literature_grounding`, `rerun_budget_spent`.

The chain contradicted itself, one re-run at a lower threshold did not resolve
it, and the system then **rejected rather than cycling**. The
`rerun_budget_spent` issue exists so the report states *why* no further attempt
was made — a rejection with no stated reason is not actionable for whoever reads
it.

---

## 6. Why the Verifier must be a separate agent

This is the strongest argument for a multi-agent design here, so it is worth
stating precisely.

An agent that checks its own output inherits the assumption that produced the
error. If the ContextAgent concludes "physical violence" from a faulty premise,
asking the same agent to review that conclusion re-applies the same premise. The
VerifierAgent sees only the **artifacts** — what each stage actually produced —
not the reasoning that produced them, and it holds no stake in any verdict.

Its audit runs in two layers, in this order:

1. **Eight deterministic rules**, which cannot be argued with. Examples: a
   violence score with zero people detected; an interaction category with fewer
   than two people; a weapon verdict with no weapon detection; an escalation
   above the evidence threshold.
2. **An LLM critique**, which may only *add* issues or *lower* confidence.

A blocking rule failure cannot be overturned by the model. That ordering is the
point: the deterministic layer is the floor, and the language model works above
it, never beneath it.

---

## 7. Measured communication load

Across all ten scenarios (`tools/load_report.py`, 239 tool and LLM calls):

| agent | tool calls | LLM calls | share |
|---|---:|---:|---:|
| `MotionAnalystAgent` | 61 | 0 | 25.5% |
| `VisionDetectorAgent` | 52 | 0 | 21.8% |
| `PrivacyReportAgent` | 40 | 0 | 16.7% |
| `IngestAgent` | 38 | 0 | 15.9% |
| `ContextAgent` | 16 | 8 | 10.0% |
| `VerifierAgent` | 16 | 8 | 10.0% |
| `OrchestratorAgent` | 0 | 0 | 0.0% |

Two observations worth defending:

**The orchestrator sits at exactly zero.** It plans and routes and does nothing
else, which the specification requires of it. The log is what proves this rather
than the claim — `test_orchestrator_does_no_domain_work` asserts it on every run.

**The busiest agent holds 25.5%, against a 40% limit.** This was not free. An
early version ran YOLO inference once per frame, which put the
VisionDetectorAgent at **77.2%** of all calls. Batching inference per clip —
faster in itself — brought it to 32.6%. Moving identity association from the
vision agent to the motion agent, where temporal reasoning belongs, settled it at
the current distribution. The role boundary and the load balance turned out to be
the same question.

---

## 8. What is logged

Every agent invocation, tool call and LLM call is appended as one JSON line, by
the runtime rather than by the agents — so an agent cannot omit its own calls.
Each record carries the timestamp, case id, agent, event type, name, success
flag, duration, attempt number, and token counts for LLM calls. Key names are
scrubbed before writing, verified by `test_secrets_never_reach_the_log`.

Per case, the typical profile is:

```
OrchestratorAgent      {'system': 2}                      plan + completion
IngestAgent            {'tool': 2, 'system': 1, 'agent': 1}
VisionDetectorAgent    {'tool': 1, 'system': 1, 'agent': 1}
MotionAnalystAgent     {'tool': 2, 'agent': 1}
ContextAgent           {'tool': 2, 'llm': 1, 'agent': 1}
VerifierAgent          {'tool': 2, 'llm': 1, 'agent': 1}
PrivacyReportAgent     {'tool': 6, 'agent': 1}
```

---

## 9. Shared state

Alongside the messages, every stage result is written to a `CaseFile` blackboard
persisted in SQLite. Messages carry data *forward*; the blackboard makes it
available *sideways* — the Verifier reads all four prior artifacts, and the
report agent reads five. It also means a crashed run leaves an inspectable
record: `safeschool show <case_id>` prints the stage statuses, the artifacts and
the notes for any case the system has ever handled.

---

## 10. Summary

| Property | Mechanism |
|---|---|
| Structured messages | pydantic models, `extra="forbid"`, inside an `AgentMessage` envelope |
| Validation | at the receiving boundary, before any work |
| Control vs domain data | `overrides` on the envelope, never in the payload |
| Flow control | one agent (the orchestrator), one decision point (the Verifier) |
| Loop safety | one re-run maximum, then a decision; identical input twice aborts |
| Shared state | `CaseFile` blackboard, persisted to SQLite |
| Auditability | every call logged by the runtime, secrets scrubbed |
| Load balance | 25.5% maximum, 0% for the orchestrator |
