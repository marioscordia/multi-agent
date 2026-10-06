# Architecture

## Interaction scheme

The specification lists four schemes — orchestrator, hierarchy, pipeline, shared
memory. This system combines three of them deliberately, because each handles a
different concern:

- **Pipeline** — the six stages have a real data dependency. Motion cannot be
  computed before detection; policy cannot be applied before there is a score.
  The order is not a convention, it is the structure of the problem.
- **Orchestrator** — something has to own the budget, the retries and the one
  control-flow decision (whether to re-run). Putting that in a worker would mean
  a worker deciding its own retry policy.
- **Shared blackboard** — the Verifier needs to see *every* earlier artifact to
  audit the chain, and the report needs all of them. Threading six payloads
  through six signatures would couple every stage to every other.

What is *not* used: hierarchy. There are no sub-agents spawning sub-agents. With
six stages and a linear dependency, a tree would add indirection and nothing else.

## Stage sequence

```
CaseRequest
   │
   ▼  OrchestratorAgent builds the plan, opens the CaseFile, starts the budget
   │
   ├─1─► IngestAgent ──────────► ClipSet          (decode, segment, keyframes)
   ├─2─► VisionDetectorAgent ──► DetectionSet     (persons, poses, weapons)
   ├─3─► MotionAnalystAgent ───► MotionAssessment (tracks, features, score)
   ├─4─► ContextAgent ─────────► ContextVerdict   (policy + literature)
   ├─5─► VerifierAgent ────────► Verification     (independent audit)
   │        │
   │        └── verdict "rerun" ──► back to stage 2 with a lower threshold
   │                                (at most once; then it must decide)
   │
   └─6─► PrivacyReportAgent ───► IncidentReport   (redact, write, escalate)
```

Every arrow is an `AgentMessage` whose payload is validated against the receiving
agent's declared input schema.

## Message schemas

All payloads inherit from `Payload` (`extra="forbid"`), so an unexpected field is
an error at the boundary rather than a silent mystery three stages later.

```python
class AgentMessage(BaseModel):
    msg_id: str
    case_id: str
    sender: str
    recipient: str
    type: MessageType
    payload: dict[str, Any]       # a Payload, dumped to JSON
    created_at: datetime
    attempt: int = 1
    causation_id: str | None = None
    overrides: dict[str, Any] = {}   # control metadata — never inside the payload
```

The `overrides` field exists for a specific reason. When the Verifier asks for a
re-run at a lower detection threshold, that threshold is *control* information,
not a fact about the incident. Putting it inside the `ClipSet` payload would
either corrupt the domain object or fail its strict schema — which is exactly
what happened during development, and is why the field is on the envelope.

| Stage | Payload | Key fields |
|---|---|---|
| — | `CaseRequest` | `video_path`, `camera_id`, `location`, `allow_fixture_fallback` |
| 0 | `Plan` | `steps[]` (stage, agent, depends_on, reason), `max_steps`, `deadline_seconds` |
| 1 | `ClipSet` | `fps`, `duration_seconds`, `width/height`, `clips[]`, `source` |
| 1 | `Clip` / `Keyframe` | time interval, frame count, keyframes tagged with `frame_index` |
| 2 | `DetectionSet` | `per_clip[]`, `model_name`, `source`, `confidence_threshold` |
| 2 | `Detection` | `frame_index`, `t`, `cls`, `confidence`, `bbox` (normalized), `track_id`, 17 `keypoints` |
| 3 | `MotionAssessment` | `per_clip[]`, `top_clip_id`, `top_score`, `scorer` |
| 3 | `MotionFeatures` | the eight named features below |
| 4 | `ContextVerdict` | `category`, `severity` 1–4, `is_child_context`, `policy_refs[]`, `citations[]` |
| 5 | `Verification` | `verdict`, `confidence`, `false_positive_risk`, `issues[]`, `rerun_hint` |
| 6 | `IncidentReport` | `escalation_tier`, `summary`, `redacted_keyframes[]`, `faces_redacted` |

`Keyframe` carries its `frame_index` for a reason: redaction must blur the faces
that are in *that* frame. An earlier version passed bare paths, and every frame's
detections were applied to every keyframe — 60 blur operations for 6 faces.

## The violence scorer

Interpretable by design. Eight named features, fixed documented weights, a
logistic combination, and a rationale string naming the three features that drove
the result. A black box would give the Verifier nothing to audit and the report
nothing to say.

| Feature | What it measures | Weight |
|---|---|---:|
| `contact_frames_ratio` | fraction of frames where two people are within 0.18 | **+2.6** |
| `limb_extension_peak` | wrist thrown from the torso, or an ankle above hip level | **+1.8** |
| `speed_variance` | burstiness of motion | +1.4 |
| `proximity` | 1 − *median* frame-wise closest distance | +1.0 |
| `peak_speed` | 95th percentile track speed, in body-lengths/s | +0.9 |
| `pose_asymmetry` | left/right upper-body imbalance | +0.6 |
| `crowding` | mean persons per frame above 2 | **−0.9** |
| | bias | −2.9 |

Three of these encode lessons from mistakes made while building it:

- **`proximity` uses the median, not the minimum.** The global minimum is the
  single closest instant in the clip, which one person walking past another
  drives to near zero. On real CCTV of a supermarket aisle that produced a
  `physical_violence` alert.
- **`limb_extension_peak` excludes ankles from the arm term.** Legs are long, so
  including them pinned the feature at ~0.6 for every upright person — it
  measured "has legs", not "is striking". Kicks are a separate term that only
  fires when an ankle rises above hip level.
- **`crowding` is negative.** A dense corridor is normal school movement. Without
  this the busiest times of day would be the loudest.

Weighting is dominated by sustained contact and limb extension rather than raw
speed, because speed alone does not separate a fight from brisk walking.

## Identity association

The detector reports boxes with no identity. Without identities, the motion stage
has to guess which box in frame *N* is which in frame *N+1*, and with more than
two people it guesses wrong — an identity switch looks like a person teleporting,
which registers as enormous speed and variance. That is a false-alarm generator.

A greedy IoU association over consecutive sampled frames fixes it. It lives in the
**motion** agent, not the vision agent: associating detections over time is
temporal reasoning, and perception should stay per-frame. (It also keeps the two
agents' call counts proportionate.)

## The adult-on-child check

Distinguishing clause SSP-2.1 from a peer altercation needs evidence of a stature
difference. The naive version — the ratio of the tallest to the shortest box in
frame — measures camera perspective, not age: a shopper near the lens is twice the
height of one down the aisle. It produced an `adult_on_child_abuse` **emergency**
alert on supermarket CCTV.

Two constraints make the ratio mean what it claims:

1. only the **closest pair** in each frame is compared — a size difference is
   evidence of an adult-child interaction only if the two are interacting;
2. their **feet must be at a similar image height**. On a ground plane, equal foot
   height means equal depth, so a remaining size difference is real stature.

Boxes clipped by the frame edge are excluded: a truncated box has an unknown
height and cannot be evidence about anyone's stature.

## Privacy boundary

Three rules, enforced structurally rather than by convention:

| Rule | Enforcement |
|---|---|
| PR-1 — no frame leaves the pipeline with an identifiable face | `PrivacyReportAgent` is the only component that opens an image file. Redaction failure withholds the frame instead of emitting it |
| PR-2 — reasoning stages receive aggregated features and text only | `ContextAgent._build_prompt` is given numbers, policy text and literature snippets; `test_context_agent_prompt_contains_no_image_paths` asserts no image path can appear |
| PR-3 — reports carry no identifiers | camera id, timestamp and track numbers only |

Redaction is mosaic-then-blur: downsampling to a 4×4 block destroys facial detail
outright, and the subsequent blur removes the block edges that would otherwise
still carry structure. A plain Gaussian at a sane kernel size left eyes legible.

## Reliability

| Concern | Mechanism | Where |
|---|---|---|
| Runaway loops | step budget, wall-clock deadline, checked before each agent | `runtime.Budget` |
| Agents ping-ponging | identical `(agent, input)` more than twice aborts | `runtime.LoopGuard` |
| Re-run cycles | at most one re-run; then the Verifier must accept or reject, and says why | `VerifierAgent`, `OrchestratorAgent` |
| Transient failures | 3 attempts, exponential backoff | `runtime.Agent.execute` |
| Unrecoverable failures | fail immediately rather than burning the budget | `SafeSchoolError.recoverable` |
| Audit | every call logged by the runtime, not by the agents | `logging_bus.CallLogger` |
| Secret leakage | key names scrubbed before writing | `logging_bus._scrub` |

## Design decisions worth defending

**Why is the scorer a heuristic rather than a trained model?** Because at this
stage an interpretable baseline that can be argued about is worth more than an
unexplainable number. The Verifier audits the *reasoning*, and a trained model
with no rationale gives it nothing to work with. The interface is built so the
classifier can replace the scorer without touching anything else.

**Why BM25 and not embeddings?** The corpus is 50 documents. A lexical index
builds in 0.04s, needs no vector database, and the retrieval is explainable — you
can see which terms matched. Embeddings would add a dependency and a model call
to solve a problem this corpus does not have.

**Why does a fixture-backed run still produce a full report?** Because the
alternative is a system that cannot be demonstrated without a GPU, weights and an
API key. The honesty constraint is handled separately: the payload is marked
`source="fixture"` and verification raises `fixture_backed_run`, so it is visible
in the output and impossible to mistake for a detection.
