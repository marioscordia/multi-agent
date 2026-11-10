# Where to find the results of a run

Every run writes its results to four places, listed here from most readable to
most detailed. Paths are relative to the repository root.

## 1. The incident report (start here)

**`data/cases/<case_id>/report.md`** is one human-readable report per case. It contains:

- **Summary for staff:** a short summary and a "what to do" sentence, written by
  the PrivacyReportAgent's language model (or by a fixed template when no model
  is available)
- the alert level, category, severity, and the Verifier's verdict
- what was detected (scores per clip)
- the reasoning of the ContextAgent and of the VerifierAgent
- the privacy section: how many faces were blurred
- the research papers cited

In the same folder:

- `report.json`: the same content as data
- `redacted/`: the key frames **with faces blurred**. These are the only images
  the system ever outputs.

## 2. The alerts that were "sent"

**`data/channel/alerts.jsonl`** has one line per alert sent to staff, admin or
emergency. In this prototype the channel is a local file, not a real messaging
system. Only cases above the `log` level appear here, so harmless videos do not.

## 3. The full work of every agent

List all saved cases:

```bash
.venv/bin/python -m safeschool.cli cases
```

Show one case in full: every agent's output, the messages between agents, and
the notes recorded along the way:

```bash
.venv/bin/python -m safeschool.cli show <case_id>
```

This is read from `data/cases.sqlite3`, the shared case file (blackboard).

## 4. The call logs (proof of the load distribution)

**`logs/run-<time>-<case_id>.jsonl`** has one line for every agent, tool and LLM
call, with timing, success or failure, and token counts. LLM calls are the lines
with `"event": "llm"`; their `name` shows the purpose and model, for example
`classify@claude-sonnet-5-5`, or `classify@mock-llm` in mock mode.

To get the load-split table (the ≤40% requirement) from the logs:

```bash
.venv/bin/python tools/load_report.py --markdown 'logs/*.jsonl'
```

## Real model or mock?

Without an `ANTHROPIC_API_KEY` the system runs in mock mode: the reasoning
agents use fixed rules and the staff summary comes from a template. Each report
says who wrote its summary (`summary_source` in `report.json`, and the
"Written by" line in `report.md`). To get results from the real model, put the
key in `.env` and run:

```bash
.venv/bin/python -m safeschool.cli demo
```

The newest folders in `data/cases/` then hold the model-written reports.

Note: `data/` and `logs/` are in `.gitignore`, so these results exist only on
the machine that ran the system and are not part of the repository.
