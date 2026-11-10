#!/usr/bin/env python3
"""Draw the figures used in docs/REPORT.md.

    python scripts/make_report_figures.py

Writes docs/figures/architecture.png and docs/figures/load_distribution.png.
The load numbers are the measured ones from a full real-model demo run
(`safeschool demo` + `tools/load_report.py`); update LOAD if you re-measure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "figures"

INK = "#1B2B2E"
TEAL = "#0F4C5C"
TEAL_LIGHT = "#E3EEF0"
AMBER = "#E36414"
AMBER_LIGHT = "#FBE9DD"
GREY = "#6B7A7D"
STORE = "#EEF1F2"

#: measured: 10 scenarios, claude-sonnet-5-5, 262 tool + LLM calls
LOAD = [
    ("MotionAnalyst", 61, 0),
    ("VisionDetector", 52, 0),
    ("PrivacyReport", 39, 8),
    ("Ingest", 38, 0),
    ("Context", 24, 8),
    ("Verifier", 24, 8),
    ("Orchestrator", 0, 0),
]


def _box(ax, x, y, w, h, title, sub, *, fill, edge, llm=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=edge, lw=1.6))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
            fontsize=8.2, color=GREY, linespacing=1.25)
    if llm:
        ax.add_patch(FancyBboxPatch((x + w - 0.62, y + h - 0.13), 0.56, 0.24,
                                    boxstyle="round,pad=0.01,rounding_size=0.08",
                                    fc=AMBER, ec="none"))
        ax.text(x + w - 0.34, y + h - 0.01, "LLM", ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white")


def _arrow(ax, a, b, *, color=INK, style="-|>", ls="-", rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=color,
                                 lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def architecture() -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.4), dpi=200)
    ax.set_xlim(0, 15.3)
    ax.set_ylim(0.3, 6.9)
    ax.axis("off")

    workers = [
        ("1 · Ingest", "decode, segment,\nkeyframes", False),
        ("2 · Vision", "persons, poses,\nweapons (YOLOv8)", False),
        ("3 · Motion", "tracks, features,\nviolence score", False),
        ("4 · Context", "policy + literature\n→ category", True),
        ("5 · Verifier", "independent audit,\nfalse-alarm risk", True),
        ("6 · Report", "blur faces, write\nsummary, escalate", True),
    ]
    w, h, gap, y = 1.98, 1.3, 0.3, 2.75
    xs = [0.2 + i * (w + gap) for i in range(len(workers))]
    right = xs[-1] + w

    # orchestrator, centred over the workers
    ow = 7.0
    _box(ax, (right + 0.2) / 2 - ow / 2, 5.55, ow, 1.15, "0 · OrchestratorAgent",
         "plans the stages, routes typed messages, enforces budgets\nno domain work",
         fill=TEAL_LIGHT, edge=TEAL)

    for (title, sub, llm), x in zip(workers, xs):
        _box(ax, x, y, w, h, title, sub, fill="white", edge=TEAL, llm=llm)
    for i in range(len(workers) - 1):
        _arrow(ax, (xs[i] + w + 0.03, y + h / 2), (xs[i + 1] - 0.03, y + h / 2))

    # orchestrator dispatch
    cx = (right + 0.2) / 2
    _arrow(ax, (cx, 5.55), (cx, 4.62), color=TEAL, ls="--", lw=1.1)
    ax.text(cx + 0.12, 5.12, "plan + dispatch", fontsize=8, color=TEAL, va="center")

    # rerun loop, drawn between the workers and the orchestrator
    _arrow(ax, (xs[4] + w * 0.5, y + h + 0.02), (xs[1] + w * 0.5, y + h + 0.02),
           color=AMBER, rad=0.22, ls="--", lw=1.3)
    ax.text(xs[0] + 0.05, 4.62, "re-run at a lower\nthreshold (max 1)",
            fontsize=8, color=AMBER, va="center", linespacing=1.2)

    # blackboard
    ax.add_patch(FancyBboxPatch((0.2, 0.55), right - 0.2, 1.0,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=STORE, ec=GREY, lw=1.2))
    ax.text((right + 0.2) / 2, 1.17, "CaseFile — shared blackboard", ha="center",
            fontsize=10, fontweight="bold", color=INK)
    ax.text((right + 0.2) / 2, 0.82, "every stage artifact and message, persisted to SQLite",
            ha="center", fontsize=8.2, color=GREY)
    for x in xs:
        _arrow(ax, (x + w / 2, y - 0.03), (x + w / 2, 1.58), color=GREY, style="<|-|>", lw=0.9)

    # output
    _arrow(ax, (right + 0.03, y + h / 2), (right + 0.55, y + h / 2), color=INK)
    ax.text(right + 0.62, y + h / 2, "alert to\nduty channel\n(redacted\nframes only)",
            fontsize=8, color=INK, va="center", linespacing=1.2)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "architecture.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def load_distribution() -> Path:
    total = sum(t + l for _, t, l in LOAD)
    names = [n for n, _, _ in LOAD][::-1]
    tools = [t for _, t, _ in LOAD][::-1]
    llm = [l for _, _, l in LOAD][::-1]

    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=200)
    ax.barh(names, tools, color=TEAL, label="tool calls")
    ax.barh(names, llm, left=tools, color=AMBER, label="LLM calls")
    for i, (t, l) in enumerate(zip(tools, llm)):
        share = (t + l) / total * 100
        ax.text(t + l + 0.8, i, f"{share:.1f}%", va="center", fontsize=9, color=INK)
    limit = 0.40 * total
    ax.axvline(limit, color=AMBER, ls="--", lw=1.2)
    ax.text(limit - 1, len(names) - 0.6, "40% limit", color=AMBER, fontsize=9, ha="right")
    ax.set_xlabel(f"calls across 10 scenarios (total {total})", color=GREY, fontsize=9)
    ax.set_xlim(0, limit + 8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C9D1D3")
    ax.tick_params(colors=INK, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="center right")

    path = OUT / "load_distribution.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    for p in (architecture(), load_distribution()):
        print(f"  wrote {p.relative_to(REPO_ROOT)}")
