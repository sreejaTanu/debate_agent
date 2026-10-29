"""Transcript generator: produces a human-readable debate transcript.

Generates both JSON (machine-readable) and Markdown (human-readable) transcripts
as required by the problem statement: "Send us the transcript of that exchange."
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from debate_state import DebateState


def save_transcript_json(state: DebateState, output_path: str) -> str:
    transcript_data = {
        "question": state["question"],
        "timestamp": datetime.now().isoformat(),
        "num_rounds": state["num_rounds"],
        "models_involved": list({a["model_label"] for a in state["answers"]}),
        "phases": state["transcript"],
        "peer_scores": state["peer_scores"],
        "aggregated_scores": state["aggregated_scores"],
        "base_answer_model": state["base_answer_model"],
        "final_answer": state["final_answer"],
        "token_usage": state["token_usage"],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(transcript_data, f, indent=2, default=str)
    return output_path


def save_transcript_markdown(state: DebateState, output_path: str) -> str:
    lines = []
    lines.append(f"# Multi-Model Debate Transcript")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Question:** {state['question']}")
    lines.append(f"**Models:** {', '.join(set(a['model_label'] for a in state['answers']))}")
    lines.append(f"**Debate Rounds:** {state['num_rounds']}")
    lines.append("")

    lines.append("---")
    lines.append("")

    lines.append("## Phase 1: Independent Answers (Round 0)")
    lines.append("")
    round_0 = [a for a in state["answers"] if a["round"] == 0]
    for a in round_0:
        lines.append(f"### {a['model_label']} ({a['vendor']})")
        lines.append("")
        lines.append(a["content"])
        lines.append("")
        lines.append(f"*Tokens: {a['input_tokens']} in / {a['output_tokens']} out | Latency: {a['latency_ms']:.0f}ms*")
        lines.append("")

    for r in range(1, state["num_rounds"] + 1):
        lines.append("---")
        lines.append("")
        lines.append(f"## Phase 2: Debate Round {r}")
        lines.append("")
        round_answers = [a for a in state["answers"] if a["round"] == r]
        for a in round_answers:
            lines.append(f"### {a['model_label']} (revised)")
            lines.append("")
            lines.append(a["content"])
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Phase 3: Peer Scoring")
    lines.append("")
    lines.append("Each model scored every other model's final answer (no self-scoring).")
    lines.append("")

    lines.append("| Scorer | Scored Model | Completeness | Accuracy | Reasoning | Nuance | Weighted |")
    lines.append("|--------|-------------|:------------:|:--------:|:---------:|:------:|:--------:|")
    for s in state["peer_scores"]:
        lines.append(
            f"| {s['scorer']} | {s['scored_model']} | "
            f"{s['completeness']:.1f} | {s['accuracy']:.1f} | "
            f"{s['reasoning']:.1f} | {s['nuance']:.1f} | "
            f"**{s['weighted_total']:.2f}** |"
        )
    lines.append("")

    lines.append("### Aggregated Scores")
    lines.append("")
    for model, score in sorted(state["aggregated_scores"].items(), key=lambda x: -x[1]):
        marker = " ← **BASE ANSWER**" if model == state["base_answer_model"] else ""
        lines.append(f"- **{model}**: {score:.2f}/10{marker}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Phase 4: Synthesis")
    lines.append("")
    lines.append(f"Base answer from **{state['base_answer_model']}** (highest peer score).")
    lines.append("Other models suggested improvements:")
    lines.append("")
    for c in state.get("synthesis_contributions", []):
        lines.append(f"### Suggestions from {c['model']}")
        lines.append("")
        lines.append(c["suggestions"])
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Final Debate-Synthesized Answer")
    lines.append("")
    lines.append(state["final_answer"])
    lines.append("")

    if state.get("token_usage"):
        lines.append("---")
        lines.append("")
        lines.append("## Token Usage")
        lines.append("")
        tu = state["token_usage"]
        lines.append(f"- Total input tokens: {tu.get('total_input_tokens', 'N/A')}")
        lines.append(f"- Total output tokens: {tu.get('total_output_tokens', 'N/A')}")
        lines.append(f"- Wall clock: {tu.get('wall_clock_seconds', 0):.1f}s")
        lines.append("")

    content = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path
