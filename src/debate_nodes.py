"""Graph nodes for the multi-model debate pipeline."""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List

from debate_state import DebateState, ModelAnswer, PeerScore
from models import DebateModel
from prompts import (
    DEBATE_ROUND_SYSTEM,
    DEBATE_ROUND_USER,
    FINAL_MERGE_SYSTEM,
    FINAL_MERGE_USER,
    INITIAL_ANSWER_SYSTEM,
    INITIAL_ANSWER_USER,
    PEER_SCORE_SYSTEM,
    PEER_SCORE_USER,
    SELF_CONSISTENCY_SYSTEM,
    SELF_CONSISTENCY_USER,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_USER,
)

logger = logging.getLogger(__name__)


def _log_transcript(state: DebateState, phase: str, model: str, content: str):
    state["transcript"].append({
        "phase": phase,
        "model": model,
        "round": state["current_round"],
        "content": content,
    })


def initial_answers_node(state: DebateState, models: List[DebateModel]) -> DebateState:
    """Round 0: Each model answers independently."""
    logger.info("=== INITIAL ANSWERS (Round 0) ===")
    state["current_round"] = 0

    for model in models:
        prompt = INITIAL_ANSWER_USER.format(question=state["question"])
        response = model.invoke(INITIAL_ANSWER_SYSTEM, prompt)

        answer: ModelAnswer = {
            "model_label": model.label,
            "vendor": model.vendor,
            "content": response.content,
            "round": 0,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
        }
        state["answers"].append(answer)
        _log_transcript(state, "initial_answer", model.label, response.content)
        logger.info(f"  {model.label}: {len(response.content)} chars, {response.latency_ms:.0f}ms")

    return state


def debate_round_node(state: DebateState, models: List[DebateModel]) -> DebateState:
    """One round of debate: each model sees others' answers and revises."""
    round_num = state["current_round"] + 1
    state["current_round"] = round_num
    logger.info(f"=== DEBATE ROUND {round_num} ===")

    prev_round = round_num - 1
    prev_answers = [a for a in state["answers"] if a["round"] == prev_round]

    new_answers = []
    for model in models:
        own = next((a for a in prev_answers if a["model_label"] == model.label), None)
        others = [a for a in prev_answers if a["model_label"] != model.label]

        # Others see each answer plus that model's critique of the rest, so challenges can be answered.
        other_text = "\n\n---\n\n".join(
            f"**{a['model_label']}:**\n{a['content']}"
            + (f"\n\n*{a['model_label']}'s critique of the others:*\n{a['critique']}" if a.get("critique") else "")
            for a in others
        )

        prompt = DEBATE_ROUND_USER.format(
            question=state["question"],
            own_answer=own["content"] if own else "(no previous answer)",
            other_answers=other_text,
        )
        response = model.invoke(DEBATE_ROUND_SYSTEM, prompt)
        # Keep the critique out of the answer itself, or it leaks into scoring and the final answer.
        critique, revised = _split_at_marker(response.content, "REVISED ANSWER")

        answer: ModelAnswer = {
            "model_label": model.label,
            "vendor": model.vendor,
            "content": revised,
            "critique": critique,
            "round": round_num,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
        }
        new_answers.append(answer)
        _log_transcript(state, f"debate_round_{round_num}", model.label, response.content)
        logger.info(f"  {model.label}: revised, {len(response.content)} chars")

    state["answers"].extend(new_answers)
    return state


def _parse_score(text: str) -> Dict[str, float]:
    """Parse dimension scores from model output.

    Tolerates the drift small models produce: "**Accuracy:** 8", "ACCURACY: 8 / 10", and
    misspelt labels such as "COMPLETENENESS: 8/10" (matched on the word stem).
    """
    stems = {"completeness": "complet", "accuracy": "accura", "reasoning": "reason", "nuance": "nuanc"}
    scores = {}
    for dim in ["completeness", "accuracy", "reasoning", "nuance"]:
        pattern = rf"\b{stems[dim]}\w*\**\s*:\s*\**\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            scores[dim] = min(float(match.group(1)), 10.0)
        else:
            logger.warning(f"Could not parse {dim} score, defaulting to 5.0. Raw text: {text[:200]}")
            scores[dim] = 5.0
    return scores


def _extract_justification(text: str) -> str:
    match = re.search(r"JUSTIFICATION:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def peer_scoring_node(state: DebateState, models: List[DebateModel], config: dict) -> DebateState:
    """Each model scores every other model's final-round answer."""
    logger.info("=== PEER SCORING ===")

    final_round = state["current_round"]
    final_answers = [a for a in state["answers"] if a["round"] == final_round]

    weights = {}
    for dim_cfg in config.get("evaluation", {}).get("scoring_dimensions", []):
        weights[dim_cfg["name"]] = dim_cfg.get("weight", 0.25)

    scores: List[PeerScore] = []

    for scorer_model in models:
        for answer in final_answers:
            if answer["model_label"] == scorer_model.label:
                continue

            prompt = PEER_SCORE_USER.format(
                question=state["question"],
                model_label=answer["model_label"],
                answer=answer["content"],
            )
            response = scorer_model.invoke(PEER_SCORE_SYSTEM, prompt)
            dim_scores = _parse_score(response.content)

            weighted_total = sum(
                dim_scores.get(dim, 5.0) * weights.get(dim, 0.25)
                for dim in weights
            )

            score: PeerScore = {
                "scorer": scorer_model.label,
                "scored_model": answer["model_label"],
                "completeness": dim_scores.get("completeness", 5.0),
                "accuracy": dim_scores.get("accuracy", 5.0),
                "reasoning": dim_scores.get("reasoning", 5.0),
                "nuance": dim_scores.get("nuance", 5.0),
                "weighted_total": weighted_total,
                "justification": _extract_justification(response.content),
            }
            scores.append(score)
            _log_transcript(
                state, "peer_score",
                scorer_model.label,
                f"Scored {answer['model_label']}: {weighted_total:.2f}/10\n{response.content}",
            )
            logger.info(f"  {scorer_model.label} → {answer['model_label']}: {weighted_total:.2f}/10")

    state["peer_scores"] = scores

    received: Dict[str, List[float]] = {}
    for s in scores:
        received.setdefault(s["scored_model"], []).append(s["weighted_total"])

    model_quality = {
        model: sum(vals) / len(vals) for model, vals in received.items()
    }

    quality_sum = sum(model_quality.values()) or 1.0
    scorer_weight = {model: model_quality.get(model, 5.0) / quality_sum for model in model_quality}

    weighted_agg: Dict[str, float] = {}
    weight_sums: Dict[str, float] = {}
    for s in scores:
        scored = s["scored_model"]
        scorer = s["scorer"]
        w = scorer_weight.get(scorer, 1.0 / len(models))
        weighted_agg[scored] = weighted_agg.get(scored, 0.0) + s["weighted_total"] * w
        weight_sums[scored] = weight_sums.get(scored, 0.0) + w

    state["aggregated_scores"] = {
        model: weighted_agg[model] / weight_sums[model]
        for model in weighted_agg
    }

    best_model = max(state["aggregated_scores"], key=state["aggregated_scores"].get)
    state["base_answer_model"] = best_model
    logger.info(f"  Scorer weights (by quality): {scorer_weight}")
    logger.info(f"  Best model by weighted peer score: {best_model} ({state['aggregated_scores'][best_model]:.2f})")

    base_scores_from_peers = [s for s in scores if s["scored_model"] == best_model]
    dim_avgs = {}
    for dim in ["completeness", "accuracy", "reasoning", "nuance"]:
        dim_avgs[dim] = sum(s[dim] for s in base_scores_from_peers) / len(base_scores_from_peers)
    state["base_dimension_scores"] = dim_avgs

    return state


def synthesis_node(state: DebateState, models: List[DebateModel]) -> DebateState:
    """Each non-base model suggests dimension-targeted improvements to the base answer."""
    logger.info("=== SYNTHESIS ===")

    final_round = state["current_round"]
    final_answers = [a for a in state["answers"] if a["round"] == final_round]
    base_answer = next(
        a for a in final_answers if a["model_label"] == state["base_answer_model"]
    )

    dim_scores = state.get("base_dimension_scores", {})
    weakest_dim = min(dim_scores, key=dim_scores.get) if dim_scores else "nuance"
    weakest_score = dim_scores.get(weakest_dim, 5.0)
    logger.info(f"  Targeting weakest dimension: {weakest_dim} ({weakest_score:.1f}/10)")

    all_answers_text = "\n\n---\n\n".join(
        f"**{a['model_label']}:**\n{a['content']}" for a in final_answers
    )

    system_prompt = SYNTHESIS_SYSTEM.format(
        weakest_dimension=weakest_dim,
        weakest_score=weakest_score,
    )

    contributions = []
    for model in models:
        if model.label == state["base_answer_model"]:
            continue

        prompt = SYNTHESIS_USER.format(
            question=state["question"],
            base_model=state["base_answer_model"],
            base_answer=base_answer["content"],
            all_answers=all_answers_text,
            weakest_dimension=weakest_dim,
            weakest_score=weakest_score,
        )
        response = model.invoke(system_prompt, prompt)
        contributions.append({
            "model": model.label,
            "suggestions": response.content,
        })
        _log_transcript(state, "synthesis", model.label, response.content)
        logger.info(f"  {model.label}: contributed suggestions targeting {weakest_dim}")

    state["synthesis_contributions"] = contributions
    return state


def _split_at_marker(text: str, marker: str):
    """Split at the last '<marker>:' line → (text before, text after). No marker → ('', text)."""
    parts = re.split(rf"^\W*{marker}:?\W*$|{marker}:", text, flags=re.IGNORECASE | re.MULTILINE)
    if len(parts) > 1 and parts[-1].strip():
        return "".join(parts[:-1]).strip(), parts[-1].strip()
    logger.warning(f"Output had no '{marker}:' marker; using the full text")
    return "", text


def _extract_final_answer(text: str) -> str:
    return _split_at_marker(text, "FINAL ANSWER")[1]


def final_merge_node(state: DebateState, models: List[DebateModel]) -> DebateState:
    """The base model merges improvements, then a self-consistency check verifies quality."""
    logger.info("=== FINAL MERGE ===")

    final_round = state["current_round"]
    base_answer = next(
        a for a in state["answers"]
        if a["round"] == final_round and a["model_label"] == state["base_answer_model"]
    )

    improvements_text = "\n\n---\n\n".join(
        f"**From {c['model']}:**\n{c['suggestions']}"
        for c in state["synthesis_contributions"]
    )

    base_model = next(m for m in models if m.label == state["base_answer_model"])

    if all("NO ADDITIONS NEEDED" in c["suggestions"].upper() for c in state["synthesis_contributions"]):
        state["final_answer"] = base_answer["content"]
        _log_transcript(state, "final_merge", base_model.label, "(no merge needed — base answer accepted)")
        logger.info(f"  Final answer: base accepted as-is ({len(state['final_answer'])} chars)")
        return state

    prompt = FINAL_MERGE_USER.format(
        question=state["question"],
        base_answer=base_answer["content"],
        improvements=improvements_text,
    )
    merge_response = base_model.invoke(FINAL_MERGE_SYSTEM, prompt)
    # Full output (with ACCEPT/REJECT decisions) goes to the transcript; only the answer is kept.
    _log_transcript(state, "final_merge", base_model.label, merge_response.content)
    merged_content = _extract_final_answer(merge_response.content)

    # Structural guard, independent of model judgement: a cut-off or collapsed merge never
    # replaces the base answer (small models have voted ACCEPT on a truncated merge).
    if merge_response.truncated or len(merged_content) < 0.5 * len(base_answer["content"]):
        reason = "truncated" if merge_response.truncated else (
            f"{len(merged_content)} chars vs base {len(base_answer['content'])}")
        state["final_answer"] = base_answer["content"]
        logger.warning(f"  Merge rejected by length guard ({reason}); using base answer")
        _log_transcript(state, "self_consistency_result", "guard", f"Merge rejected ({reason}); base answer kept")
        return state

    # Every non-base model votes on the merge, so no single model decides whether it stands.
    check_prompt = SELF_CONSISTENCY_USER.format(
        question=state["question"],
        base_answer=base_answer["content"],
        merged_answer=merged_content,
    )
    checkers = [m for m in models if m.label != base_model.label]
    reverts = 0
    for checker in checkers:
        check_response = checker.invoke(SELF_CONSISTENCY_SYSTEM, check_prompt)
        _log_transcript(state, "self_consistency", checker.label, check_response.content)
        # Read the VERDICT line only; a bare substring check also matches an echoed "ACCEPT or REVERT".
        verdict = re.search(r"VERDICT\**\s*:\**\s*(ACCEPT|REVERT)", check_response.content, re.IGNORECASE)
        if verdict and verdict.group(1).upper() == "REVERT":
            reverts += 1

    tally = f"{reverts}/{len(checkers)} voted REVERT"
    if reverts > len(checkers) / 2:
        state["final_answer"] = base_answer["content"]
        logger.info(f"  Quality gate: {tally} — merge degraded quality, using base answer")
        _log_transcript(state, "self_consistency_result", "gate", f"REVERTED to base answer ({tally})")
    else:
        state["final_answer"] = merged_content
        logger.info(f"  Quality gate: {tally} — merge approved")
        _log_transcript(state, "self_consistency_result", "gate", f"Merge accepted ({tally})")

    logger.info(f"  Final answer produced: {len(state['final_answer'])} chars")
    return state
