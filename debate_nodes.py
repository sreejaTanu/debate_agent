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

        other_text = "\n\n---\n\n".join(
            f"**{a['model_label']}:**\n{a['content']}" for a in others
        )

        prompt = DEBATE_ROUND_USER.format(
            question=state["question"],
            own_answer=own["content"] if own else "(no previous answer)",
            other_answers=other_text,
        )
        response = model.invoke(DEBATE_ROUND_SYSTEM, prompt)

        answer: ModelAnswer = {
            "model_label": model.label,
            "vendor": model.vendor,
            "content": response.content,
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
    """Parse dimension scores from model output."""
    scores = {}
    for dim in ["completeness", "accuracy", "reasoning", "nuance"]:
        pattern = rf"{dim.upper()}:\s*(\d+(?:\.\d+)?)\s*/\s*10"
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
    merged_content = merge_response.content
    _log_transcript(state, "final_merge", base_model.label, merged_content)

    checker = next((m for m in models if m.label != base_model.label), base_model)
    check_prompt = SELF_CONSISTENCY_USER.format(
        question=state["question"],
        base_answer=base_answer["content"],
        merged_answer=merged_content,
    )
    check_response = checker.invoke(SELF_CONSISTENCY_SYSTEM, check_prompt)
    _log_transcript(state, "self_consistency", checker.label, check_response.content)

    if "REVERT" in check_response.content.upper():
        state["final_answer"] = base_answer["content"]
        logger.info(f"  Self-consistency check: REVERT — merge degraded quality, using base answer")
        _log_transcript(state, "self_consistency_result", checker.label, "REVERTED to base answer")
    else:
        state["final_answer"] = merged_content
        logger.info(f"  Self-consistency check: ACCEPT — merge approved")

    logger.info(f"  Final answer produced: {len(state['final_answer'])} chars")
    return state
