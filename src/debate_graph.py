"""LangGraph-based multi-model debate pipeline.

Flow:
  initial_answers → debate_round (×N, with early stop) → peer_scoring → synthesis → final_merge
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from langgraph.graph import END, StateGraph

from connections import load_config
from debate_nodes import (
    debate_round_node,
    final_merge_node,
    initial_answers_node,
    peer_scoring_node,
    synthesis_node,
)
from debate_state import DebateState
from models import DebateModel, get_spend_tracker, initialize_models

logger = logging.getLogger(__name__)


def _cosine_similarity(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity — robust to rephrasing and word reordering."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text_a, text_b])
    return float(sk_cosine(tfidf[0:1], tfidf[1:2])[0][0])


def _check_convergence(state: DebateState, threshold: float = 0.85) -> bool:
    """Check if models' answers converged using cosine similarity between rounds."""
    current = state["current_round"]
    if current < 1:
        return False

    prev_answers = {a["model_label"]: a["content"] for a in state["answers"] if a["round"] == current - 1}
    curr_answers = {a["model_label"]: a["content"] for a in state["answers"] if a["round"] == current}

    similarities = []
    for label in prev_answers:
        if label in curr_answers:
            sim = _cosine_similarity(prev_answers[label], curr_answers[label])
            similarities.append(sim)
            logger.info(f"  {label}: round {current-1}→{current} cosine similarity = {sim:.3f}")

    if not similarities:
        return False

    avg_sim = sum(similarities) / len(similarities)
    converged = avg_sim >= threshold
    if converged:
        logger.info(f"  Convergence detected: avg cosine similarity {avg_sim:.3f} >= {threshold} — skipping remaining rounds")
    else:
        logger.info(f"  No convergence yet: avg cosine similarity {avg_sim:.3f} < {threshold}")
    return converged


def _should_continue_debate(state: DebateState, num_rounds: int) -> str:
    if state.get("converged", False):
        return "peer_scoring"
    if state["current_round"] < num_rounds:
        return "debate_round"
    return "peer_scoring"


def build_debate_graph(models: List[DebateModel], config: dict) -> StateGraph:
    num_rounds = config.get("debate", {}).get("num_rounds", 2)

    graph = StateGraph(DebateState)

    convergence_threshold = config.get("debate", {}).get("convergence_threshold", 0.85)

    def _debate_round_with_convergence(s):
        s = debate_round_node(s, models)
        s["converged"] = _check_convergence(s, convergence_threshold)
        return s

    graph.add_node("initial_answers", lambda s: initial_answers_node(s, models))
    graph.add_node("debate_round", _debate_round_with_convergence)
    graph.add_node("peer_scoring", lambda s: peer_scoring_node(s, models, config))
    graph.add_node("synthesis", lambda s: synthesis_node(s, models))
    graph.add_node("final_merge", lambda s: final_merge_node(s, models))

    graph.set_entry_point("initial_answers")

    graph.add_conditional_edges(
        "initial_answers",
        lambda s: _should_continue_debate(s, num_rounds),
        {"debate_round": "debate_round", "peer_scoring": "peer_scoring"},
    )
    graph.add_conditional_edges(
        "debate_round",
        lambda s: _should_continue_debate(s, num_rounds),
        {"debate_round": "debate_round", "peer_scoring": "peer_scoring"},
    )
    graph.add_edge("peer_scoring", "synthesis")
    graph.add_edge("synthesis", "final_merge")
    graph.add_edge("final_merge", END)

    return graph.compile()


def run_debate(question: str, config: Optional[dict] = None) -> DebateState:
    if config is None:
        config = load_config()

    models = initialize_models(config)
    compiled_graph = build_debate_graph(models, config)

    initial_state: DebateState = {
        "question": question,
        "num_rounds": config.get("debate", {}).get("num_rounds", 2),
        "current_round": -1,
        "answers": [],
        "peer_scores": [],
        "aggregated_scores": {},
        "base_answer_model": "",
        "base_dimension_scores": {},
        "synthesis_contributions": [],
        "final_answer": "",
        "transcript": [],
        "token_usage": {},
        "converged": False,
    }

    logger.info(f"Starting debate on: {question[:100]}...")
    tracker = get_spend_tracker()
    before = tracker.snapshot()
    start = time.time()

    result = compiled_graph.invoke(initial_state)

    elapsed = time.time() - start
    logger.info(f"Debate completed in {elapsed:.1f}s")

    # Every model call in this debate (answers, scores, synthesis, merge, gate), not just answers.
    after = tracker.snapshot()
    result["token_usage"] = {
        "model_calls": after["calls"] - before["calls"],
        "total_input_tokens": after["input_tokens"] - before["input_tokens"],
        "total_output_tokens": after["output_tokens"] - before["output_tokens"],
        "debate_spend_usd": round(after["usd"] - before["usd"], 4),
        "cumulative_spend_usd": round(tracker.total_usd, 4),
        "spend_ceiling_usd": tracker.max_usd,
        "wall_clock_seconds": elapsed,
    }

    return result
