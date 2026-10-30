"""State definition for the multi-model debate graph."""

from __future__ import annotations

from typing import Any, Dict, List, NotRequired, Optional, TypedDict


class ModelAnswer(TypedDict):
    model_label: str
    vendor: str
    content: str
    critique: NotRequired[str]  # debate rounds only: this model's critique of the others
    round: int
    input_tokens: int
    output_tokens: int
    latency_ms: float


class PeerScore(TypedDict):
    scorer: str
    scored_model: str
    completeness: float
    accuracy: float
    reasoning: float
    nuance: float
    weighted_total: float
    justification: str


class DebateState(TypedDict):
    question: str
    num_rounds: int
    current_round: int
    answers: List[ModelAnswer]
    peer_scores: List[PeerScore]
    aggregated_scores: Dict[str, float]
    base_answer_model: str
    base_dimension_scores: Dict[str, float]
    synthesis_contributions: List[Dict[str, str]]
    final_answer: str
    transcript: List[Dict[str, Any]]
    token_usage: Dict[str, Any]
    converged: bool
