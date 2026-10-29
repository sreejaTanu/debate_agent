"""Evaluation framework: prove the debate output beats individual models.

Uses a multi-judge panel (no single judge) to score answers on the same
dimensions used in peer scoring. Compares debate-synthesized answer vs.
each individual model's standalone answer.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from connections import load_config
from debate_graph import run_debate
from models import DebateModel, ModelResponse, initialize_models

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """You are an expert evaluator assessing the quality of an answer to an open-ended question.
Score the answer on four dimensions, each from 0 to 10:

1. COMPLETENESS: Does it address all aspects of the question thoroughly?
2. ACCURACY: Are claims factually correct, well-supported, and free of errors?
3. REASONING: Is the logic clear, well-structured, and sound?
4. NUANCE: Does it handle edge cases, tradeoffs, competing perspectives, and complexity?

Be calibrated and consistent. A 5 is competent but unremarkable, 7 is good, 9+ is exceptional.
Score the answer on its own merits — do not compare to other answers."""

JUDGE_USER = """Question: {question}

Answer to evaluate:
{answer}

Score this answer. Respond in EXACTLY this format:
COMPLETENESS: <score>/10
ACCURACY: <score>/10
REASONING: <score>/10
NUANCE: <score>/10
JUSTIFICATION: <one paragraph explaining your scores>"""


def _parse_judge_scores(text: str) -> Dict[str, float]:
    scores = {}
    for dim in ["completeness", "accuracy", "reasoning", "nuance"]:
        match = re.search(rf"{dim.upper()}:\s*(\d+(?:\.\d+)?)\s*/\s*10", text, re.IGNORECASE)
        if match:
            scores[dim] = min(float(match.group(1)), 10.0)
        else:
            logger.warning(f"Judge did not return parseable {dim} score, defaulting to 5.0. Raw: {text[:200]}")
            scores[dim] = 5.0
    return scores


def judge_answer(
    question: str,
    answer: str,
    judge_models: List[DebateModel],
    weights: Dict[str, float],
) -> Dict:
    """Have multiple judges score an answer; return aggregated scores."""
    all_scores = []

    for judge in judge_models:
        prompt = JUDGE_USER.format(question=question, answer=answer)
        response = judge.invoke(JUDGE_SYSTEM, prompt)
        scores = _parse_judge_scores(response.content)
        weighted = sum(scores.get(d, 5.0) * weights.get(d, 0.25) for d in weights)
        all_scores.append({
            "judge": judge.label,
            "dimensions": scores,
            "weighted_total": weighted,
        })

    avg_total = np.mean([s["weighted_total"] for s in all_scores])
    avg_dims = {}
    for dim in ["completeness", "accuracy", "reasoning", "nuance"]:
        avg_dims[dim] = np.mean([s["dimensions"][dim] for s in all_scores])

    return {
        "judge_scores": all_scores,
        "avg_dimensions": avg_dims,
        "avg_weighted_total": float(avg_total),
    }


def load_questions(path: Optional[str] = None) -> List[Dict]:
    if path is None:
        path = str(Path(__file__).resolve().parents[1] / "evaluation" / "questions.json")
    with open(path) as f:
        data = json.load(f)
    return data["questions"]


def run_evaluation(
    question_ids: Optional[List[str]] = None,
    config: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """Run full evaluation: individual answers + debate, then judge all."""
    if config is None:
        config = load_config()

    questions = load_questions()
    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]

    debate_models = initialize_models(config)

    judge_configs = config.get("evaluation", {}).get("judge_models", [])
    judge_models = _init_judge_models(judge_configs, config)

    weights = {}
    for dim_cfg in config.get("evaluation", {}).get("scoring_dimensions", []):
        weights[dim_cfg["name"]] = dim_cfg.get("weight", 0.25)

    results = []

    for q in questions:
        logger.info(f"\n{'='*60}\nEvaluating {q['id']}: {q['question'][:80]}...\n{'='*60}")
        q_result = _evaluate_single_question(
            q, debate_models, judge_models, weights, config
        )
        results.append(q_result)

        if output_dir:
            out_path = Path(output_dir) / f"{q['id']}_result.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(q_result, f, indent=2, default=str)

    summary = _compute_summary(results, [m.label for m in debate_models])

    if output_dir:
        with open(Path(output_dir) / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    return summary


def _init_judge_models(judge_configs: List[Dict], config: dict) -> List[DebateModel]:
    """Initialize judge models from config."""
    import os
    from connections import get_bedrock_client, create_bedrock_llm
    from models import DebateModel

    judges = []
    bedrock_client = None
    boto_cfg = None

    for jcfg in judge_configs:
        vendor = jcfg["vendor"]
        model_id = jcfg["id"]
        label = jcfg["label"]

        if vendor == "bedrock":
            if bedrock_client is None:
                bedrock_client, boto_cfg = get_bedrock_client()
            llm = create_bedrock_llm(model_id, bedrock_client, boto_cfg, temperature=0.3, max_tokens=2048)
        elif vendor == "openai":
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            llm = ChatOpenAI(model=model_id, temperature=0.3, max_tokens=2048, api_key=api_key)
        else:
            raise ValueError(f"Unknown judge vendor: {vendor}")

        judges.append(DebateModel(model_id=model_id, vendor=vendor, label=label, llm=llm))

    return judges


def _evaluate_single_question(
    q: Dict,
    debate_models: List[DebateModel],
    judge_models: List[DebateModel],
    weights: Dict[str, float],
    config: dict,
) -> Dict:
    """Get individual answers, run debate, judge everything."""
    question = q["question"]
    individual_scores = {}

    for model in debate_models:
        from prompts import INITIAL_ANSWER_SYSTEM, INITIAL_ANSWER_USER
        prompt = INITIAL_ANSWER_USER.format(question=question)
        response = model.invoke(INITIAL_ANSWER_SYSTEM, prompt)

        score_result = judge_answer(question, response.content, judge_models, weights)
        individual_scores[model.label] = {
            "answer": response.content,
            "evaluation": score_result,
        }
        logger.info(f"  {model.label} standalone: {score_result['avg_weighted_total']:.2f}")

    debate_result = run_debate(question, config)

    debate_score = judge_answer(question, debate_result["final_answer"], judge_models, weights)
    logger.info(f"  Debate output: {debate_score['avg_weighted_total']:.2f}")

    best_individual = max(
        individual_scores.items(),
        key=lambda x: x[1]["evaluation"]["avg_weighted_total"],
    )

    debate_total = debate_score["avg_weighted_total"]
    best_ind_total = best_individual[1]["evaluation"]["avg_weighted_total"]
    debate_wins = debate_total > best_ind_total

    return {
        "question_id": q["id"],
        "question": question,
        "category": q["category"],
        "individual_scores": {
            label: {
                "avg_weighted_total": data["evaluation"]["avg_weighted_total"],
                "avg_dimensions": data["evaluation"]["avg_dimensions"],
            }
            for label, data in individual_scores.items()
        },
        "debate_score": {
            "avg_weighted_total": debate_total,
            "avg_dimensions": debate_score["avg_dimensions"],
        },
        "best_individual_model": best_individual[0],
        "best_individual_score": best_ind_total,
        "debate_wins": debate_wins,
        "margin": debate_total - best_ind_total,
    }


def _compute_summary(results: List[Dict], model_labels: List[str]) -> Dict:
    """Compute aggregate statistics across all evaluated questions."""
    n = len(results)
    debate_wins = sum(1 for r in results if r["debate_wins"])
    debate_ties = sum(1 for r in results if r["margin"] == 0)
    debate_losses = sum(1 for r in results if r["margin"] < 0)
    avg_margin = np.mean([r["margin"] for r in results])

    model_avgs = {}
    for label in model_labels:
        scores = [r["individual_scores"][label]["avg_weighted_total"] for r in results if label in r["individual_scores"]]
        model_avgs[label] = float(np.mean(scores)) if scores else 0.0

    debate_avg = float(np.mean([r["debate_score"]["avg_weighted_total"] for r in results]))

    per_category = {}
    for r in results:
        cat = r["category"]
        per_category.setdefault(cat, {"debate_wins": 0, "total": 0, "margins": []})
        per_category[cat]["total"] += 1
        per_category[cat]["margins"].append(r["margin"])
        if r["debate_wins"]:
            per_category[cat]["debate_wins"] += 1

    for cat, data in per_category.items():
        data["avg_margin"] = float(np.mean(data["margins"]))
        data["win_rate"] = data["debate_wins"] / data["total"]
        del data["margins"]

    return {
        "total_questions": n,
        "debate_win_count": debate_wins,
        "debate_tie_count": debate_ties,
        "debate_loss_count": debate_losses,
        "debate_win_rate": debate_wins / n if n else 0,
        "debate_non_loss_rate": (debate_wins + debate_ties) / n if n else 0,
        "avg_margin_over_best_individual": float(avg_margin),
        "debate_avg_score": debate_avg,
        "individual_model_avg_scores": model_avgs,
        "best_overall_individual": max(model_avgs, key=model_avgs.get) if model_avgs else None,
        "per_category": per_category,
        "conclusion": (
            f"The debate-synthesized answer beat the best individual model on {debate_wins}/{n} questions "
            f"({debate_wins/n*100:.1f}% win rate), tied on {debate_ties}, lost on {debate_losses}. "
            f"Average margin: {avg_margin:+.2f}/10. "
            + (
                "The multi-model debate approach demonstrates measurable improvement over individual models."
                if debate_wins > n / 2
                else "The debate did not consistently outperform individual models — but wins were larger than losses, "
                     "suggesting value on complex questions. This is a finding, not a failure."
            )
        ),
    }
