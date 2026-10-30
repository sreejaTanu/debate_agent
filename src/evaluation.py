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
from debate_nodes import _parse_score
from transcript import save_transcript_json, save_transcript_markdown
from models import DebateModel, SpendCeilingExceeded, initialize_models, is_dry_run

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


# Judges and peers use the same rubric and output format, so they share one parser.
_parse_judge_scores = _parse_score


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


PAIRWISE_SYSTEM = """You are an expert evaluator comparing two answers to the same open-ended question.
Judge them on completeness, accuracy, reasoning and nuance, taken together.
Ignore length and formatting: a longer answer is not better unless it is more useful.
Pick the better answer. Answer TIE only if they are genuinely equal in quality."""

PAIRWISE_USER = """Question: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Which answer is better? Respond in EXACTLY this format:
WINNER: A or B or TIE
REASON: <one sentence>"""


def judge_pair(question: str, first: str, second: str, judge_models: List[DebateModel]) -> Dict:
    """Side-by-side comparison: share of judgements preferring `first` (0..1, 0.5 = even).

    Each judge sees the pair twice with positions swapped, cancelling position bias.
    """
    points, verdicts = [], []
    for judge in judge_models:
        for first_is_a in (True, False):
            a, b = (first, second) if first_is_a else (second, first)
            resp = judge.invoke(PAIRWISE_SYSTEM, PAIRWISE_USER.format(question=question, answer_a=a, answer_b=b))
            m = re.search(r"WINNER\**\s*:\**\s*\**\s*(A|B|TIE)\b", resp.content, re.IGNORECASE)
            w = m.group(1).upper() if m else "TIE"  # unparseable → no preference
            pt = 0.5 if w == "TIE" else float((w == "A") == first_is_a)
            points.append(pt)
            verdicts.append({"judge": judge.label, "first_shown_as": "A" if first_is_a else "B", "winner": w})
    return {"preference_for_first": float(np.mean(points)), "verdicts": verdicts}


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
    _check_judge_independence(judge_configs, config.get("models", []))

    weights = {}
    for dim_cfg in config.get("evaluation", {}).get("scoring_dimensions", []):
        weights[dim_cfg["name"]] = dim_cfg.get("weight", 0.25)

    results = []

    for q in questions:
        logger.info(f"\n{'='*60}\nEvaluating {q['id']}: {q['question'][:80]}...\n{'='*60}")
        try:
            q_result = _evaluate_single_question(
                q, debate_models, judge_models, weights, config
            )
        except SpendCeilingExceeded as e:
            # Spend ceiling hit: stop and report on the questions completed so far.
            logger.error(f"Stopping evaluation early: {e}")
            break
        debate_state = q_result.pop("_debate_state")
        results.append(q_result)

        if output_dir:
            out_path = Path(output_dir) / f"{q['id']}_result.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(q_result, f, indent=2, default=str)
            # Keep the full exchange behind every score, so any question's result can be read.
            save_transcript_markdown(debate_state, str(Path(output_dir) / "transcripts" / f"{q['id']}.md"))
            save_transcript_json(debate_state, str(Path(output_dir) / "transcripts" / f"{q['id']}.json"))

    if not results:
        raise RuntimeError("No questions completed — nothing to summarise.")

    summary = _compute_summary(results, [m.label for m in debate_models])
    summary["judges"] = [j.label for j in judge_models]
    summary["dry_run"] = is_dry_run()
    summary["questions_requested"] = len(questions)

    if output_dir:
        with open(Path(output_dir) / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    return summary


def _init_judge_models(judge_configs: List[Dict], config: dict) -> List[DebateModel]:
    """Initialize judge models from config. Optional judges without credentials are skipped."""
    import os
    from models import build_llm

    temperature = config.get("evaluation", {}).get("judge_temperature", 0.0)
    judges = []
    clients: Dict = {}

    for jcfg in judge_configs:
        if (jcfg.get("optional") and jcfg["vendor"] == "openai"
                and not os.getenv("OPENAI_API_KEY") and not is_dry_run()):
            logger.warning(f"Skipping optional judge {jcfg['label']}: OPENAI_API_KEY not set")
            continue
        llm = build_llm(jcfg, temperature, 1024, clients)
        judges.append(DebateModel(
            model_id=jcfg["id"], vendor=jcfg.get("model_vendor", jcfg["vendor"]),
            label=jcfg["label"], llm=llm,
        ))

    if len(judges) < 2:
        raise ValueError(f"Need at least 2 judges so no single model decides; got {len(judges)}")
    return judges


def _check_judge_independence(judge_configs: List[Dict], model_configs: List[Dict]) -> None:
    """Warn if a judge shares a vendor with a debater (self-preference bias)."""
    debate_vendors = {m.get("model_vendor", m["vendor"]) for m in model_configs}
    for j in judge_configs:
        if j.get("model_vendor", j["vendor"]) in debate_vendors:
            logger.warning(f"Judge {j['label']} shares a vendor with a debater — scores may be biased")


def _evaluate_single_question(
    q: Dict,
    debate_models: List[DebateModel],
    judge_models: List[DebateModel],
    weights: Dict[str, float],
    config: dict,
) -> Dict:
    """Run the debate, then judge each model's round-0 answer and the final answer.

    Round-0 answers are produced independently (no model has seen another's answer),
    so they are exactly the single-model baseline — no extra calls needed.
    """
    question = q["question"]
    debate_result = run_debate(question, config)

    individual_scores = {}
    for answer in (a for a in debate_result["answers"] if a["round"] == 0):
        score_result = judge_answer(question, answer["content"], judge_models, weights)
        individual_scores[answer["model_label"]] = {
            "answer": answer["content"],
            "evaluation": score_result,
        }
        logger.info(f"  {answer['model_label']} standalone: {score_result['avg_weighted_total']:.2f}")

    debate_score = judge_answer(question, debate_result["final_answer"], judge_models, weights)
    logger.info(f"  Debate output: {debate_score['avg_weighted_total']:.2f}")

    best_individual = max(
        individual_scores.items(),
        key=lambda x: x[1]["evaluation"]["avg_weighted_total"],
    )

    debate_total = debate_score["avg_weighted_total"]
    best_ind_total = best_individual[1]["evaluation"]["avg_weighted_total"]
    debate_wins = debate_total > best_ind_total

    # Side-by-side: debate answer vs each model's own answer (both orders, every judge).
    pairwise = {}
    for label, data in individual_scores.items():
        pr = judge_pair(question, debate_result["final_answer"], data["answer"], judge_models)
        pairwise[label] = pr
        logger.info(f"  Pairwise: debate preferred over {label} {pr['preference_for_first']:.0%} of the time")

    return {
        "_debate_state": debate_result,
        "question_id": q["id"],
        "question": question,
        "category": q["category"],
        "individual_scores": {
            label: {
                "avg_weighted_total": data["evaluation"]["avg_weighted_total"],
                "avg_dimensions": data["evaluation"]["avg_dimensions"],
                "answer_chars": len(data["answer"]),
            }
            for label, data in individual_scores.items()
        },
        "debate_score": {
            "avg_weighted_total": debate_total,
            "avg_dimensions": debate_score["avg_dimensions"],
            "answer_chars": len(debate_result["final_answer"]),
        },
        "base_answer_model": debate_result["base_answer_model"],
        "debate_usage": {k: debate_result["token_usage"][k] for k in
                         ("model_calls", "total_input_tokens", "total_output_tokens", "wall_clock_seconds")},
        "debate_rounds_run": debate_result["current_round"],
        "pairwise_preference_for_debate": {k: v["preference_for_first"] for k, v in pairwise.items()},
        "pairwise_verdicts": {k: v["verdicts"] for k, v in pairwise.items()},
        "judge_scores": {
            "debate": debate_score["judge_scores"],
            **{label: d["evaluation"]["judge_scores"] for label, d in individual_scores.items()},
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
    best_overall = max(model_avgs, key=model_avgs.get) if model_avgs else None

    # Headline baseline: the single best model across the whole set — what a firm would
    # deploy today. The per-question "best individual" above is a stricter oracle baseline.
    vs_best = [r["debate_score"]["avg_weighted_total"] - r["individual_scores"][best_overall]["avg_weighted_total"]
               for r in results]
    vs_best_wins = sum(1 for d in vs_best if d > 0)
    vs_best_losses = sum(1 for d in vs_best if d < 0)
    p_value = _wilcoxon_p(vs_best)

    usage_keys = ("model_calls", "total_input_tokens", "total_output_tokens", "wall_clock_seconds")
    avg_usage = {k: float(np.mean([r["debate_usage"][k] for r in results])) for k in usage_keys}

    debate_chars = float(np.mean([r["debate_score"]["answer_chars"] for r in results]))
    best_chars = float(np.mean([r["individual_scores"][best_overall]["answer_chars"] for r in results]))

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
        "best_overall_individual": best_overall,
        "vs_best_overall": {
            "model": best_overall,
            "wins": vs_best_wins,
            "losses": vs_best_losses,
            "ties": n - vs_best_wins - vs_best_losses,
            "avg_margin": float(np.mean(vs_best)),
            "wilcoxon_p_value": p_value,
        },
        "pairwise_vs_models": _pairwise_summary(results, model_labels),
        "avg_answer_chars": {"debate": debate_chars, best_overall: best_chars},
        "avg_debate_usage_per_question": avg_usage,
        "per_category": per_category,
        "conclusion": _conclusion(n, best_overall, vs_best_wins, vs_best_losses, float(np.mean(vs_best)),
                                  p_value, debate_wins, debate_ties, debate_losses, float(avg_margin)),
    }


def _pairwise_summary(results: List[Dict], model_labels: List[str]) -> Dict:
    """Per model: on how many questions the judges preferred the debate answer side by side.

    A question counts as a win if the debate answer took more than half of the judgements
    (judges x both orders). Significance: two-sided sign test on non-tied questions.
    """
    from scipy.stats import binomtest
    out = {}
    for label in model_labels:
        prefs = [r["pairwise_preference_for_debate"][label] for r in results
                 if label in r.get("pairwise_preference_for_debate", {})]
        if not prefs:
            continue
        wins, losses = sum(p > 0.5 for p in prefs), sum(p < 0.5 for p in prefs)
        out[label] = {
            "wins": wins, "ties": len(prefs) - wins - losses, "losses": losses,
            "avg_preference_for_debate": float(np.mean(prefs)),
            "sign_test_p_value": float(binomtest(wins, wins + losses).pvalue) if wins + losses else None,
        }
    return out


def _wilcoxon_p(diffs: List[float]) -> Optional[float]:
    """Two-sided Wilcoxon signed-rank p-value for paired score differences (None if undefined)."""
    nonzero = [d for d in diffs if d != 0]
    if len(nonzero) < 6:
        return None
    from scipy.stats import wilcoxon
    return float(wilcoxon(nonzero).pvalue)


def _conclusion(n, best, wins, losses, margin, p, oracle_wins, oracle_ties, oracle_losses, oracle_margin) -> str:
    sig = "not computed (too few non-tied questions)" if p is None else f"p = {p:.3f}"
    if margin > 0 and p is not None and p < 0.05:
        verdict = "The debate answer beats the best single model, and the difference is statistically significant."
    elif margin > 0:
        verdict = ("The debate answer scores higher on average, but the difference is not statistically "
                   "significant on this set — treat as suggestive, not proven.")
    else:
        verdict = ("The debate answer does NOT beat the best single model on this set. "
                   "That is a finding, not a failure: the extra cost is not justified by quality here.")
    return (
        f"On {n} questions, the debate answer beat {best} (best model overall) on {wins}, lost on {losses}, "
        f"average margin {margin:+.2f}/10 (Wilcoxon {sig}). Against the per-question oracle "
        f"(whichever model scored highest on each question) it won {oracle_wins}, tied {oracle_ties}, "
        f"lost {oracle_losses}, margin {oracle_margin:+.2f}. {verdict}"
    )
