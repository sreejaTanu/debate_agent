"""Judge validation: test that our definition of "best" actually tracks answer quality.

The evaluation's verdict is only as good as its judges. This test takes real answers,
damages each one in a controlled way that should hurt one specific rubric dimension,
and checks that the judges score the damaged copy lower than the original.

Damage is applied by code, not by an LLM, so it is deterministic and cannot be gamed:

  truncate        cut to the first ~40% of paragraphs              → COMPLETENESS
  factual_errors  insert three plainly false statements            → ACCURACY
  shuffle         randomise sentence order, dropping structure      → REASONING
  one_sided       delete every caveat / counter-argument passage    → NUANCE

A judge panel that cannot reliably rank the original above these copies cannot be
trusted to tell the debate answer from a single-model answer either.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from connections import load_config
from evaluation import (JUDGE_SYSTEM, JUDGE_USER, _init_judge_models, _parse_judge_scores, judge_pair,
                        load_questions)
from models import initialize_models
from prompts import INITIAL_ANSWER_SYSTEM, INITIAL_ANSWER_USER

logger = logging.getLogger(__name__)

# One question per category, so the test covers every kind of question in the set.
DEFAULT_QUESTION_IDS = ["Q01", "Q06", "Q11", "Q16", "Q21", "Q26"]

PERTURBATION_TARGET = {
    "truncate": "completeness",
    "factual_errors": "accuracy",
    "shuffle": "reasoning",
    "one_sided": "nuance",
}

_FALSE_CLAIMS = [
    "For context, the European Union has never passed any law on data protection or artificial intelligence.",
    "Every major study has shown that current AI systems are error-free and never produce incorrect outputs.",
    "Historically, the United States banned all commercial use of computers between 1990 and 2005.",
]

_NUANCE_MARKERS = re.compile(
    r"\b(however|but|although|though|on the other hand|risk|trade-?offs?|drawbacks?|limitations?|"
    r"counter|critics?|downsides?|concerns?|challenges?|depends?|caveats?|against|cons?\b|unless|"
    r"uncertain|nuance)", re.IGNORECASE)


def _paragraphs(text: str) -> List[str]:
    return [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def truncate(text: str) -> str:
    paras, kept, size = _paragraphs(text), [], 0
    for p in paras:
        if kept and size + len(p) > 0.4 * len(text):
            break
        kept.append(p)
        size += len(p)
    return "\n\n".join(kept)


def factual_errors(text: str) -> str:
    paras = _paragraphs(text)
    for i, claim in enumerate(_FALSE_CLAIMS):
        pos = min(len(paras), 1 + i * max(1, len(paras) // 3))
        paras.insert(pos, claim)
    return "\n\n".join(paras)


def shuffle(text: str) -> str:
    flat = re.sub(r"[#*>\-•]+\s*", " ", text)  # drop headings/bullets: structure is what we remove
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]
    random.Random(42).shuffle(sentences)
    return " ".join(sentences)


def one_sided(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not _NUANCE_MARKERS.search(ln)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


PERTURBATIONS = {"truncate": truncate, "factual_errors": factual_errors,
                 "shuffle": shuffle, "one_sided": one_sided}


def _source_answers(question_ids: List[str], config: dict, cache_path: Path) -> List[Dict]:
    """Real answers to perturb: one per question, rotating across the debate models. Cached."""
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if [a["question_id"] for a in cached] == question_ids:
            logger.info(f"Using cached source answers from {cache_path}")
            return cached

    questions = {q["id"]: q for q in load_questions()}
    models = initialize_models(config)
    answers = []
    for i, qid in enumerate(question_ids):
        model = models[i % len(models)]
        resp = model.invoke(INITIAL_ANSWER_SYSTEM, INITIAL_ANSWER_USER.format(question=questions[qid]["question"]))
        logger.info(f"  source answer {qid} from {model.label}: {len(resp.content)} chars")
        answers.append({"question_id": qid, "question": questions[qid]["question"],
                        "category": questions[qid]["category"], "model": model.label, "answer": resp.content})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    return answers


def run_judge_validation(question_ids: Optional[List[str]] = None, config: Optional[dict] = None,
                         output_dir: str = "evaluation/judge_validation", method: str = "absolute") -> Dict:
    """method: "absolute" (score each answer 0-10 alone) or "pairwise" (pick the better of two)."""
    config = config or load_config()
    question_ids = question_ids or DEFAULT_QUESTION_IDS
    out = Path(output_dir)

    sources = _source_answers(question_ids, config, out / "source_answers.json")
    judges = _init_judge_models(config.get("evaluation", {}).get("judge_models", []), config)
    weights = {d["name"]: d.get("weight", 0.25) for d in config["evaluation"]["scoring_dimensions"]}

    # Every (question, variant) text to be scored; "original" is the undamaged answer.
    items = []
    for src in sources:
        items.append({**src, "variant": "original", "text": src["answer"]})
        for name, fn in PERTURBATIONS.items():
            damaged = fn(src["answer"])
            items.append({**src, "variant": name, "text": damaged,
                          "chars_removed_pct": round(100 * (1 - len(damaged) / len(src["answer"])), 1)})

    if method == "pairwise":
        summary = _run_pairwise(sources, judges)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary_pairwise.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    # Judge-major order keeps one model loaded at a time (model swaps dominate local runtime).
    scores: Dict[tuple, Dict] = {}
    for judge in judges:
        logger.info(f"=== Judging {len(items)} answers with {judge.label} ===")
        for it in items:
            resp = judge.invoke(JUDGE_SYSTEM, JUDGE_USER.format(question=it["question"], answer=it["text"]))
            dims = _parse_judge_scores(resp.content)
            total = sum(dims[d] * weights[d] for d in weights)
            scores[(judge.label, it["question_id"], it["variant"])] = {"dims": dims, "total": total}
            logger.info(f"  {it['question_id']} {it['variant']:15} {total:.2f}")

    summary = _summarise(sources, judges, scores, items)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary_absolute.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _run_pairwise(sources, judges) -> Dict:
    """Original vs each damaged copy, side by side, both orders, every judge."""
    per_perturbation = {}
    for name, target in PERTURBATION_TARGET.items():
        prefs = []
        for src in sources:
            r = judge_pair(src["question"], src["answer"], PERTURBATIONS[name](src["answer"]), judges)
            prefs.append(r["preference_for_first"])
            logger.info(f"  {src['question_id']} {name:15} original preferred {r['preference_for_first']:.2f}")
        detected = sum(p > 0.5 for p in prefs)
        per_perturbation[name] = {"target_dimension": target,
                                  "panel": {"detected": detected, "ties": sum(p == 0.5 for p in prefs),
                                            "n": len(prefs), "avg_preference_for_original": float(np.mean(prefs))}}
    n_pairs = len(sources) * len(PERTURBATIONS)
    rate = sum(v["panel"]["detected"] for v in per_perturbation.values()) / n_pairs
    return _with_verdict({"method": "pairwise", "questions": [s["question_id"] for s in sources],
                          "pairs_per_judge": n_pairs, "judges": [j.label for j in judges],
                          "detection_rate": {"panel": rate}, "per_perturbation": per_perturbation},
                         rate, n_pairs, len(sources))


def _with_verdict(summary: Dict, rate: float, n_pairs: int, n_questions: int) -> Dict:
    summary.update({
        "pass_threshold": 0.8,
        "passed": rate >= 0.8,
        "conclusion": (
            f"The judge panel ranked the original above the damaged copy in {rate:.0%} of {n_pairs} "
            f"pairs ({n_questions} questions x {len(PERTURBATIONS)} kinds of damage). "
            + ("It reliably detects quality loss." if rate >= 0.8 else
               "Below the 80% bar: it cannot reliably detect quality loss.")),
    })
    return summary


def _summarise(sources, judges, scores, items) -> Dict:
    judge_labels = [j.label for j in judges] + ["panel"]

    def score(judge, qid, variant):
        if judge == "panel":  # the panel average is what the evaluation actually uses
            per = [scores[(j.label, qid, variant)] for j in judges]
            return {"total": float(np.mean([p["total"] for p in per])),
                    "dims": {d: float(np.mean([p["dims"][d] for p in per])) for d in per[0]["dims"]}}
        return scores[(judge, qid, variant)]

    per_perturbation = {}
    for name, target in PERTURBATION_TARGET.items():
        row = {"target_dimension": target}
        for jl in judge_labels:
            detected = ties = 0
            target_drops, total_drops = [], []
            for src in sources:
                o, d = score(jl, src["question_id"], "original"), score(jl, src["question_id"], name)
                detected += d["total"] < o["total"]
                ties += d["total"] == o["total"]
                total_drops.append(o["total"] - d["total"])
                target_drops.append(o["dims"][target] - d["dims"][target])
            row[jl] = {"detected": detected, "ties": ties, "n": len(sources),
                       "avg_total_drop": round(float(np.mean(total_drops)), 2),
                       "avg_target_dim_drop": round(float(np.mean(target_drops)), 2)}
        per_perturbation[name] = row

    n_pairs = len(sources) * len(PERTURBATIONS)
    detection = {jl: sum(per_perturbation[p][jl]["detected"] for p in PERTURBATIONS) / n_pairs
                 for jl in judge_labels}
    panel_rate = detection["panel"]
    return _with_verdict({
        "method": "absolute",
        "questions": [s["question_id"] for s in sources],
        "pairs_per_judge": n_pairs,
        "judges": [j.label for j in judges],
        "detection_rate": detection,
        "per_perturbation": per_perturbation,
        "avg_chars_removed_pct": {
            p: round(float(np.mean([it["chars_removed_pct"] for it in items if it["variant"] == p])), 1)
            for p in PERTURBATIONS},
    }, panel_rate, n_pairs, len(sources))
