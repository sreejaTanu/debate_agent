#!/usr/bin/env python3
"""CLI entry point for the multi-model debate agent.

Usage:
    # Single question debate
    python run_debate.py debate "Your question here"

    # Run evaluation on all questions
    python run_debate.py evaluate

    # Run evaluation on specific questions
    python run_debate.py evaluate --questions Q01,Q05,Q10

    # Single question with transcript output
    python run_debate.py debate "Your question here" --transcript-dir transcripts/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()


def cmd_debate(args):
    from debate_graph import run_debate
    from transcript import save_transcript_json, save_transcript_markdown

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    result = run_debate(args.question)

    print("\n" + "=" * 60)
    print("FINAL DEBATE-SYNTHESIZED ANSWER")
    print("=" * 60)
    print(result["final_answer"])
    print("=" * 60)

    print(f"\nAggregated peer scores:")
    for model, score in sorted(result["aggregated_scores"].items(), key=lambda x: -x[1]):
        marker = " ← base" if model == result["base_answer_model"] else ""
        print(f"  {model}: {score:.2f}/10{marker}")

    tu = result["token_usage"]
    print(f"\nSpend: ${tu['cumulative_spend_usd']:.4f} of ${tu['spend_ceiling_usd']:.2f} ceiling | "
          f"{tu['wall_clock_seconds']:.0f}s")

    if args.transcript_dir:
        ts_dir = Path(args.transcript_dir)
        json_path = save_transcript_json(result, str(ts_dir / "debate_transcript.json"))
        md_path = save_transcript_markdown(result, str(ts_dir / "debate_transcript.md"))
        print(f"\nTranscripts saved:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")


def cmd_evaluate(args):
    from evaluation import run_evaluation

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    question_ids = args.questions.split(",") if args.questions else None
    output_dir = args.output_dir

    summary = run_evaluation(question_ids=question_ids, output_dir=output_dir)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions evaluated: {summary['total_questions']}")
    print(f"Debate win rate: {summary['debate_win_count']}/{summary['total_questions']} "
          f"({summary['debate_win_rate']*100:.1f}%)")
    print(f"Average margin over per-question best individual: {summary['avg_margin_over_best_individual']:+.2f}/10")
    vb = summary["vs_best_overall"]
    p = "n/a" if vb["wilcoxon_p_value"] is None else f"{vb['wilcoxon_p_value']:.3f}"
    print(f"Vs best overall model ({vb['model']}): {vb['wins']}W / {vb['ties']}T / {vb['losses']}L, "
          f"margin {vb['avg_margin']:+.2f}/10, Wilcoxon p={p}")
    print(f"\nDebate average score: {summary['debate_avg_score']:.2f}/10")
    print(f"\nIndividual model averages:")
    for model, score in sorted(summary["individual_model_avg_scores"].items(), key=lambda x: -x[1]):
        print(f"  {model}: {score:.2f}/10")

    print(f"\nPer-category results:")
    for cat, data in summary["per_category"].items():
        print(f"  {cat}: {data['debate_wins']}/{data['total']} wins, "
              f"avg margin {data['avg_margin']:+.2f}")

    print(f"\n{summary['conclusion']}")
    print(f"\nFull results saved to: {output_dir}/")


def cmd_validate_judges(args):
    from judge_validation import run_judge_validation

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    question_ids = args.questions.split(",") if args.questions else None
    summary = run_judge_validation(question_ids=question_ids, output_dir=args.output_dir, method=args.method)

    print("\n" + "=" * 60)
    print(f"JUDGE VALIDATION ({args.method}) — do the judges detect deliberately damaged answers?")
    print("=" * 60)
    for name, row in summary["per_perturbation"].items():
        judges = [j for j in summary["judges"] + ["panel"] if j in row]
        cells = "  ".join(f"{j.replace(' (judge)', '')}: {row[j]['detected']}/{row[j]['n']}" for j in judges)
        print(f"{name:15} (→ {row['target_dimension']:12}) {cells}")
    print(f"\nDetection rate: " + ", ".join(f"{j}: {r:.0%}" for j, r in summary["detection_rate"].items()))
    print(f"\n{summary['conclusion']}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Debate Agent: get the best answer through structured debate"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Use offline mock models (no keys, no spend) to test the full pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    debate_parser = subparsers.add_parser("debate", help="Run a single debate on a question")
    debate_parser.add_argument("question", help="The question to debate")
    debate_parser.add_argument("--transcript-dir", default=None,
                               help="Directory to save transcripts (default: transcripts/)")
    debate_parser.set_defaults(func=cmd_debate)

    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation across question set")
    eval_parser.add_argument("--questions", help="Comma-separated question IDs (e.g., Q01,Q05)")
    eval_parser.add_argument("--output-dir", default=None,
                             help="Directory for results (default: evaluation/results/)")
    eval_parser.set_defaults(func=cmd_evaluate)

    val_parser = subparsers.add_parser("validate-judges",
                                       help="Test that the judges detect deliberately damaged answers")
    val_parser.add_argument("--questions", help="Comma-separated question IDs (default: one per category)")
    val_parser.add_argument("--output-dir", default=None,
                            help="Directory for results (default: evaluation/judge_validation/)")
    val_parser.add_argument("--method", choices=["absolute", "pairwise"], default="absolute",
                            help="absolute: score each answer 0-10 alone; pairwise: pick the better of two")
    val_parser.set_defaults(func=cmd_validate_judges)

    args = parser.parse_args()
    if args.command == "validate-judges" and args.output_dir is None:
        args.output_dir = "evaluation/judge_validation" + ("_dry_run" if args.dry_run else "")
    if args.dry_run:
        # Mock outputs go to separate folders so they never overwrite real results.
        os.environ["DEBATE_DRY_RUN"] = "1"
    suffix = "_dry_run" if args.dry_run else ""
    if args.command == "debate" and args.transcript_dir is None:
        args.transcript_dir = f"transcripts{suffix}/"
    if args.command == "evaluate" and args.output_dir is None:
        args.output_dir = f"evaluation/results{suffix}"
    args.func(args)


if __name__ == "__main__":
    main()
