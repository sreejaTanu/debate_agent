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
    output_dir = args.output_dir or "evaluation/results"

    summary = run_evaluation(question_ids=question_ids, output_dir=output_dir)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions evaluated: {summary['total_questions']}")
    print(f"Debate win rate: {summary['debate_win_count']}/{summary['total_questions']} "
          f"({summary['debate_win_rate']*100:.1f}%)")
    print(f"Average margin over best individual: {summary['avg_margin_over_best_individual']:+.2f}/10")
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


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Debate Agent: get the best answer through structured debate"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    debate_parser = subparsers.add_parser("debate", help="Run a single debate on a question")
    debate_parser.add_argument("question", help="The question to debate")
    debate_parser.add_argument("--transcript-dir", default="transcripts/",
                               help="Directory to save transcripts (default: transcripts/)")
    debate_parser.set_defaults(func=cmd_debate)

    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation across question set")
    eval_parser.add_argument("--questions", help="Comma-separated question IDs (e.g., Q01,Q05)")
    eval_parser.add_argument("--output-dir", default="evaluation/results",
                             help="Directory for results (default: evaluation/results/)")
    eval_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
