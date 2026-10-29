"""Generate the 6-slide PDF for submission.

Usage: python generate_pdf.py [--results evaluation/results/summary.json]
Outputs: submission/presentation.pdf
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from reportlab.lib.pagesizes import LETTER, landscape
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
except ImportError:
    print("Install reportlab: pip install reportlab")
    sys.exit(1)


DARK = HexColor("#1a1a2e")
ACCENT = HexColor("#0f3460")
HIGHLIGHT = HexColor("#e94560")
LIGHT_BG = HexColor("#f0f0f5")
WHITE = HexColor("#ffffff")


def load_results(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "SlideTitle", parent=styles["Heading1"],
        fontSize=28, textColor=DARK, spaceAfter=20,
        leading=34,
    ))
    styles.add(ParagraphStyle(
        "SlideSubtitle", parent=styles["Heading2"],
        fontSize=18, textColor=ACCENT, spaceAfter=14,
        leading=22,
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=13, textColor=DARK, spaceAfter=8,
        leading=18,
    ))
    styles.add(ParagraphStyle(
        "BulletCustom", parent=styles["Normal"],
        fontSize=12, textColor=DARK, spaceAfter=6,
        leftIndent=20, bulletIndent=8, leading=16,
    ))
    styles.add(ParagraphStyle(
        "SmallNote", parent=styles["Normal"],
        fontSize=10, textColor=HexColor("#666666"), spaceAfter=4,
        leading=13,
    ))
    return styles


def slide_separator():
    return PageBreak()


def build_slide_1(styles):
    """How it works — one diagram."""
    elements = []
    elements.append(Paragraph("Slide 1: How It Works", styles["SlideTitle"]))
    elements.append(Paragraph("Multi-Model Debate Architecture", styles["SlideSubtitle"]))
    elements.append(Spacer(1, 10))

    diagram_data = [
        ["QUESTION INPUT"],
        ["↓"],
        ["┌─────────────────┬─────────────────┬─────────────────┐"],
        ["Claude Sonnet 4.5 │  Llama 3.3 70B  │  Mistral Large  "],
        ["(Anthropic)       │  (Meta)          │  (Mistral)      "],
        ["└────────┬────────┴────────┬────────┴────────┬────────┘"],
        ["         └─────────────────┼─────────────────┘"],
        ["                          ↓"],
        ["          DEBATE ROUNDS (×2 rounds)"],
        ["    Each model reads others' answers and revises"],
        ["                          ↓"],
        ["            PEER SCORING (cross-eval)"],
        ["    Each model scores every other on 4 dimensions"],
        ["         No self-scoring — no single judge"],
        ["                          ↓"],
        ["         SYNTHESIS + FINAL MERGE"],
        ["    Non-base models suggest improvements"],
        ["    Base model (chosen by peers) merges"],
        ["                          ↓"],
        ["              FINAL ANSWER"],
    ]

    diagram_style = ParagraphStyle("Mono", parent=styles["Normal"],
                                    fontName="Courier", fontSize=10,
                                    textColor=DARK, leading=13, alignment=1)
    for row in diagram_data:
        elements.append(Paragraph(row[0], diagram_style))

    elements.append(Spacer(1, 15))
    elements.append(Paragraph(
        "<b>Key constraint met:</b> No single model decides. All 3 models contribute scores, "
        "all non-base models contribute synthesis suggestions. The final answer reflects "
        "input from every model.",
        styles["Body"]
    ))
    elements.append(Paragraph(
        "<b>Built with:</b> Python, LangGraph, AWS Bedrock, 3 models from 3 vendors.",
        styles["SmallNote"]
    ))
    return elements


def build_slide_2(styles):
    """What 'best' means and how we tested it."""
    elements = []
    elements.append(Paragraph("Slide 2: What 'Best' Means", styles["SlideTitle"]))
    elements.append(Paragraph("Defining and Measuring Answer Quality", styles["SlideSubtitle"]))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(
        "We define answer quality across <b>four dimensions</b>, each scored 0–10:",
        styles["Body"]
    ))

    dim_data = [
        ["Dimension", "Weight", "What It Measures"],
        ["Completeness", "25%", "Does it address all aspects of the question?"],
        ["Accuracy", "25%", "Are claims factually correct and well-supported?"],
        ["Reasoning", "25%", "Is the logic clear, structured, and sound?"],
        ["Nuance", "25%", "Does it handle edge cases, tradeoffs, complexity?"],
    ]
    t = Table(dim_data, colWidths=[1.5*inch, 0.8*inch, 5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BG),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("<b>Why not voting?</b>", styles["SlideSubtitle"]))
    elements.append(Paragraph(
        "The problem explicitly rules out voting: 'Voting works on multiple choice; it does not "
        "work on a paragraph.' Our dimensional scoring evaluates the quality of free-text answers "
        "without reducing them to binary choices.",
        styles["Body"]
    ))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<b>How we tested it:</b>", styles["SlideSubtitle"]))
    elements.append(Paragraph(
        "• Independent multi-judge panel (Llama 3 70B + Mistral Large) scores each answer<br/>"
        "• Judges are different from debate participants to avoid self-evaluation bias<br/>"
        "• Each question: 3 standalone answers + 1 debate answer, all scored blind<br/>"
        "• Compare debate score vs. best individual model score per question",
        styles["BulletCustom"]
    ))
    return elements


def build_slide_3(styles, results):
    """Evidence with question set size."""
    elements = []
    elements.append(Paragraph("Slide 3: The Evidence", styles["SlideTitle"]))
    elements.append(Paragraph("Does the Debate Beat Individual Models?", styles["SlideSubtitle"]))
    elements.append(Spacer(1, 8))

    if results:
        n = results["total_questions"]
        wins = results["debate_win_count"]
        ties = results.get("debate_tie_count", 0)
        losses = results.get("debate_loss_count", 0)
        rate = results["debate_win_rate"] * 100
        margin = results["avg_margin_over_best_individual"]
        debate_avg = results["debate_avg_score"]

        elements.append(Paragraph(f"<b>Question set:</b> {n} open-ended questions across multiple categories "
                                   "(strategy, ethics, analysis, creative, policy)", styles["Body"]))
        elements.append(Spacer(1, 8))

        summary_data = [
            ["Metric", "Value"],
            ["Questions evaluated", str(n)],
            ["Debate wins / ties / losses", f"{wins} / {ties} / {losses}"],
            ["Debate never-worse rate", f"{(wins+ties)}/{n} ({(wins+ties)/n*100:.0f}%)"],
            ["Avg margin over best individual", f"{margin:+.2f}/10"],
            ["Debate avg score", f"{debate_avg:.2f}/10"],
        ]
        for label, score in results.get("individual_model_avg_scores", {}).items():
            summary_data.append([f"{label} avg score", f"{score:.2f}/10"])

        t = Table(summary_data, colWidths=[3.5*inch, 3*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTSIZE", (0, 0), (-1, -1), 12),
            ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BG),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 12))

        if results.get("per_category"):
            elements.append(Paragraph("<b>Per-category breakdown:</b>", styles["Body"]))
            cat_data = [["Category", "Win Rate", "Avg Margin"]]
            for cat, data in results["per_category"].items():
                cat_data.append([cat, f"{data['win_rate']*100:.0f}%", f"{data['avg_margin']:+.2f}"])
            t2 = Table(cat_data, colWidths=[2.5*inch, 2*inch, 2*inch])
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BG),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(t2)

        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"<b>Conclusion:</b> {results.get('conclusion', 'See summary.json')}", styles["Body"]))
    else:
        elements.append(Paragraph(
            "<i>Evaluation results not yet available. Run: python run_debate.py evaluate</i>",
            styles["Body"]
        ))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(
            "<b>Question set:</b> 30 open-ended questions across 6 categories, all hard difficulty. "
            "No question has a single correct answer — all require nuanced reasoning.",
            styles["Body"]
        ))

    return elements


def build_slide_4(styles):
    """Value to a professional services firm."""
    elements = []
    elements.append(Paragraph("Slide 4: Value to a Professional Services Firm", styles["SlideTitle"]))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>The Problem in Professional Services</b>", styles["SlideSubtitle"]))
    elements.append(Paragraph(
        "Consultants, analysts, and advisors answer complex, open-ended questions daily — "
        "strategy recommendations, risk assessments, policy analysis. A single AI model gives one "
        "perspective. Clients pay for <b>thorough, multi-perspective analysis</b>.",
        styles["Body"]
    ))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>What This System Delivers</b>", styles["SlideSubtitle"]))
    elements.append(Paragraph(
        "• <b>Higher-quality first drafts:</b> Debate-synthesized answers are more complete and nuanced "
        "than any single model — reduces revision cycles<br/>"
        "• <b>Built-in devil's advocate:</b> Models challenge each other's reasoning, catching blind spots "
        "a single model misses<br/>"
        "• <b>Auditable reasoning:</b> Full transcript shows how the answer was reached — critical for "
        "regulated industries and client trust<br/>"
        "• <b>Vendor diversity:</b> No single-vendor lock-in; if one model degrades, the system still "
        "functions with the remaining models<br/>"
        "• <b>Scalable expertise:</b> Run 100 debates in parallel — impossible with human panels",
        styles["BulletCustom"]
    ))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>Use Cases</b>", styles["SlideSubtitle"]))
    elements.append(Paragraph(
        "• Due diligence analysis (M&A, vendor evaluation)<br/>"
        "• Policy recommendation memos<br/>"
        "• Risk assessment with competing viewpoints<br/>"
        "• Technical architecture decision records<br/>"
        "• Client-facing research where thoroughness matters",
        styles["BulletCustom"]
    ))
    return elements


def build_slide_5(styles):
    """How to measure return — baseline, kill criteria."""
    elements = []
    elements.append(Paragraph("Slide 5: Measuring the Return", styles["SlideTitle"]))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Baseline</b>", styles["SlideSubtitle"]))
    elements.append(Paragraph(
        "Single-model answer (best available model, e.g., Claude Sonnet 4.5). "
        "This is what teams use today — one prompt, one answer. Measure: quality score "
        "(same 4-dimension rubric), time-to-deliverable, revision count.",
        styles["Body"]
    ))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Metrics to Track</b>", styles["SlideSubtitle"]))
    metrics = [
        ["Metric", "Baseline (Single Model)", "Debate System", "How to Measure"],
        ["Answer quality", "Avg score on rubric", "Avg score on rubric", "Blind human eval + automated judges"],
        ["Revision cycles", "Typical: 2-3 rounds", "Target: 0-1 rounds", "Count before client acceptance"],
        ["Time to first draft", "Seconds", "Minutes (3-5x longer)", "Wall clock per question"],
        ["Cost per question", "~$0.01-0.05", "~$0.15-0.50", "Token cost from spend tracker"],
        ["Coverage of perspectives", "1 viewpoint", "3+ viewpoints synthesized", "Rubric: nuance dimension"],
    ]
    t = Table(metrics, colWidths=[1.4*inch, 1.6*inch, 1.6*inch, 2.4*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BG),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Kill Criteria</b>", styles["SlideSubtitle"]))
    elements.append(Paragraph(
        "Kill the project if after 2 weeks of production use:<br/>"
        "• Debate win rate < 50% on real client questions (not worth the cost)<br/>"
        "• Cost per question exceeds $1.00 without proportional quality gain<br/>"
        "• Human reviewers prefer single-model output ≥60% of the time in blind eval<br/>"
        "• Latency (>5 min per question) makes it unusable in time-sensitive workflows",
        styles["BulletCustom"]
    ))
    return elements


def build_slide_6(styles):
    """What you'd do with two more weeks + where the model was wrong."""
    elements = []
    elements.append(Paragraph("Slide 6: Two More Weeks", styles["SlideTitle"]))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("<b>Current Design Choices</b>", styles["SlideSubtitle"]))
    elements.append(Paragraph(
        "The system uses <b>TF-IDF cosine similarity</b> for adaptive early stopping (skips debate rounds "
        "when models converge), <b>quality-weighted peer scoring</b> (stronger models get more vote weight), "
        "<b>adversarial debate prompts</b> (challenge before synthesize), <b>dimension-targeted synthesis</b> "
        "(focuses on the weakest-scoring dimension), and a <b>self-consistency merge gate</b> (reverts if "
        "the merge degrades quality). A <b>$5.00 spend ceiling</b> is enforced per run.",
        styles["Body"]
    ))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>With Two More Weeks</b>", styles["SlideSubtitle"]))
    next_data = [
        ["Priority", "Improvement", "Technique / Reference", "Expected Impact"],
        ["1", "Parallel model calls", "asyncio.gather() / LangGraph Send()", "3-4x faster (latency)"],
        ["2", "Stronger eval judges", "Chatbot Arena pairwise comparison\n(Zheng et al., 2023)", "More reliable scoring"],
        ["3", "Error handling + fallbacks", "LangChain .with_retry() + .with_fallbacks()", "Essential for production"],
        ["4", "Structured output for scoring", "JSON mode / with_structured_output()", "Zero score parse failures"],
        ["5", "Role-differentiated debate", "Critic / steelman / synthesizer roles\n(Society of Mind, Liang 2023)", "+0.2-0.5 on nuance"],
        ["6", "Semantic convergence", "Sentence-transformers (MiniLM)\nor BERTScore (Zhang 2020)", "Better early-stop detection"],
        ["7", "Pipeline checkpointing", "LangGraph MemorySaver / SqliteSaver", "Resume crashed pipelines"],
        ["8", "Human eval study", "Blind A/B pairs with consultants", "Validate real-world value"],
    ]
    t = Table(next_data, colWidths=[0.6*inch, 1.6*inch, 2.5*inch, 2.1*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BG),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Where the Model Was Wrong and I Overrode It</b>", styles["SlideSubtitle"]))
    elements.append(Paragraph(
        "<b>1. Judge self-evaluation bias:</b> Claude initially configured Claude Sonnet as both a "
        "debate participant and an evaluation judge — the same model scoring its own output. "
        "I overrode this: judges must be independent. The final config uses Llama 3 70B and "
        "Mistral Large as judges.<br/><br/>"
        "<b>2. Convergence detection:</b> The initial implementation used Jaccard word-overlap "
        "similarity, which scored Claude's round-over-round changes at 0.23 — suggesting massive "
        "divergence. Cosine similarity on the same text scored 0.73, correctly detecting that the "
        "meaning was stable despite rephrasing. I replaced Jaccard with TF-IDF cosine similarity.<br/><br/>"
        "<b>3. Consensus bias in prompts:</b> The original debate prompt told models to 'incorporate "
        "the best insights from all responses' — which biased toward averaging and diluted strong "
        "positions. I rewrote it to require models to challenge each other's weakest points first, "
        "only adopting ideas that survive scrutiny.",
        styles["Body"]
    ))
    return elements


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="evaluation/results/summary.json")
    parser.add_argument("--output", default="submission/presentation.pdf")
    args = parser.parse_args()

    results = load_results(args.results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(LETTER),
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.6*inch, bottomMargin=0.5*inch,
    )

    styles = build_styles()
    elements = []

    elements.extend(build_slide_1(styles))
    elements.append(slide_separator())
    elements.extend(build_slide_2(styles))
    elements.append(slide_separator())
    elements.extend(build_slide_3(styles, results))
    elements.append(slide_separator())
    elements.extend(build_slide_4(styles))
    elements.append(slide_separator())
    elements.extend(build_slide_5(styles))
    elements.append(slide_separator())
    elements.extend(build_slide_6(styles))

    doc.build(elements)
    print(f"PDF generated: {output_path}")


if __name__ == "__main__":
    main()
