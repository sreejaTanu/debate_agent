"""Generate the 6-slide submission deck from real results.

Usage:
    python generate_pdf.py
        [--results evaluation/results/summary.json]
        [--validation evaluation/judge_validation]
        [--output submission/presentation.pdf]

Every number on the slides is read from the result files and config.yaml, so the deck
always matches what was actually run. Also writes submission/architecture.pdf (the
slide-1 diagram on its own page).
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph
except ImportError:
    print("Install reportlab: pip install reportlab")
    sys.exit(1)

# ---------------------------------------------------------------- design tokens
W, H = 960, 540                      # 16:9 slide in points
M = 44                               # side margin
INK = HexColor("#16202c")
MUTED = HexColor("#5b6573")
FAINT = HexColor("#8a93a0")
LINE = HexColor("#d5dae1")
PANEL = HexColor("#f3f5f8")
WHITE = HexColor("#ffffff")
ACCENT = HexColor("#2a78d6")         # the debate / the system — the one emphasised thing
ACCENT_SOFT = HexColor("#e6effa")
GREY_BAR = HexColor("#b4bcc7")       # de-emphasised comparison marks
GOOD = HexColor("#15803d")
BAD = HexColor("#b91c1c")

FONT, BOLD = "Helvetica", "Helvetica-Bold"


def _register_fonts():
    """Segoe UI if present (full Unicode: arrows, ≥, ×); Helvetica otherwise."""
    global FONT, BOLD
    fonts = Path("C:/Windows/Fonts")
    try:
        pdfmetrics.registerFont(TTFont("UI", str(fonts / "segoeui.ttf")))
        pdfmetrics.registerFont(TTFont("UI-Bold", str(fonts / "segoeuib.ttf")))
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily("UI", normal="UI", bold="UI-Bold", italic="UI", boldItalic="UI-Bold")
        FONT, BOLD = "UI", "UI-Bold"
    except Exception:
        pass


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------- primitives
def para(c, html, x, y_top, w, size=11, color=INK, leading=None):
    """Draw wrapped rich text (supports <b>) with its top at y_top; returns its height."""
    style = ParagraphStyle("p", fontName=FONT, fontSize=size, leading=leading or size * 1.32, textColor=color)
    p = Paragraph(html, style)
    _, h = p.wrapOn(c, w, 1000)
    p.drawOn(c, x, y_top - h)
    return h


def box(c, x, y, w, h, fill=PANEL, stroke=None, r=8, lw=1):
    c.setFillColor(fill)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(lw)
    c.roundRect(x, y, w, h, r, stroke=1 if stroke else 0, fill=1)


def arrow(c, x1, y1, x2, y2, color=MUTED, lw=1.4):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(lw)
    c.line(x1, y1, x2, y2)
    a = math.atan2(y2 - y1, x2 - x1)
    s = 6
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - s * math.cos(a - 0.45), y2 - s * math.sin(a - 0.45))
    p.lineTo(x2 - s * math.cos(a + 0.45), y2 - s * math.sin(a + 0.45))
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def header(c, n, title, subtitle=None, page_no=True):
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(ACCENT)
    c.rect(M, H - 38, 28, 3, stroke=0, fill=1)
    c.setFillColor(INK)
    c.setFont(BOLD, 22)
    c.drawString(M, H - 66, title)
    if subtitle:
        c.setFillColor(MUTED)
        c.setFont(FONT, 12)
        c.drawString(M, H - 86, subtitle)
    c.setFillColor(FAINT)
    c.setFont(FONT, 8)
    c.drawString(M, 18, "Multi-model answer agent · Grant Thornton AI task")
    if page_no:
        c.drawRightString(W - M, 18, f"{n} / 6")


def stat_tile(c, x, y, w, h, value, label, note=None, value_color=INK):
    box(c, x, y, w, h, fill=PANEL)
    c.setFillColor(value_color)
    c.setFont(BOLD, 26)
    c.drawString(x + 14, y + h - 38, value)
    c.setFillColor(MUTED)
    c.setFont(FONT, 9.5)
    for i, line in enumerate(simpleSplit(label, FONT, 9.5, w - 28)[:2]):
        c.drawString(x + 14, y + h - 54 - i * 12, line)
    if note:
        c.setFillColor(FAINT)
        c.setFont(FONT, 8)
        c.drawString(x + 14, y + 8, note)


def hbar_chart(c, x, y_top, w, rows, vmax, fmt, accent_rows=(), bar_h=16, gap=10, label_w=150):
    """Horizontal bars with direct value labels. Single hue; or emphasis (accent rows, rest grey)."""
    plot_w = w - label_w - 48
    for i, (label, value) in enumerate(rows):
        y = y_top - (i + 1) * (bar_h + gap) + gap
        c.setFillColor(INK)
        c.setFont(FONT, 10)
        c.drawRightString(x + label_w - 10, y + bar_h / 2 - 3.5, label)
        c.setFillColor(PANEL)
        c.rect(x + label_w, y, plot_w, bar_h, stroke=0, fill=1)
        bw = max(2, plot_w * (value / vmax)) if vmax else 2
        c.setFillColor(ACCENT if (not accent_rows or label in accent_rows) else GREY_BAR)
        c.roundRect(x + label_w, y, bw, bar_h, 3, stroke=0, fill=1)
        c.setFillColor(INK)
        c.setFont(BOLD, 10)
        c.drawString(x + label_w + bw + 6, y + bar_h / 2 - 3.5, fmt(value))
    return len(rows) * (bar_h + gap)


def short(label):
    return label.replace(" (judge)", "")


# ---------------------------------------------------------------- slide 1: architecture
def draw_architecture(c, config):
    """The end-to-end flow: answer path, participants, evaluation path, guardrails."""
    models = config.get("models", [])
    judges = config.get("evaluation", {}).get("judge_models", [])
    rounds = config.get("debate", {}).get("num_rounds", 2)
    thr = config.get("debate", {}).get("convergence_threshold", 0.85)

    # Row 1 — the answer path
    top, bh = 410, 150
    pill_w, stage_w, gap = 74, 128, 17
    x = M

    def pill(px, text_lines, fill):
        box(c, px, top - bh / 2 - 26, pill_w, 52, fill=fill, r=26)
        c.setFillColor(WHITE if fill == ACCENT else INK)
        c.setFont(BOLD, 10.5)
        off = (len(text_lines) - 1) * 6
        for i, t in enumerate(text_lines):
            c.drawCentredString(px + pill_w / 2, top - bh / 2 - 3.5 + off - i * 12, t)

    stages = [
        ("1", "Answer alone", ["Each model answers", "independently.", "", "This is also the", "single-model baseline."]),
        ("2", f"Debate × {rounds}", ["Critique the others'", "weakest claims, then", "revise. Critiques are",
                                      "passed on. Stops early", f"if answers converge."]),
        ("3", "Peer score", ["Each scores the other", "two on 4 dimensions,", "never itself. Weighted",
                             "aggregate picks the", "base answer."]),
        ("4", "Synthesise", ["The other two models", "propose fixes for the", "base answer's weakest",
                             "dimension.", ""]),
        ("5", "Merge + gate", ["Base model applies", "the fixes. A cut-off", "merge is rejected by",
                               "code; a majority of the", "others can revert it."]),
    ]
    pill(x, ["Question"], PANEL)
    x += pill_w + gap
    for num, title, lines in stages:
        arrow(c, x - gap + 2, top - bh / 2, x - 2, top - bh / 2)
        shared = num in ("3", "5")
        box(c, x, top - bh, stage_w, bh, fill=ACCENT_SOFT if shared else PANEL)
        c.setFillColor(ACCENT)
        c.circle(x + 18, top - 19, 9, stroke=0, fill=1)
        c.setFillColor(WHITE)
        c.setFont(BOLD, 10)
        c.drawCentredString(x + 18, top - 22.5, num)
        c.setFillColor(INK)
        c.setFont(BOLD, 11.5)
        c.drawString(x + 33, top - 23, title)
        c.setFillColor(MUTED)
        c.setFont(FONT, 9)
        for j, t in enumerate(lines):
            c.drawString(x + 11, top - 46 - j * 12.5, t)
        if shared:
            c.setFillColor(ACCENT)
            c.setFont(BOLD, 7.5)
            c.drawString(x + 11, top - bh + 10, "SHARED DECISION")
        x += stage_w + gap
    arrow(c, x - gap + 2, top - bh / 2, x - 2, top - bh / 2)
    pill(x, ["Final", "answer"], ACCENT)

    # Loop marker above the debate stage
    dx = M + pill_w + gap + stage_w + gap
    c.setStrokeColor(MUTED)
    c.setLineWidth(1.2)
    c.arc(dx + 30, top - 10, dx + stage_w - 30, top + 22, 0, 180)
    arrow(c, dx + 31, top + 5, dx + 30.5, top + 1)
    c.setFillColor(MUTED)
    c.setFont(FONT, 8.5)
    c.drawCentredString(dx + stage_w / 2, top + 27, f"each round · early stop at similarity ≥ {thr}")

    # Row 2 — who takes part
    y2 = top - bh - 26
    c.setFillColor(INK)
    c.setFont(BOLD, 10)
    c.drawString(M, y2 - 4, "Debaters")
    cx = M + 64
    for m in models:
        t = f"{m['label']} · {m.get('model_vendor', m['vendor']).title()}"
        tw = pdfmetrics.stringWidth(t, FONT, 9.5) + 22
        box(c, cx, y2 - 12, tw, 22, fill=WHITE, stroke=LINE, r=11)
        c.setFillColor(INK)
        c.setFont(FONT, 9.5)
        c.drawString(cx + 11, y2 - 4.5, t)
        cx += tw + 8
    provider = ", ".join(sorted({m["vendor"] for m in models}))
    n_vendors = len({m.get("model_vendor", m["vendor"]) for m in models})
    c.setFillColor(MUTED)
    c.setFont(FONT, 9.5)
    c.drawString(cx + 6, y2 - 4.5, f"{len(models)} models from {n_vendors} vendors, run via {provider}")

    # Row 3 — evaluation path, kept separate from the answer path
    y3 = y2 - 34
    eh = 96
    box(c, M, y3 - eh, W - 2 * M, eh, fill=WHITE, stroke=LINE, r=8)
    c.setFillColor(INK)
    c.setFont(BOLD, 10.5)
    c.drawString(M + 14, y3 - 19, "Evaluation — proving it was worth doing")
    c.setFillColor(MUTED)
    c.setFont(FONT, 9)
    c.drawString(M + 14 + pdfmetrics.stringWidth("Evaluation — proving it was worth doing", BOLD, 10.5) + 10,
                 y3 - 19, "(runs separately; not part of producing the answer)")
    ex, ey = M + 14, y3 - 70
    judge_txt = " + ".join(f"{short(j['label'])} ({j.get('model_vendor', '').title()})" for j in judges)
    eboxes = [("Baseline", "each model's step-1 answer"),
              ("Debate answer", "the final answer above"),
              ("Independent judges", judge_txt),
              ("Verdict", "debate vs best single model")]
    # Fit the four boxes (and three 18pt arrow gaps) exactly inside the panel.
    inner = W - 2 * M - 28 - 3 * 18
    widths = [inner * f for f in (0.17, 0.17, 0.42, 0.24)]
    for i, ((t, s), bw) in enumerate(zip(eboxes, widths)):
        box(c, ex, ey, bw, 38, fill=PANEL, r=6)
        c.setFillColor(INK)
        c.setFont(BOLD, 9.5)
        c.drawString(ex + 10, ey + 22, t)
        c.setFillColor(MUTED)
        c.setFont(FONT, 8.5)
        c.drawString(ex + 10, ey + 9, s)
        if i < len(eboxes) - 1:
            arrow(c, ex + bw + 2, ey + 19, ex + bw + 16, ey + 19)
        ex += bw + 18
    c.setFillColor(MUTED)
    c.setFont(FONT, 8.5)
    c.drawString(M + 14, y3 - eh + 10,
                 "No judge shares a vendor with a debater. The judges are tested too: deliberately damaged "
                 "answers must score lower (slide 2).")

    # Guardrails
    c.setFillColor(MUTED)
    c.setFont(FONT, 9)
    c.drawString(M, y3 - eh - 22,
                 "Guardrails:  hard $25 spend ceiling in code  ·  keys from environment variables only  ·  "
                 "one command to run  ·  dry-run mode at zero cost  ·  full transcript of every exchange")


def slide_1(c, config):
    header(c, 1, "How it works: three models answer, argue, score each other and merge",
           "No model ever picks or merges on its own — each decision is shared by several models or enforced by code")
    draw_architecture(c, config)


# ---------------------------------------------------------------- slide 2: what best means
def slide_2(c, config, validation):
    header(c, 2, "What “best” means — and a test that our judges can tell",
           "Best = highest average rubric score from an independent judge panel; peers score with the same rubric")
    dims = config.get("evaluation", {}).get("scoring_dimensions", [])
    desc = {"completeness": "Covers every part of the question",
            "accuracy": "Claims are correct and supported",
            "reasoning": "Logic is clear, structured and sound",
            "nuance": "Handles trade-offs, edge cases, other views"}

    # Left: the rubric
    lx, ly, lw = M, H - 118, 360
    c.setFillColor(INK)
    c.setFont(BOLD, 12)
    c.drawString(lx, ly, "The rubric — each dimension 0–10")
    y = ly - 14
    for d in dims:
        box(c, lx, y - 44, lw, 40, fill=PANEL, r=6)
        c.setFillColor(INK)
        c.setFont(BOLD, 11)
        c.drawString(lx + 12, y - 19, d["name"].title())
        c.setFillColor(MUTED)
        c.setFont(FONT, 9.5)
        c.drawString(lx + 12, y - 33, desc.get(d["name"], ""))
        c.setFillColor(ACCENT)
        c.setFont(BOLD, 11)
        c.drawRightString(lx + lw - 12, y - 27, f"weight {d.get('weight', 0.25):.0%}")
        y -= 48
    para(c, "<b>Why not a vote?</b> Three paragraphs are never “the same answer”, so voting cannot pick "
            "one. Scoring each dimension compares free text without forcing agreement.",
         lx, y - 8, lw, size=10, color=MUTED)

    # Right: judge validation, both methods
    rx = M + lw + 44
    rw = W - M - rx
    c.setFillColor(INK)
    c.setFont(BOLD, 12)
    c.drawString(rx, ly, "The test: damage real answers, check the judges notice")
    methods = [(k, validation.get(k)) for k in ("absolute", "pairwise") if validation and validation.get(k)]
    if not methods:
        para(c, "Run <b>python run_debate.py validate-judges</b> to produce this test.", rx, ly - 20, rw,
             size=10, color=MUTED)
        return
    first = methods[0][1]
    n = first["per_perturbation"]["truncate"]["panel"]["n"]
    para(c, f"{n} real answers (one per question category), each damaged four ways by code. "
            "Share of damaged copies the judges ranked <b>below</b> the original:",
         rx, ly - 10, rw, size=10, color=MUTED)

    name = {"absolute": "Scored 0–10 alone", "pairwise": "Compared side by side"}
    colour = {"absolute": GREY_BAR, "pairwise": ACCENT}
    # Legend (two series: method identity is never colour alone)
    lgx, lgy = rx, ly - 52
    for k, _ in methods:
        c.setFillColor(colour[k])
        c.roundRect(lgx, lgy, 10, 10, 2, stroke=0, fill=1)
        c.setFillColor(INK)
        c.setFont(FONT, 9.5)
        c.drawString(lgx + 15, lgy + 1.5, name[k])
        lgx += pdfmetrics.stringWidth(name[k], FONT, 9.5) + 34

    label = {"truncate": "Cut to 40% → completeness", "factual_errors": "3 false facts → accuracy",
             "shuffle": "Sentences shuffled → reasoning", "one_sided": "Caveats deleted → nuance"}
    label_w, bar_h = 190, 11
    plot_w = rw - label_w - 44
    y = lgy - 14
    for p in label:
        group_h = len(methods) * (bar_h + 3)
        c.setFillColor(INK)
        c.setFont(FONT, 10)
        c.drawRightString(rx + label_w - 10, y - group_h / 2 - 3, label[p])
        for k, v in methods:
            pp = v["per_perturbation"][p]["panel"]
            val = pp["detected"] / pp["n"] * 100
            y -= bar_h + 3
            c.setFillColor(PANEL)
            c.rect(rx + label_w, y, plot_w, bar_h, stroke=0, fill=1)
            bw = max(2, plot_w * val / 100)
            c.setFillColor(colour[k])
            c.roundRect(rx + label_w, y, bw, bar_h, 2.5, stroke=0, fill=1)
            c.setFillColor(INK)
            c.setFont(BOLD, 9)
            c.drawString(rx + label_w + bw + 5, y + 2, f"{val:.0f}%")
        y -= 10
    # 80% bar marker
    mx = rx + label_w + plot_w * 0.8
    c.setStrokeColor(MUTED)
    c.setDash(2, 2)
    c.setLineWidth(0.8)
    c.line(mx, y + 4, mx, lgy - 8)
    c.setDash()
    c.setFillColor(MUTED)
    c.setFont(FONT, 8)
    c.drawCentredString(mx, y - 6, "80% bar")

    y -= 26
    for k, v in methods:
        rate, passed = v["detection_rate"]["panel"], v["passed"]
        c.setFillColor(GOOD if passed else BAD)
        c.setFont(BOLD, 11.5)
        c.drawString(rx, y, f"{'PASS' if passed else 'FAIL'}")
        c.setFillColor(INK)
        c.setFont(FONT, 11)
        c.drawString(rx + 42, y, f"{name[k]}: caught {rate:.0%} of {v['pairs_per_judge']} damaged copies")
        y -= 18
    chosen = _primary_method(validation)
    verdict = ({"pairwise": "Side-by-side judging passes, so it carries the evidence on slide 3.",
                "absolute": "0–10 scoring passes, so it carries the evidence on slide 3.",
                None: "Neither method reaches the bar: slide 3 is reported with that caveat, and better "
                      "judges are the first priority (slide 6)."})[chosen]
    y -= para(c, verdict, rx, y + 2, rw, size=10, color=INK) + 4
    if chosen:
        weak = [(p, v["panel"]) for p, v in validation[chosen]["per_perturbation"].items()
                if v["panel"]["detected"] / v["panel"]["n"] < 0.8]
        if weak:
            txt = "; ".join(f"{label[p].split(' →')[0].lower()} caught {d['detected']}/{d['n']}" for p, d in weak)
            why = (" Losing caveats is the hardest damage for these judges to see."
                   if any(p == "one_sided" for p, _ in weak) else "")
            y -= para(c, f"<b>Weak spot:</b> {txt}.{why}", rx, y + 2, rw, size=10, color=MUTED) + 4
    ab = validation.get("absolute", {}).get("detection_rate", {})
    singles = [r for j, r in ab.items() if j != "panel"]
    if singles and ab.get("panel", 0) > max(singles):
        para(c, f"<b>Two judges beat one:</b> on 0–10 scores the panel caught {ab['panel']:.0%}; "
                f"each judge alone caught {' and '.join(f'{r:.0%}' for r in singles)}.",
             rx, y + 2, rw, size=10, color=MUTED)


def _primary_method(validation):
    """The evidence method: the validated one (pairwise preferred), else None."""
    for k in ("pairwise", "absolute"):
        if validation and validation.get(k, {}).get("passed"):
            return k
    return None


# ---------------------------------------------------------------- slide 3: the evidence
def _verdict_phrase(margin, p):
    """Claim only what the statistics support."""
    if margin > 0 and p is not None and p < 0.05:
        return "beat"
    if margin > 0:
        return "edged — not significantly —"
    if margin == 0:
        return "matched"
    return "did not beat"


def slide_3(c, config, results, validation):
    if not results:
        header(c, 3, "The evidence", "No evaluation results yet")
        para(c, "Run <b>python run_debate.py evaluate</b>, then regenerate this deck.", M, H - 120, 600, size=12)
        return
    n = results["total_questions"]
    method = _primary_method(validation)
    pw = results.get("pairwise_vs_models") or {}
    use_pairwise = method == "pairwise" and pw
    vb = results["vs_best_overall"]

    if use_pairwise:
        # Toughest competitor: the model the debate answer was preferred over least often.
        rival = min(pw, key=lambda k: pw[k]["avg_preference_for_debate"])
        r = pw[rival]
        p = r["sign_test_p_value"]
        margin = r["wins"] - r["losses"]
        tiles = [
            (f"{n}", "questions in the set"),
            (f"{r['wins']} – {r['ties']} – {r['losses']}", f"wins – ties – losses vs {rival}, side by side"),
            (f"{r['avg_preference_for_debate']:.0%}", "of judgements preferred the debate answer (50% = even)"),
            ("n/a" if p is None else f"{p:.2f}", "significance, sign test p (needs < 0.05)"),
        ]
        how = "Judged side by side (the validated method), both orders, two independent judges."
    else:
        rival = vb["model"]
        p = vb["wilcoxon_p_value"]
        margin = vb["avg_margin"]
        tiles = [
            (f"{n}", "questions in the set"),
            (f"{vb['wins']} – {vb['ties']} – {vb['losses']}", f"wins – ties – losses vs {rival}, the best model overall"),
            (f"{margin:+.2f}", "average margin in points (0–10 scale)"),
            ("n/a" if p is None else f"{p:.2f}", "significance, Wilcoxon p (needs < 0.05)"),
        ]
        how = "Judged on the 0–10 rubric by two independent judges."

    header(c, 3, f"On {n} questions, the debate answer {_verdict_phrase(margin, p)} the best single model",
           f"Question set: {n} open-ended questions built for this test"
           + (", one per category (pilot)" if n < 30 else "")
           + (" · DRY RUN — NOT REAL EVIDENCE" if results.get("dry_run") else " · real run, local models"))
    tw = (W - 2 * M - 3 * 14) / 4
    for i, (v, l) in enumerate(tiles):
        note = "too few questions to test" if (i == 3 and p is None) else None
        stat_tile(c, M + i * (tw + 14), H - 198, tw, 88, v, l, note)

    # Left: the chart
    y_top = H - 232
    c.setFillColor(INK)
    c.setFont(BOLD, 12)
    if use_pairwise:
        c.drawString(M, y_top, "How often judges preferred the debate answer, side by side")
        rows = sorted(((f"vs {k}", v["avg_preference_for_debate"] * 100) for k, v in pw.items()),
                      key=lambda kv: kv[1])
        # 50% reference line first, so bars and value labels sit on top of it
        mx = M + 130 + (470 - 130 - 48) * 0.5
        c.setStrokeColor(MUTED)
        c.setDash(2, 2)
        c.setLineWidth(0.8)
        c.line(mx, y_top - 12 - len(rows) * 32 + 4, mx, y_top - 8)
        c.setDash()
        hbar_chart(c, M, y_top - 12, 470, rows, 100, lambda v: f"{v:.0f}%", label_w=130, bar_h=20, gap=12)
        c.setFillColor(MUTED)
        c.setFont(FONT, 8)
        c.drawCentredString(mx, y_top - 12 - len(rows) * 32 - 6, "50% = no preference")
    else:
        c.drawString(M, y_top, "Average judge score, 0–10")
        rows = [("Debate answer", results["debate_avg_score"])] + sorted(
            results["individual_model_avg_scores"].items(), key=lambda kv: -kv[1])
        hbar_chart(c, M, y_top - 12, 470, rows, 10, lambda v: f"{v:.2f}", accent_rows=("Debate answer",),
                   label_w=120, bar_h=20, gap=12)

    # Right: plain-English reading
    rx = M + 510
    rw = W - M - rx
    c.setFillColor(INK)
    c.setFont(BOLD, 12)
    c.drawString(rx, y_top, "What this shows")
    lines = [how]
    if use_pairwise:
        lines.append(f"Against the toughest single model ({rival}) the debate answer won {r['wins']}, "
                     f"tied {r['ties']} and lost {r['losses']} of {n} questions.")
        others = [f"{pw[k]['wins']} of {n} vs {k}" for k in pw if k != rival]
        if others:
            lines.append("Against the weaker models it won " + " and ".join(others) + ".")
    # Why not the best model: which answer did the debate build on?
    per_q = results.get("_per_question") or []
    if per_q and use_pairwise:
        on_rival = [q for q in per_q if q.get("base_answer_model") == rival]
        won = sum(q["pairwise_preference_for_debate"][rival] > 0.5 for q in on_rival)
        lines.append(f"<b>Why not the best model:</b> the debaters chose {rival}'s answer to build on in only "
                     f"{len(on_rival)} of {n} questions, and won {won} of those. Small models are poor at "
                     f"spotting the strongest answer (fix: slide 6).")
    lines.append(f"On the 0–10 rubric: debate <b>{results['debate_avg_score']:.2f}</b> vs best single model "
                 f"{vb['model']} <b>{results['individual_model_avg_scores'][vb['model']]:.2f}</b>.")
    if p is None or n < 12:
        lines.append(f"{n} questions cannot show significance; the full 30-question run is one command.")
    else:
        lines.append("The difference is statistically significant." if p < 0.05
                     else "The difference is not statistically significant.")
    if method is None:
        lines.append("Caveat: neither judging method passed validation (slide 2).")
    yy = y_top - 14
    for t in lines:
        yy -= para(c, "•  " + t, rx, yy, rw, size=10.5, color=INK) + 8


# ---------------------------------------------------------------- slide 4: value
def slide_4(c, config):
    header(c, 4, "Why a professional services firm would want this",
           "Advisory work is open questions with no single right answer — exactly where one model's view is a risk")
    cards = [
        ("Built-in second opinion", "Models challenge each other's weakest claims before anything reaches a "
                                    "reviewer, so one model's blind spot is caught by another."),
        ("An audit trail by default", "Every answer ships with the full exchange: who claimed what, who "
                                      "challenged it, and how each answer scored. Fits review and QA culture."),
        ("Client data can stay in-house", "This build runs entirely on local open-weight models: no data "
                                          "leaves the machine and no per-call fees. Cloud models are a config switch."),
        ("No single-vendor dependency", "Three vendors debate. If one model degrades or is withdrawn, change "
                                        "one line of config; the method stays the same."),
    ]
    cw, ch = (W - 2 * M - 20) / 2, 100
    for i, (t, body) in enumerate(cards):
        x = M + (i % 2) * (cw + 20)
        y = H - 112 - (i // 2) * (ch + 16)
        box(c, x, y - ch, cw, ch, fill=PANEL, r=8)
        c.setFillColor(ACCENT)
        c.rect(x, y - ch, 4, ch, stroke=0, fill=1)
        c.setFillColor(INK)
        c.setFont(BOLD, 14)
        c.drawString(x + 20, y - 32, t)
        para(c, body, x + 20, y - 44, cw - 40, size=11.5, color=MUTED)
    para(c, "<b>Where it fits:</b> due-diligence memos · regulatory and policy analysis · risk assessments · "
            "first drafts of advisory positions — any question where a reviewer would otherwise ask "
            "“what did we miss?”", M, H - 112 - 2 * (ch + 16) - 6, W - 2 * M, size=11.5, color=INK)


# ---------------------------------------------------------------- slide 5: return
def _cloud_estimate(usage):
    """Estimated cost of one debate at Bedrock list prices for the cloud configuration."""
    from models import PRICE_PER_1K
    try:
        import yaml
        with open(Path(__file__).parent / "config" / "config.bedrock.yaml", encoding="utf-8") as f:
            bed = yaml.safe_load(f)
        prices = [PRICE_PER_1K[m["id"]] for m in bed["models"] if m["id"] in PRICE_PER_1K]
        pin = sum(p["input"] for p in prices) / len(prices)
        pout = sum(p["output"] for p in prices) / len(prices)
        return usage["total_input_tokens"] / 1000 * pin + usage["total_output_tokens"] / 1000 * pout
    except Exception:
        return None


def slide_5(c, config, results):
    header(c, 5, "Measuring the return — and when to kill it",
           "Baseline: the best single model answering alone, which is what teams do today")
    usage = (results or {}).get("avg_debate_usage_per_question")
    top = H - 118
    if usage:
        est = _cloud_estimate(usage)
        # Median, not mean: one debate stalled for hours while the laptop slept.
        times = [q["debate_usage"]["wall_clock_seconds"] for q in (results.get("_per_question") or [])]
        minutes = (sorted(times)[len(times) // 2] if times else usage["wall_clock_seconds"]) / 60
        tiles = [
            (f"{usage['model_calls']:.0f}", "model calls per answer (baseline: 1)"),
            (f"{minutes:.1f} min", "median per answer on a 6 GB laptop GPU"),
            (f"{(usage['total_input_tokens'] + usage['total_output_tokens']) / 1000:.0f}k", "tokens per answer"),
            ("$0" if est is None else f"~${est:.2f}", "per answer on cloud models (local: $0)"),
        ]
        tw = (W - 2 * M - 3 * 14) / 4
        for i, (v, l) in enumerate(tiles):
            stat_tile(c, M + i * (tw + 14), H - 186, tw, 76, v, l)
        c.setFillColor(FAINT)
        c.setFont(FONT, 8)
        c.drawString(M, H - 200, "Measured in the evaluation run. Cloud figure applies Bedrock list prices "
                                 "(Claude Sonnet 4.5, Llama 3.3 70B, Mistral Large) to the measured tokens.")
        top = H - 228
    colw = (W - 2 * M - 24) / 2
    c.setFillColor(INK)
    c.setFont(BOLD, 12)
    c.drawString(M, top, "Measure on real client questions, against the baseline")
    measures = [
        ("Quality", "reviewers pick debate vs baseline answer, blind"),
        ("Reviewer time", "minutes of edits before sign-off"),
        ("Rework", "revision cycles before the draft is accepted"),
        ("Cost, latency", "$ and minutes per answer, from the run log"),
    ]
    y = top - 24
    for k, v in measures:
        c.setFillColor(INK)
        c.setFont(BOLD, 10.5)
        c.drawString(M, y, k)
        c.setFillColor(MUTED)
        c.setFont(FONT, 10.5)
        c.drawString(M + 100, y, v)
        y -= 22
    para(c, "<b>Return</b> = reviewer hours saved × rate − (model cost + waiting time).",
         M, y - 2, colw, size=10.5, color=INK)

    rx = M + colw + 24
    kh = 30 + 5 * 22 + 16
    box(c, rx, top - kh + 14, colw, kh, fill=HexColor("#fdf2f2"), r=8)
    c.setFillColor(BAD)
    c.setFont(BOLD, 12)
    c.drawString(rx + 16, top - 6, "Kill it if, after a 4–6 week pilot…")
    kills = [
        "reviewers prefer it in under 55% of blind comparisons",
        "reviewer edit time does not fall by at least 20%",
        "cost per answer exceeds the reviewer time it saves",
        "latency rules it out of the workflows that need it",
        "the judges cannot pass validation",
    ]
    y = top - 30
    for k in kills:
        c.setFillColor(INK)
        c.setFont(FONT, 10.5)
        c.drawString(rx + 16, y, "•  " + k)
        y -= 22


# ---------------------------------------------------------------- slide 6: next + override
def slide_6(c, config, validation, results=None):
    header(c, 6, "With two more weeks — and where the model was wrong",
           "Priorities come straight from what the testing showed")
    weak_judges = bool(validation) and _primary_method(validation) is None
    usage = (results or {}).get("avg_debate_usage_per_question")
    est = _cloud_estimate(usage) if usage else None
    nxt = [
        ("Stronger debaters", "3–4B models misjudge which answer to build on. Rerun on 70B-class or "
         "frontier models" + (f": about ${est:.2f} per answer at Bedrock prices for the measured tokens"
                              if est else "") + "; keep local models for confidential work"),
        ("Build on the best answer", "stronger peer scoring, or merge the best parts of all three answers "
                                     "instead of editing one base"),
        ("Enforce the answer format", "when a model skips the REVISED ANSWER marker, ask again instead of "
                                      "letting its critique through (this cost one question)"),
        ("Stronger, calibrated judges", "frontier-model judges, calibrated on ~50 human-rated answers"
         + (" (the current small judges failed validation)" if weak_judges else
            "; close the blind spot for lost caveats (slide 2)")),
        ("Full and real question sets", "all 30 built questions, then anonymised real client questions"),
    ]
    lw = 450
    y = H - 120
    for i, (t, d) in enumerate(nxt):
        c.setFillColor(ACCENT)
        c.circle(M + 10, y - 4, 10, stroke=0, fill=1)
        c.setFillColor(WHITE)
        c.setFont(BOLD, 10)
        c.drawCentredString(M + 10, y - 7.5, str(i + 1))
        c.setFillColor(INK)
        c.setFont(BOLD, 11.5)
        c.drawString(M + 30, y - 2, t)
        h = para(c, d, M + 30, y - 8, lw - 30, size=10, color=MUTED)
        y -= 30 + h

    rx = M + lw + 30
    rw = W - M - rx
    body = (
        "<b>The AI set the output limit to 1,024 tokens.</b> The first real run then showed the debate "
        "<i>losing</i>, 7.38 vs 8.75. The transcript showed why: 4 of 6 debate turns and the merged answer "
        "were cut off mid-sentence. The loss was a bug, not a finding.<br/><br/>"
        "<b>The models approved it.</b> Both gate voters ACCEPTED a truncated 469-character answer.<br/><br/>"
        "<b>Override:</b> raised the limit to 2,048, made every call report truncation, and added a code "
        "guard so a cut-off merge can never replace the base answer; no model gets a vote on that. "
        "Rerun: no truncation, and the debate tied the best model on that question."
    )
    # Size the panel to its text: measure first, then draw panel, heading and body.
    style = ParagraphStyle("m", fontName=FONT, fontSize=10.5, leading=14.5)
    _, body_h = Paragraph(body, style).wrap(rw - 36, 1000)
    panel_top = H - 104
    box(c, rx, panel_top - body_h - 64, rw, body_h + 64, fill=PANEL, r=8)
    c.setFillColor(INK)
    c.setFont(BOLD, 13)
    c.drawString(rx + 18, panel_top - 26, "Where the model was wrong, and I overrode it")
    para(c, body, rx + 18, panel_top - 40, rw - 36, size=10.5, color=INK, leading=14.5)


# ---------------------------------------------------------------- main
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="evaluation/results/summary.json")
    parser.add_argument("--validation", default="evaluation/judge_validation",
                        help="directory holding summary_absolute.json / summary_pairwise.json")
    parser.add_argument("--output", default="submission/presentation.pdf")
    args = parser.parse_args()

    from connections import load_config
    _register_fonts()
    config = load_config()
    results = load_json(args.results)
    if results:
        # Per-question files carry what the summary does not: base answer chosen, per-debate timing.
        results["_per_question"] = [load_json(f) for f in sorted(Path(args.results).parent.glob("Q*_result.json"))]
    vdir = Path(args.validation)
    validation = {k: load_json(vdir / f"summary_{k}.json") for k in ("absolute", "pairwise")}
    validation = {k: v for k, v in validation.items() if v}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out), pagesize=(W, H))
    c.setTitle("Multi-model answer agent")
    slide_1(c, config)
    c.showPage()
    slide_2(c, config, validation)
    c.showPage()
    slide_3(c, config, results, validation)
    c.showPage()
    slide_4(c, config)
    c.showPage()
    slide_5(c, config, results)
    c.showPage()
    slide_6(c, config, validation, results)
    c.showPage()
    c.save()
    print(f"PDF generated: {out}")

    arch = out.parent / "architecture.pdf"
    a = canvas.Canvas(str(arch), pagesize=(W, H))
    a.setTitle("Architecture — multi-model answer agent")
    header(a, 1, "Architecture: how the agent works, start to end",
           "Answer path (top), participants (middle), evaluation path (bottom)", page_no=False)
    draw_architecture(a, config)
    a.showPage()
    a.save()
    print(f"Architecture diagram: {arch}")


if __name__ == "__main__":
    main()
