# Multi-Model Debate Agent

A multi-model debate system that takes a question, sends it to 3+ models from 2+ vendors, and returns the best answer through structured debate — not by asking one model to pick.

## The Problem It Solves

Different models answer the same question differently. The best answer may not come from any single model. This system orchestrates a structured debate between models so they **engage with each other's answers**, then produces a synthesized answer that demonstrably outperforms any individual model.

## How It Works

```
Question → Independent Answers → Debate Rounds → Peer Scoring → Synthesis → Final Answer
                (3 models)         (2 rounds)      (cross-eval)   (merge)
```

### Architecture (LangGraph Pipeline)

```
┌──────────────────────────────────────────────────────────────┐
│                        Question Input                         │
└───────────────────────────┬──────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Claude   │  │  Llama   │  │ Mistral  │  Round 0: Independent
        │  Sonnet   │  │ 3.3 70B  │  │  Large   │  answers
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                   ┌────────────────┐
                   │  Debate Rounds │  Round 1-2: Each model reads
                   │   (×2 rounds)  │  others' answers and revises
                   └────────┬───────┘
                            ▼
                   ┌────────────────┐
                   │  Peer Scoring  │  Each model scores the others
                   │  (cross-eval)  │  on 4 dimensions (0-10 each)
                   └────────┬───────┘
                            ▼
                   ┌────────────────┐
                   │   Synthesis    │  Non-base models suggest
                   │  + Final Merge │  improvements → base merges
                   └────────┬───────┘
                            ▼
                   ┌────────────────┐
                   │  Final Answer  │  + full debate transcript
                   └────────────────┘
```

### How Each Constraint Is Met

| Constraint | How It's Addressed |
|-----------|-------------------|
| **3+ models, 2+ vendors** | Claude Sonnet (Anthropic), Llama 3.3 70B (Meta), Mistral Large (Mistral) — all via AWS Bedrock |
| **No single model decides** | Peer scoring: each model scores every other model on 4 dimensions. Scores are aggregated across all models. No single judge. |
| **Works on open questions** | Dimensional scoring (completeness, accuracy, reasoning, nuance) replaces voting. Synthesis adds to the base answer rather than selecting one. |
| **Prove it was worth doing** | Evaluation framework runs 30 questions, uses an independent multi-judge panel (Llama 3.3 70B + GPT-4o — no overlap with debate's base model selector) to score debate output vs. individual models |

### What "Best" Means

We define answer quality across four dimensions, each scored 0-10:

1. **Completeness** (25%): Does it address all aspects of the question?
2. **Accuracy** (25%): Are claims correct and well-supported?
3. **Reasoning** (25%): Is the logic clear, structured, and sound?
4. **Nuance** (25%): Does it handle edge cases, tradeoffs, and complexity?

The weighted total determines the score. "Best" = highest average score across an independent multi-judge panel (no single judge in evaluation either).

### The Consensus Mechanism (Why No Single Model Decides)

After debate rounds, consensus is reached through **structured peer scoring + collaborative synthesis**:

1. **Peer Evaluation**: Each model scores every other model's final answer on 4 dimensions. This produces a cross-evaluation matrix — the aggregate determines the "base" answer.
2. **Synthesis**: Non-base models suggest specific improvements to the base answer. The base model merges accepted improvements.
3. **No single model picks**: All 3 models contribute scores. All non-base models contribute synthesis suggestions. The final answer reflects input from every model.

## Setup

### Prerequisites
- Python 3.10+
- AWS credentials with Bedrock access (Claude, Llama, Mistral)
- (Optional) OpenAI API key for GPT-based evaluation judges

### Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
```

### Configuration

Edit `config/config.yaml` to change models, debate rounds, or scoring weights.

## Usage

### Run a single debate

```bash
python run_debate.py debate "Should companies be required to disclose when customers are interacting with an AI?"
```

This produces:
- The final debate-synthesized answer (printed to stdout)
- A full debate transcript in `transcripts/debate_transcript.md`
- Machine-readable transcript in `transcripts/debate_transcript.json`

### Run the evaluation

```bash
# Full evaluation (30 questions — takes a while and uses tokens)
python run_debate.py evaluate

# Subset of questions
python run_debate.py evaluate --questions Q01,Q06,Q13,Q19,Q26
```

Output: `evaluation/results/summary.json` with win rates, margins, and per-category breakdowns.

## Project Structure

```
multi-model-debate-agent/
├── config/
│   └── config.yaml           # Model IDs, debate config, scoring weights
├── evaluation/
│   ├── questions.json         # 30 open-ended evaluation questions
│   └── results/               # Evaluation output (gitignored)
├── src/
│   ├── connections.py         # AWS Bedrock session and client setup
│   ├── models.py              # Multi-vendor model abstraction layer
│   ├── debate_state.py        # TypedDict state definition
│   ├── debate_nodes.py        # LangGraph node implementations
│   ├── debate_graph.py        # LangGraph pipeline orchestration
│   ├── prompts.py             # All prompt templates
│   ├── transcript.py          # JSON + Markdown transcript generation
│   └── evaluation.py          # Evaluation framework and scoring
├── transcripts/               # Debate transcripts (gitignored)
├── run_debate.py              # CLI entry point
├── requirements.txt
└── README.md
```

## Research Background

Based on ["Improving Factuality and Reasoning in Language Models through Multiagent Debate"](https://composable-models.github.io/llm_debate/) (Du et al., ICML 2024):

- Multiple LLM instances debate over rounds, naturally converging to better answers
- Cross-model debate (different models) can solve problems neither model solves alone
- 3 agents × 2 rounds is the empirically optimal cost/quality tradeoff

**Our extension beyond the paper**: The paper tests on factual/math benchmarks with ground truth. We handle **open-ended questions** where there is no single correct answer — this requires dimensional scoring instead of voting, and collaborative synthesis instead of convergence checking.

## Alternative Approaches Considered

We evaluated several multi-model consensus strategies before choosing structured debate. Here's why:

| Approach | How It Works | Why We Didn't Use It |
|----------|-------------|---------------------|
| **Mixture of Agents (MoA)** ([Wang et al., 2024](https://arxiv.org/abs/2406.04692)) | Layer models: layer N reads all layer N-1 outputs and produces a refined answer. A single aggregator at the top produces the final output. | The final aggregator is a single model deciding — violates constraint #1. Also, no genuine back-and-forth engagement between models. |
| **LLM Cascading** ([Chen et al., 2023](https://arxiv.org/abs/2305.05176)) | Route easy questions to weaker models, hard ones to stronger models. | Optimizes cost, not quality. Does not combine knowledge from multiple models — only one model answers each question. |
| **Society of Mind** ([Du et al., 2023](https://arxiv.org/abs/2305.14325)) | Multiple agents discuss in rounds, then a single summarizer produces the final answer. | The summarizer is a single model deciding, violating constraint #1. The discussion format is close to debate but lacks structured scoring. |
| **Best-of-N Sampling** | Generate N answers from one or more models, score them, pick the best. | No engagement between answers — models never see or respond to each other. Works for factual questions with verifiable answers, not open-ended ones. |
| **Voting / Majority Consensus** | Models answer, then vote on the best answer. | The problem statement explicitly rules this out: "Voting works on multiple choice; it does not work on a paragraph." |
| **Ensemble Averaging** | Combine model outputs numerically (e.g., average logits or embeddings). | Only works for classification/numeric tasks. Cannot meaningfully average free-text paragraphs. |

### Why Structured Debate Wins for This Problem

1. **Models genuinely engage**: Each model reads others' answers, critiques them, and revises its own. This is the only approach where models respond to specific points in each other's reasoning.
2. **No single model decides**: Peer scoring is fully distributed — every model scores every other model. The base answer is selected by aggregate vote, and non-base models contribute improvements. The final merge is done by a collectively-chosen model following others' instructions.
3. **Works on open-ended questions**: Dimensional scoring (completeness, accuracy, reasoning, nuance) replaces binary voting. Synthesis adds missing points rather than picking a winner.
4. **Proven foundation**: Du et al. demonstrated that debate improves factual accuracy. We extend to open-ended domains with structured scoring.

### Potential Improvements

- **Hybrid MoA + Debate**: Use debate for engagement, then MoA-style layering for synthesis — could capture benefits of both.
- **Adaptive rounds**: Stop debate early if answers converge (measure similarity between rounds), saving tokens.
- **Weighted peer scoring**: Weight each model's scoring ability by historical calibration instead of equal weighting.
- **Diverse judge panel**: Use models from vendors not involved in the debate as judges to eliminate any bias.
