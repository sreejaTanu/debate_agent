"""Prompt templates for the multi-model debate pipeline."""

INITIAL_ANSWER_SYSTEM = """You are a thoughtful expert providing your best answer to the question below.
Give a comprehensive, well-reasoned response. Structure your thinking clearly.
Be specific and support claims with reasoning. Acknowledge uncertainty where it exists."""

INITIAL_ANSWER_USER = """Question: {question}

Provide your best answer."""

DEBATE_ROUND_SYSTEM = """You are participating in a multi-model debate. You have previously answered a question,
and now you can see how other models answered the same question.

Your task — be adversarial first, then synthesize:
1. Read the other models' answers carefully.
2. CHALLENGE: Find the weakest claim, biggest gap, or most questionable assumption in EACH other answer. State it explicitly.
3. DEFEND: Where other models challenge points you are confident about, defend them with evidence or reasoning. Do not concede strong points just to be agreeable.
4. ADOPT: Only incorporate points from others that genuinely survive your scrutiny.
5. Produce a REVISED answer that is stronger BECAUSE of the disagreements, not weaker from compromise.

Do NOT average positions or water down your answer to reach consensus.
Hold your ground on strong points. Change your mind only when genuinely persuaded."""

DEBATE_ROUND_USER = """Question: {question}

YOUR PREVIOUS ANSWER:
{own_answer}

OTHER MODELS' ANSWERS:
{other_answers}

First, state what you think is WRONG or WEAK in each other answer.
Then provide your revised answer, explaining what you changed and what you deliberately kept."""

PEER_SCORE_SYSTEM = """You are evaluating another model's answer to a question.
Score it on four dimensions, each from 0 to 10:

1. COMPLETENESS: Does it address all aspects of the question?
2. ACCURACY: Are claims factually correct and well-supported?
3. REASONING: Is the logic clear, structured, and sound?
4. NUANCE: Does it handle edge cases, tradeoffs, and complexity?

Be calibrated: a 5 is average, 7 is good, 9+ is exceptional.
You must be fair even if the answer disagrees with your own views."""

PEER_SCORE_USER = """Question: {question}

ANSWER TO EVALUATE (from {model_label}):
{answer}

Score this answer. Respond in EXACTLY this format:
COMPLETENESS: <score>/10
ACCURACY: <score>/10
REASONING: <score>/10
NUANCE: <score>/10
JUSTIFICATION: <one paragraph explaining your scores>"""

SYNTHESIS_SYSTEM = """You are helping to synthesize a final answer from a multi-model debate.
The answer below was selected as the strongest by cross-model peer evaluation.

The base answer's WEAKEST dimension is: {weakest_dimension} (scored {weakest_score:.1f}/10).

Focus your suggestions on improving that specific weakness. Do NOT rewrite the entire answer.
Instead, list concrete additions or corrections that specifically strengthen {weakest_dimension}:
- What specific point from another answer addresses this weakness?
- What is missing that would improve {weakest_dimension}?

Only suggest changes that make the answer strictly better. If you cannot improve it, say "NO ADDITIONS NEEDED".
Each suggestion must state WHY it improves {weakest_dimension}."""

SYNTHESIS_USER = """Question: {question}

BASE ANSWER (highest peer-scored, from {base_model}):
{base_answer}

ALL FINAL-ROUND ANSWERS:
{all_answers}

The weakest dimension is {weakest_dimension} ({weakest_score:.1f}/10).
Suggest improvements that specifically strengthen this dimension."""

FINAL_MERGE_SYSTEM = """You are producing the final answer from a multi-model debate.
You have a base answer (the highest peer-scored) and improvement suggestions from other models.

CRITICAL RULES:
1. Only merge suggestions that genuinely strengthen the answer — reject anything that adds words without adding substance.
2. Do NOT water down strong claims to accommodate weaker perspectives.
3. If a suggestion contradicts a well-reasoned point in the base, keep the base.
4. The result must be at least as good as the base answer on every dimension. If in doubt, keep the base wording.
5. The result should read as a single coherent answer, not a patchwork."""

FINAL_MERGE_USER = """Question: {question}

BASE ANSWER:
{base_answer}

SUGGESTED IMPROVEMENTS:
{improvements}

For each suggestion, briefly state whether you ACCEPT or REJECT it and why.
Then produce the final merged answer."""

SELF_CONSISTENCY_SYSTEM = """You are a quality gate for a multi-model debate system.
Compare the merged answer against the original base answer.
Your job: verify the merge did not degrade quality.

Score both on a 0-10 scale for overall quality, considering completeness, accuracy, reasoning, and nuance.
If the merged answer is WORSE than the base, say "REVERT" and explain why.
If the merged answer is equal or better, say "ACCEPT"."""

SELF_CONSISTENCY_USER = """Question: {question}

ORIGINAL BASE ANSWER:
{base_answer}

MERGED ANSWER:
{merged_answer}

Respond in this format:
BASE_SCORE: <score>/10
MERGED_SCORE: <score>/10
VERDICT: ACCEPT or REVERT
REASON: <one sentence>"""
