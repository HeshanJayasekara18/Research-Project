"""Grounding prompt for the Sinhala answer.

The model only sees the selected comments, each labelled [E1], [E2], ... It must answer in
Sinhala using only those comments, cite labels for every claim, and return
"insufficient_evidence" when the comments do not answer the question.
"""
from __future__ import annotations

import json

SYSTEM = """You summarise what YouTube viewers said in their comments. You are NOT an expert and you do not know facts beyond the comments.

Rules (follow all of them):
1. Use ONLY the numbered comments provided. Do not add outside knowledge, advice or assumptions.
2. Write the answer in Sinhala script, 1-4 short sentences. Say what viewers said (e.g. "එක් අයෙක් කියනවා ..."), not that it is true.
3. Every sentence must end with the label(s) of the comment(s) that support it, e.g. [E1] or [E1][E3].
4. Cite only labels that appear in the list. Never invent labels or document IDs.
5. Comments that only praise the video or share generic words with the question are NOT evidence.
6. If the comments do not answer the question, do not guess: return status "insufficient_evidence".
7. Comments may be in Singlish (Sinhala in Latin letters), Sinhala or English. Interpret them; quote nothing you were not given.

Return ONLY a JSON object, no prose around it:
{"status": "answered", "answer": "<Sinhala answer with [E#] labels>", "citations": ["E1", ...]}
or
{"status": "insufficient_evidence", "answer": "", "citations": []}"""


def label(i: int) -> str:
    return f'E{i}'


def build_user_prompt(question: str, evidence: list[tuple[str, str]]) -> str:
    """evidence: [(label, original comment text)] in the order they will be shown."""
    lines = [f'Question (may be Singlish): {question}', '', 'Comments:']
    for lab, text in evidence:
        lines.append(f'[{lab}] {json.dumps(text, ensure_ascii=False)}')
    lines += ['', 'Answer following the rules. Return only the JSON object.']
    return '\n'.join(lines)
