"""Naive turn-level baseline: keyword match on the CURRENT turn only.

No memory of prior turns, no embeddings, no lineage. This is the
comparison point the whole prototype exists to beat -- a classifier that
can only ever see the most recent message. It flags a turn as risky iff
that single turn's text literally contains one of the abstracted
restricted-content placeholders.
"""

import re

RESTRICTED_KEYWORDS = [
    "restricted objective x",
    "restricted procedure y",
    "restricted domain z",
    "restricted topic w",
]

_KEYWORD_RE = re.compile("|".join(re.escape(k) for k in RESTRICTED_KEYWORDS), re.IGNORECASE)


def classify_turn(text: str) -> str:
    """Returns 'HIGH' if the current turn's text contains a restricted
    keyword, else 'LOW'. Ignores every other turn in the conversation."""
    return "HIGH" if _KEYWORD_RE.search(text) else "LOW"


def classify_trajectory(turns) -> str:
    """Baseline's verdict for a trajectory: whatever it says about the
    final turn, since that's the turn a real system would be classifying
    right now. (No memory means earlier turns are irrelevant to it.)"""
    return classify_turn(turns[-1])
