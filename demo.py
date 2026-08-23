"""Terminal demo: walks case D1 (boundary re-entry) turn by turn.

Shows the sequence the eval targets: an initial request hits a safety
boundary, the conversation diverts to something benign, then the same
request comes back reformulated as a hypothetical -- and the scorer
resolves it back to the earlier boundary instead of treating it as new.

Run: python demo.py
"""

import contextlib
import io
import json
import os

from intent_layer import IntentGraph, score_node
from intent_layer.extractor import embedding_backend

CASE_ID = "D1"
TRAJECTORIES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval", "trajectories.json"
)


def load_case(case_id):
    with open(TRAJECTORIES_PATH) as f:
        cases = json.load(f)
    for c in cases:
        if c["id"] == case_id:
            return c
    raise KeyError(case_id)


def main():
    case = load_case(CASE_ID)
    # Backend selection prints a one-time fallback warning to stderr the
    # first time it runs; we report the same fact cleanly below, so quiet
    # that duplicate noise for a screen-recordable run.
    with contextlib.redirect_stderr(io.StringIO()):
        backend = embedding_backend()

    print(f"IntentGraph demo -- case {case['id']} ({case['category']})")
    print(f"expected risk: {case['expected_risk']}  |  embedding backend: {backend}\n")

    graph = IntentGraph()
    last_node = None
    for i, text in enumerate(case["turns"], start=1):
        print(f"Turn {i}: {text}")
        node = graph.add_turn(text, timestamp=i)
        if node is None:
            print("  -> annotation marker, no intent node created")
            if graph.nodes and graph.nodes[-1].safety_boundary:
                print("  -> safety boundary recorded on the prior intent")
        else:
            last_node = node
            print(
                f"  -> intent: domain={node.domain}, "
                f"confidence={node.confidence:.2f}, direction={node.direction}"
            )
            if node.is_reformulation_cue or node.is_backreference_cue:
                print("  -> reformulation / back-reference language detected -> "
                      "resolved to the earlier boundary intent")
        print()

    result = score_node(last_node, graph)
    print(f"Final re-entry score: {result.score:.2f} -> {result.risk}")
    print(f"Why: {result.explanation}")


if __name__ == "__main__":
    main()
