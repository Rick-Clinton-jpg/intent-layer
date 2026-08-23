"""Runs the real IntentGraph scorer over eval/trajectories_candidate.json and
reports whether each generated trajectory's proposed expected_risk actually
matches what the system computes -- catches model-mislabeled ground truth
before it's trusted. Read-only: does not modify trajectories_candidate.json
or trajectories.json.

Usage: python3 eval/check_candidates.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent_layer import IntentGraph, score_node  # noqa: E402

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories_candidate.json")


def is_flagged(risk):
    return risk != "LOW"


def main():
    with open(PATH) as f:
        candidates = json.load(f)

    mismatches = 0
    for c in candidates:
        graph = IntentGraph()
        created = graph.run_conversation(c["turns"])
        if not created:
            actual_risk, score = "LOW", 0.0
        else:
            result = score_node(created[-1], graph)
            actual_risk, score = result.risk, result.score

        expected = c["expected_risk"]
        match = (expected == "HIGH") == is_flagged(actual_risk)
        tag = "OK " if match else "MISMATCH"
        if not match:
            mismatches += 1
        print(f"{tag}  {c['id']:8} expected={expected:5} actual={actual_risk:6} score={score:.2f}  last_turn={c['turns'][-1]!r}")

    print(f"\n{len(candidates) - mismatches}/{len(candidates)} candidates match what the real scorer computes.")
    if mismatches:
        print("Mismatches mean the model's proposed label is wrong per the system's own rule -- fix or discard those before merging.")


if __name__ == "__main__":
    main()
