"""Runs the naive baseline and IntentGraph over eval/trajectories.json and
prints an honest comparison: detection rate, false-positive rate, and a
per-category breakdown.

Usage: python3 eval/run_eval.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baseline  # noqa: E402
from intent_layer import IntentGraph, score_node  # noqa: E402
from intent_layer.extractor import embedding_backend  # noqa: E402

TRAJECTORIES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories.json")

RISK_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def is_flagged(risk: str) -> bool:
    """Binarized verdict used for detection-rate / false-positive-rate
    metrics: MEDIUM or HIGH both count as 'the system raised a flag'."""
    return risk != "LOW"


def run_intentgraph(turns):
    graph = IntentGraph()
    created = graph.run_conversation(turns)
    if not created:
        return "LOW", None
    last_node = created[-1]
    result = score_node(last_node, graph)
    return result.risk, result


def main():
    with open(TRAJECTORIES_PATH) as f:
        trajectories = json.load(f)

    backend = embedding_backend()
    print(f"Embedding backend in use: {backend}\n")

    rows = []
    for traj in trajectories:
        turns = traj["turns"]
        expected = traj["expected_risk"]

        baseline_pred = baseline.classify_trajectory(turns)
        ig_pred, ig_result = run_intentgraph(turns)

        rows.append({
            "id": traj["id"],
            "category": traj["category"],
            "expected": expected,
            "baseline_pred": baseline_pred,
            "ig_pred": ig_pred,
            "ig_score": ig_result.score if ig_result else 0.0,
        })

    # --- per-trajectory table -------------------------------------------
    print(f"{'ID':4} {'Category':28} {'Expected':9} {'Baseline':10} {'IntentGraph':12} {'IG score':9}")
    print("-" * 76)
    for r in rows:
        b_mark = "OK" if r["baseline_pred"] == r["expected"] else "X"
        ig_mark = "OK" if r["ig_pred"] == r["expected"] else "X"
        print(
            f"{r['id']:4} {r['category']:28} {r['expected']:9} "
            f"{r['baseline_pred']:8}{b_mark:>2} {r['ig_pred']:10}{ig_mark:>2} {r['ig_score']:9.2f}"
        )

    # --- aggregate metrics ------------------------------------------------
    def metrics_for(pred_key):
        positives = [r for r in rows if r["expected"] == "HIGH"]
        negatives = [r for r in rows if r["expected"] == "LOW"]
        tp = sum(1 for r in positives if is_flagged(r[pred_key]))
        fp = sum(1 for r in negatives if is_flagged(r[pred_key]))
        detection_rate = tp / len(positives) if positives else float("nan")
        fp_rate = fp / len(negatives) if negatives else float("nan")
        return detection_rate, fp_rate, len(positives), len(negatives)

    print("\n=== Overall ===")
    for name, key in [("Baseline (turn-level keyword)", "baseline_pred"), ("IntentGraph", "ig_pred")]:
        det, fpr, n_pos, n_neg = metrics_for(key)
        print(f"{name}: detection rate = {det:.0%} ({n_pos} HIGH cases), "
              f"false-positive rate = {fpr:.0%} ({n_neg} LOW cases)")

    print("\n=== Per-category ===")
    categories = sorted(set(r["category"] for r in rows))
    print(f"{'Category':28} {'Expected':9} {'Baseline':10} {'IntentGraph':12}")
    print("-" * 60)
    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        expected = cat_rows[0]["expected"]
        b_correct = sum(1 for r in cat_rows if is_flagged(r["baseline_pred"]) == (expected == "HIGH"))
        ig_correct = sum(1 for r in cat_rows if is_flagged(r["ig_pred"]) == (expected == "HIGH"))
        n = len(cat_rows)
        print(f"{cat:28} {expected:9} {f'{b_correct}/{n}':10} {f'{ig_correct}/{n}':12}")

    print("\n=== Exact risk-level match (LOW/MEDIUM/HIGH vs expected LOW/HIGH) ===")
    for name, key in [("Baseline", "baseline_pred"), ("IntentGraph", "ig_pred")]:
        exact = sum(1 for r in rows if r[key] == r["expected"])
        print(f"{name}: {exact}/{len(rows)} exact matches")


if __name__ == "__main__":
    main()
