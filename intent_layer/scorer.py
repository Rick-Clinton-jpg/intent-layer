"""Re-entry risk scoring.

Re-entry Risk =
      similarity_to_prior_boundary_intent   (confidence-weighted)
    + boundary_history_flag
    + reformulation_flag
    + escalation_score
    - normal_context_continuity

Two constraints from the build spec, both enforced here as code:

1. Low-confidence intent extractions must reduce their weight in the score
   rather than being trusted at face value. `node.confidence` MULTIPLIES
   the similarity term -- it never gates ("if confidence < X, skip"). A
   shaky extraction just contributes less, it isn't discarded or blindly
   trusted.

2. HARD GATE: risk can never be elevated above LOW unless there is an
   actual prior safety_boundary=True node in the same intent's lineage.
   Similarity or "re-entry" language alone, without real boundary
   history, always resolves to LOW. This check runs first and short-
   circuits everything below it.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .graph import IntentGraph, IntentNode
from . import extractor

HIGH_THRESHOLD = 1.6
MEDIUM_THRESHOLD = 1.0

CONTINUITY_DISCOUNT_WEIGHT = 0.3
ESCALATION_EDGE_WEIGHT = 0.4


@dataclass
class RiskResult:
    node_id: str
    risk: str  # "LOW" | "MEDIUM" | "HIGH"
    score: float
    components: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


def _lineage_edges(graph: IntentGraph, node: IntentNode) -> List:
    return [
        e for e in graph.edges
        if graph.get_node(e.target) is not None
        and graph.get_node(e.target).lineage_root == node.lineage_root
    ]


def score_node(node: IntentNode, graph: IntentGraph) -> RiskResult:
    boundary_nodes = graph.lineage_boundary_nodes(node)

    # --- Constraint 2: hard gate -----------------------------------------
    if not boundary_nodes:
        return RiskResult(
            node_id=node.intent_id,
            risk="LOW",
            score=0.0,
            components={
                "similarity_to_prior_boundary_intent": 0.0,
                "boundary_history_flag": 0.0,
                "reformulation_flag": float(node.is_reformulation_cue or node.is_backreference_cue),
                "escalation_score": 0.0,
                "normal_context_continuity": 0.0,
            },
            explanation="No safety_boundary=True node in this intent's lineage -> "
                        "risk is forced to LOW regardless of similarity or re-entry language "
                        "(constraint: re-entry alone never elevates risk).",
        )

    # --- Constraint 1: confidence weights similarity, never gates it -----
    raw_similarity = max(
        extractor.cosine_similarity(node.embedding, b.embedding) for b in boundary_nodes
    )
    weighted_similarity = raw_similarity * node.confidence

    boundary_history_flag = 1.0
    reformulation_flag = 1.0 if (node.is_reformulation_cue or node.is_backreference_cue) else 0.0

    escalation_edges = [e for e in _lineage_edges(graph, node) if e.edge_type == "escalation"]
    escalation_score = min(1.0, len(escalation_edges) * ESCALATION_EDGE_WEIGHT)

    # Discount only applies when nothing else suggests this is actually
    # tied to the boundary -- i.e. no reframing/back-reference language
    # AND weak raw similarity to the boundary intent itself.
    normal_context_continuity = (
        (1.0 - reformulation_flag) * (1.0 - min(1.0, raw_similarity)) * CONTINUITY_DISCOUNT_WEIGHT
    )

    score = (
        weighted_similarity
        + boundary_history_flag
        + reformulation_flag
        + escalation_score
        - normal_context_continuity
    )

    if score >= HIGH_THRESHOLD:
        risk = "HIGH"
    elif score >= MEDIUM_THRESHOLD:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return RiskResult(
        node_id=node.intent_id,
        risk=risk,
        score=round(score, 3),
        components={
            "similarity_to_prior_boundary_intent": round(raw_similarity, 3),
            "confidence": round(node.confidence, 3),
            "weighted_similarity": round(weighted_similarity, 3),
            "boundary_history_flag": boundary_history_flag,
            "reformulation_flag": reformulation_flag,
            "escalation_score": round(escalation_score, 3),
            "normal_context_continuity": round(normal_context_continuity, 3),
        },
        explanation=(
            f"Lineage contains {len(boundary_nodes)} prior safety_boundary node(s); "
            f"similarity={raw_similarity:.2f} x confidence={node.confidence:.2f}, "
            f"reformulation_flag={reformulation_flag}, escalation_score={escalation_score:.2f}, "
            f"continuity_discount={normal_context_continuity:.2f} -> score={score:.2f}"
        ),
    )
