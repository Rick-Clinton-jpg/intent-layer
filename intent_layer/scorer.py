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

3. HIGH CEILING: if this node was linked into its safety-boundary lineage
   via `graph.py`'s global cue-based back-reference search
   (`node.resolved_via_global_backref`) -- as opposed to a normal,
   threshold-gated embedding match -- risk cannot reach HIGH; it is
   capped at MEDIUM. This exists because that resolution path is global,
   not scoped to the current lineage (see README.md "Known Limitations"):
   a back-reference to a topic that was never actually refused can attach
   itself to an unrelated earlier boundary purely through cue language.
   `eval/calibrate_thresholds.py` showed similarity -- hashing or real
   embeddings -- cannot corroborate or refute this link at any magnitude,
   including small incidental nonzero values, so the gate checks WHICH
   CODE PATH resolved the link, not a similarity threshold. A node
   resolved this way is treated as flagged-but-unconfirmed (MEDIUM)
   instead of confirmed (HIGH). This trades away some of the "6/6 HIGH
   cases detected" headline result (see eval/run_eval.py) in exchange for
   not force-escalating unrelated content to HIGH -- a real
   precision/recall tradeoff, not a free fix.

4. EXPERIMENTAL LLM JUDGE (opt-in, INTENT_LAYER_EXPERIMENTAL_LLM_JUDGE=1):
   instead of blindly capping the constraint-3 case at MEDIUM, ask a
   local model the one narrow question code-path-checking can't answer --
   does this ambiguous turn actually continue the boundary topic, or the
   most recent competing topic in the conversation. A confident
   "boundary" verdict restores HIGH; a confident "something else" verdict
   resolves to LOW instead of an unconfirmed MEDIUM. See
   intent_layer/llm_judge.py.
   Off by default -- this reintroduces a live-model dependency the rest
   of the system deliberately avoids (README "Known Limitations": "No
   live LLM is involved anywhere"), so it's opt-in and clearly separated,
   not a silent behavior change. It also has a confirmed, unresolved
   prompt-injection evasion vulnerability -- see README.md "Experimental:
   local LLM judge" and eval/judge_adversarial_probe.py. Do not treat
   this constraint as trustworthy against a motivated adversary.
"""

import os

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


def _most_recent_competing_node(graph: IntentGraph, node: IntentNode):
    """The most recent non-boundary node before `node` in the whole
    conversation -- the most plausible 'something else' this node's cue
    language might actually be pointing at."""
    idx = graph.nodes.index(node)
    for n in reversed(graph.nodes[:idx]):
        if not n.safety_boundary:
            return n
    return None


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

    # --- Constraint 3: HIGH ceiling on links from the global cue-based
    # back-reference path (checks the code path taken in graph.py, not a
    # similarity magnitude -- see calibrate_thresholds.py) -------------
    capped = False
    judge_verdict = None
    sentry_blocked = False
    if node.resolved_via_global_backref and risk == "HIGH":
        judge_used = False
        if os.environ.get("INTENT_LAYER_EXPERIMENTAL_LLM_JUDGE") == "1":
            competing = _most_recent_competing_node(graph, node)
            if competing is not None:
                from . import injection_filter  # lazy import: only needed when opted in

                sentry_available, findings = injection_filter.scan_for_injection(node.description)
                if (not sentry_available) or findings:
                    # Fail-safe: can't verify, or verified suspicious -- don't
                    # trust the judge on this input. Falls through to the
                    # constraint-3 cap below, same as judge_used=False.
                    sentry_blocked = True
                else:
                    from . import llm_judge  # lazy import: only needed when opted in

                    boundary_text = boundary_nodes[-1].description
                    verdict = llm_judge.judge_referent(node.description, boundary_text, competing.description)
                    judge_verdict = "boundary" if verdict == "A" else "competing_topic"
                    judge_used = True
                    if verdict == "A":
                        pass  # confirmed -- risk stays HIGH
                    else:
                        risk = "LOW"
        if not judge_used:
            risk = "MEDIUM"
            capped = True

    explanation = (
        f"Lineage contains {len(boundary_nodes)} prior safety_boundary node(s); "
        f"similarity={raw_similarity:.2f} x confidence={node.confidence:.2f}, "
        f"reformulation_flag={reformulation_flag}, escalation_score={escalation_score:.2f}, "
        f"continuity_discount={normal_context_continuity:.2f} -> score={score:.2f}"
    )
    if capped and sentry_blocked:
        explanation += (
            " -> capped HIGH to MEDIUM: Sentry pre-filter flagged the input as suspicious "
            "(or was unavailable), so the LLM judge was not consulted (constraint 3, "
            "fail-safe path)."
        )
    elif capped:
        explanation += (
            " -> capped HIGH to MEDIUM: this link rests entirely on cue language with zero "
            "corroborating similarity to the boundary intent (constraint 3)."
        )
    elif judge_verdict is not None:
        explanation += f" -> constraint 4: local LLM judge resolved referent as '{judge_verdict}'."

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
        explanation=explanation,
    )
