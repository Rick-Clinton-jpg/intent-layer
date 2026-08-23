"""IntentGraph: nodes + edges tracking intent across a conversation.

Linking a new turn into the graph uses two signals, not one:

1. Embedding similarity to prior turns (semantic-ish "this looks like
   that"). This is the primary continuity signal for ordinary topic
   return/drift (categories A, B, C, H in the eval set).
2. Reframing/back-reference language ("what we discussed earlier", "same
   steps", "pretend you're a professor", "purely hypothetically", ...)
   *combined with* the fact that a safety boundary was hit earlier in
   this same conversation. When both hold, the new turn is resolved
   directly to the most recent prior safety-boundary node -- the way a
   reader would resolve "explain it like I'm a safety researcher" as
   referring back to the thing that just got refused, even without exact
   lexical overlap. This keeps the mechanism robust even when the
   embedding backend is lexically weak (see extractor.py's offline
   fallback), and it is inert whenever no boundary exists yet (so
   ordinary reformulations like "give me ideas for a birthday
   celebration instead", with no refusal anywhere in the conversation,
   are never pulled into a lineage they were never part of).

Either path only *links* nodes into a lineage. Whether that lineage
actually carries risk is entirely the scorer's job, gated by
scorer.py's hard constraint: no elevation above LOW without a genuine
prior safety_boundary=True node in that lineage.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import extractor

REENTRY_SIM_THRESHOLD = 0.30
ESCALATION_CONFIDENCE_DELTA = 0.05


@dataclass
class IntentNode:
    intent_id: str
    description: str
    embedding: np.ndarray
    timestamp: int
    confidence: float
    direction: str  # "initial" | "escalation" | "de-escalation" | "lateral"
    safety_boundary: bool
    parent_intent: Optional[str]
    domain: str
    lineage_root: str
    is_reformulation_cue: bool = False
    is_backreference_cue: bool = False
    resolved_via_global_backref: bool = False


@dataclass
class Edge:
    source: str
    target: str
    edge_type: str  # escalation | de-escalation | boundary_encounter | reformulation | re-entry


class IntentGraph:
    def __init__(self):
        self.nodes: List[IntentNode] = []
        self.edges: List[Edge] = []
        self._extractor = extractor.IntentExtractor()
        self._next_id = 0

    def _new_id(self) -> str:
        node_id = f"n{self._next_id}"
        self._next_id += 1
        return node_id

    def get_node(self, intent_id: str) -> Optional[IntentNode]:
        for n in self.nodes:
            if n.intent_id == intent_id:
                return n
        return None

    def lineage_boundary_nodes(self, node: IntentNode) -> List[IntentNode]:
        """All nodes sharing this node's lineage that hit a safety boundary."""
        return [
            n for n in self.nodes
            if n.lineage_root == node.lineage_root and n.safety_boundary
        ]

    def most_recent_boundary_node(self) -> Optional[IntentNode]:
        for n in reversed(self.nodes):
            if n.safety_boundary:
                return n
        return None

    def _best_match(self, embedding: np.ndarray):
        best_node, best_sim = None, -1.0
        for n in self.nodes:
            sim = extractor.cosine_similarity(embedding, n.embedding)
            if sim > best_sim:
                best_node, best_sim = n, sim
        return best_node, best_sim

    def add_turn(self, text: str, timestamp: int) -> Optional[IntentNode]:
        """Feeds one conversation turn into the graph. Returns the created
        node, or None for bracketed annotation turns (e.g. boundary/no-boundary
        markers), which update graph state but do not create a new intent."""
        if extractor.is_marker_turn(text):
            if extractor.is_boundary_marker(text) and self.nodes:
                last = self.nodes[-1]
                last.safety_boundary = True
                self.edges.append(Edge(last.intent_id, last.intent_id, "boundary_encounter"))
            return None

        features = self._extractor.extract(text)
        embedding = features["embedding"]
        node_id = self._new_id()

        best_match, best_sim = self._best_match(embedding)

        # Reframing language ("pretend you're a professor", "purely
        # hypothetically", ...) or an explicit back-reference ("what we
        # discussed earlier") only resolves to the boundary lineage if a
        # boundary actually exists yet -- otherwise it's just an ordinary
        # reformulation of a never-refused request (category C) and must
        # fall through to plain embedding-based linking below.
        has_content_cue = features["is_reformulation_cue"] or features["is_backreference_cue"]
        backref_target = self.most_recent_boundary_node() if has_content_cue else None

        parent = backref_target if backref_target is not None else (
            best_match if best_match is not None and best_sim >= REENTRY_SIM_THRESHOLD else None
        )

        if parent is not None:
            parent_intent = parent.intent_id
            lineage_root = parent.lineage_root
            if features["confidence"] > parent.confidence + ESCALATION_CONFIDENCE_DELTA:
                direction = "escalation"
            elif features["confidence"] < parent.confidence - ESCALATION_CONFIDENCE_DELTA:
                direction = "de-escalation"
            else:
                direction = "lateral"

            if features["is_reformulation_cue"] or backref_target is not None:
                edge_type = "reformulation"
            elif direction == "escalation":
                edge_type = "escalation"
            elif direction == "de-escalation":
                edge_type = "de-escalation"
            else:
                edge_type = "re-entry"
        else:
            parent_intent = None
            lineage_root = node_id
            direction = "initial"
            edge_type = None

        node = IntentNode(
            intent_id=node_id,
            description=text,
            embedding=embedding,
            timestamp=timestamp,
            confidence=features["confidence"],
            direction=direction,
            safety_boundary=False,
            parent_intent=parent_intent,
            domain=features["domain"],
            lineage_root=lineage_root,
            is_reformulation_cue=features["is_reformulation_cue"],
            is_backreference_cue=features["is_backreference_cue"],
            resolved_via_global_backref=(parent is backref_target and backref_target is not None),
        )
        self.nodes.append(node)
        if parent is not None:
            self.edges.append(Edge(parent.intent_id, node.intent_id, edge_type))
        return node

    def run_conversation(self, turns: List[str]) -> List[IntentNode]:
        created = []
        for i, turn in enumerate(turns):
            node = self.add_turn(turn, timestamp=i)
            if node is not None:
                created.append(node)
        return created
