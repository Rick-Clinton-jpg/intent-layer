"""Turns raw conversation text into an intent node's raw features.

Domain labeling uses simple keyword matching (sufficient for this
prototype's abstracted placeholder categories). Confidence is derived
from embedding similarity to a known category's canonical description,
never treated as ground truth -- a low similarity means low confidence,
not "no match at all".

Embedding backend: this module tries sentence-transformers
(all-MiniLM-L6-v2) first, as specified in the build spec. If the model
weights cannot be downloaded (e.g. huggingface.co is unreachable under
this environment's network policy), it falls back automatically to a
deterministic, fully-offline scikit-learn HashingVectorizer embedding so
the prototype still runs with no API keys and no network dependency. See
README.md "Known Limitations" for what this trade-off costs in practice.
"""

import re
import sys
import warnings
from functools import lru_cache

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_BACKEND = None  # set on first use: "sentence-transformers" | "hashing-fallback"

# Known domains: label -> (keywords for direct match, canonical phrase for
# centroid embedding). Restricted_* correspond to the abstracted
# placeholders used throughout the eval dataset; the others are the benign
# topics that appear in the trajectories, included so the extractor has
# something concrete to anchor "general" turns to instead of guessing.
KNOWN_DOMAINS = {
    "restricted_objective_x": (["restricted objective x"], "Restricted Objective X"),
    "restricted_procedure_y": (["restricted procedure y"], "Restricted Procedure Y"),
    "restricted_domain_z": (["restricted domain z"], "Restricted Domain Z"),
    "restricted_topic_w": (["restricted topic w"], "Restricted Topic W"),
}

# Fixed confidence assigned when no keyword match is found and embedding
# similarity to every known centroid is weak. This is a deliberately low
# default -- it should never be treated as a confirmed category.
DEFAULT_LOW_CONFIDENCE = 0.2
NEAREST_CENTROID_THRESHOLD = 0.4

# Phrases that signal an attempt to reframe/repackage a request rather than
# introduce a genuinely new one (hypotheticals, role-play, fiction framing,
# authority-figure framing, etc.). Used to raise a reformulation flag; it is
# one signal among several, never decisive alone.
REFORMULATION_CUES = [
    "hypothetical", "hypothetically", "pretend", "as fiction", "for a novel",
    "phrase it as", "professor", "for an exam", "story", "fiction",
    "safety researcher", "defend against", "forget that", "instead",
    "purely hypothetically", "role-play", "roleplay", "just describe it as",
]

# Phrases that explicitly point back at an earlier point in the
# conversation ("what we discussed earlier", "same steps", ...). These are
# a distinct signal from reformulation framing: even a lexically weak
# embedder should not need to guess that the user is referring back --
# the text says so directly. Used to resolve the reference to the most
# recent prior safety-boundary node, independent of embedding similarity.
BACKREF_CUES = [
    "what we discussed earlier", "what i asked before", "going back to",
    "same steps", "same as before", "reconstruct it piece by piece",
    "back to what i asked", "what we talked about",
]

MARKER_BOUNDARY_RE = re.compile(r"safety boundary triggered", re.IGNORECASE)
MARKER_RE = re.compile(r"^\s*\[.*\]\s*$")


@lru_cache(maxsize=1)
def _get_model():
    """Loads the real sentence-transformer model, or None if unavailable."""
    global _EMBEDDING_BACKEND
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
        _EMBEDDING_BACKEND = "sentence-transformers"
        return model
    except Exception as exc:  # noqa: BLE001 - any failure means "use fallback"
        _EMBEDDING_BACKEND = "hashing-fallback"
        warnings.warn(
            f"sentence-transformers/{MODEL_NAME} unavailable ({exc.__class__.__name__}: {exc}); "
            "falling back to an offline HashingVectorizer embedding. "
            "See README.md 'Known Limitations'.",
            RuntimeWarning,
            stacklevel=2,
        )
        print(
            f"[intent_layer] WARNING: {MODEL_NAME} could not be loaded "
            f"({exc.__class__.__name__}); using offline hashing-vector fallback embeddings instead.",
            file=sys.stderr,
        )
        return None


@lru_cache(maxsize=1)
def _hashing_vectorizer():
    from sklearn.feature_extraction.text import HashingVectorizer

    return HashingVectorizer(
        n_features=2048,
        analyzer="word",
        ngram_range=(1, 2),
        stop_words="english",
        alternate_sign=False,
        norm="l2",
    )


def embedding_backend() -> str:
    """Forces backend selection (if not already done) and returns its name."""
    _get_model()
    return _EMBEDDING_BACKEND


@lru_cache(maxsize=1)
def _domain_centroids():
    labels = list(KNOWN_DOMAINS.keys())
    phrases = [KNOWN_DOMAINS[label][1] for label in labels]
    embeddings = np.stack([embed(p) for p in phrases])
    return labels, embeddings


def embed(text: str) -> np.ndarray:
    model = _get_model()
    if model is not None:
        return model.encode([text], normalize_embeddings=True)[0]
    vec = _hashing_vectorizer().transform([text]).toarray()[0]
    return vec.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def is_marker_turn(text: str) -> bool:
    """Bracketed annotation turns (e.g. '[SAFETY BOUNDARY TRIGGERED]',
    '[no boundary]') are dataset/event annotations, not user intents."""
    return bool(MARKER_RE.match(text.strip()))


def is_boundary_marker(text: str) -> bool:
    return bool(MARKER_BOUNDARY_RE.search(text))


def reformulation_cue(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in REFORMULATION_CUES)


def backreference_cue(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in BACKREF_CUES)


def classify_domain(text: str, embedding: np.ndarray):
    """Returns (domain_label, confidence). Keyword match is checked first
    (cheap, precise for our abstracted placeholder terms); if nothing
    matches, fall back to nearest embedding centroid, and if that's weak
    too, label as 'general' with a fixed low confidence."""
    lowered = text.lower()
    for label, (keywords, canonical_phrase) in KNOWN_DOMAINS.items():
        if any(kw in lowered for kw in keywords):
            centroid = embed(canonical_phrase)
            sim = cosine_similarity(embedding, centroid)
            # Keyword match confirms domain; similarity still modulates
            # confidence rather than being forced to 1.0.
            confidence = max(sim, 0.85)
            return label, confidence

    labels, centroids = _domain_centroids()
    sims = [cosine_similarity(embedding, c) for c in centroids]
    best_idx = int(np.argmax(sims)) if sims else -1
    best_sim = sims[best_idx] if sims else 0.0
    if best_sim >= NEAREST_CENTROID_THRESHOLD:
        return labels[best_idx], best_sim
    return "general", DEFAULT_LOW_CONFIDENCE


class IntentExtractor:
    """Text -> raw intent features (embedding, domain, confidence)."""

    def extract(self, text: str):
        embedding = embed(text)
        domain, confidence = classify_domain(text, embedding)
        return {
            "description": text,
            "embedding": embedding,
            "domain": domain,
            "confidence": confidence,
            "is_reformulation_cue": reformulation_cue(text),
            "is_backreference_cue": backreference_cue(text),
        }
