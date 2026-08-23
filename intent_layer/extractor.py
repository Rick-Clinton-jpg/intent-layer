"""Turns raw conversation text into an intent node's raw features.

Domain labeling uses simple keyword matching (sufficient for this
prototype's abstracted placeholder categories). Confidence is derived
from embedding similarity to a known category's canonical description,
never treated as ground truth -- a low similarity means low confidence,
not "no match at all".

Embedding backend: three tiers, tried in order, each falling through to
the next on failure:

1. sentence-transformers (all-MiniLM-L6-v2), as specified in the build
   spec -- usually unreachable in a locked-down environment since it
   requires downloading weights from huggingface.co.
2. A local OpenAI-compatible embedding endpoint (LM Studio/Bionic on
   localhost:1234), if one is running with an embedding model loaded --
   AND ONLY if INTENT_LAYER_EXPERIMENTAL_LOCAL_EMBEDDING=1 is set.
   This stays offline in the sense that matters here -- no internet
   egress, no API key -- but it IS a network call to another local
   process, which is a real dependency the fully-offline tier below
   doesn't have. It is opt-in, not auto-detected, because
   eval/calibrate_thresholds.py found that REENTRY_SIM_THRESHOLD has no
   value that separates genuine re-entries from unrelated topic pairs
   under this embedding model -- see README.md "Known Limitations".
   Without the env var, this tier is skipped entirely.
3. A deterministic, fully-offline scikit-learn HashingVectorizer
   embedding, so the prototype always runs even with nothing else
   available.

See README.md "Known Limitations" for what each tier costs in practice.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
import warnings
from functools import lru_cache

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_EMBEDDING_URL = "http://localhost:1234/v1/embeddings"
LOCAL_EMBEDDING_MODELS_URL = "http://localhost:1234/v1/models"
LOCAL_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
_EMBEDDING_BACKEND = None  # set on first use: "sentence-transformers" | "local-lm-studio" | "hashing-fallback"

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
    except Exception as exc:  # noqa: BLE001 - any failure means "try the next tier"
        print(
            f"[intent_layer] {MODEL_NAME} unavailable ({exc.__class__.__name__}); "
            "trying a local LM Studio embedding endpoint next.",
            file=sys.stderr,
        )

    if os.environ.get("INTENT_LAYER_EXPERIMENTAL_LOCAL_EMBEDDING") == "1" and _local_embedding_available():
        _EMBEDDING_BACKEND = "local-lm-studio"
        print(
            f"[intent_layer] EXPERIMENTAL: using local embedding endpoint at "
            f"{LOCAL_EMBEDDING_URL} ({LOCAL_EMBEDDING_MODEL}). Calibration in "
            "eval/calibrate_thresholds.py found no threshold that separates "
            "genuine re-entries from unrelated pairs under this embedding model -- "
            "see README.md 'Known Limitations'. Not the default backend.",
            file=sys.stderr,
        )
        return None

    _EMBEDDING_BACKEND = "hashing-fallback"
    warnings.warn(
        "No real embedding backend available (sentence-transformers and the local "
        "LM Studio endpoint both unreachable); falling back to an offline "
        "HashingVectorizer embedding. See README.md 'Known Limitations'.",
        RuntimeWarning,
        stacklevel=2,
    )
    print(
        "[intent_layer] WARNING: no embedding backend reachable; "
        "using offline hashing-vector fallback embeddings instead.",
        file=sys.stderr,
    )
    return None


def _local_embedding_available() -> bool:
    try:
        req = urllib.request.Request(LOCAL_EMBEDDING_MODELS_URL, method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return any(m.get("id") == LOCAL_EMBEDDING_MODEL for m in body.get("data", []))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False


def _local_embed(text: str) -> np.ndarray:
    payload = {"model": LOCAL_EMBEDDING_MODEL, "input": [text]}
    req = urllib.request.Request(
        LOCAL_EMBEDDING_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return np.array(body["data"][0]["embedding"], dtype=np.float32)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, OSError) as exc:
        raise RuntimeError(
            f"Local embedding backend was available at startup but failed mid-run: {exc}. "
            "Not silently falling back mid-conversation -- that would mix embedding "
            "spaces within one graph. Restart once LM Studio/Bionic is stable again."
        ) from exc


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
    if _EMBEDDING_BACKEND == "sentence-transformers":
        return model.encode([text], normalize_embeddings=True)[0]
    if _EMBEDDING_BACKEND == "local-lm-studio":
        return _local_embed(text)
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
