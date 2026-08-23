"""Pre-filter for llm_judge.py inputs: a deterministic circuit-breaker in
front of the probabilistic judge, not another model trying to catch the
first model's manipulation.

Uses Sentry's regex engine (https://github.com/Rick-Clinton-jpg/Sentry, a
sibling repo, not the Sentry.io product) as the scanning mechanism, but
against TWO rulesets:

1. Sentry's own default ruleset -- general agent tool-use manipulation
   (direct agent-address, hidden comment instructions, confirmation-bypass
   phrasing, env/network exfiltration). Real coverage, but built for a
   different threat model than this one.
2. judge_injection_rules.json (this package) -- written specifically
   against the evasion payloads found in eval/judge_adversarial_probe.py:
   fake system/override directives, "ignore prior context" claims,
   explicit answer-injection ("respond with B"), and reframe-the-message
   claims aimed at a classifier rather than a conversational partner.

If either ruleset reports a HIGH-severity finding, or Sentry's engine is
unavailable and nothing can be checked at all, the judge is not consulted
-- scorer.py falls back to the constraint-3 MEDIUM cap instead of trusting
a (possibly manipulated) judge verdict. Fail-safe, not fail-open: "cannot
verify" is treated the same as "flagged," never as "clean."

This closes the SPECIFIC evasion payloads tested so far (see
eval/judge_adversarial_probe.py section 5) -- it is a regex ruleset
against known attack phrasing, not a general solution to prompt
injection. A differently-worded injection that avoids all four patterns
in judge_injection_rules.json would still get through. Treat this as
"raises the cost of the specific attacks we've tried," not "solved."
"""

import json
import os
import sys

_SIBLING_SENTRY_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "Sentry", "src"
)
_LOCAL_RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_injection_rules.json")


def _import_sentry():
    try:
        import sentry

        return sentry
    except ImportError:
        pass

    sentry_src = os.path.abspath(_SIBLING_SENTRY_SRC)
    if os.path.isdir(sentry_src) and sentry_src not in sys.path:
        sys.path.insert(0, sentry_src)
    try:
        import sentry

        return sentry
    except ImportError:
        return None


def scan_for_injection(text: str):
    """Returns (available, high_severity_findings). available=False means
    the scanning engine could not be loaded at all -- callers must treat
    that as 'cannot verify', not 'clean', per the fail-safe design above."""
    sentry = _import_sentry()
    if sentry is None:
        return False, []

    findings = list(sentry.scan(text))  # Sentry's own default ruleset

    try:
        local_rules = sentry.load_rules(_LOCAL_RULES_PATH)
        findings += sentry.scan(text, rules=local_rules)
    except Exception:  # noqa: BLE001 - local ruleset failing to load shouldn't silently
        # widen the gap; treat as unavailable so the caller falls back safe.
        return False, []

    high = [m for m in findings if m.severity == "HIGH"]
    return True, high
