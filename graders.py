"""
Graders for the VisaDrift environment.

Each grader receives an EnvironmentState and returns a reward in [0.05, 0.95].
Graders are fully deterministic and programmatic — no LLM in the loop.

Structure mirrors the Smart Meeting Scheduler graders.py:
  - _SCORE_MIN / _SCORE_MAX clamps (0.05 / 0.95)
  - Private component helpers (`_completion`, `_detection_rate`, ...)
  - One weighted grade function per task tier (easy / medium / hard)
  - GRADERS dispatch table + top-level grade(state)
  - One extra public helper `component_scores()` for training-time logging
"""
from __future__ import annotations

from typing import Callable, Dict

from models import EnvironmentState


_SCORE_MIN = 0.05
_SCORE_MAX = 0.95


def _clamp(score: float) -> float:
    return round(min(_SCORE_MAX, max(_SCORE_MIN, score)), 4)


# ── Component helpers ────────────────────────────────────────────────────────

def _completion(state: EnvironmentState) -> float:
    """Fraction of the five workflow sections accepted by the portal."""
    return state.completion_ratio()


def _detection_rate(state: EnvironmentState) -> float:
    """
    Fraction of fired drifts the agent queried within the detection window
    (window size is enforced in environment.py when flipping the detection
    bit, so here we just tally).

    Returns 1.0 when no drifts have fired yet — an agent cruising through a
    no-drift stretch should not be penalised on this axis.
    """
    if not state.fired_mutations:
        return 1.0
    return state.detection_count() / len(state.fired_mutations)


def _recovery_rate(state: EnvironmentState) -> float:
    """
    Fraction of fired drifts whose affected section has been re-accepted by
    the portal under the new schema. Vacuously 1.0 when no drifts fired.
    """
    if not state.fired_mutations:
        return 1.0
    return state.recovery_count() / len(state.fired_mutations)


def _anti_loop(state: EnvironmentState) -> float:
    """
    1.0 baseline; subtract 0.2 for every repeated identical failed submission
    *beyond the first*. Floor at 0.0.

    `state.repeat_fail_count == 1` means the first failure on this fingerprint
    and is free; every subsequent fail with the same section+payload counts.
    """
    repeats_beyond_first = max(0, state.repeat_fail_count - 1)
    return max(0.0, 1.0 - 0.2 * repeats_beyond_first)


# ── Task graders ─────────────────────────────────────────────────────────────

def grade_easy(state: EnvironmentState) -> float:
    """
    Easy task: F-1 student visa, one late rule drift on documents.
    Completion dominates; detection + recovery make the single-drift
    handling worth noticing; anti-loop is a tie-breaker.

    Score: 50% completion · 20% detection · 25% recovery · 5% anti-loop.
    """
    completion = _completion(state)
    detection = _detection_rate(state)
    recovery = _recovery_rate(state)
    anti_loop = _anti_loop(state)

    score = (
        0.50 * completion
        + 0.20 * detection
        + 0.25 * recovery
        + 0.05 * anti_loop
    )
    return _clamp(score)


def grade_medium(state: EnvironmentState) -> float:
    """
    Medium task: B-1/B-2 tourist visa, schema drift + policy drift.
    Detection weight rises because two drifts produce a meaningful rate
    rather than a binary flag.

    Score: 40% completion · 30% detection · 25% recovery · 5% anti-loop.
    """
    completion = _completion(state)
    detection = _detection_rate(state)
    recovery = _recovery_rate(state)
    anti_loop = _anti_loop(state)

    score = (
        0.40 * completion
        + 0.30 * detection
        + 0.25 * recovery
        + 0.05 * anti_loop
    )
    return _clamp(score)


def grade_hard(state: EnvironmentState) -> float:
    """
    Hard task: H-1B work visa, three drifts spanning all three kinds.
    Drift handling dominates; completion still matters but a trained
    model must earn most of its points from detection + recovery.

    Score: 30% completion · 35% detection · 30% recovery · 5% anti-loop.
    """
    completion = _completion(state)
    detection = _detection_rate(state)
    recovery = _recovery_rate(state)
    anti_loop = _anti_loop(state)

    score = (
        0.30 * completion
        + 0.35 * detection
        + 0.30 * recovery
        + 0.05 * anti_loop
    )
    return _clamp(score)


# ── Dispatch ─────────────────────────────────────────────────────────────────

GRADERS: Dict[str, Callable[[EnvironmentState], float]] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade(state: EnvironmentState) -> float:
    """Grade the current state for the active task_id."""
    grader = GRADERS.get(state.task_id)
    if grader is None:
        raise ValueError(f"No grader for task_id {state.task_id!r}")
    return grader(state)


# ── Training-time introspection ──────────────────────────────────────────────

def component_scores(state: EnvironmentState) -> Dict[str, float]:
    """
    Return the raw, pre-weighting component scores for the current state.

    Intended for the TRL GRPO training notebook: by logging each component
    independently through trackio, we get four separate curves instead of a
    single noisy aggregate — which is the evidence the judging rubric
    rewards under 'reward improvement'.
    """
    return {
        "completion": _completion(state),
        "detection": _detection_rate(state),
        "recovery": _recovery_rate(state),
        "anti_loop": _anti_loop(state),
    }