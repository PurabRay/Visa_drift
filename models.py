"""
Typed Pydantic v2 models for the VisaDrift environment.
Inherits Action / Observation base classes from openenv-core.

Structure mirrors the Smart Meeting Scheduler models.py:
  - Enums for domain constants
  - Plain Pydantic BaseModel classes for domain objects, with helper methods
    where the math is used in more than one place
  - Literal-discriminated Pydantic payload classes for each action_type
  - A single `message: str` Action; a flat Observation; a flat EnvironmentState
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Domain constants ─────────────────────────────────────────────────────────

class VisaType(str, Enum):
    """Visa categories one task at a time — mirrors scheduler.Priority shape."""
    F1 = "F-1"
    B1B2 = "B-1/B-2"
    H1B = "H-1B"


class DriftKind(str, Enum):
    """Three categories of silent mid-episode portal change."""
    SCHEMA = "schema"   # a field was renamed in a section
    POLICY = "policy"   # a numeric threshold or required value changed
    RULE = "rule"       # a new required field or policy constraint appeared


class ErrorCode(str, Enum):
    """Structured rejection reasons the agent can condition on."""
    UNKNOWN_SECTION = "unknown_section"
    UNKNOWN_FIELD = "unknown_field"
    MISSING_FIELD = "missing_field"
    POLICY_VIOLATION = "policy_violation"
    INVALID_PAYLOAD = "invalid_payload"


# The five workflow sections, in canonical order.
SECTIONS: List[str] = [
    "personal_info",
    "travel_info",
    "documents",
    "appointment",
    "payment",
]

# Steps after a drift fires within which a query counts as "detected".
DETECTION_WINDOW: int = 2


# ── Domain models ────────────────────────────────────────────────────────────

class MutationSpec(BaseModel):
    """
    A scheduled drift. Applied in-place to the active schemas at the top of
    VisaEnvironment.step() when step_number == trigger_step.

    The shape of `change` is constrained by `kind`:
        SCHEMA : {"rename": {"<old>": "<new>", ...}}
        POLICY : {"update_policy": {"<key>": <value>, ...}}
        RULE   : {"add_required_field": "<name>"}  and/or
                 {"update_policy": {"<key>": <value>, ...}}
    """
    trigger_step: int = Field(..., ge=1, description="1-indexed step on which this drift fires.")
    kind: DriftKind
    section: str
    change: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("section")
    @classmethod
    def _section_must_exist(cls, v: str) -> str:
        if v not in SECTIONS:
            raise ValueError(f"section {v!r} not in {SECTIONS}")
        return v

    @model_validator(mode="after")
    def _validate_change_shape(self) -> "MutationSpec":
        change = self.change
        if self.kind == DriftKind.SCHEMA:
            rename = change.get("rename")
            if not isinstance(rename, dict) or not rename:
                raise ValueError(
                    "schema drift requires change['rename'] = {old: new, ...}"
                )
            for old, new in rename.items():
                if not isinstance(old, str) or not isinstance(new, str):
                    raise ValueError("rename keys and values must be strings")
        elif self.kind == DriftKind.POLICY:
            update = change.get("update_policy")
            if not isinstance(update, dict) or not update:
                raise ValueError(
                    "policy drift requires change['update_policy'] = {key: value, ...}"
                )
        elif self.kind == DriftKind.RULE:
            has_add = isinstance(change.get("add_required_field"), str)
            has_upd = isinstance(change.get("update_policy"), dict) and change["update_policy"]
            if not (has_add or has_upd):
                raise ValueError(
                    "rule drift requires change['add_required_field'] "
                    "and/or change['update_policy']"
                )
        return self

    def describe(self) -> str:
        """Human-readable summary for logs and debug dumps."""
        return (
            f"{self.kind.value} drift on '{self.section}' "
            f"at step {self.trigger_step}: {self.change}"
        )


class PortalResponse(BaseModel):
    """Structured result of a submit_section attempt."""
    status: Literal["accepted", "rejected"] = "accepted"
    section: str
    error_code: Optional[ErrorCode] = None
    error_detail: Optional[str] = None

    @property
    def is_rejected(self) -> bool:
        return self.status == "rejected"

    @property
    def is_accepted(self) -> bool:
        return self.status == "accepted"


# ── Action payloads (Literal-discriminated; mirrors scheduler) ───────────────

class QueryRequirements(BaseModel):
    """Ask the portal for the currently-valid schema for a section (or all)."""
    action_type: Literal["query_requirements"] = "query_requirements"
    section: Optional[str] = None   # None → return every section

    @field_validator("section")
    @classmethod
    def _section_valid_if_set(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in SECTIONS:
            raise ValueError(f"section {v!r} not in {SECTIONS}")
        return v


class SubmitSection(BaseModel):
    """Submit a filled-in section payload for validation."""
    action_type: Literal["submit_section"] = "submit_section"
    section: str
    payload: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("section")
    @classmethod
    def _section_must_exist(cls, v: str) -> str:
        if v not in SECTIONS:
            raise ValueError(f"section {v!r} not in {SECTIONS}")
        return v


class Done(BaseModel):
    """Signal that the agent believes the application is complete."""
    action_type: Literal["done"] = "done"
    message: str = "Application complete."


VisaActionPayload = Union[QueryRequirements, SubmitSection, Done]


# ── OpenEnv Action / Observation ─────────────────────────────────────────────

class VisaAction(Action):
    """
    Action: the agent sends a single JSON-encoded command in `message`.
    Matches the scheduler's SchedulerAction pattern for dispatch simplicity.
    """
    message: str = Field(
        ...,
        description=(
            'JSON action, e.g. {"action_type": "submit_section", '
            '"section": "personal_info", '
            '"payload": {"full_name": "Jane Doe", "date_of_birth": "1998-03-15", '
            '"passport_number": "P1234567", "nationality": "IN"}}'
        ),
    )


class VisaObservation(Observation):
    """Observation returned after each step."""
    result: str = Field(
        default="",
        description="Human-readable outcome of the last action.",
    )
    workflow_progress: str = Field(
        default="",
        description="e.g. '2/5 sections accepted'",
    )
    current_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Mapping of section name → schema dict. Populated only on steps "
            "where the agent called query_requirements."
        ),
    )
    last_portal_response: Optional[PortalResponse] = Field(
        default=None,
        description="Populated only on submit_section steps.",
    )
    drift_hint: Optional[str] = Field(
        default=None,
        description=(
            "Set to 'something_changed_recently' on any step after a drift has "
            "fired but before the agent has queried the portal."
        ),
    )
    drifts_fired_total: int = Field(
        default=0,
        description="Cumulative count of drifts that have fired this episode.",
    )
    task_instructions: str = Field(
        default="",
        description="Task description.",
    )
    visa_type: str = Field(
        default="",
        description="Current task's visa type, e.g. 'F-1'.",
    )
    step_number: int = Field(default=0)
    max_steps: int = Field(default=30)


# ── Environment state ────────────────────────────────────────────────────────

class EnvironmentState(BaseModel):
    """
    Full internal state of a VisaDrift episode.

    Mirrors scheduler.EnvironmentState: a flat Pydantic model held directly
    on the Environment. Graders read from this; the environment mutates it.
    """
    task_id: str
    visa_type: VisaType

    # Currently-valid schemas. Start as baseline, mutated by drifts.
    active_schemas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Sections the agent has successfully submitted (keyed by section name).
    accepted_sections: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Drift bookkeeping — `fired_mutations`, `detections`, `recoveries`
    # are index-aligned: detections[i] / recoveries[i] correspond to
    # fired_mutations[i].
    scheduled_mutations: List[MutationSpec] = Field(default_factory=list)
    fired_mutations: List[MutationSpec] = Field(default_factory=list)
    detections: List[bool] = Field(default_factory=list)
    recoveries: List[bool] = Field(default_factory=list)
    drift_hint_active: bool = False

    # Anti-loop bookkeeping.
    last_failed_fingerprint: Optional[str] = None
    repeat_fail_count: int = 0

    # Progress.
    step_number: int = 0
    max_steps: int = 30
    done: bool = False
    total_reward: float = 0.0
    action_log: List[Dict[str, Any]] = Field(default_factory=list)

    # ── Convenience accessors (mirrors CalendarEvent's helper-method style) ──

    def completion_ratio(self) -> float:
        """Fraction of the five workflow sections the agent has completed."""
        return len(self.accepted_sections) / max(1, len(SECTIONS))

    def detection_count(self) -> int:
        return sum(1 for x in self.detections if x)

    def recovery_count(self) -> int:
        return sum(1 for x in self.recoveries if x)

    def pending_sections(self) -> List[str]:
        """Sections in canonical order that have not yet been accepted."""
        return [s for s in SECTIONS if s not in self.accepted_sections]

    def is_finished(self) -> bool:
        """True when every section has been accepted."""
        return all(s in self.accepted_sections for s in SECTIONS)

    def has_pending_drift_on(self, section: str) -> bool:
        """Whether a fired-but-unqueried drift affects this section."""
        return self.drift_hint_active and any(
            m.section == section for m in self.fired_mutations
        )

    @staticmethod
    def submission_fingerprint(section: str, payload: Dict[str, Any]) -> str:
        """Stable string key for detecting repeated identical failed submits."""
        return json.dumps(
            {"section": section, "payload": payload},
            sort_keys=True,
            default=str,
        )