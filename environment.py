
"""
VisaDrift — core environment logic.

All task definitions, drift application, section validation, and state
transitions live here. The HTTP layer is in server/app.py.

Structure mirrors the Smart Meeting Scheduler environment.py exactly:
  - Module-level BASELINE_SCHEMAS and TASKS dicts
  - Module-level helper functions (_apply_mutation, _validate_submission)
  - VisaEnvironment class with reset() / step() / state / close()
  - Private action handlers (_handle_query, _handle_submit, _handle_done)
  - _make_obs() observation builder
  - Detection window logic in _mark_detection_on_query()
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    DETECTION_WINDOW,
    DriftKind,
    EnvironmentState,
    MutationSpec,
    PortalResponse,
    SECTIONS,
    VisaAction,
    VisaObservation,
    VisaType,
)
from graders import grade


# ── Baseline schemas (pre-drift) ──────────────────────────────────────────────
#
# These are the schemas the portal starts every episode with.
# _apply_mutation() modifies deepcopies of these on EnvironmentState;
# BASELINE_SCHEMAS itself is never mutated.

BASELINE_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "personal_info": {
        "fields": {
            "full_name": "required",
            "date_of_birth": "required",
            "passport_number": "required",
            "nationality": "required",
        },
        "policy": {},
    },
    "travel_info": {
        "fields": {
            "purpose": "required",
            "entry_date": "required",
            "duration_days": "required",
            "address_abroad": "required",
        },
        "policy": {"max_duration_days": 180},
    },
    "documents": {
        "fields": {
            "passport_photo": "required",
            "bank_statement": "required",
        },
        "policy": {"photo_spec": "35x45mm"},
    },
    "appointment": {
        "fields": {"date": "required"},
        "policy": {"booking_window_days": 60},
    },
    "payment": {
        "fields": {"amount_usd": "required"},
        "policy": {"amount_usd": 160},
    },
}


# ── Task definitions ──────────────────────────────────────────────────────────
#
# Mirrors the scheduler's TASKS dict exactly:
# each entry has description, visa_type, max_steps, scheduled_mutations.

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "description": (
            "Apply for an F-1 student visa. Complete all 5 sections in order: "
            "personal_info, travel_info, documents, appointment, payment. "
            "Actions available: query_requirements, submit_section, done. "
            "Portal requirements may change mid-application — "
            "if a submission is rejected, call query_requirements to check "
            "the current schema before resubmitting."
        ),
        "visa_type": VisaType.F1,
        "max_steps": 20,
        "scheduled_mutations": [
            # Single rule drift on documents at step 5.
            # photo_spec changes from "35x45mm" to "2x2in".
            # An agent that submits documents before querying after step 5
            # will be rejected on the photo_spec policy check.
            MutationSpec(
                trigger_step=5,
                kind=DriftKind.RULE,
                section="documents",
                change={"add_required_field": "supporting_letter"},
            ),
        ],
    },
    "medium": {
        "description": (
            "Apply for a B-1/B-2 tourist visa. Two portal updates may occur "
            "during your application. If a submission is rejected, call "
            "query_requirements to check the current schema before resubmitting."
        ),
        "visa_type": VisaType.B1B2,
        "max_steps": 25,
        "scheduled_mutations": [
            # Schema drift at step 3: travel_info renames entry_date → arrival_date.
            # Any subsequent submit_section for travel_info using entry_date
            # will get an unknown_field rejection.
            MutationSpec(
                trigger_step=3,
                kind=DriftKind.SCHEMA,
                section="travel_info",
                change={"rename": {"entry_date": "arrival_date"}},
            ),
            # Policy drift at step 8: payment fee rises from 160 to 185.
            # Any subsequent submit_section for payment using amount_usd=160
            # will get a policy_violation rejection.
            MutationSpec(
                trigger_step=8,
                kind=DriftKind.POLICY,
                section="payment",
                change={"update_policy": {"amount_usd": 185}},
            ),
        ],
    },
    "hard": {
        "description": (
            "Apply for an H-1B work visa. Multiple portal changes are active "
            "or pending throughout the application. Check current requirements "
            "proactively — especially after any rejection."
        ),
        "visa_type": VisaType.H1B,
        "max_steps": 35,
        "scheduled_mutations": [
            # Schema drift at step 2: personal_info renames date_of_birth → dob.
            # Fires very early — catches agents that front-load all submissions.
            MutationSpec(
                trigger_step=2,
                kind=DriftKind.SCHEMA,
                section="personal_info",
                change={"rename": {"date_of_birth": "dob"}},
            ),
            # Rule drift at step 6: documents gains a new required field.
            # Agent must re-query documents schema and include approval_notice.
            MutationSpec(
                trigger_step=6,
                kind=DriftKind.RULE,
                section="documents",
                change={"add_required_field": "approval_notice"},
            ),
            # Policy drift at step 10: appointment booking window tightens.
            # An agent submitting appointment without re-querying may hit
            # a stale booking_window_days value (not directly validated here,
            # but contributes to the detection/recovery bookkeeping).
            MutationSpec(
                trigger_step=10,
                kind=DriftKind.POLICY,
                section="appointment",
                change={"update_policy": {"booking_window_days": 14}},
            ),
        ],
    },
}


# ── Drift application ─────────────────────────────────────────────────────────

def _apply_mutation(
    schemas: Dict[str, Dict[str, Any]],
    mutation: MutationSpec,
) -> None:
    """
    Apply a single MutationSpec to the schemas dict in place.

    Called at the top of step() for every mutation whose trigger_step
    matches the current step number. Mirrors the scheduler's
    per-action helper pattern.
    """
    section = schemas.setdefault(mutation.section, {"fields": {}, "policy": {}})
    fields = section.setdefault("fields", {})
    policy = section.setdefault("policy", {})
    change = mutation.change

    if mutation.kind == DriftKind.SCHEMA:
        # Rename one or more fields in the section's field map.
        for old, new in change.get("rename", {}).items():
            if old in fields:
                fields[new] = fields.pop(old)

    elif mutation.kind == DriftKind.POLICY:
        # Update one or more policy values (thresholds, fees, etc.).
        policy.update(change.get("update_policy", {}))

    elif mutation.kind == DriftKind.RULE:
        # Add a new required field and/or update policy values.
        new_field = change.get("add_required_field")
        if new_field:
            fields[new_field] = "required"
        policy.update(change.get("update_policy", {}))


# ── Section validation ────────────────────────────────────────────────────────

def _validate_submission(
    schemas: Dict[str, Dict[str, Any]],
    section: str,
    payload: Dict[str, Any],
) -> PortalResponse:
    """
    Validate a section submission payload against the currently-active schema.

    Checks (in order):
      1. Section exists in active schemas.
      2. No unknown fields in payload.
      3. All required fields present in payload.
      4. Section-specific policy checks (payment amount, photo spec,
         travel duration, appointment window).

    Returns a PortalResponse with status "accepted" or "rejected".
    Mirrors the scheduler's _has_conflict / _within_working_hours pattern:
    validation is pure and stateless — it reads schemas and payload only.
    """
    if section not in schemas:
        return PortalResponse(
            status="rejected",
            section=section,
            error_code="unknown_section",
            error_detail=f"No such section {section!r}. "
                         f"Valid sections: {SECTIONS}.",
        )

    schema = schemas[section]
    fields = schema.get("fields", {})
    policy = schema.get("policy", {})

    # Check for unknown fields first — gives the clearest signal after a
    # schema drift (renamed field appears as unknown_field, not missing_field).
    for key in payload.keys():
        if key not in fields:
            return PortalResponse(
                status="rejected",
                section=section,
                error_code="unknown_field",
                error_detail=(
                    f"Field {key!r} is not recognised in section {section!r}. "
                    f"Call query_requirements to see the current schema."
                ),
            )

    # Check for missing required fields.
    for fname, requirement in fields.items():
        if requirement == "required" and fname not in payload:
            return PortalResponse(
                status="rejected",
                section=section,
                error_code="missing_field",
                error_detail=(
                    f"Required field {fname!r} is missing from section {section!r}. "
                    f"Call query_requirements to see the current schema."
                ),
            )

    # ── Section-specific policy checks ───────────────────────────────────────

    if section == "payment":
        # Fee must exactly match the current policy amount.
        expected_fee = policy.get("amount_usd")
        submitted_fee = payload.get("amount_usd")
        if expected_fee is not None and submitted_fee != expected_fee:
            return PortalResponse(
                status="rejected",
                section=section,
                error_code="policy_violation",
                error_detail=(
                    f"Fee mismatch. The portal currently requires a specific "
                    f"amount. Call query_requirements on 'payment' to check "
                    f"the current fee."
                ),
            )

    elif section == "documents":
        # Documents section: only field presence is validated.
        # photo_spec is informational only — not checked against submitted value.
        pass

    elif section == "travel_info":
        # duration_days must not exceed the current max_duration_days policy.
        max_days = policy.get("max_duration_days")
        submitted_days = payload.get("duration_days")
        if (
            max_days is not None
            and submitted_days is not None
            and isinstance(submitted_days, (int, float))
            and submitted_days > max_days
        ):
            return PortalResponse(
                status="rejected",
                section=section,
                error_code="policy_violation",
                error_detail=(
                    f"duration_days {submitted_days} exceeds the maximum "
                    f"allowed stay of {max_days} days for this visa type."
                ),
            )

    elif section == "appointment":
        # booking_window_days is informational — we record the policy check
        # but do not reject on it (the agent can't know the appointment date
        # context without more scaffolding). Detection/recovery bookkeeping
        # still fires via the mutation tracking in step().
        pass

    return PortalResponse(status="accepted", section=section)


# ── Environment ───────────────────────────────────────────────────────────────

class VisaEnvironment(Environment):
    """
    OpenEnv-compliant VisaDrift environment.

    The agent completes a 5-section visa application while portal schemas,
    policies, and rules silently drift mid-episode. The key behaviour the
    environment trains is: call query_requirements() after a rejection (or
    when a drift_hint is present) rather than retrying the same payload.

    Mirrors the scheduler's MeetingSchedulerEnvironment class structure
    exactly: same __init__ shape, same reset/step/state/close signatures,
    same _make_obs() builder, same per-handler return (result_str, data)
    tuples.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False
    MAX_STEPS: int = 35

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=None, step_count=0)
        task = TASKS["easy"]
        self._current_task = task
        self._env_state = EnvironmentState(
            task_id="easy",
            visa_type=task["visa_type"],
            active_schemas=deepcopy(BASELINE_SCHEMAS),
            scheduled_mutations=deepcopy(task["scheduled_mutations"]),
            max_steps=task["max_steps"],
        )

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> VisaObservation:
        task_id = kwargs.get("task_id", "easy")
        if task_id not in TASKS:
            task_id = "easy"

        task = TASKS[task_id]
        self._current_task = task
        self._env_state = EnvironmentState(
            task_id=task_id,
            visa_type=task["visa_type"],
            active_schemas=deepcopy(BASELINE_SCHEMAS),
            scheduled_mutations=deepcopy(task["scheduled_mutations"]),
            max_steps=task["max_steps"],
        )
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return self._make_obs(
            result=(
                "Environment reset. Begin the visa application workflow. "
                f"Sections to complete (in order): {', '.join(SECTIONS)}. "
                "Call query_requirements to see the current schema for any section."
            ),
            reward=grade(self._env_state),
        )

    def step(
        self,
        action: VisaAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> VisaObservation:
        self._state.step_count += 1
        self._env_state.step_number += 1
        step = self._env_state.step_number

        # ── 1. Fire scheduled drifts for this step BEFORE handling the action.
        #
        # This means the agent's action at step N is validated against the
        # schema that has already been updated by any drift firing at step N.
        # That's the correct semantics: the portal changed, then you submitted.
        fired_now: List[MutationSpec] = [
            m for m in self._env_state.scheduled_mutations
            if m.trigger_step == step
        ]
        for m in fired_now:
            _apply_mutation(self._env_state.active_schemas, m)
            self._env_state.fired_mutations.append(m)
            self._env_state.detections.append(False)
            self._env_state.recoveries.append(False)
        if fired_now:
            self._env_state.drift_hint_active = True

        # ── 2. Parse action JSON — identical pattern to the scheduler.
        try:
            payload = json.loads(action.message)
        except (json.JSONDecodeError, AttributeError) as exc:
            return self._make_obs(
                result=(
                    f"Error: Invalid JSON — {exc}. "
                    'Expected e.g. {"action_type": "submit_section", '
                    '"section": "personal_info", "payload": {...}}'
                ),
                reward=grade(self._env_state),
            )

        act_type = payload.get("action_type", "")

        # ── 3. Dispatch to handler.
        if act_type == "query_requirements":
            result, schema = self._handle_query(payload)
            # Clear the drift hint — agent has now queried.
            self._env_state.drift_hint_active = False
            # Mark detection for any drift within the window.
            self._mark_detection_on_query()
            obs = self._make_obs(
                result=result,
                reward=grade(self._env_state),
                current_schema=schema,
            )

        elif act_type == "submit_section":
            result, portal_resp = self._handle_submit(payload)
            obs = self._make_obs(
                result=result,
                reward=grade(self._env_state),
                last_portal_response=portal_resp,
            )

        elif act_type == "done":
            result = self._handle_done()
            self._env_state.done = True
            obs = self._make_obs(
                result=result,
                reward=grade(self._env_state),
                done=True,
            )

        else:
            obs = self._make_obs(
                result=(
                    f"Unknown action_type {act_type!r}. "
                    "Valid: query_requirements | submit_section | done"
                ),
                reward=grade(self._env_state),
            )

        # ── 4. Step-limit termination — mirrors scheduler exactly.
        if (
            self._env_state.step_number >= self._env_state.max_steps
            and not self._env_state.done
        ):
            self._env_state.done = True
            obs.done = True
            obs.result += " | Episode timed out."

        # ── 5. Log the step.
        self._env_state.action_log.append({
            "step": step,
            "action_type": act_type,
            "reward": obs.reward,
        })
        self._env_state.total_reward += obs.reward or 0.0

        return obs

    @property
    def state(self) -> State:
        self._state.env_snapshot = {
            "task_id": self._env_state.task_id,
            "visa_type": self._env_state.visa_type.value,
            "accepted_sections": list(self._env_state.accepted_sections.keys()),
            "pending_sections": self._env_state.pending_sections(),
            "drifts_fired": len(self._env_state.fired_mutations),
            "detections": list(self._env_state.detections),
            "recoveries": list(self._env_state.recoveries),
            "total_reward": self._env_state.total_reward,
        }
        return self._state

    def close(self) -> None:
        pass

    # ── Action handlers ───────────────────────────────────────────────────────

    def _handle_query(
        self, payload: dict
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Return the currently-valid schema for one section or all sections.

        Clears drift_hint_active and triggers detection bookkeeping in step()
        after this handler returns. The handler itself is pure — it only reads
        state, never writes it.
        """
        section = payload.get("section")
        if section:
            if section not in SECTIONS:
                return (
                    f"Unknown section {section!r}. Valid: {SECTIONS}.",
                    {},
                )
            schema = self._env_state.active_schemas.get(section, {})
            return (
                f"Current requirements for section '{section}'.",
                {section: schema},
            )
        # No section specified — return all.
        return (
            "Current requirements for all sections.",
            dict(self._env_state.active_schemas),
        )

    def _handle_submit(
        self, payload: dict
    ) -> Tuple[str, PortalResponse]:
        """
        Validate a section payload against the current schema and update state.

        On acceptance:
          - Adds section to accepted_sections.
          - Marks recovery for any fired drift on this section.
          - Resets the anti-loop counter.

        On rejection:
          - Updates the repeat-fail fingerprint and counter.

        Mirrors the scheduler's _handle_create / _handle_reschedule pattern.
        """
        section = payload.get("section", "")
        sub_payload = payload.get("payload", {})

        if not isinstance(sub_payload, dict):
            sub_payload = {}

        resp = _validate_submission(
            self._env_state.active_schemas, section, sub_payload,
        )

        if resp.is_accepted:
            self._env_state.accepted_sections[section] = deepcopy(sub_payload)

            # Mark recovery for any fired drift on this section that hasn't
            # been recovered yet — the agent successfully resubmitted under
            # the new schema.
            for i, m in enumerate(self._env_state.fired_mutations):
                if m.section == section and not self._env_state.recoveries[i]:
                    self._env_state.recoveries[i] = True

            # Reset anti-loop bookkeeping on acceptance.
            self._env_state.last_failed_fingerprint = None
            self._env_state.repeat_fail_count = 0

            n_done = len(self._env_state.accepted_sections)
            n_total = len(SECTIONS)
            return (
                f"'{section}' accepted. {n_done}/{n_total} sections complete.",
                resp,
            )

        # Rejected — update anti-loop bookkeeping.
        fingerprint = EnvironmentState.submission_fingerprint(section, sub_payload)
        if fingerprint == self._env_state.last_failed_fingerprint:
            self._env_state.repeat_fail_count += 1
        else:
            self._env_state.last_failed_fingerprint = fingerprint
            self._env_state.repeat_fail_count = 1

        return (
            f"'{section}' rejected: {resp.error_code} — {resp.error_detail}",
            resp,
        )

    def _handle_done(self) -> str:
        n = len(self._env_state.accepted_sections)
        total = len(SECTIONS)
        pending = self._env_state.pending_sections()
        if pending:
            return (
                f"Done. {n}/{total} sections accepted. "
                f"Still pending: {', '.join(pending)}."
            )
        return f"Done. All {total}/{total} sections accepted. Application complete."

    # ── Detection bookkeeping ─────────────────────────────────────────────────

    def _mark_detection_on_query(self) -> None:
        """
        Flip detections[i] = True for any fired drift whose trigger_step is
        within DETECTION_WINDOW steps of the current step.

        Called after every query_requirements action. The window is defined
        in models.py as DETECTION_WINDOW = 2, matching the plan's spec.
        """
        step = self._env_state.step_number
        for i, m in enumerate(self._env_state.fired_mutations):
            if self._env_state.detections[i]:
                continue  # already detected
            if step - m.trigger_step <= DETECTION_WINDOW:
                self._env_state.detections[i] = True

    # ── Observation builder ───────────────────────────────────────────────────

    def _make_obs(
        self,
        result: str,
        reward: float = 0.0,
        done: Optional[bool] = None,
        current_schema: Optional[Dict[str, Any]] = None,
        last_portal_response: Optional[PortalResponse] = None,
    ) -> VisaObservation:
        """
        Build a VisaObservation from current environment state.
        Mirrors the scheduler's _make_obs() pattern exactly.
        """
        if done is None:
            done = self._env_state.done

        drift_hint = (
            "something_changed_recently"
            if self._env_state.drift_hint_active
            else None
        )

        return VisaObservation(
            result=result,
            workflow_progress=(
                f"{len(self._env_state.accepted_sections)}/{len(SECTIONS)} "
                f"sections accepted"
            ),
            current_schema=current_schema,
            last_portal_response=last_portal_response,
            drift_hint=drift_hint,
            drifts_fired_total=len(self._env_state.fired_mutations),
            task_instructions=self._current_task.get("description", ""),
            visa_type=self._env_state.visa_type.value,
            step_number=self._env_state.step_number,
            max_steps=self._env_state.max_steps,
            done=done,
            reward=reward,
        )

