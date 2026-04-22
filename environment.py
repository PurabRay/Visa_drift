"""
VisaDrift — core environment logic (FIXED).

Key fix vs. original:
  * SUPPORTS_CONCURRENT_SESSIONS is now False. The original value of True,
    combined with clients (inference.py / eval notebook) that do not thread
    episode_id through every /step call, caused the server to treat each
    HTTP request as a fresh episode — which is why progress stayed at
    "1/5 sections accepted" no matter how many sections you submitted.
    With this flag off, the server holds a single persistent env across
    the lifetime of the process, which is what inference.py actually expects.
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
            MutationSpec(
                trigger_step=3,
                kind=DriftKind.SCHEMA,
                section="travel_info",
                change={"rename": {"entry_date": "arrival_date"}},
            ),
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
            MutationSpec(
                trigger_step=2,
                kind=DriftKind.SCHEMA,
                section="personal_info",
                change={"rename": {"date_of_birth": "dob"}},
            ),
            MutationSpec(
                trigger_step=6,
                kind=DriftKind.RULE,
                section="documents",
                change={"add_required_field": "approval_notice"},
            ),
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

def _apply_mutation(schemas, mutation):
    section = schemas.setdefault(mutation.section, {"fields": {}, "policy": {}})
    fields = section.setdefault("fields", {})
    policy = section.setdefault("policy", {})
    change = mutation.change

    if mutation.kind == DriftKind.SCHEMA:
        for old, new in change.get("rename", {}).items():
            if old in fields:
                fields[new] = fields.pop(old)
    elif mutation.kind == DriftKind.POLICY:
        policy.update(change.get("update_policy", {}))
    elif mutation.kind == DriftKind.RULE:
        new_field = change.get("add_required_field")
        if new_field:
            fields[new_field] = "required"
        policy.update(change.get("update_policy", {}))


# ── Section validation ────────────────────────────────────────────────────────

def _validate_submission(schemas, section, payload):
    if section not in schemas:
        return PortalResponse(
            status="rejected", section=section,
            error_code="unknown_section",
            error_detail=f"No such section {section!r}. Valid: {SECTIONS}.",
        )

    schema = schemas[section]
    fields = schema.get("fields", {})
    policy = schema.get("policy", {})

    for key in payload.keys():
        if key not in fields:
            return PortalResponse(
                status="rejected", section=section,
                error_code="unknown_field",
                error_detail=f"Field {key!r} is not recognised in section {section!r}. Call query_requirements to see the current schema.",
            )
    for fname, requirement in fields.items():
        if requirement == "required" and fname not in payload:
            return PortalResponse(
                status="rejected", section=section,
                error_code="missing_field",
                error_detail=f"Required field {fname!r} is missing from section {section!r}. Call query_requirements to see the current schema.",
            )
    if section == "payment":
        expected = policy.get("amount_usd")
        submitted = payload.get("amount_usd")
        if expected is not None and submitted != expected:
            return PortalResponse(
                status="rejected", section=section,
                error_code="policy_violation",
                error_detail="Fee mismatch. Call query_requirements on 'payment' to check the current fee.",
            )
    elif section == "travel_info":
        max_days = policy.get("max_duration_days")
        sub_days = payload.get("duration_days")
        if max_days is not None and isinstance(sub_days, (int, float)) and sub_days > max_days:
            return PortalResponse(
                status="rejected", section=section,
                error_code="policy_violation",
                error_detail=f"duration_days {sub_days} exceeds the maximum allowed stay of {max_days} days for this visa type.",
            )

    return PortalResponse(status="accepted", section=section)


# ── Environment ───────────────────────────────────────────────────────────────

class VisaEnvironment(Environment):
    """
    OpenEnv-compliant VisaDrift environment.

    NOTE: SUPPORTS_CONCURRENT_SESSIONS is False so the openenv HTTP server
    reuses one env instance across /reset + /step calls. With True and a
    client that doesn't thread episode_id (as inference.py does not), every
    /step call creates a fresh env and progress resets.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False
    MAX_STEPS: int = 35

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=None, step_count=0)
        task = TASKS["easy"]
        self._current_task = task
        self._env_state = EnvironmentState(
            task_id="easy", visa_type=task["visa_type"],
            active_schemas=deepcopy(BASELINE_SCHEMAS),
            scheduled_mutations=deepcopy(task["scheduled_mutations"]),
            max_steps=task["max_steps"],
        )

    def reset(self, seed=None, episode_id=None, **kwargs):
        task_id = kwargs.get("task_id", "easy")
        if task_id not in TASKS:
            task_id = "easy"
        task = TASKS[task_id]
        self._current_task = task
        self._env_state = EnvironmentState(
            task_id=task_id, visa_type=task["visa_type"],
            active_schemas=deepcopy(BASELINE_SCHEMAS),
            scheduled_mutations=deepcopy(task["scheduled_mutations"]),
            max_steps=task["max_steps"],
        )
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return self._make_obs(
            result=(
                "Environment reset. Begin the visa application workflow. "
                f"Sections to complete (in order): {', '.join(SECTIONS)}. "
                "Call query_requirements to see the current schema."
            ),
            reward=grade(self._env_state),
        )

    def step(self, action, timeout_s=None, **kwargs):
        if self._env_state.done:
            return self._make_obs(
                result="Episode already complete. Call /reset to start a new one.",
                reward=grade(self._env_state), done=True,
            )

        self._state.step_count += 1
        self._env_state.step_number += 1
        step = self._env_state.step_number

        fired_now = [m for m in self._env_state.scheduled_mutations if m.trigger_step == step]
        for m in fired_now:
            _apply_mutation(self._env_state.active_schemas, m)
            self._env_state.fired_mutations.append(m)
            self._env_state.detections.append(False)
            self._env_state.recoveries.append(False)
        if fired_now:
            self._env_state.drift_hint_active = True

        try:
            payload = json.loads(action.message)
        except (json.JSONDecodeError, AttributeError) as exc:
            return self._make_obs(
                result=f"Error: Invalid JSON — {exc}. Expected e.g. {{\"action_type\": \"submit_section\", ...}}",
                reward=grade(self._env_state),
            )

        act_type = payload.get("action_type", "")

        if act_type == "query_requirements":
            result, schema = self._handle_query(payload)
            self._env_state.drift_hint_active = False
            self._mark_detection_on_query()
            obs = self._make_obs(result=result, reward=grade(self._env_state), current_schema=schema)
        elif act_type == "submit_section":
            result, portal_resp = self._handle_submit(payload)
            obs = self._make_obs(result=result, reward=grade(self._env_state), last_portal_response=portal_resp)
        elif act_type == "done":
            result = self._handle_done()
            self._env_state.done = True
            obs = self._make_obs(result=result, reward=grade(self._env_state), done=True)
        else:
            obs = self._make_obs(
                result=f"Unknown action_type {act_type!r}. Valid: query_requirements | submit_section | done",
                reward=grade(self._env_state),
            )

        if self._env_state.step_number >= self._env_state.max_steps and not self._env_state.done:
            self._env_state.done = True
            obs.done = True
            obs.result += " | Episode timed out."

        self._env_state.action_log.append({"step": step, "action_type": act_type, "reward": obs.reward})
        self._env_state.total_reward += obs.reward or 0.0
        return obs

    @property
    def state(self):
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

    def close(self):
        pass

    def _handle_query(self, payload):
        section = payload.get("section")
        if section:
            if section not in SECTIONS:
                return f"Unknown section {section!r}. Valid: {SECTIONS}.", {}
            schema = self._env_state.active_schemas.get(section, {})
            return f"Current requirements for section '{section}'.", {section: schema}
        return "Current requirements for all sections.", dict(self._env_state.active_schemas)

    def _handle_submit(self, payload):
        section = payload.get("section", "")
        sub_payload = payload.get("payload", {})
        if not isinstance(sub_payload, dict):
            sub_payload = {}
        resp = _validate_submission(self._env_state.active_schemas, section, sub_payload)
        if resp.is_accepted:
            self._env_state.accepted_sections[section] = deepcopy(sub_payload)
            for i, m in enumerate(self._env_state.fired_mutations):
                if m.section == section and not self._env_state.recoveries[i]:
                    self._env_state.recoveries[i] = True
            self._env_state.last_failed_fingerprint = None
            self._env_state.repeat_fail_count = 0
            n_done = len(self._env_state.accepted_sections)
            return f"'{section}' accepted. {n_done}/{len(SECTIONS)} sections complete.", resp

        fingerprint = EnvironmentState.submission_fingerprint(section, sub_payload)
        if fingerprint == self._env_state.last_failed_fingerprint:
            self._env_state.repeat_fail_count += 1
        else:
            self._env_state.last_failed_fingerprint = fingerprint
            self._env_state.repeat_fail_count = 1
        return f"'{section}' rejected: {resp.error_code} — {resp.error_detail}", resp

    def _handle_done(self):
        n = len(self._env_state.accepted_sections)
        total = len(SECTIONS)
        pending = self._env_state.pending_sections()
        if pending:
            return f"Done. {n}/{total} sections accepted. Still pending: {', '.join(pending)}."
        return f"Done. All {total}/{total} sections accepted. Application complete."

    def _mark_detection_on_query(self):
        step = self._env_state.step_number
        for i, m in enumerate(self._env_state.fired_mutations):
            if self._env_state.detections[i]:
                continue
            if step - m.trigger_step <= DETECTION_WINDOW:
                self._env_state.detections[i] = True

    def _make_obs(self, result, reward=0.0, done=None, current_schema=None, last_portal_response=None):
        if done is None:
            done = self._env_state.done
        drift_hint = "something_changed_recently" if self._env_state.drift_hint_active else None
        return VisaObservation(
            result=result,
            workflow_progress=f"{len(self._env_state.accepted_sections)}/{len(SECTIONS)} sections accepted",
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
