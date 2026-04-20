"""
inference.py — VisaDrift
========================
Baseline inference script. Runs an LLM agent against the VisaDrift
environment using the OpenAI API client.

Mirrors the Smart Meeting Scheduler inference.py exactly:
  - Same environment variable names (API_KEY / OPENAI_API_KEY / HF_TOKEN,
    API_BASE_URL, MODEL_NAME, ENV_BASE_URL, TASK_NAME)
  - Same [START] / [STEP] / [END] log format
  - Same api_reset / api_step / wait_for_server helpers
  - Same get_model_action with trimmed history and regex JSON fallback
  - Same run_task / main structure

Usage:
    # Against local server:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
    export API_KEY=sk-...
    python inference.py

    # Against HF Space:
    export ENV_BASE_URL=https://<your-space>.hf.space
    export API_KEY=sk-...
    python inference.py

    # Run all tasks:
    export TASK_NAME=all
    python inference.py

Environment variables:
    API_KEY / OPENAI_API_KEY / HF_TOKEN  - required (checked in order)
    API_BASE_URL / OPENAI_BASE_URL       - optional; set for open models
    MODEL_NAME                           - model to use (default: gpt-4o-mini)
    ENV_BASE_URL                         - environment server URL
                                           (default: http://localhost:7860)
    TASK_NAME                            - easy | medium | hard | all
                                           (default: easy)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


# -- Configuration -------------------------------------------------------------

OPENAI_API_KEY  = (
    os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("HF_TOKEN", "no-key")
)
OPENAI_BASE_URL = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
MODEL_NAME      = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL    = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASK_NAME       = os.getenv("TASK_NAME", "easy")
BENCHMARK       = "visa_drift"
_base_url_kwargs = {"base_url": OPENAI_BASE_URL} if OPENAI_BASE_URL else {}

client = OpenAI(
    api_key=OPENAI_API_KEY,
    **_base_url_kwargs,
)


# -- Structured logging --------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# -- System prompt -------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert visa application agent. Your job is to complete a \
5-section visa application on a government portal. The five sections \
are (in order): personal_info, travel_info, documents, appointment, payment.

IMPORTANT: Portal requirements can change silently mid-application. \
If you see drift_hint = "something_changed_recently" in the observation, \
or if a section is rejected, you MUST call query_requirements before \
resubmitting. Never retry a rejected payload without querying first.

Each step you receive a JSON observation with:
  - result:              feedback from your last action
  - workflow_progress:   e.g. "2/5 sections accepted"
  - current_schema:      section schema (only present after query_requirements)
  - last_portal_response: rejection details (only present after submit_section)
  - drift_hint:          "something_changed_recently" if portal changed, else null
  - drifts_fired_total:  how many drifts have fired this episode
  - step_number / max_steps: progress counters
  - task_instructions:   task description

Respond with EXACTLY ONE JSON action object - no markdown fences, no prose:

1. Query requirements for a section (or all sections):
{"action_type": "query_requirements", "section": "personal_info"}
{"action_type": "query_requirements"}

2. Submit a section:
{"action_type": "submit_section", "section": "personal_info",
 "payload": {"full_name": "Jane Doe", "date_of_birth": "1998-03-15",
             "passport_number": "P1234567", "nationality": "IN"}}

3. Signal completion when all sections are accepted:
{"action_type": "done", "message": "Application complete."}

Rules:
- Complete sections in order: personal_info -> travel_info -> documents ->
  appointment -> payment.
- BEFORE every action, check workflow_progress. If a section already
  appears as accepted, NEVER submit it again. Move to the next pending section.
- Always call query_requirements on a section before submitting it for
  the first time, so you know the exact current field names.
- If a submit_section is rejected, IMMEDIATELY call query_requirements
  on that section - do not retry the same payload.
- Use the field names and policy values from current_schema exactly.
- Once a section is accepted, move on. Do not re-query or re-submit it.
- Call done only when workflow_progress shows all 5 sections accepted.
- Respond ONLY with the raw JSON. No explanation.\
"""


# -- HTTP helpers --------------------------------------------------------------

def _post(path: str, body: Any = None, timeout: int = 30) -> Dict[str, Any]:
    url = f"{ENV_BASE_URL}{path}"
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get(path: str, timeout: int = 10) -> Dict[str, Any]:
    url = f"{ENV_BASE_URL}{path}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_reset(task_id: str) -> Dict[str, Any]:
    """POST /reset with task_id kwarg - openenv reset body format."""
    return _post("/reset", {"kwargs": {"task_id": task_id}})


def api_step(action_json: str) -> Dict[str, Any]:
    """POST /step with correct openenv action envelope."""
    return _post("/step", {"action": {"message": action_json}})


# -- LLM action selection ------------------------------------------------------

def get_model_action(
    obs: Dict[str, Any],
    step: int,
    history: List[Dict],
) -> Dict[str, Any]:
    user_content = (
        f"STEP {step}\n\n"
        f"OBSERVATION:\n{json.dumps(obs, indent=2)}\n\n"
        "Respond with your next action as a JSON object."
    )

    trimmed = history[-6:] if len(history) > 6 else list(history)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *trimmed,
        {"role": "user", "content": user_content},
    ]
    history.append({"role": "user", "content": user_content})

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=512,
            messages=messages,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        raw = '{"action_type": "done", "message": "LLM error."}'

    history.append({"role": "assistant", "content": raw})

    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {"action_type": "done", "message": "JSON parse error."}


# -- Task runner ---------------------------------------------------------------

def run_task(task_id: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    reset_resp = api_reset(task_id)
    obs = reset_resp.get("observation", reset_resp)

    history: List[Dict] = []
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    try:
        max_steps = obs.get("max_steps", 35)
        for step in range(1, max_steps + 5):
            action_dict = get_model_action(obs, step, history)
            action_str = json.dumps(action_dict, separators=(",", ":"))

            step_resp = api_step(action_str)

            obs    = step_resp.get("observation", {})
            reward = float(step_resp.get("reward", 0.05))
            done   = bool(step_resp.get("done", False))
            error  = (
                step_resp.get("info", {}).get("error")
                if step_resp.get("info")
                else None
            )

            rewards.append(reward)
            steps_taken = step
            final_score = reward

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        success = final_score > 0.5

    except Exception as exc:
        print(f"[DEBUG] run_task({task_id}) error: {exc}", flush=True)

    finally:
        final_score = max(0.05, min(0.95, final_score))
        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards,
        )

    return final_score


# -- Server readiness check ----------------------------------------------------

def wait_for_server(timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                print(f"Server ready at {ENV_BASE_URL}", flush=True)
                return
        except Exception:
            pass
        attempt += 1
        print(f"Waiting for server... ({attempt})", flush=True)
        time.sleep(2)
    print("ERROR: Server did not start in time.", file=sys.stderr)
    sys.exit(1)


# -- Entry point ---------------------------------------------------------------

def main() -> None:
    wait_for_server()

    tasks_to_run = (
        ["easy", "medium", "hard"] if TASK_NAME == "all" else [TASK_NAME]
    )
    results = {}
    for task_id in tasks_to_run:
        score = run_task(task_id)
        results[task_id] = score

    print("\n=== Baseline Results ===")
    for tid, sc in results.items():
        print(f"  {tid:8s}: {sc:.4f}")
    print("========================")


if __name__ == "__main__":
    main()