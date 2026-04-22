"""
inference.py — VisaDrift (FIXED)

Changes vs. original:
  * Thread episode_id from /reset through every /step call.
  * obs_to_text matches the training and eval notebook format byte-for-byte.
  * History trimmed to last 2 turns in the prompt (was full replay — blew the
    context window on hard task).
  * Score = last step's reward (was cumulative sum — meaningless).
  * Smoke-check that step_number advances after two /step calls — bail loudly
    if the server has lost state persistence.
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


# ── Configuration ────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "no-key")
OPENAI_BASE_URL = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
MODEL_NAME      = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL    = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASK_NAME       = os.getenv("TASK_NAME", "easy")
BENCHMARK       = "visa_drift"
MAX_TURNS       = 40
HISTORY_TRIM    = 2

_base_url_kwargs = {"base_url": OPENAI_BASE_URL} if OPENAI_BASE_URL else {}
client = OpenAI(api_key=OPENAI_API_KEY, **_base_url_kwargs)


# ── Logging ──────────────────────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.4f} done={str(done).lower()} error={err}", flush=True)

def log_end(success, steps, score, rewards):
    r = ",".join(f"{x:.4f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={r}", flush=True)


# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a visa application agent. Complete a multi-section visa application workflow.

Always respond in this exact format:
<reasoning>
Your step-by-step thinking about the current state: what is the next pending section, is the schema known, was the last action rejected, has a drift fired. Be specific, not generic.
</reasoning>
<action>
{"action_type": "...", ...}
</action>

Available action_types:
- query_requirements — learn current field requirements (call after any rejection or drift_hint)
- submit_section — submit a section: {"action_type": "submit_section", "section": "<name>", "payload": {...}}
- done — signal completion after all sections accepted

Rules:
- Submit sections in order: personal_info, travel_info, documents, appointment, payment
- If rejected, call query_requirements before retrying — never retry with the same payload
- If drift_hint is set, call query_requirements immediately
- Use ONLY fields listed in the current schema — do not use old field names after a drift"""


# ── Observation formatter — match training & eval notebooks EXACTLY ──────────
def obs_to_text(obs: Dict[str, Any]) -> str:
    if not isinstance(obs, dict):
        return str(obs)
    parts = []
    if obs.get("task_instructions"):
        parts.append(f"Task: {obs['task_instructions']}")
    if obs.get("result"):
        parts.append(f"Last result: {obs['result']}")
    if obs.get("current_schema"):
        parts.append(f"Current schema: {json.dumps(obs['current_schema'])}")
    if obs.get("workflow_progress"):
        parts.append(f"Progress: {obs['workflow_progress']}")
    if obs.get("last_portal_response"):
        parts.append(f"Portal response: {obs['last_portal_response']}")
    if obs.get("drift_hint"):
        parts.append(f"\u26a0\ufe0f DRIFT HINT: {obs['drift_hint']} — call query_requirements now!")
    return "\n".join(parts) if parts else str(obs)


# ── HTTP helpers ─────────────────────────────────────────────────────────────
def _post(path, body=None, timeout=60):
    r = requests.post(f"{ENV_BASE_URL}{path}", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _get(path, timeout=10):
    r = requests.get(f"{ENV_BASE_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()

def api_reset(task_id: str) -> Dict[str, Any]:
    return _post("/reset", {"kwargs": {"task_id": task_id}})

def api_step(action_json: str, episode_id: Optional[str] = None) -> Dict[str, Any]:
    body = {"action": {"message": action_json}}
    if episode_id:
        body["episode_id"] = episode_id
    return _post("/step", body)

def wait_for_server(max_wait=120):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            _get("/health")
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"Server at {ENV_BASE_URL} did not become healthy in {max_wait}s")


# ── Action extraction ────────────────────────────────────────────────────────
def extract_action(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    else:
        matches = re.findall(r"\{[^{}]*\}", text)
        if not matches:
            return None
        raw = matches[-1]
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if "action" in obj and "action_type" not in obj:
        obj["action_type"] = obj.pop("action")
    if "data" in obj and "payload" not in obj:
        obj["payload"] = obj.pop("data")
    return obj


# ── Action selection via the LLM ─────────────────────────────────────────────
def get_model_action(obs, history):
    obs_text = obs_to_text(obs)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in history[-HISTORY_TRIM:]:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": obs_text})

    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=messages,
        temperature=0.2, max_tokens=512,
    )
    text = resp.choices[0].message.content or ""
    action = extract_action(text)
    return text, action, obs_text


# ── Smoke check: confirm /step actually advances step_number ─────────────────
def smoke_state_check():
    print("=== smoke test: is server state-persistent across /step calls? ===", flush=True)
    d0 = api_reset("easy")
    eid = d0.get("episode_id") or d0.get("observation", {}).get("episode_id")
    d1 = api_step(json.dumps({"action_type": "query_requirements"}), episode_id=eid)
    d2 = api_step(json.dumps({"action_type": "query_requirements"}), episode_id=eid)
    sn1 = d1.get("observation", {}).get("step_number")
    sn2 = d2.get("observation", {}).get("step_number")
    print(f"  step_number after /step #1: {sn1}", flush=True)
    print(f"  step_number after /step #2: {sn2}", flush=True)
    if sn1 != 1 or sn2 != 2:
        raise RuntimeError(
            "STATE PERSISTENCE BROKEN. step_number must be 1 then 2. "
            f"Got {sn1} then {sn2}. Check SUPPORTS_CONCURRENT_SESSIONS=False in "
            "environment.py, max_concurrent_envs=1 in server/app.py, "
            "and uvicorn --workers 1."
        )
    print("  passed.", flush=True)


# ── Episode runner ───────────────────────────────────────────────────────────
def run_task(task_id: str) -> Dict[str, Any]:
    log_start(task_id, ENV_BASE_URL, MODEL_NAME)

    d0 = api_reset(task_id)
    eid = d0.get("episode_id") or d0.get("observation", {}).get("episode_id")
    obs = d0.get("observation", {})
    last_reward = d0.get("reward", 0.0)
    rewards: List[float] = [last_reward]
    history: List[tuple] = []

    steps = 0
    success = False
    error = None

    for turn in range(MAX_TURNS):
        turn_obs = dict(obs)
        if turn > 0:
            turn_obs.pop("task_instructions", None)

        try:
            raw, action, obs_text = get_model_action(turn_obs, history)
        except Exception as e:
            error = f"llm_error: {e}"
            break

        if action is None:
            error = "parse_error"
            break

        at = action.get("action_type", "?")
        history.append((obs_text, raw))

        try:
            d = api_step(json.dumps(action), episode_id=eid)
        except Exception as e:
            error = f"step_http_error: {e}"
            break

        obs = d.get("observation", {})
        last_reward = d.get("reward", last_reward)
        rewards.append(last_reward)
        steps = turn + 1
        done = bool(d.get("done"))
        log_step(steps, at, last_reward, done, None)
        if done:
            success = (obs.get("workflow_progress", "").startswith("5/5"))
            break

    score = last_reward
    log_end(success, steps, score, rewards)
    return {
        "task_id": task_id, "score": score, "steps": steps,
        "success": success, "rewards": rewards, "error": error,
    }


def main():
    wait_for_server()
    smoke_state_check()

    tasks = [TASK_NAME] if TASK_NAME != "all" else ["easy", "medium", "hard"]
    results = {}
    for t in tasks:
        results[t] = run_task(t)

    os.makedirs("outputs", exist_ok=True)
    out = {
        "environment": BENCHMARK,
        "model": MODEL_NAME,
        "results": {t: {k: v for k, v in r.items() if k != "rewards"}
                    for t, r in results.items()},
    }
    with open("outputs/baseline_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote outputs/baseline_results.json", flush=True)


if __name__ == "__main__":
    main()
