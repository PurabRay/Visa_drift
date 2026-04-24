"""
inference.py — VisaDrift (FAIR BASELINE)
========================================
Updated baseline inference script. This version levels the playing field 
by using the exact same System Prompt, Observation text cleaner, Context Window, 
and Fallback extraction logic that the trained GRPO model gets.
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

# -- Agent Logic Matching Notebook Exactly -------------------------------------

SYSTEM_PROMPT = """You are a visa application agent. Complete a multi-section visa application workflow.
Always respond in this exact format:
<reasoning>
Think step by step about what to do next based on the current observation.
</reasoning>
<action>
{"action_type": "submit_section", "section": "personal_info", "payload": {"full_name": "Jane Doe", "date_of_birth": "1999-01-01", "passport_number": "P12345", "nationality": "US"}}
</action>

Available action_types:
- query_requirements — learn current field requirements (call this after any rejection or when you see drift_hint)
- submit_section — submit a section with a fully complete payload dictionary.
- done — signal completion after all sections accepted

Rules:
- Submit sections in order: personal_info, travel_info, documents, appointment, payment
- If rejected, call query_requirements before retrying — never retry with the same payload
- If drift_hint is set, call query_requirements immediately
- Use ONLY fields listed in the current schema — do not use old field names after a drift
- CRITICAL: NEVER use "..." or output incomplete JSON. You must generate the full dictionary with all required fields.
"""

def obs_to_text(obs: Dict[str, Any]) -> str:
    if isinstance(obs, dict):
        parts = []
        step = obs.get("step_number", 0)
        max_s = obs.get("max_steps", 20)
        prog = obs.get("workflow_progress", "0/5 sections accepted")
        parts.append(f"Step {step}/{max_s} | Progress: {prog}")
        
        if obs.get("drift_hint"):
            parts.append(f"⚠️ DRIFT HINT: {obs['drift_hint']} — call query_requirements now!")
            
        res = obs.get("result", "")
        if res:
            parts.append(f"Result: {res}")
            
        pr = obs.get("last_portal_response")
        if pr:
            if isinstance(pr, dict):
                parts.append(f"Portal response: {pr.get('status', 'unknown')}")
            else:
                parts.append(f"Portal response: {pr}")
            
        if obs.get("current_schema"):
            parts.append(f"Current schema: {json.dumps(obs['current_schema'])}")
            
        return "\n".join(parts)
    return str(obs)

def get_model_action(
    obs: Dict[str, Any],
    step: int,
    history: List[Dict],
) -> Dict[str, Any]:
    
    # 1. Clean the observation
    user_content = obs_to_text(obs)

    # 2. Emulate the window_size = 2 (meaning last 2 pairs, 4 items total)
    trimmed = history[-4:] if len(history) > 4 else list(history)
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
            temperature=0.1, # Match your notebook's temp
            messages=messages,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        raw = '<action>\n{"action_type": "query_requirements"}\n</action>'

    history.append({"role": "assistant", "content": raw})

    # 3. Exact matching extraction logic with safe fallback
    text = raw
    if "<action>" in text:
        text = text.split("<action>")[-1].split("</action>")[0].strip()
        
    try:
        start = text.find('{')
        if start != -1:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(text[start:])
            
            if "action" in obj and "action_type" not in obj:
                obj["action_type"] = obj.pop("action")
            if "data" in obj and "payload" not in obj:
                obj["payload"] = obj.pop("data")
            return obj
    except Exception:
        pass
        
    # Safe fallback
    return {"action_type": "query_requirements"}


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
    return _post("/reset", {"kwargs": {"task_id": task_id}})

def api_step(action_json: str) -> Dict[str, Any]:
    return _post("/step", {"action": {"message": action_json}})


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
        max_steps = obs.get("max_steps", 30) # Match notebook
        for step in range(1, max_steps + 1):
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
        final_score = max(0.00, min(1.00, final_score)) # Removed 0.05 min clamp
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

    print("\n=== Fair Baseline Results ===")
    for tid, sc in results.items():
        print(f"  {tid:8s}: {sc:.4f}")
    print("=============================")


if __name__ == "__main__":
    main()
