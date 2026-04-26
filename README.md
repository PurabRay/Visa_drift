
## 📌 Hackathon Submission Links
* **Hugging Face Space (Live Demo):** [https://huggingface.co/spaces/Rayewoie922/Visadrift_eval_space](https://huggingface.co/spaces/Rayewoie922/Visadrift_eval_space)
* **Colab Notebook (Training Code):** [https://colab.research.google.com/drive/1FF_z1QiRSpwKQqcQ07cgpfWqvQKU0idR?usp=sharing](https://colab.research.google.com/drive/1FF_z1QiRSpwKQqcQ07cgpfWqvQKU0idR?usp=sharing)
* **Code Repository:** [https://github.com/PurabRay/Visa_drift](https://github.com/PurabRay/Visa_drift)
* **Blog Post:** [https://huggingface.co/spaces/Rayewoie922/Visadrift_eval_space/blob/main/Blog.md]
* **Evaluation Notebook:**:**[https://colab.research.google.com/drive/1EkhKsWSgplI2ow5nEs1JSZ3KQIDkrA1t?usp=sharing]**
* **Training Environment:**:**[https://huggingface.co/spaces/Rayewoie922/VisaDrift]**
* **Blog-post(second link):**:**[https://huggingface.co/spaces/Rayewoie922/VisaDrift-Blog]** 
---

## 📖 Environment Description
**Real-world task:** Consumer-facing workflows (visa applications, insurance claims, tax filings) change constantly with no warning. Field names change, fees update, new required documents appear. Any agent trained on yesterday's portal breaks the moment a form updates. VisaDrift captures this failure mode as a trainable signal.

An OpenEnv-compliant RL environment where an AI agent acts as a visa application assistant inside a simulated embassy portal where schemas, policies, and rules silently change mid-episode. The agent must complete a 5-section application workflow while learning to **detect and recover from portal drift** rather than blindly retrying failed submissions.

## 👁️ Observation Space
| Field | Type | Description |
| :--- | :--- | :--- |
| `result` | string | Human-readable outcome of the last action |
| `workflow_progress` | string | e.g. "2/5 sections accepted" |
| `current_schema` | dict | Populated only after `query_requirements` |
| `last_portal_response` | dict | Rejection details after `submit_section` |
| `drift_hint` | string | "something_changed_recently" if drift fired, else null |
| `drifts_fired_total` | int | Cumulative count of drifts this episode |
| `task_instructions` | string | Task description |
| `visa_type` | string | e.g. "F-1" |
| `step_number` | int | Current step |
| `max_steps` | int | Step limit for this task |
| `done` | bool | Episode finished flag |
| `reward` | float | Grader score after this step (0.0–1.0) |

## 🕹️ Action Space
The agent sends a single JSON object in `action.message`:

```json
// 1. Query current requirements for a section (or all sections)
{"action_type": "query_requirements", "section": "personal_info"}
{"action_type": "query_requirements"}

// 2. Submit a section
{"action_type": "submit_section", "section": "personal_info",
"payload": {"full_name": "Jane Doe", "date_of_birth": "1998-03-15",
"passport_number": "P1234567", "nationality": "IN"}}

// 3. Signal completion
{"action_type": "done", "message": "Application complete."}
🎯 TasksTask IDVisaDriftsMax StepsChallengeeasyF-1 Student1 (rule, step 5)20New required field added to documentsmediumB-1/B-2 Tourist2 (schema step 3, policy step 8)25Field renamed + fee changeshardH-1B Work3 (schema step 2, rule step 6, policy step 10)35Multiple drift types across sections⚖️ Task Graders (0.0 – 1.0)Scores are clamped to [0.05, 0.95] so partial progress always receives signal.Easy (graders.grade_easy): 50% completion ratio, 20% drift detection rate, 25% drift recovery rate, 5% anti-loop score.Medium (graders.grade_medium): 40% completion ratio, 30% drift detection rate, 25% drift recovery rate, 5% anti-loop score.Hard (graders.grade_hard): 30% completion ratio, 35% drift detection rate, 30% drift recovery rate, 5% anti-loop score.🧬 Drift TaxonomyThree categories of silent mid-episode portal change:Schema drift — a field is renamed in a section (e.g. entry_date → arrival_date). An unadapted agent gets an unknown_field rejection.Policy drift — a numeric threshold or value changes (e.g. fee rises from $160 to $185). An unadapted agent gets a policy_violation rejection.Rule drift — a new required field or document appears (e.g. supporting_letter added to documents). An unadapted agent gets a missing_field rejection.The agent cannot see which drifts are armed — only their effects. A drift_hint flag signals that something changed recently.🏅 Reward FunctionPer-step reward is a weighted sum of four components:Completion: fraction of the five sections accepted.Detection rate: fraction of drifts the agent queried within 2 steps of firing.Recovery rate: fraction of drifts whose section was eventually re-accepted.Anti-loop: penalises repeated identical failed submissions.The key training signal: Baseline LLMs retry rejected payloads without querying. Trained models learn to call query_requirements immediately after any rejection or drift hint.📊 Baseline ScoresMeasured with llama-3.1-8b-instant via Groq API:TaskScoreeasy0.5000medium0.6000hard0.6000🚀 Setup & UsagePrerequisites: Python 3.10+, uv (or pip), Docker (optional).Local RunBash# Install dependencies
pip install openenv-core openai python-dotenv uvicorn fastapi pydantic

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal, run the baseline agent
export API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini
export TASK_NAME=all # easy | medium | hard | all
python inference.py
DockerBashdocker build -t visa-drift .
docker run -e API_KEY=sk-... -p 7860:7860 visa-drift
Validate with openenvBashpip install openenv-core
openenv validate .
🔌 API EndpointsMethodPathDescriptionPOST/resetReset environment; pass {"kwargs": {"task_id": "easy"}}POST/stepExecute action; pass {"action": {"message": "<json>"}}GET/stateCurrent environment stateGET/schemaAction / observation JSON schemasGET/healthHealth checkWS/wsWebSocket for persistent sessions
