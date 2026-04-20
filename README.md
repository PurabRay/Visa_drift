VisaDrift — OpenEnv Environment
An OpenEnv-compliant RL environment where an AI agent acts as a visa application assistant inside a simulated embassy portal where schemas, policies, and rules silently change mid-episode. The agent must complete a 5-section application workflow while learning to detect and recover from portal drift rather than blindly retrying failed submissions.

Environment Description
Real-world task: consumer-facing workflows (visa applications, insurance claims, tax filings) change constantly with no warning. Field names change, fees update, new required documents appear. Any agent trained on yesterday's portal breaks the moment a form updates. VisaDrift captures this failure mode as a trainable signal.
The agent operates on a simulated embassy portal with five workflow sections. At each step it receives the current portal state and must emit one JSON action. The environment injects silent schema, policy, and rule changes mid-episode and rewards the agent for detecting drift and recovering — not just completing the form.

Observation Space
FieldTypeDescriptionresultstringHuman-readable outcome of the last actionworkflow_progressstringe.g. "2/5 sections accepted"current_schemadictPopulated only after query_requirementslast_portal_responsedictRejection details after submit_sectiondrift_hintstring"something_changed_recently" if drift fired, else nulldrifts_fired_totalintCumulative count of drifts this episodetask_instructionsstringTask descriptionvisa_typestringe.g. "F-1"step_numberintCurrent stepmax_stepsintStep limit for this taskdoneboolEpisode finished flagrewardfloatGrader score after this step (0.0–1.0)

Action Space
The agent sends a single JSON object in action.message:
json// 1. Query current requirements for a section (or all sections)
{"action_type": "query_requirements", "section": "personal_info"}
{"action_type": "query_requirements"}

// 2. Submit a section
{"action_type": "submit_section", "section": "personal_info",
 "payload": {"full_name": "Jane Doe", "date_of_birth": "1998-03-15",
             "passport_number": "P1234567", "nationality": "IN"}}

// 3. Signal completion
{"action_type": "done", "message": "Application complete."}

Tasks
Task IDVisaDriftsMax StepsChallengeeasyF-1 Student1 (rule, step 5)20New required field added to documentsmediumB-1/B-2 Tourist2 (schema step 3, policy step 8)25Field renamed + fee changeshardH-1B Work3 (schema step 2, rule step 6, policy step 10)35Multiple drift types across sections
Task Graders (0.0 – 1.0)
Easy — graders.grade_easy

50% completion ratio (sections accepted / total)
20% drift detection rate
25% drift recovery rate
5% anti-loop score

Medium — graders.grade_medium

40% completion ratio
30% drift detection rate
25% drift recovery rate
5% anti-loop score

Hard — graders.grade_hard

30% completion ratio
35% drift detection rate
30% drift recovery rate
5% anti-loop score

Scores are clamped to [0.05, 0.95] so partial progress always receives signal.

Drift Taxonomy
Three categories of silent mid-episode portal change:

Schema drift — a field is renamed in a section (e.g. entry_date → arrival_date). An unadapted agent gets an unknown_field rejection.
Policy drift — a numeric threshold or value changes (e.g. fee rises from $160 to $185). An unadapted agent gets a policy_violation rejection.
Rule drift — a new required field or document appears (e.g. supporting_letter added to documents). An unadapted agent gets a missing_field rejection.

The agent cannot see which drifts are armed — only their effects. A drift_hint flag signals that something changed recently.

Reward Function
Per-step reward is a weighted sum of four components:

Completion — fraction of the five sections accepted
Detection rate — fraction of drifts the agent queried within 2 steps of firing
Recovery rate — fraction of drifts whose section was eventually re-accepted
Anti-loop — penalises repeated identical failed submissions

The key training signal: baseline LLMs retry rejected payloads without querying. Trained models learn to call query_requirements immediately after any rejection or drift hint.

Baseline Scores
Measured with llama-3.1-8b-instant via Groq API:
TaskScoreeasy0.6000medium0.5000hard0.6000

Setup & Usage
Prerequisites

Python 3.10+
uv (or pip)
Docker (optional, for containerised run)

Local run
bash# Install dependencies
pip install openenv-core openai python-dotenv uvicorn fastapi pydantic

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal, run the baseline agent
export API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini
export TASK_NAME=all   # easy | medium | hard | all
python inference.py
Docker
bashdocker build -t visa-drift .
docker run -e API_KEY=sk-... -p 7860:7860 visa-drift
Validate with openenv
bashpip install openenv-core
openenv validate .

API Endpoints
MethodPathDescriptionPOST/resetReset environment; pass {"kwargs": {"task_id": "easy"}}POST/stepExecute action; pass {"action": {"message": "<json action>"}}GET/stateCurrent environment stateGET/schemaAction / observation JSON schemasGET/healthHealth checkWS/wsWebSocket for persistent sessions
