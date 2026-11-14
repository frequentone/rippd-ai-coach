# RIPPD AI Coach

The AI coach from RIPPD, a fitness app that's live on the Play Store. It knows
each user's logged training, nutrition and body weight, answers questions about
their own data, and can act in the app (build a program, log food, swap an
exercise) with the user confirming every change.

This repo holds only the coach. The rest of the app isn't included, so imports
such as `diet_plan` won't resolve and no keys are included: it's here to read.

## How a message flows in production

1. **`ai/main.py`, `/chat`** loads a cached snapshot of the user's data
   (`ai/context_builder.py`).
2. **`ai/rag_pipeline.py`, `chat()`** handles the turn:
   - **Safety first, in code.** `_classify_intent_regex()` sends crisis
     messages, medical red flags and off-topic messages to fixed replies. They
     never reach a model.
   - **Intent.** A small classifier (`classify_intent_smart()`) picks the path.
   - **Model.** `complete_for_intent()` routes by intent: DeepSeek, with Groq as
     the fallback.
   - **Programs.** `_handle_tool_intent()`: the model calls a tool and returns
     structured JSON. Code then checks it: `_snap_to_db_exercise()` (no invented
     exercises), `_enforce_program_rules()`, `_enforce_difficulty_ceiling()`.
   - **Food logs.** `_ground_food_macros()` recalculates macros from the foods
     table instead of trusting the model's arithmetic.
3. **Nothing is written until the user confirms**: `execute_program_creation()`,
   `execute_actions()`.
4. **The app side**: `src/app/(tabs)/coach.tsx` (chat and confirm cards) and
   `src/services/coachService.ts`.

The pattern throughout: the model proposes, code validates, the user confirms.

## Agent loop (prototype, not in production)

**`ai/agent_loop.py`.** In production, keyword matching decides which of the
user's data the model sees. Here the model decides. It calls read-only tools
(`get_user_data`, `get_lift_history`, `search_knowledge`, `lookup_food`), reads
the results, and can call more, for up to 4 rounds, before it answers.

Kept from production on purpose:
- Safety runs first, in code.
- The user's profile (goals, injuries, standing facts) is always in the prompt.
- Tools only read. A change comes back as a proposal (`propose_change`) for the
  user to confirm.
- The loop is bounded, and an empty answer raises instead of becoming a reply.

```bash
python ai/agent_loop.py <user_id> "is my bench ready to go up?"
```

`ai/test_agent_loop.py` checks it offline with a scripted fake model.

## Evals

- **`ai/fixed_evals.py`**: deterministic checks (routing, safety regex, prompt
  assembly, parsing, failures surfacing as errors). No model calls.
- **`ai/judge_evals.py`**: real answers scored by an LLM judge against rubrics.
- **Persona simulation**: `personas.py` generates synthetic users, `simulate.py`
  runs multi-turn chats against `ai/eval_server.py`, and `analyze.py` writes
  `report.md`. See `EVAL_HARNESS.md`.

In the run saved in `results.jsonl`, 31 of 72 replies were the pipeline's own
"Something went wrong" message returned with HTTP 200. The harness logged them
as successes and the judge scored them as bad coaching. Failures in the
pipeline now raise, and `/chat` returns a 503.
