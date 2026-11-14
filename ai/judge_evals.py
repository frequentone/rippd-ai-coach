"""
judge_evals.py — LLM-graded answer-QUALITY evals for the RIPPD coach.

Unlike fixed_evals.py (deterministic plumbing), this actually GENERATES coach
answers and has an LLM judge score them against a per-case rubric. It measures
the stuff fixed_evals can't: teaching the "why", emotional attunement, grounding
in the user's data, format/safety adherence, accuracy.

COSTS TOKENS. Not for CI on every push — run it when you change prompts, the
knowledge base, or model routing, and want to know if answer *quality* moved.

Design:
  - Isolated context: a fixed synthetic user (no DB) → reproducible.
  - Generation mirrors prod: same build_system_prompt + knowledge injection +
    model routing the app uses (Groq 70b for questions).
  - Judge: DeepSeek if DEEPSEEK_API_KEY is set (cross-model, less self-bias),
    else falls back to Groq 70b. Judge runs at temperature 0.
  - A case passes if mean(rubric scores) >= PASS_THRESHOLD and not hard_fail.

Run from ai/:  python judge_evals.py
"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("SUPABASE_URL", "https://dummy.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "dummy")

from groq import Groq  # noqa: E402
from openai import OpenAI  # noqa: E402
from rag_pipeline import (  # noqa: E402
    build_system_prompt,
    search_gym_knowledge,
    MODEL,
    MODEL_FAST,
    MODEL_FAST_70B,
)

PASS_THRESHOLD = 3.5  # mean rubric score (1-5) required to pass a case

_GROQ_KEY = os.environ.get("GROQ_API_KEY")
_DS_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not _GROQ_KEY:
    print("GROQ_API_KEY missing — cannot generate answers. Set it in ai/.env and retry.")
    sys.exit(2)

groq_client = Groq(api_key=_GROQ_KEY)
ds_client = OpenAI(api_key=_DS_KEY, base_url="https://api.deepseek.com") if _DS_KEY else None

# Judge: prefer cross-model (DeepSeek) to reduce self-grading bias; else Groq 70b.
if ds_client:
    _judge_client, _judge_model, _judge_name = ds_client, MODEL, "deepseek"
else:
    _judge_client, _judge_model, _judge_name = groq_client, MODEL_FAST_70B, "groq-70b (fallback)"

CREATOR = {"name": "Coach Rip",
           "coaching_philosophy": "Progressive overload, no fluff. Compounds first.",
           "diet_philosophy": "High protein, flexible dieting.",
           "tone": "direct, motivating"}


def make_ctx(height=183, weight=72, gender="male", with_macros=True):
    return {
        "user": {"full_name": "Ashish Kumar"},
        "macros": {"calories": 2400, "protein_g": 180, "carbs_g": 240,
                   "fat_g": 70, "water_liters": 3.5, "fiber_g": 30} if with_macros else None,
        "program": {"programs": {"title": "Upper/Lower Split"},
                    "current_week": 3, "current_day": 2},
        "preferences": {"diet_preference": "veg", "diet_goal": "gain muscle",
                        "equipment": "full gym", "fitness_level": "intermediate",
                        "height": height, "weight": weight, "age": 24, "gender": gender},
        # Most-recent-first (prod order): 71 -> 72 reads as "up 1kg", matching the
        # muscle-gain goal. Keep this consistent with each case's context_note or
        # the judge grades the model against a fiction.
        "weight_logs": [{"body_weight": 72.0}, {"body_weight": 71.0}],
        "personal_bests": [{"exercise_name": "Bench Press", "weight_kg": 85, "reps": 5}],
        "streak": 12,
        "todays_workout": None,
    }


# ── generation — mirrors chat()'s assembly + routing ─────────────────────────
def generate(intent: str, user_message: str, context: dict) -> str:
    system_prompt = build_system_prompt(context, CREATOR, intent=intent)
    if intent in ("question", "medical"):
        chunks = search_gym_knowledge(user_message, top_k=2)
        if chunks:
            system_prompt += (
                "\n\nVERIFIED BIOMECHANICS & TRAINING SCIENCE "
                "(use these exact facts — no substitutions):\n\n"
                + "\n\n---\n\n".join(chunks)
                + "\n\nTEACH THE MECHANISM: give the user the ONE reason each pick works, "
                "in plain words. Spell out any jargon the first time."
            )
    if intent == "question":
        model, max_tokens, temp = MODEL_FAST_70B, 800, 0.6
    elif intent == "greeting":
        model, max_tokens, temp = MODEL_FAST, 250, 0.75
    else:  # medical + others route to the fast model in prod
        model, max_tokens, temp = MODEL_FAST, 500, 0.7
    resp = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_message}],
        max_tokens=max_tokens, temperature=temp, top_p=0.9,
    )
    return (resp.choices[0].message.content or "").strip()


# ── judge ─────────────────────────────────────────────────────────────────────
def judge(case: dict, answer: str) -> dict:
    rubric = "\n".join(f'- {k}: {v} (1-5)' for k, v in case["rubric"].items())
    prompt = f"""You are a STRICT evaluator of an AI fitness coach's reply. Be harsh — 5 is reserved for genuinely excellent.

User asked: "{case['user_message']}"
What the coach knew about the user: {case['context_note']}

Coach's reply:
\"\"\"
{answer}
\"\"\"

Score each criterion 1-5:
{rubric}

Set "hard_fail": true ONLY if the reply {case['hard_fail']}.

Reply with ONLY a JSON object, no other text:
{{"scores": {{{", ".join(f'"{k}": <int>' for k in case["rubric"])}}}, "hard_fail": <true|false>, "note": "<one short line>"}}"""
    resp = _judge_client.chat.completions.create(
        model=_judge_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300, temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    m = re.search(r"\{[\s\S]+\}", raw)
    if not m:
        return {"scores": {}, "hard_fail": True, "note": "judge returned no JSON", "_error": True}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {"scores": {}, "hard_fail": True, "note": "judge JSON parse failed", "_error": True}


# ── cases — each targets a goal from the coach upgrade ────────────────────────
CASES = [
    {
        "name": "look_bigger_5'4",
        "intent": "question",
        "user_message": "my height is 5'4 and i look skinny and tiny, what do i do to look bigger? "
                        "but then those short and bulky ppl look chopped",
        "context": make_ctx(height=163),
        "context_note": "5'4 (163cm), skinny, goal is muscle gain, full gym, intermediate.",
        "rubric": {
            "attunement": "addresses the 'skinny / look chopped' worry, not just a workout dump",
            "teach_why": "explains WHY its picks build the look (e.g. delts widen the frame)",
            "actionable": "gives specific exercises/sets, not vague advice",
            "accuracy": "advice is sound, nothing dangerous or wrong",
        },
        "hard_fail": "gives dangerous or clearly incorrect fitness advice",
    },
    {
        "name": "protein_target",
        "intent": "question",
        "user_message": "how much protein should i eat daily?",
        "context": make_ctx(),
        "context_note": "72kg, muscle gain goal, protein target 180g stored.",
        "rubric": {
            "grounding": "uses the user's weight/goal or stated target, not generic",
            "specificity": "gives a concrete gram number",
            "accuracy": "figure is sane (roughly 1.2-2.5 g/kg)",
        },
        "hard_fail": "gives a protein figure wildly outside 1.2-2.5 g per kg bodyweight",
    },
    {
        "name": "shoulder_pain",
        "intent": "medical",
        "user_message": "my shoulder hurts when i do overhead press",
        "context": make_ctx(),
        "context_note": "intermediate lifter, shoulder discomfort on OHP.",
        "rubric": {
            "empathy": "acknowledges the discomfort",
            "concrete_fix": "gives a specific training adjustment (swap/grip/ROM), not a punt",
            "tone": "authoritative and safe, not alarmist",
        },
        "hard_fail": "just tells them to see a doctor with no concrete training adjustment",
    },
    {
        "name": "on_track",
        "intent": "question",
        "user_message": "am i on track with my goals?",
        "context": make_ctx(),
        "context_note": "12-day streak, weight trend UP 1kg (now 72kg), muscle gain goal, 180g protein target.",
        "rubric": {
            "grounding": "cites at least one real number from the user's data",
            "directness": "gives a clear read, not waffle",
            "actionable": "names one thing to keep or fix",
        },
        "hard_fail": "invents specific numbers that contradict the given context",
    },
    {
        "name": "height_contradiction",
        "intent": "question",
        "user_message": "i'm 5'4 by the way — what should my daily calorie target be?",
        "context": make_ctx(height=183),
        "context_note": "SAVED height is 183cm, but the user now says they are 5'4 (163cm).",
        "rubric": {
            "catches_contradiction": "flags that 5'4 conflicts with the saved 183cm profile",
            "guides_fix": "tells them to update it in Profile/settings so macros recalc",
            "still_helpful": "still gives a useful answer rather than stonewalling",
        },
        "hard_fail": "silently accepts 5'4 with no mention of the profile mismatch",
    },
    {
        "name": "what_to_eat",
        "intent": "question",
        "user_message": "what should i eat right now?",
        "context": make_ctx(),
        "context_note": "2400 kcal / 180g protein targets; veg; muscle gain.",
        "rubric": {
            "grounding": "references the user's targets or that they're vegetarian",
            "concrete": "gives specific meal/snack options with rough calories or protein",
            "actionable": "the user could act on it immediately",
        },
        "hard_fail": "suggests non-vegetarian food despite the veg preference",
    },
    {
        "name": "beginner_start",
        "intent": "question",
        "user_message": "i've never worked out in my life, where do i even start?",
        "context": make_ctx(),
        "context_note": "complete beginner.",
        "rubric": {
            "teach": "explains the approach, not just a raw list",
            "not_overwhelming": "simple and structured, doesn't dump 12 things",
            "actionable": "gives a clear first step",
        },
        "hard_fail": "recommends an advanced or unsafe program for a total beginner",
    },
    {
        "name": "massive_shoulders",
        "intent": "question",
        "user_message": "how do i build massive shoulders?",
        "context": make_ctx(),
        "context_note": "intermediate, full gym.",
        "rubric": {
            "teach_why": "explains delt mechanics (e.g. side delts drive width)",
            "uses_knowledge": "biomechanically sound (laterals isolate side delt, etc.)",
            "actionable": "specific exercises with sets/reps",
        },
        "hard_fail": "claims compound pressing alone maximally builds the side delts",
    },
]


def main() -> int:
    print(f"Generator: Groq (question→{MODEL_FAST_70B}, other→{MODEL_FAST})")
    print(f"Judge: {_judge_name} ({_judge_model}), temp 0, pass>= {PASS_THRESHOLD} mean & no hard_fail")
    print(f"Cases: {len(CASES)}\n")

    results = []
    for c in CASES:
        try:
            answer = generate(c["intent"], c["user_message"], c["context"])
            verdict = judge(c, answer)
        except Exception as e:  # noqa: BLE001
            results.append((c["name"], None, None, False, f"ERROR: {e}"))
            print(f"  ERROR  {c['name']}: {e}")
            continue
        scores = verdict.get("scores", {}) or {}
        vals = [v for v in scores.values() if isinstance(v, (int, float))]
        mean = round(sum(vals) / len(vals), 2) if vals else 0.0
        hard = bool(verdict.get("hard_fail"))
        ok = (mean >= PASS_THRESHOLD) and not hard and not verdict.get("_error")
        results.append((c["name"], mean, scores, ok, verdict.get("note", "")))
        mark = "PASS" if ok else "FAIL"
        flag = " HARD_FAIL" if hard else ""
        breakdown = " ".join(f"{k}={v}" for k, v in scores.items())
        print(f"  {mark}  {c['name']}  mean={mean}{flag}")
        print(f"        {breakdown}")
        print(f"        judge: {verdict.get('note','')}\n")

    passed = sum(1 for *_, ok, _ in results if ok)
    graded = [m for _, m, _, _, _ in results if isinstance(m, (int, float))]
    overall = round(sum(graded) / len(graded), 2) if graded else 0.0
    print(f"{passed}/{len(results)} cases passed · overall mean quality {overall}/5")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
