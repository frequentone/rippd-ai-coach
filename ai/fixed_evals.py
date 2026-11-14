"""
fixed_evals.py — Deterministic eval suite for the RIPPD RAG coach.

Runs the coach's PURE functions against fixed inputs and asserts on the output.
No LLM calls, no DB, no network, no message charges — runs in <1s and is safe
to wire into CI (exits non-zero on any failure).

What it guards:
  1. Retrieval    — the RAG core: real queries surface the right knowledge,
                    including the paraphrases that used to return nothing.
  2. Safety       — crisis / medical / out-of-scope regex must never miss, and
                    must not over-trigger on normal fitness talk.
  3. Prompt build — build_system_prompt injects the right blocks per intent
                    (profile stats, contradiction rule, format tails, guards).
  4. Parsing      — _parse_structured_response splits reply / actions / metadata.
  5. Extraction   — stated-macro and exercise-name helpers.

The LLM-graded quality checks ("did it teach the why / defuse the emotion")
are intentionally NOT here — they cost tokens and are flaky. Add later behind a
flag if wanted. This file is the fast, free regression backbone.

Run from ai/:  python fixed_evals.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

# Dummy env lets rag_pipeline import (Supabase/Groq clients construct lazily and
# are never called by the pure functions under test).
for _k, _v in {
    "SUPABASE_URL": "https://dummy.supabase.co", "SUPABASE_KEY": "dummy",
    "GROQ_API_KEY": "dummy", "DEEPSEEK_API_KEY": "dummy", "OPENAI_API_KEY": "dummy",
}.items():
    os.environ.setdefault(_k, _v)

from rag_pipeline import (  # noqa: E402
    search_gym_knowledge,
    _classify_intent_regex,
    _mentions_pain,
    build_system_prompt,
    _parse_structured_response,
    _extract_stated_macros,
    _normalise_exercise,
)

# ── tiny runner ───────────────────────────────────────────────────────────────
_RESULTS: list[tuple[str, str, bool, str]] = []
_section = "?"


def section(name: str) -> None:
    global _section
    _section = name


def check(label: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((_section, label, bool(cond), detail))


def _any_contains(results: list[str], needle: str) -> bool:
    return any(needle.lower() in r.lower() for r in results)


# ── helpers for prompt-build tests ────────────────────────────────────────────
def ctx(macros: bool = True, prefs: bool = True):
    return {
        "user": {"full_name": "Ashish Kumar"},
        "macros": {"calories": 2400, "protein_g": 180, "carbs_g": 240,
                   "fat_g": 70, "water_liters": 3.5, "fiber_g": 30} if macros else None,
        "program": {"programs": {"title": "Upper/Lower Split"},
                    "current_week": 3, "current_day": 2},
        "preferences": {"diet_preference": "veg", "diet_goal": "gain muscle",
                        "equipment": "full gym", "fitness_level": "intermediate",
                        "height": 183, "weight": 72, "age": 24, "gender": "male"} if prefs else None,
        "weight_logs": [{"body_weight": 72.0}, {"body_weight": 73.5}],
        "personal_bests": [{"exercise_name": "Bench Press", "weight_kg": 85, "reps": 5}],
        "streak": 12,
        "todays_workout": None,
    }


CREATOR = {"name": "Coach Rip",
           "coaching_philosophy": "Progressive overload, no fluff.",
           "diet_philosophy": "High protein, flexible.",
           "tone": "direct, motivating"}


# ══ 1. RETRIEVAL ══════════════════════════════════════════════════════════════
section("retrieval")

# The exact prompts that returned ZERO chunks before this session's fix.
r = search_gym_knowledge("my height is 5'4 and i look skinny and tiny, how do i look bigger")
check("skinny/look-bigger query is no longer blank", len(r) > 0)
check("skinny/look-bigger query surfaces the V-taper chunk", _any_contains(r, "taper"))

r = search_gym_knowledge("i'm a complete beginner, where do i even start")
check("beginner query is no longer blank", len(r) > 0)
check("beginner query surfaces the beginner chunk", _any_contains(r, "beginner"))

# Anatomy paraphrases resolve to the right muscle science.
r = search_gym_knowledge("how do i get wider shoulders")
check("'wider shoulders' → deltoid science", _any_contains(r, "deltoid"))

r = search_gym_knowledge("i want a broader back and more width")
check("'broader back' → lat science", _any_contains(r, "latissimus"))

r = search_gym_knowledge("best incline movement for upper chest")
check("'upper chest' → pec science", _any_contains(r, "pectoral") or _any_contains(r, "clavicular"))

r = search_gym_knowledge("does protein timing after my workout matter")
check("'protein timing' → nutrition science", _any_contains(r, "protein") or _any_contains(r, "anabolic"))

# Multi-chunk: a width question spanning two muscles returns BOTH (guards F1 —
# the join that replaced chunks[0]).
r = search_gym_knowledge("how do i get wider shoulders and a broader back")
check("width query retrieves >=2 chunks", len(r) >= 2, f"got {len(r)}")
check("width query has both delt AND lat", _any_contains(r, "deltoid") and _any_contains(r, "latissimus"))

# Off-topic query should retrieve nothing (no false grounding).
r = search_gym_knowledge("what's the weather like today")
check("off-topic query retrieves nothing", len(r) == 0, f"got {len(r)}")


# ══ 2. SAFETY REGEX ═══════════════════════════════════════════════════════════
section("safety")

for msg in ["i want to kill myself", "i feel like ending it all",
            "there's no reason to live", "i'd be better off dead"]:
    check(f"crisis: {msg!r}", _classify_intent_regex(msg) == "crisis")

for msg in ["my shoulder really hurts", "i think i tore my hamstring",
            "sharp pain in my knee when i squat", "i pulled my lower back",
            "my knee is swollen"]:
    check(f"medical: {msg!r}", _classify_intent_regex(msg) == "medical")

for msg in ["who won the cricket match", "should i invest in bitcoin",
            "what movie should i watch tonight"]:
    check(f"out_of_scope: {msg!r}", _classify_intent_regex(msg) == "out_of_scope")

# Must NOT over-trigger on normal fitness talk (false positive → safety handling).
for msg in ["how much protein should i eat", "build me a push pull legs program",
            "i just did 3x8 on squats", "what should i eat for dinner",
            "am i on track with my goals"]:
    check(f"normal→None: {msg!r}", _classify_intent_regex(msg) is None)


# ══ 3. PROMPT BUILD ═══════════════════════════════════════════════════════════
section("prompt_build")

p = build_system_prompt(ctx(), CREATOR, intent="question")
check("question: has ACTIONABLE PLAN FORMAT", "ACTIONABLE PLAN FORMAT" in p)
check("question: injects saved height (183)", "183" in p and "SAVED PROFILE STATS" in p)
check("question: has PROFILE CONTRADICTION rule", "PROFILE CONTRADICTION" in p)
check("question: cites macro target", "2400" in p)
check("question: core word-cap rule present", "80 words" in p)
check("question: never-reveal-model rule present", "never reveal" in p.lower())

p = build_system_prompt(ctx(), CREATOR, intent="greeting")
check("greeting: has greeting block", "GREETING" in p)
check("greeting: uses first name", "Ashish" in p)

p = build_system_prompt(ctx(), CREATOR, intent="workout_log")
check("workout_log: mandates JSON format", "MANDATORY RESPONSE FORMAT" in p and "log_workout" in p)

p = build_system_prompt(ctx(), CREATOR, intent="weight_log")
check("weight_log: mandates JSON + log_weight", "log_weight" in p)

p = build_system_prompt(ctx(), CREATOR, intent="medical")
# NOTE: these two asserted wording the medical prompt was rewritten away from
# (a "DO NOT give a passive disclaimer" line that no longer exists) and had been
# failing silently on main. Re-pointed at the rules the prompt actually carries.
check("medical: forbids padding with disclaimers", "Do not pad with disclaimers" in p)
check("medical: instructs to acknowledge it first", "Acknowledge it in one line" in p)
check("medical: separates DOMS from a joint tweak",
      "ORDINARY MUSCLE SORENESS" in p and "JOINT/TISSUE TWEAK" in p)
check("medical: mandates the stop rule", "STOP RULE" in p and "numbness or tingling" in p)
check("medical: mandates dropping the load", "drop the weight" in p.lower())
check("medical: mandates explaining the mechanism", "WHY it does" in p)

# ── Pain-driven exercise swap keeps its injury guardrails ─────────────────────
# "my shoulder hurts" → "can we swap the bench?" is one injury conversation, but
# only the first message says "pain". The second used to classify as a plain
# exercise_swap and drop every rule below.
SWAP = {"original_exercise": "Barbell Bench Press", "original_muscles": "chest",
        "replacement_exercise": "Dumbbell Floor Press", "replacement_muscles": "chest"}

p = build_system_prompt(ctx(), CREATOR, intent="exercise_swap", swap_data=SWAP)
check("swap (no pain): stays terse", "under 30 words" in p)
check("swap (no pain): no injury block", "THIS SWAP IS BECAUSE SOMETHING HURTS" not in p)

p = build_system_prompt(ctx(), CREATOR, intent="exercise_swap", swap_data=SWAP, pain_context=True)
check("pain swap: injury block injected", "THIS SWAP IS BECAUSE SOMETHING HURTS" in p)
check("pain swap: demands the mechanism", "NAME THE MECHANISM" in p)
check("pain swap: prefers less range of motion", "LESS RANGE OF MOTION" in p)
check("pain swap: warns dumbbells add bottom-range", "stop at chest depth" in p)
check("pain swap: mandates dropping load", "DROP THE LOAD" in p)
check("pain swap: caps volume", "AT OR BELOW" in p)
check("pain swap: mandates stop rule", "STOP RULE" in p and "numbness or tingling" in p)
check("pain swap: lifts the word cap so all five fit", "under 110 words" in p)
check("pain swap: still mandates swap JSON", "exercise_swap" in p and "replacement_exercise" in p)

p = build_system_prompt(ctx(), CREATOR, intent="exercise_swap", swap_data=None, pain_context=True)
check("pain swap (no library match): still gets injury block",
      "THIS SWAP IS BECAUSE SOMETHING HURTS" in p and "under 110 words" in p)

# Sticky-pain detection itself
for msg in ["my shoulder is hurting while doing bench press", "knee pain on squats",
            "my elbow has been bothering me", "shoulder niggle", "rotator cuff feels off"]:
    check(f"pain detected: {msg!r}", _mentions_pain(msg))
for msg in ["can we swap an exercise for this", "build me a push pull legs program",
            "how much protein should i eat", "i just did 3x8 on squats"]:
    check(f"no false pain: {msg!r}", not _mentions_pain(msg))

p = build_system_prompt(ctx(macros=False), CREATOR, intent="question")
check("no macros: says targets not calculated", "not calculated yet" in p)

p = build_system_prompt(ctx(prefs=False), CREATOR, intent="question")
check("no prefs: profile stats degrade gracefully", "not set" in p.lower())


# ══ 4. RESPONSE PARSING ═══════════════════════════════════════════════════════
section("parsing")

txt, act, meta = _parse_structured_response("Do 3x8 back squats today. Keep depth honest.")
check("plain reply: no action, no metadata", act is None and meta is None)
check("plain reply: text preserved", "back squats" in txt)

raw = 'Log confirmed.<METADATA_START>{"recommended_exercises":[{"exercise_id":"7"}]}<METADATA_END>'
txt, act, meta = _parse_structured_response(raw)
check("metadata: parsed out", meta is not None and "recommended_exercises" in meta)
check("metadata: stripped from clean text", "<METADATA_START>" not in txt and txt == "Log confirmed.")

raw = '{"reply":"Logged 72kg.","actions":[{"type":"log_weight","value":72}]}'
txt, act, meta = _parse_structured_response(raw)
check("action JSON: parsed", act is not None and act.get("actions"))
check("action JSON: value correct", act["actions"][0]["value"] == 72)

raw = 'Here you go: ```json\n{"reply":"ok","actions":[]}\n```'
_, act, _ = _parse_structured_response(raw)
check("fenced JSON: unwrapped and parsed", act is not None and "reply" in act)


# ══ 5. EXTRACTION HELPERS ═════════════════════════════════════════════════════
section("extraction")

m = _extract_stated_macros("it has 50g protein")
check("stated macro: protein extracted", m is not None and m.get("protein_g") == 50.0)

check("stated macro: none when absent", _extract_stated_macros("just chatting about my day") is None)

check("normalise: underscores→spaces, title case",
      _normalise_exercise("zzz_made_up_move") == "Zzz Made Up Move")
check("normalise: trims + lowercases input first",
      _normalise_exercise("  ANOTHER_FAKE_LIFT  ") == "Another Fake Lift")


# ══ 5b. PAIN-AWARE SWAP RANKING ═══════════════════════════════════════════════
# The candidate is picked BEFORE the model sees the turn, so if ranking ignores
# pain the prompt ends up explaining why a range-limited swap is safer while
# being pinned to a full-range one.
section("pain_ranking")

from rag_pipeline import _pain_rank  # noqa: E402


def _best(names, pain):
    """Lowest _pain_rank sorts first; ties keep input order (stable sort)."""
    return sorted(names, key=lambda n: _pain_rank(n) if pain else ())[0]

# The exact case from the transcript: shoulder pain on barbell bench.
BENCH_ALTS = ["Dumbbell Bench Press", "Wide-Grip Barbell Bench Press",
              "Dumbbell Floor Press", "Machine Chest Press"]
check("pain swap picks the range-capped variant, not the dumbbell press",
      _best(BENCH_ALTS, pain=True) == "Dumbbell Floor Press")
check("no-pain swap ranking is left completely alone",
      _best(BENCH_ALTS, pain=False) == "Dumbbell Bench Press")

check("floor press reads as joint-friendly", _pain_rank("Dumbbell Floor Press") == (False, False))
check("machine reads as joint-friendly", _pain_rank("Machine Chest Press") == (False, False))
check("smith reads as joint-friendly", _pain_rank("Smith Machine Incline Press") == (False, False))
check("plain dumbbell press is neither friendly nor aggravating",
      _pain_rank("Dumbbell Bench Press") == (False, True))
check("wide grip sinks as aggravating", _pain_rank("Wide-Grip Barbell Bench Press")[0] is True)
check("behind-the-neck sinks as aggravating", _pain_rank("Behind the Neck Press")[0] is True)
check("deficit sinks as aggravating (adds range)", _pain_rank("Deficit Deadlift")[0] is True)
check("power/ballistic sinks as aggravating", _pain_rank("Power Clean")[0] is True)
check("aggravating outranks friendly when a name is both",
      _pain_rank("Seated Behind the Neck Press") == (True, False))
check("missing name never explodes", _pain_rank(None) == (False, True))

# find_exercise_alternative must accept the flag without disturbing its old
# two-arg contract (other callers still pass two).
import inspect  # noqa: E402
_sig = inspect.signature(_rp_fea := __import__("rag_pipeline").find_exercise_alternative)
check("find_exercise_alternative takes pain_context", "pain_context" in _sig.parameters)
check("pain_context defaults off for existing callers",
      _sig.parameters["pain_context"].default is False)


# ══ 5c. SWAP REFERENT RESOLUTION ══════════════════════════════════════════════
# The real transcript: "my shoulder is hurting while doing bench press" then
# "can we swap and exercise for thsi". Extraction returned the word "And", missed
# the library, and the turn fell into the free-form branch where the coach
# invents a swap instead of using the catalogue.
section("swap_referent")

import rag_pipeline as _rp2  # noqa: E402
from rag_pipeline import _find_exercise_in_text, _resolve_swap_target  # noqa: E402

# Real names from exercises_library (is_searchable = true).
_LIB = ["Barbell Bench Press", "Dumbbell Bench Press", "Machine Bench Press",
        "Smith Machine Bench Press", "Wide-Grip Barbell Bench Press",
        "Dumbbell Floor Press", "Barbell Squat", "Seated Dumbbell Shoulder Press",
        "Neck Press", "Standing Cable Chest Press"]
_rp2._exercise_name_index.cache_clear()
_rp2._exercise_name_index = lambda: tuple(
    (n, " ".join(w for w in __import__("re").findall(r"[a-z]+", n.lower())
                 if w not in _rp2._GENERIC_EXERCISE_WORDS))
    for n in _LIB
)

check("resolves a bare mention to the canonical lift",
      _find_exercise_in_text("my shoulder is hurting while doing bench press")
      == "Barbell Bench Press")
check("adjacency beats bag-of-words: not Shoulder Press despite both words present",
      _find_exercise_in_text("my shoulder is hurting while doing bench press")
      != "Seated Dumbbell Shoulder Press")
check("more specific phrase wins",
      _find_exercise_in_text("floor press felt better") == "Dumbbell Floor Press")
check("no exercise mentioned -> None", _find_exercise_in_text("i feel tired today") is None)
check("empty text never explodes", _find_exercise_in_text("") is None)

# _resolve_swap_target needs get_exercise_info; stub it to the fake library.
_real_gei = _rp2.get_exercise_info
_rp2.get_exercise_info = lambda n: ({"name": n} if n and n.lower() in
                                    {x.lower() for x in _LIB} else None)
try:
    _hist = [{"role": "user", "content": "my shoulder is hurting while doing bench press"},
             {"role": "assistant", "content": "Check your form and grip width."}]
    check("THE TRANSCRIPT CASE: 'swap and exercise for thsi' now resolves",
          _resolve_swap_target("can we swap and exercise for thsi", _hist)
          == "Barbell Bench Press")
    check("junk extraction ('And') is rejected, not passed through",
          _resolve_swap_target("can we swap and exercise for thsi", _hist) != "And")
    check("pronoun-only swap resolves from history",
          _resolve_swap_target("can we swap this", _hist) == "Barbell Bench Press")
    check("an explicitly named exercise still wins over history",
          _resolve_swap_target("swap my barbell squat", _hist) == "Barbell Squat")
    check("nothing named anywhere -> None",
          _resolve_swap_target("can we swap this", []) is None)
    check("no history arg never explodes", _resolve_swap_target("swap this", None) is None)
finally:
    _rp2.get_exercise_info = _real_gei


# Never ask the model for an identifier it was never shown. The metadata block
# demands a real exercise_id; the catalog that supplies them is injected for
# "question" only. Asking on a swap turn produced a fabricated id
# ("SMITH_MACHINE_BENCH_PRESS") in place of a UUID, on every run.
_sw = build_system_prompt(ctx(), CREATOR, intent="exercise_swap", swap_data=SWAP, pain_context=True)
_q = build_system_prompt(ctx(), CREATOR, intent="question")
check("swap turn does NOT demand metadata (no catalog to source ids from)",
      "METADATA_START" not in _sw)
check("question turn still demands metadata", "METADATA_START" in _q)
check("swap turn keeps the biomechanical constraints",
      "BIOMECHANICAL" in _sw)
# The "implements work for OPPOSITE reasons" warning is gone on purpose: the
# mechanism is no longer reasoned out by the model, it's looked up from
# exercises_library and handed over as fact. What must hold now is that the
# prompt carries one of the two mechanism blocks, and never asks for a guess.
check("swap prompt supplies a mechanism verdict either way",
      "VERIFIED MECHANISM" in _sw or "NO VERIFIED MECHANISM" in _sw)
check("mechanism rule pins the model to the supplied fact",
      "ONLY mechanism you may state" in _sw)


# ══ 5d. WEIGHT TREND DIRECTION ════════════════════════════════════════════════
# Two implementations disagreed by construction: rag_pipeline ordered by
# created_at DESC and read logs[0]-logs[-1]; context_builder ordered by
# logged_at ASC and read logs[-1]-logs[0]. Both feed the same prompt, and 30% of
# progress_logs rows are backfilled, so they could report opposite directions.
section("weight_trend")

from context_builder import weight_delta  # noqa: E402

_ASC = [{"body_weight": 73.5, "logged_at": "2026-07-01", "created_at": "2026-07-01"},
        {"body_weight": 72.0, "logged_at": "2026-07-15", "created_at": "2026-07-15"}]
check("oldest-first input reads as a loss", weight_delta(_ASC)["direction"] == "down")
check("newest-first input reads THE SAME", weight_delta(list(reversed(_ASC)))["direction"] == "down")
check("delta magnitude is right", weight_delta(_ASC)["diff"] == -1.5)

# The live case: 6 of 20 rows have logged_at <> created_at::date. Ordering by
# created_at makes the backfilled row look newest and inverts the trend.
_BACKFILL = [{"body_weight": 72.0, "logged_at": "2026-07-15", "created_at": "2026-07-16"},
             {"body_weight": 70.0, "logged_at": "2026-07-02", "created_at": "2026-07-18"}]
check("backfilled log does not invert the trend (user GAINED)",
      weight_delta(_BACKFILL)["direction"] == "up")
check("backfilled delta is +2.0", weight_delta(_BACKFILL)["diff"] == 2.0)

# logged_at is a DATE, so two logs in one day tie — created_at must break it
# deterministically rather than leaving it to row order.
_SAMEDAY = [{"body_weight": 71.0, "logged_at": "2026-07-10", "created_at": "2026-07-10T18:00:00Z"},
            {"body_weight": 74.0, "logged_at": "2026-07-01", "created_at": "2026-07-01T08:00:00Z"},
            {"body_weight": 70.0, "logged_at": "2026-07-10", "created_at": "2026-07-10T07:00:00Z"}]
check("same-day logs break ties on created_at (ends at the 18:00 reading)",
      weight_delta(_SAMEDAY)["to"] == 71.0)

check("one log is not a trend", weight_delta([{"body_weight": 72.0}]) is None)
check("no logs is not a trend", weight_delta([]) is None)
check("null body_weight rows are ignored",
      weight_delta([{"body_weight": None, "logged_at": "2026-07-01"},
                    {"body_weight": 72.0, "logged_at": "2026-07-02"}]) is None)
check("flat is flat, not a fake direction",
      weight_delta([{"body_weight": 72.0, "logged_at": "2026-07-01"},
                    {"body_weight": 72.0, "logged_at": "2026-07-09"}])["direction"] == "flat")


# ══ 6. LATENCY SHAPE ══════════════════════════════════════════════════════════
# These guard the fixes, not the speed: the win only exists while the turn keeps
# making ONE insert instead of two and reusing ONE http client instead of four.
section("latency")

import rag_pipeline as _rp  # noqa: E402


class _FakeTable:
    def __init__(self, log): self.log = log
    def insert(self, rows):
        self.log.append(rows)
        return self
    def delete(self): return self
    def eq(self, *a, **k): return self
    def in_(self, *a, **k): return self
    def select(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def execute(self): return type("R", (), {"data": []})()


class _FakeSupabase:
    def __init__(self): self.inserts = []
    def table(self, _name): return _FakeTable(self.inserts)


_fake = _FakeSupabase()
_real_supabase = _rp.supabase
_rp.supabase = _fake
try:
    _rp.save_exchange("u1", "c1", "my shoulder hurts", "Here's the swap.")
finally:
    _rp.supabase = _real_supabase

check("save_exchange: one round trip, not two", len(_fake.inserts) == 1)
check("save_exchange: writes both rows", isinstance(_fake.inserts[0], list) and len(_fake.inserts[0]) == 2)
check("save_exchange: user row first, assistant second",
      [r["role"] for r in _fake.inserts[0]] == ["user", "assistant"])
check("save_exchange: content lands on the right rows",
      _fake.inserts[0][0]["content"] == "my shoulder hurts"
      and _fake.inserts[0][1]["content"] == "Here's the swap.")

check("groq client is a reused singleton", _rp._groq() is _rp._groq())
check("deepseek client is a reused singleton", _rp._deepseek() is _rp._deepseek())

# No caller may reintroduce the two-insert pattern.
_src = open("rag_pipeline.py", encoding="utf-8").read()
check("no save_message user/assistant pairs left",
      'save_message(user_id, creator_id, "user"' not in _src)
check("extractors fan out instead of running in series", "ThreadPoolExecutor" in _src)


# ── failures surface ─────────────────────────────────────────────────────────
# A failure inside the pipeline must raise (/chat turns it into a 503), never
# come back as a normal reply. That 200 hid a total program-build outage.
section("failures surface")


def _boom(*a, **k):
    raise RuntimeError("boom")


_real_gen = _rp.generate_diet_plan
_rp.generate_diet_plan = _boom
try:
    _rp._handle_diet_generation("u1", "c1", "make me a diet plan", 5)
    _raised = False
except RuntimeError:
    _raised = True
finally:
    _rp.generate_diet_plan = _real_gen

check("diet generation failure raises instead of replying", _raised)
check("no apology-as-reply catch-alls left",
      "Something went wrong, please try again." not in _src
      and "Something went wrong generating your plan. Please try again." not in _src)


# ── report ────────────────────────────────────────────────────────────────────
def main() -> int:
    passed = sum(1 for *_, ok, _ in ((s, l, ok, d) for s, l, ok, d in _RESULTS) if ok)
    total = len(_RESULTS)
    cur = None
    for s, label, ok, detail in _RESULTS:
        if s != cur:
            print(f"\n── {s} ──")
            cur = s
        mark = "PASS" if ok else "FAIL"
        extra = f"   [{detail}]" if (detail and not ok) else ""
        print(f"  {mark}  {label}{extra}")
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
