"""
context_builder.py — Builds and caches per-user AI coach context.

Queries all user data from Supabase, formats 4 context slices + coaching
directives, and upserts into context_cache so /chat can do 1 read instead
of 6+ queries per message.

Usage:
    from context_builder import get_or_build_context, detect_topics, build_prompt_context

    slices = get_or_build_context(user_id)              # cached, or built on miss
    text   = build_prompt_context(slices, detect_topics(message))
"""

import asyncio
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date as _date, datetime
from typing import Literal, Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from supabase import create_client
import os

from tz import is_before_local_today, local_date_str, today_local, utc_now_iso

logger = logging.getLogger(__name__)

_supabase_url = os.environ.get("SUPABASE_URL", "")
_supabase_key = os.environ.get("SUPABASE_KEY", "")
supabase = create_client(_supabase_url, _supabase_key) if _supabase_url and _supabase_key else None

Topic = Literal["nutrition", "training", "progress", "general"]

# ─── Topic detection ──────────────────────────────────────────────────────────

_NUTRITION_KW = {"food", "eat", "diet", "calories", "protein", "meal", "hungry",
                 "recipe", "macro", "carb", "fat", "nutrition", "drink", "calorie",
                 "snack", "cook", "cheat", "sugar", "fibre", "fiber"}

# The list above is entirely ABSTRACT nutrition vocabulary, and that was a hole
# big enough to drive the core loop through: a user naming what they ate scored
# ZERO. "i had 2 rotis and dal" -> ["general"] -> build_prompt_context appends
# no nutrition slice -> the coach is handed a prompt with no "TODAY so far"
# line at all, and cheerfully plans dinner as if breakfast never happened. The
# staleness users report is usually this: not an out-of-date number, an ABSENT
# one. So also score the way people actually talk about food.
_FOOD_LOG_KW = {"i had", " had ", "i ate", "ate ", "eaten", "having",
                "log ", "logged", "logging", "drank", "consumed", "finished"}

# Deliberately the STAPLES, not a catalogue — foods has 11k rows and chasing
# them here would be a second, worse food database. These plus _PORTION_RE
# cover the overwhelming majority of real messages.
_COMMON_FOOD_KW = {"roti", "chapati", "rice", "dal", "daal", "paneer", "curd",
                   "dahi", "chicken", "egg", "milk", "banana", "oats", "poha",
                   "idli", "dosa", "sabzi", "rajma", "chana", "chole", "rajmah",
                   "biryani", "khichdi", "upma", "paratha", "bread", "apple",
                   "whey", "shake", "coffee", "tea", "juice", "salad", "curry",
                   "fish", "mutton", "soya", "tofu", "almond", "peanut", "ghee",
                   "butter", "samosa", "pizza", "maggi", "poori", "puri", "lassi"}

# A number attached to a food unit IS a portion, and a portion means the user is
# talking about eating — whatever the food happens to be called. This is what
# makes the staple list above good enough: "250g bhindi masala" has no keyword
# hit at all, but it is unmistakably a food message.
_PORTION_RE = re.compile(
    r"\b\d+\s*(g|gm|gms|gram|grams|kg|ml|l|litre|liter|"
    r"katori|katoris|bowl|bowls|plate|plates|piece|pieces|slice|slices|"
    r"cup|cups|glass|glasses|spoon|spoons|tbsp|tsp|scoop|scoops|"
    r"roti|rotis|chapati|chapatis|idli|idlis|dosa|dosas|egg|eggs)\b",
    re.IGNORECASE,
)


def _nutrition_score(lower: str) -> int:
    """Nutrition signal for topic routing. Shared so detect_topic and
    detect_topics can never drift apart on what counts as a food message."""
    score = sum(1 for k in _NUTRITION_KW if k in lower)
    score += sum(1 for k in _FOOD_LOG_KW if k in lower)
    score += sum(1 for k in _COMMON_FOOD_KW if k in lower)
    if _PORTION_RE.search(lower):
        score += 1
    return score
_TRAINING_KW  = {"workout", "exercise", "lift", "gym", "sets", "reps", "push",
                 "pull", "training", "bench", "squat", "deadlift", "run", "cardio",
                 "muscle", "strength", "session", "form", "technique",
                 # program-structure asks — these must route to the training
                 # slice, which now carries the full current-week program
                 "program", "my plan", "split", "routine"}
_PROGRESS_KW  = {"weight", "streak", " pr ", "personal best", "progress", "results",
                 "gain", "lose", "body", "measurement", "bmi", "fat loss", "bulk"}

_COMPOUND_RE = re.compile(
    r"bench|squat|deadlift|overhead.press|barbell.row|ohp|pull.up|dip|press",
    re.IGNORECASE,
)


# ─── Progressive Overload / Stall Detection ───────────────────────────────────

def detect_stalled_exercises(user_id: str) -> list[dict]:
    """
    Returns exercises where max weight hasn't increased across 14 days with >= 3 sessions.
    Only counts sets where weight_kg > 0 (skips bodyweight exercises).
    Used for proactive progressive overload coaching and daily nudges.
    """
    if not supabase:
        return []

    cutoff = _local_date(14)
    try:
        rows = (
            supabase.table("workout_set_logs")
            .select("exercise_name, weight_kg, logged_at")
            .eq("user_id", user_id)
            .gte("logged_at", cutoff)
            .execute()
            .data or []
        )
    except Exception:
        return []

    # Group by exercise → by session day → max weight per session
    by_exercise: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        ex = (row.get("exercise_name") or "").strip()
        day = str(row.get("logged_at", ""))[:10]
        w = row.get("weight_kg")
        if ex and w is not None and float(w) > 0:
            by_exercise[ex][day].append(float(w))

    stalled = []
    for exercise_name, days_data in by_exercise.items():
        if len(days_data) < 3:
            continue
        session_maxes = [max(weights) for weights in days_data.values()]
        peak = max(session_maxes)
        trough = min(session_maxes)
        if peak - trough <= 0.5:  # no meaningful weight increase
            increment = 5.0 if _COMPOUND_RE.search(exercise_name) else 2.5
            stalled.append({
                "exercise_name": exercise_name,
                "current_weight_kg": peak,
                "suggested_weight_kg": peak + increment,
                "sessions_count": len(days_data),
                "increment": increment,
            })

    return stalled


def detect_progression_signals(user_id: str) -> list[dict]:
    """
    Rep-performance-based progression, complementing detect_stalled_exercises
    (which only sees flat weight): compares the last 2 logged sessions per
    exercise against the rep prescription in the user's active program.
      - every set at/above the top of the prescribed reps, 2 sessions running → +2.5%
      - below the bottom of the prescribed reps, 2 sessions running → -10% deload
      - otherwise → hold (not returned; only actionable signals are)
    Prescriptions come from the slash-encoded exercises rows ("10/8/6"), so
    bottom/top = min/max across the prescribed sets.
    """
    if not supabase:
        return []

    # 1. Rep/weight prescriptions from the active program
    try:
        up = _maybe(
            supabase.table("user_programs")
            .select("program_id")
            .eq("user_id", user_id)
            .eq("status", "active")
            .limit(1)
            .execute()
        )
        if not up or not up.get("program_id"):
            return []
        w_rows = (
            supabase.table("workouts")
            .select("id")
            .eq("program_id", up["program_id"])
            .execute()
            .data or []
        )
        workout_ids = [w["id"] for w in w_rows if w.get("id")]
        if not workout_ids:
            return []
        ex_rows = (
            supabase.table("exercises")
            .select("name, reps")
            .in_("workout_id", workout_ids)
            .execute()
            .data or []
        )
    except Exception:
        return []

    # exercise name → (bottom, top, prescribed set count). The same exercise
    # can appear on multiple days/weeks — keep the widest range seen so a
    # single lighter day never triggers a premature increase.
    prescriptions: dict[str, tuple[int, int, int]] = {}
    for ex in ex_rows:
        name = (ex.get("name") or "").strip()
        rep_parts = [r.strip() for r in str(ex.get("reps") or "").split("/") if r.strip().isdigit()]
        if not name or not rep_parts:
            continue
        reps = [int(r) for r in rep_parts]
        bottom, top, n_sets = min(reps), max(reps), len(reps)
        if name.lower() in prescriptions:
            b0, t0, s0 = prescriptions[name.lower()][1:]
            prescriptions[name.lower()] = (name, min(b0, bottom), max(t0, top), max(s0, n_sets))
        else:
            prescriptions[name.lower()] = (name, bottom, top, n_sets)
    if not prescriptions:
        return []

    # 2. Logged sets, last 30 days
    try:
        rows = (
            supabase.table("workout_set_logs")
            .select("exercise_name, reps, weight_kg, logged_at")
            .eq("user_id", user_id)
            .gte("logged_at", _local_date(30))
            .execute()
            .data or []
        )
    except Exception:
        return []

    by_exercise: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        name = (row.get("exercise_name") or "").strip().lower()
        day = str(row.get("logged_at", ""))[:10]
        try:
            reps_n = int(row.get("reps") or 0)
            weight = float(row.get("weight_kg") or 0)
        except (TypeError, ValueError):
            continue
        if name and reps_n > 0:
            by_exercise[name][day].append((reps_n, weight))

    signals = []
    for log_name, days_data in by_exercise.items():
        rx = prescriptions.get(log_name)
        if not rx or len(days_data) < 2:
            continue
        display_name, bottom, top, n_sets = rx
        last_two = [days_data[d] for d in sorted(days_data.keys())[-2:]]
        # % progression needs a working weight — skip bodyweight/unweighted work
        current = max((w for _, w in last_two[-1] if w > 0), default=0.0)
        if current <= 0:
            continue

        hit_top = all(all(r >= top for r, _ in session) for session in last_two)
        missed_bottom = all(any(r < bottom for r, _ in session) for session in last_two)

        if hit_top:
            new_weight = round(current * 1.025, 1)
            action = "increase"
            reason = f"hit the top of the {bottom}-{top} rep prescription in both of the last 2 sessions"
        elif missed_bottom:
            new_weight = round(current * 0.90, 1)
            action = "deload"
            reason = f"missed the bottom of the {bottom}-{top} rep prescription in both of the last 2 sessions"
        else:
            continue  # hold — no note by design

        signals.append({
            "exercise_name": display_name,
            "action": action,
            "current_weight_kg": current,
            "new_weight_kg": new_weight,
            # ready to paste into exercises.target_weight ("62.5/62.5/62.5")
            "target_weight_str": "/".join([str(new_weight)] * n_sets),
            "reason": reason,
        })

    return signals


def _program_structure_block(program_id: str, current_week: int) -> str:
    """
    Compact one-line-per-day rendering of the user's current program week —
    the coach can't say "follow this exactly" about a plan it can't see.
    Falls back to week 1 if the program defines fewer weeks than the user's
    current_week (older programs sometimes only seed week 1).
    """
    if not supabase or not program_id:
        return ""
    try:
        week = int(current_week or 1)
        week_workouts = (
            supabase.table("workouts")
            .select("id, title, day_number, is_rest_day")
            .eq("program_id", program_id)
            .eq("week_number", week)
            .order("day_number")
            .execute()
            .data or []
        )
        if not week_workouts and week != 1:
            week = 1
            week_workouts = (
                supabase.table("workouts")
                .select("id, title, day_number, is_rest_day")
                .eq("program_id", program_id)
                .eq("week_number", 1)
                .order("day_number")
                .execute()
                .data or []
            )
        if not week_workouts:
            return ""

        workout_ids = [w["id"] for w in week_workouts if w.get("id") and not w.get("is_rest_day")]
        ex_by_workout: dict[str, list[dict]] = defaultdict(list)
        if workout_ids:
            ex_rows = (
                supabase.table("exercises")
                .select("workout_id, name, sets, reps, target_weight, order_index")
                .in_("workout_id", workout_ids)
                .order("order_index")
                .execute()
                .data or []
            )
            for ex in ex_rows:
                ex_by_workout[ex.get("workout_id")].append(ex)

        day_lines = []
        for w in week_workouts:
            day_no = w.get("day_number", "?")
            if w.get("is_rest_day"):
                day_lines.append(f"  Day {day_no} — {w.get('title') or 'Rest'}: REST")
                continue
            ex_strs = []
            for ex in ex_by_workout.get(w.get("id"), []):
                s = f"{ex.get('name', '?')} {ex.get('sets', '?')}x{ex.get('reps', '?')}"
                tw = str(ex.get("target_weight") or "").strip()
                # only show weight when at least one set has a real target
                if tw and any(part not in ("", "0", "0.0") for part in tw.split("/")):
                    s += f" @ {tw}kg"
                ex_strs.append(s)
            day_lines.append(
                f"  Day {day_no} — {w.get('title') or 'Workout'}: " + ("; ".join(ex_strs) or "no exercises")
            )
        return (
            f"\nCURRENT WEEK'S PROGRAM (week {week} — cite these exact exercises/sets/reps "
            f"when discussing their plan; never invent a different one):\n" + "\n".join(day_lines)
        )
    except Exception:
        return ""


def _adaptive_macro_suggestion(
    prefs: dict,
    macros: dict | None,
    prog_rows: list[dict],
    avg_cal: float,
) -> str:
    """
    Deterministic plateau adjustment, suggest-only: when logged weight has been
    flat while the goal says it should be moving, compute the new calorie
    target in code and hand the finished number to the coach to present.
    Never writes user_macros — the user applies it via Profile → Edit Profile.
    Mirrors the fat-loss -7% / muscle-gain +7% convention, with clinical
    calorie floors and an adherence-first guard.
    """
    goal = (prefs.get("primary_goal") or "").lower()
    target_cal = float((macros or {}).get("calories") or 0)
    if not target_cal:
        return ""
    is_loss = any(k in goal for k in ("fat", "lose", "cut", "lean", "tone"))
    is_gain = any(k in goal for k in ("muscle", "gain", "bulk"))
    if not (is_loss or is_gain):
        return ""  # recomp/maintenance/athletic: flat weight is expected, not a plateau

    window_days, min_span = (14, 10) if is_loss else (21, 15)
    cutoff = local_date_str(window_days)
    recent = [
        r for r in prog_rows
        if str(r.get("logged_at", ""))[:10] >= cutoff and r.get("body_weight") is not None
    ]
    if len(recent) < 3:
        return ""
    weights = [float(r["body_weight"]) for r in recent]
    if max(weights) - min(weights) > 0.6:
        return ""  # weight is moving — no plateau
    try:
        first = datetime.fromisoformat(str(recent[0]["logged_at"])[:10])
        last = datetime.fromisoformat(str(recent[-1]["logged_at"])[:10])
    except ValueError:
        return ""
    span_days = (last - first).days
    if span_days < min_span:
        return ""

    if is_loss:
        # Adherence first: if intake is above target, the target isn't the problem.
        if avg_cal > target_cal * 1.05:
            return (
                f"\nPLATEAU DETECTED (deterministic, from logged data): weight flat for {span_days} days on a "
                f"fat-loss goal, but 7-day intake (~{int(avg_cal)} kcal) is ABOVE the {int(target_cal)} kcal "
                f"target — fix adherence first; do NOT suggest lowering the target yet."
            )
        floor = {"male": 1500, "female": 1200}.get((prefs.get("gender") or "").lower(), 1350)
        new_cal = max(floor, round(target_cal * 0.93))
        if new_cal >= target_cal:
            return (
                f"\nPLATEAU DETECTED: weight flat for {span_days} days but the {int(target_cal)} kcal target is "
                f"already at the safe minimum — do not cut further; suggest a diet break or more daily movement instead."
            )
        return (
            f"\nADAPTIVE MACRO SUGGESTION (computed deterministically from {span_days} days of flat weight logs — "
            f"treat as ground truth, state it confidently): their maintenance has likely adapted; recommend dropping "
            f"the calorie target from {int(target_cal)} to {new_cal} kcal/day, protein unchanged. Tell them to apply "
            f"it via Profile → Edit Profile — you cannot change targets yourself."
        )

    # gain: only suggest more food if they're actually eating near target already
    if avg_cal < target_cal * 0.9:
        return ""
    new_cal = round(target_cal * 1.07)
    return (
        f"\nADAPTIVE MACRO SUGGESTION (computed deterministically from {span_days} days of flat weight logs — "
        f"treat as ground truth, state it confidently): no weight gain despite hitting a surplus target; recommend "
        f"raising the calorie target from {int(target_cal)} to {new_cal} kcal/day. Tell them to apply it via "
        f"Profile → Edit Profile — you cannot change targets yourself."
    )


def detect_topic(message: str) -> Topic:
    """The single best-matching topic. Kept for callers that need one label;
    prompt building uses detect_topics() so cross-domain asks keep both slices."""
    lower = message.lower()
    ns = _nutrition_score(lower)
    ts = sum(1 for k in _TRAINING_KW  if k in lower)
    ps = sum(1 for k in _PROGRESS_KW  if k in lower)
    best = max(ns, ts, ps)
    if best == 0:
        return "general"
    if ns == best:
        return "nutrition"
    if ts == best:
        return "training"
    return "progress"


def detect_topics(message: str) -> list[Topic]:
    """
    Every topic the message actually touches, strongest first.

    Winner-take-all was dropping real context: "what should I eat after my bench
    session?" scores training > nutrition, so the nutrition slice — today's
    intake, the protein gap — got stripped out of a food question. The slices
    are pre-built and cached, so carrying a second one costs a few hundred
    tokens and buys the cross-domain answers ("eat for my training", "why isn't
    my weight moving despite lifting") that are exactly where a coach should
    look smart.
    """
    lower = message.lower()
    scores: list[tuple[int, Topic]] = [
        (_nutrition_score(lower), "nutrition"),
        (sum(1 for k in _TRAINING_KW  if k in lower), "training"),
        (sum(1 for k in _PROGRESS_KW  if k in lower), "progress"),
    ]
    hits = sorted([s for s in scores if s[0] > 0], key=lambda s: -s[0])
    if not hits:
        return ["general"]
    # Cap at 2: a third slice is nearly always noise from an incidental keyword.
    return [topic for _, topic in hits[:2]]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _maybe(res) -> Optional[dict]:
    if res is None:
        return None
    data = res.data if hasattr(res, "data") else res
    if not data:
        return None
    return data[0] if isinstance(data, list) else data



def adherence_status(
    days_per_week: int,
    account_created,           # datetime.date | None
    sessions: int,
    today,                     # datetime.date
) -> str:
    """
    The adherence line injected into the coach's prompt.

    Expectation is scaled to how long the account has ACTUALLY existed rather
    than a flat 14 days. A user who signed up yesterday has not "missed" eight
    sessions, they have had time for at most one. The unscaled version made
    every brand-new user trip the behind-plan branch, so the coach opened
    first-ever conversations by demanding to know why they were 0 for 8.

    Under a week of history there is no ratio worth reporting at all, so that
    case returns an explicit "too new" note instead of a number the model would
    otherwise read as failure.
    """
    days_on_plan = 14
    if account_created:
        try:
            days_on_plan = max(0, min(14, (today - account_created).days + 1))
        except Exception:
            days_on_plan = 14

    if days_on_plan < 7:
        return (
            "\nAdherence: not enough history yet, this account is "
            f"{days_on_plan} day(s) old. Do NOT treat the zero counts above as "
            "missed sessions or backsliding, there was no time to miss anything. "
            "Treat it as a fresh start: welcome them and help them get session "
            "one done."
        )

    expected = round(days_per_week * days_on_plan / 7)
    if expected <= 0:
        return ""

    line = (
        f"\nAdherence (last {days_on_plan} days): {sessions} of {expected} "
        "planned sessions completed"
    )
    if sessions < expected * 0.6:
        line += ", well behind plan; address it directly, don't gloss over it"
    return line

def _local_date(days_ago: int = 0) -> str:
    """The user's calendar date N days ago — not the server's. See tz.py."""
    return local_date_str(days_ago)


def weight_delta(rows: list[dict]) -> dict | None:
    """Oldest -> newest weight change. SORTS INTERNALLY, on purpose.

    There were two implementations of this and they disagreed by construction:
    rag_pipeline ordered by created_at DESC and read logs[0] - logs[-1];
    context_builder ordered by logged_at ASC and read logs[-1] - logs[0]. Each
    was right about its own query and neither could survive the other's. 30% of
    progress_logs rows are backfilled (logged_at <> created_at::date), so the
    two orderings genuinely disagree about which reading is most recent — and
    both feed the SAME prompt. A coach telling someone they're losing weight
    while they're gaining is the kind of wrong that ends trust.

    Sorting here means the caller's query order is irrelevant and the mistake
    cannot be made again. logged_at is the truth (the day the user says they
    weighed); created_at only breaks ties, so two logs on one day are
    deterministic instead of arbitrary.

    Returns {'from','to','diff','direction'} or None when there's nothing to
    compare.
    """
    usable = [r for r in (rows or []) if r.get("body_weight") is not None]
    if len(usable) < 2:
        return None
    usable.sort(key=lambda r: (str(r.get("logged_at") or ""), str(r.get("created_at") or "")))
    first = float(usable[0]["body_weight"])
    last = float(usable[-1]["body_weight"])
    diff = round(last - first, 1)
    return {
        "from": first,
        "to": last,
        "diff": diff,
        "direction": "up" if diff > 0 else "down" if diff < 0 else "flat",
    }


def _weight_trend(logs: list[dict]) -> str:
    d = weight_delta(logs)
    if not d:
        return "stable"
    if d["diff"] > 0.5:
        return "gaining"
    if d["diff"] < -0.5:
        return "losing"
    return "stable"


_FACT_TYPE_LABELS = {
    "injury_limitation": "Limitation",
    "preference": "Preference",
    "override": "Exception",
}


def fetch_active_facts(user_id: str) -> list[dict]:
    """Durable coach-memory facts the user has confirmed (injuries, standing
    preferences, negotiated exceptions)."""
    if not supabase:
        return []
    try:
        return (
            supabase.table("coach_memory_facts")
            .select("fact_type, content")
            .eq("user_id", user_id)
            .eq("status", "active")
            .order("created_at")
            .execute()
            .data or []
        )
    except Exception:
        return []


def get_facts_block(user_id: str) -> str:
    """Formatted coach-memory block for callers outside the slice pipeline
    (e.g. the program builder, which builds its own prompt from scratch)."""
    return _format_facts(fetch_active_facts(user_id))


def _format_facts(rows: list[dict]) -> str:
    """Formats persistent coach-memory facts (injuries/preferences/exceptions the
    user has confirmed) so they always ride along with the base slice — this is
    durable memory, distinct from the live-state slices below."""
    if not rows:
        return ""
    lines = ["KNOWN FACTS ABOUT THIS USER (persistent — treat as ground truth unless the user contradicts it):"]
    for r in rows:
        label = _FACT_TYPE_LABELS.get(r.get("fact_type"), r.get("fact_type"))
        lines.append(f"  - [{label}] {r.get('content', '')}")
    return "\n".join(lines)


# ─── Today's intake — the one line that can't be cached ───────────────────────
# Everything else in the nutrition slice moves on the scale of days. Today's
# intake moves every time the user logs a meal, and the app writes food_logs
# straight to Postgres (services/mealService.logMeal) without telling this
# service, so context_cache.invalidated_at stays null. A row built at 13:51 UTC
# with 701 kcal in it was still being quoted as "701 kcal already in" at 16:37,
# 1290 kcal later, and the coach planned a cheat meal around a budget the user
# had already spent. Recomputed on read instead of trusting every current and
# future write path to remember to invalidate.

_TODAY_LINE_RE = re.compile(r"^TODAY so far:.*$", re.M)


def _today_intake_line(
    today_cal: float, today_prot: float, water_glasses, macros: Optional[dict]
) -> str:
    m = macros or {}
    line = f"TODAY so far: {int(today_cal)} kcal, {int(today_prot)}g protein eaten"
    if m.get("calories"):
        line += (
            f" — {max(0, int(m.get('calories', 0)) - int(today_cal))} kcal and "
            f"{max(0, int(m.get('protein_g', 0)) - int(today_prot))}g protein remaining"
        )
    return line + f" | Water today: {water_glasses or 0}/8 glasses"


def refresh_today_intake(nutrition: str, user_id: str) -> str:
    """Re-point the cached TODAY line at what food_logs says right now.

    Three small indexed reads. Cheap next to the LLM call they feed, and the
    alternative — a full build_user_context — is a dozen queries plus stall
    detection. On any failure the cached line is returned untouched: a slightly
    stale number beats no nutrition context at all.
    """
    if not supabase or not nutrition or not _TODAY_LINE_RE.search(nutrition):
        return nutrition
    try:
        today = _local_date(0)
        food_rows = (
            supabase.table("food_logs")
            .select("calories, protein_g, logged_at")
            .eq("user_id", user_id)
            .gte("logged_at", today)
            .execute()
            .data or []
        )
        water_row = _maybe(
            supabase.table("water_logs")
            .select("glasses").eq("user_id", user_id).eq("log_date", today).limit(1).execute()
        )
        macros = _maybe(
            supabase.table("user_macros")
            .select("calories, protein_g").eq("user_id", user_id).limit(1).execute()
        )
    except Exception:
        logger.warning("[context_builder] today-intake refresh failed for %s", user_id)
        return nutrition

    # logged_at is a date on new rows but a timestamp on older ones, so match on
    # the prefix the way build_user_context does rather than trusting gte alone.
    today_rows = [r for r in food_rows if str(r.get("logged_at", ""))[:10] == today]
    cal, prot = _avg_food(today_rows)  # one day in, so the per-day average IS the sum
    fresh = _today_intake_line(cal, prot, (water_row or {}).get("glasses", 0), macros)
    # lambda, not a replacement string — the line carries user data and a stray
    # backslash would be read as a group reference.
    return _TODAY_LINE_RE.sub(lambda _: fresh, nutrition, count=1)


def _avg_food(logs: list[dict]) -> tuple[float, float]:
    by_date: dict[str, dict] = {}
    for row in logs:
        day = str(row.get("logged_at", ""))[:10]
        if day not in by_date:
            by_date[day] = {"cal": 0.0, "prot": 0.0}
        by_date[day]["cal"]  += float(row.get("calories", 0) or 0)
        by_date[day]["prot"] += float(row.get("protein_g", 0) or 0)
    days = list(by_date.values())
    if not days:
        return 0.0, 0.0
    return (
        sum(d["cal"]  for d in days) / len(days),
        sum(d["prot"] for d in days) / len(days),
    )


# ─── Coaching directives ──────────────────────────────────────────────────────

def _coaching_directives(
    prefs: dict,
    macros: dict | None,
    avg_cal: float,
    avg_prot: float,
    weight_trend: str,
    workouts_this_week: int,
) -> str:
    """
    Derives goal-specific coaching directives from real user data.
    These tell the LLM HOW to coach this specific person — proactively,
    not just when asked.
    """
    goal = (prefs.get("primary_goal") or "general fitness").lower().strip()
    fitness_level = (prefs.get("fitness_level") or "").lower()
    target_cal = float((macros or {}).get("calories") or 0)
    target_prot = float((macros or {}).get("protein_g") or 0)
    available_days = int(prefs.get("available_days_per_week") or 0)
    gender = (prefs.get("gender") or "").lower()
    injuries_raw = prefs.get("injuries") or []
    injuries = injuries_raw if isinstance(injuries_raw, list) else [injuries_raw]
    injuries_clean = [i for i in injuries if i]

    lines = ["COACHING DIRECTIVES — read and follow proactively:"]

    # ── Goal-specific primary directive ───────────────────────────────────────
    # Strength is checked BEFORE muscle: the two used to share one hypertrophy
    # branch, which had the coach preaching "surplus is mandatory" and 6-12
    # reps to users whose macros were set to maintenance and whose programs
    # are built around heavy triples.
    if "strength" in goal or "stronger" in goal:
        lines += [
            "PRIMARY GOAL: Get Stronger (maximal strength)",
            "  • Every exercise recommendation → default to heavy 3-6 rep work on the big compounds (squat, bench, deadlift, press, row); accessories can sit at 6-10",
            "  • Rest is part of the set: 3+ minutes between heavy compound sets — never suggest cutting rest to 'keep intensity up'",
            "  • Progress = weight on the bar. Talk in terms of adding load, bar speed, and rep quality — not pump or size",
            "  • Nutrition: maintenance calories with high protein is the plan — do NOT push a big surplus; strength comes from progressive loading and neural adaptation, not extra food",
            "  • SAFETY — non-negotiable on heavy work: proper warm-up ramp, a spotter or safety pins on bench and squat, and 1-2 reps always left in reserve. If they mention maxing out alone, push back",
        ]
    elif "muscle" in goal or "build" in goal:
        lines += [
            "PRIMARY GOAL: Build Muscle",
            "  • Every exercise recommendation → default to 6-12 rep hypertrophy range and progressive overload cues",
            "  • On any food question → aggressively push protein sources first; muscle growth cannot happen without adequate protein",
            "  • On check-ins → ask about sleep quality (muscle protein synthesis peaks during deep sleep), recovery, and protein intake",
            "  • Caloric surplus is mandatory for muscle growth — NEVER suggest cutting calories unless the user explicitly wants a fat-loss phase",
            "  • Recovery is training — if they mention fatigue or soreness, address it as part of the program, not a reason to skip",
        ]
    elif "fat" in goal or "lose" in goal or "cut" in goal:
        lines += [
            "PRIMARY GOAL: Fat Loss",
            "  • Frame every food recommendation around: caloric deficit + maximum protein (muscle-sparing cut)",
            "  • Push high-satiety, high-fiber, high-protein foods. Help them feel full on fewer calories",
            "  • Strength training must stay — the goal is losing FAT, not muscle. Never suggest dropping weights for cardio",
            "  • If they hit a plateau, check adherence first (weekends, liquid calories, underestimating portions) before suggesting changes",
        ]
    elif "recomp" in goal or "recomposition" in goal:
        lines += [
            "PRIMARY GOAL: Body Recomposition",
            "  • Simultaneous fat loss + muscle gain — requires 2g+ protein per kg bodyweight and near-maintenance calories",
            "  • SET EXPECTATIONS: the scale may barely move. Progress = strength going up + body fat going down. Track measurements, not just weight",
            "  • Push: high protein at every meal, progressive overload on compound lifts, sleep, patience",
        ]
    elif "athletic" in goal or "performance" in goal or "sport" in goal:
        lines += [
            "PRIMARY GOAL: Athletic Performance",
            "  • Train for power, explosiveness, and sport-specific movement patterns — not just aesthetics",
            "  • Carbohydrates are CRITICAL for high-intensity athletes — do not suggest low-carb approaches",
            "  • Periodization: push phases MUST include planned deloads or performance will regress",
        ]
    else:
        lines += [
            "PRIMARY GOAL: General Fitness",
            "  • Focus on building sustainable habits and consistency. Don't overwhelm with advanced techniques",
            "  • Push balanced nutrition, 3-4 compound movements per session, adequate sleep",
        ]

    # ── Experience-level coaching style ───────────────────────────────────────
    lines.append("")
    if "complete beginner" in fitness_level or "beginner" in fitness_level:
        lines += [
            "EXPERIENCE LEVEL: Beginner",
            "  • Keep advice SIMPLE. 3-4 compound movements, basic form cues, consistency over complexity",
            "  • Do NOT recommend advanced techniques (periodization, RPE tracking, drop sets, supersets) — it will overwhelm them",
            "  • Celebrate small wins. A beginner adding 2.5kg to a lift is significant",
            "  • Explain WHY behind each recommendation — beginners need to understand, not just follow",
        ]
    elif "trained before" in fitness_level or "somewhat active" in fitness_level:
        lines += [
            "EXPERIENCE LEVEL: Intermediate",
            "  • Can handle progressive overload discussion, basic periodization, RIR (reps in reserve) tracking",
            "  • Don't over-explain fundamentals. They know what a squat is",
        ]
    elif "seriously training" in fitness_level or "advanced" in fitness_level:
        lines += [
            "EXPERIENCE LEVEL: Advanced",
            "  • Speak at a high level — periodization, deload timing, RIR, advanced techniques (rest-pause, myo-reps, drop sets) are fair game",
            "  • They've heard basic advice 100 times. Give them edge-case, nuanced coaching they won't find on Google",
        ]

    # ── Routine structure → how much to impose ────────────────────────────────
    routine = (prefs.get("routine_structure") or "").strip().lower()
    if routine:
        lines.append("")
        if "wing" in routine:
            lines += [
                "TRAINING STRUCTURE: Wings it — decides in the gym",
                "  • When training comes up, offer a concrete session to do TODAY. Don't assume a fixed split they'll stick to",
            ]
        elif "flex" in routine:
            lines += [
                "TRAINING STRUCTURE: Rough plan, stays flexible",
                "  • Give structure but make swaps easy — they adapt day to day. Offer alternatives, not mandates",
            ]
        elif "plan" in routine:
            lines += [
                "TRAINING STRUCTURE: Follows a fixed plan",
                "  • Respect their existing split — refine and progressively overload it, don't overhaul it unprompted",
            ]

    # ── Muscle priorities → bias every training suggestion ─────────────────────
    grow_m    = [x for x in (prefs.get("target_muscles_grow") or []) if x]
    define_m  = [x for x in (prefs.get("target_muscles_define") or []) if x]
    exclude_m = [x for x in (prefs.get("target_muscles_exclude") or []) if x]
    if grow_m or define_m or exclude_m:
        lines.append("")
        lines.append("MUSCLE PRIORITIES (from onboarding — reflect these whenever you suggest training):")
        if grow_m:
            lines.append(f"  • Prioritise volume for: {', '.join(grow_m)} — add extra work here first")
        if define_m:
            lines.append(f"  • Maintain, don't grow: {', '.join(define_m)} — keep at minimal effective volume")
        if exclude_m:
            lines.append(f"  • Minimise direct work for: {', '.join(exclude_m)} — only as secondary movers in compounds")

    # ── Active injury flags ────────────────────────────────────────────────────
    if injuries_clean:
        lines.append("")
        lines.append(f"INJURY CONSTRAINTS — CRITICAL, never violate these:")
        for inj in injuries_clean:
            lines.append(f"  ⛔ {inj}: NEVER suggest movements that load this area. Always provide an alternative that avoids it")

    # ── Nutrition analysis ────────────────────────────────────────────────────
    lines.append("")
    lines.append("NUTRITION STATUS (based on last 7 days of logged food):")
    if target_prot > 0 and avg_prot > 0:
        gap = target_prot - avg_prot
        pct = int((avg_prot / target_prot) * 100)
        if gap > 40:
            lines.append(
                f"  ⚠️ CRITICAL PROTEIN GAP: averaging {int(avg_prot)}g/day vs {int(target_prot)}g target — "
                f"SHORT BY {int(gap)}g ({pct}% of target). "
                f"This is their single biggest obstacle to their goal. "
                f"Actively mention it in food conversations and suggest high-protein options every time."
            )
        elif gap > 15:
            lines.append(
                f"  → Protein: hitting {pct}% of target ({int(avg_prot)}g vs {int(target_prot)}g). "
                f"Nudge them to add one more protein source per day"
            )
        else:
            lines.append(f"  ✓ Protein on track: {int(avg_prot)}g vs {int(target_prot)}g target — good")
    elif target_prot > 0 and avg_prot == 0:
        lines.append(f"  → No food logged in last 7 days. Target is {int(target_prot)}g protein/day. "
                     f"Ask about their diet when relevant — we have no data")
    else:
        lines.append("  → No macro targets set yet")

    if target_cal > 0 and avg_cal > 0:
        cal_diff = avg_cal - target_cal
        if ("muscle" in goal or "strength" in goal) and cal_diff < -200:
            lines.append(
                f"  ⚠️ CALORIE DEFICIT on muscle-gain goal: eating {int(avg_cal)} kcal vs {int(target_cal)} kcal target. "
                f"User is eating too little to grow. Proactively push them to eat more"
            )
        elif ("fat" in goal or "lose" in goal) and cal_diff > 300:
            lines.append(
                f"  ⚠️ CALORIE SURPLUS on fat-loss goal: eating {int(avg_cal)} kcal vs {int(target_cal)} kcal target. "
                f"Gently flag that they're eating above their target and help them find where the excess is coming from"
            )
        elif abs(cal_diff) <= 150:
            lines.append(f"  ✓ Calories on track: {int(avg_cal)} kcal vs {int(target_cal)} kcal target")

    # ── Weight trend vs goal alignment ────────────────────────────────────────
    lines.append("")
    lines.append("WEIGHT TREND ANALYSIS:")
    if weight_trend == "gaining" and ("fat" in goal or "lose" in goal):
        lines.append(
            "  ⚠️ MISALIGNMENT: Body weight is trending UP but goal is fat loss. "
            "If the user checks in on progress, proactively address this — ask about diet adherence, weekends, hidden calories"
        )
    elif weight_trend == "losing" and ("muscle" in goal or "strength" in goal):
        lines.append(
            "  ⚠️ MISALIGNMENT: Body weight is trending DOWN but goal is muscle gain. "
            "User may be in too large a caloric deficit. Push them to eat more — gaining muscle in a deficit is extremely difficult"
        )
    elif weight_trend == "gaining" and ("muscle" in goal or "strength" in goal):
        lines.append("  ✓ Weight trending up — consistent with muscle-gain goal")
    elif weight_trend == "losing" and ("fat" in goal or "lose" in goal):
        lines.append("  ✓ Weight trending down — consistent with fat-loss goal")
    else:
        lines.append(f"  → Weight trend: {weight_trend}")

    # ── Workout frequency ─────────────────────────────────────────────────────
    if available_days > 0:
        lines.append("")
        if workouts_this_week >= available_days:
            lines.append(f"CONSISTENCY: ✓ Completed {workouts_this_week}/{available_days} planned sessions this week — on track")
        elif workouts_this_week >= available_days - 1:
            lines.append(f"CONSISTENCY: {workouts_this_week}/{available_days} sessions this week — nearly on track")
        else:
            lines.append(
                f"CONSISTENCY: ⚠️ Only {workouts_this_week}/{available_days} planned sessions completed this week. "
                f"If they check in, gently explore what's getting in the way — don't lecture, ask"
            )

    return "\n".join(lines)


# ─── Main builder ─────────────────────────────────────────────────────────────

def build_user_context(user_id: str) -> dict:
    """
    Fetches all user data, formats 4 context slices + coaching directives,
    and upserts to context_cache. Returns { base, nutrition, training, progress }.
    """
    seven_days_ago  = _local_date(7)
    thirty_days_ago = _local_date(30)

    today_str    = _local_date(0)
    tomorrow_str = _local_date(-1)

    # Twelve independent reads. They used to run one after another — ~200ms each
    # from the Singapore dyno, so ~2.5s of pure round trips before the coach
    # could begin thinking, on EVERY cache miss. And misses are common: the
    # cache expires at local midnight (is_before_local_today on built_at) and is
    # invalidated by every food log, workout log and saved memory fact. So the
    # first chat of the day, and the next chat after logging anything, paid it.
    #
    # The comment here used to read "Parallel fetches". They never were.
    # Concurrent supabase reads are already proven in chat(), which fans out its
    # own context/history/pending reads the same way.
    _fetches = {
        "profile":      lambda: _maybe(supabase.table("profiles").select("full_name, created_at").eq("id", user_id).limit(1).execute()),
        "prefs":        lambda: _maybe(supabase.table("user_preferences").select("*").eq("user_id", user_id).limit(1).execute()),
        "macros":       lambda: _maybe(supabase.table("user_macros").select("*").eq("user_id", user_id).limit(1).execute()),
        "streak_row":   lambda: _maybe(supabase.table("user_streaks").select("current_streak, longest_streak").eq("user_id", user_id).limit(1).execute()),
        "pb_rows":      lambda: supabase.table("personal_bests").select("exercise_name, weight_kg, reps").eq("user_id", user_id).order("weight_kg", desc=True).limit(8).execute().data or [],
        "wl_rows":      lambda: supabase.table("workout_logs").select("date, exercises_done, workout_id").eq("user_id", user_id).eq("completed", True).gte("date", seven_days_ago).order("date", desc=True).execute().data or [],
        "prog_rows":    lambda: supabase.table("progress_logs").select("body_weight, logged_at").eq("user_id", user_id).gte("logged_at", thirty_days_ago).order("logged_at").execute().data or [],
        "food_rows":    lambda: supabase.table("food_logs").select("calories, protein_g, logged_at").eq("user_id", user_id).gte("logged_at", seven_days_ago).execute().data or [],
        "water_row":    lambda: _maybe(supabase.table("water_logs").select("glasses").eq("user_id", user_id).eq("log_date", today_str).limit(1).execute()),
        "planned_rows": lambda: supabase.table("planned_meals").select("meal_type, food_name, calories").eq("user_id", user_id).eq("plan_date", tomorrow_str).execute().data or [],
        "fact_rows":    lambda: fetch_active_facts(user_id),
        # Independent of everything above — only needs user_id — so they ride the
        # fan-out instead of adding two more sequential round trips later.
        "stalled":      lambda: detect_stalled_exercises(user_id),
        "wl14_rows":    lambda: supabase.table("workout_logs").select("date").eq("user_id", user_id).eq("completed", True).gte("date", _local_date(14)).execute().data or [],
        "user_program": lambda: _maybe(
            supabase.table("user_programs")
            .select("program_id, current_week, current_day, programs(title, difficulty)")
            .eq("user_id", user_id)
            .eq("status", "active")
            .limit(1)
            .execute()
        ),
    }
    with ThreadPoolExecutor(max_workers=len(_fetches)) as _pool:
        _futs = {k: _pool.submit(fn) for k, fn in _fetches.items()}
        # .result() re-raises exactly as the sequential version did — a failed
        # read still fails the build rather than silently yielding empty context.
        _r = {k: f.result() for k, f in _futs.items()}

    profile      = _r["profile"]
    prefs        = _r["prefs"]
    macros       = _r["macros"]
    streak_row   = _r["streak_row"]
    pb_rows      = _r["pb_rows"]
    wl_rows      = _r["wl_rows"]
    prog_rows    = _r["prog_rows"]
    food_rows    = _r["food_rows"]
    water_row    = _r["water_row"]
    planned_rows = _r["planned_rows"]
    fact_rows    = _r["fact_rows"]
    user_program = _r["user_program"]
    stalled_exs  = _r["stalled"]
    wl14_rows    = _r["wl14_rows"]

    # Resolve workout titles for recent sessions. One IN query, not one lookup
    # per session — this loop was up to five more sequential round trips, and it
    # could only start once wl_rows had landed, so it sat entirely outside the
    # fan-out above.
    _recent   = wl_rows[:5]
    _wids     = [r["workout_id"] for r in _recent if r.get("workout_id")]
    _titles   = {}
    if _wids:
        _trows = supabase.table("workouts").select("id, title").in_("id", _wids).execute().data or []
        _titles = {t["id"]: (t.get("title") or "Workout") for t in _trows}
    recent_sessions = [
        f"{_titles.get(row.get('workout_id'), 'Workout')} on {row['date']}"
        for row in _recent
    ]

    # Derived values
    p                = prefs or {}
    injuries_raw     = p.get("injuries") or []
    injuries_list    = injuries_raw if isinstance(injuries_raw, list) else [injuries_raw]
    injuries_clean   = [i for i in injuries_list if i]
    injuries_str     = ", ".join(injuries_clean) or "None"
    allergies_raw    = p.get("allergies_or_dislikes") or []
    allergies_list   = allergies_raw if isinstance(allergies_raw, list) else [allergies_raw]
    allergies_str    = ", ".join([a for a in allergies_list if a]) or "None"
    training_days    = ", ".join(p.get("training_days") or []) or "Not set"
    grow_m           = [x for x in (p.get("target_muscles_grow") or []) if x]
    define_m         = [x for x in (p.get("target_muscles_define") or []) if x]
    exclude_m        = [x for x in (p.get("target_muscles_exclude") or []) if x]
    rest_pref        = p.get("rest_time_seconds")
    routine_pref     = (p.get("routine_structure") or "").strip()
    # Session length was collected at onboarding and shapes the starter plan
    # (src/lib/starterPlan.ts budgets exercises against it), but it was never
    # projected here — so the coach proposed programs with no idea how long the
    # user actually has, and routinely wrote hour-plus days for a 30-minute user.
    session_pref     = p.get("session_duration_mins")
    _train_pref_lines = []
    if rest_pref or routine_pref or session_pref:
        _train_pref_lines.append(
            f"Preferences: session length ~{session_pref or '?'} min | "
            f"default rest ~{rest_pref or '?'}s | approach: {routine_pref or 'not set'}"
        )
    if grow_m or define_m or exclude_m:
        _train_pref_lines.append(
            f"Muscle priorities — grow: {', '.join(grow_m) or '—'} | "
            f"define: {', '.join(define_m) or '—'} | minimize: {', '.join(exclude_m) or '—'}"
        )
    train_pref_block = ("\n" + "\n".join(_train_pref_lines)) if _train_pref_lines else ""
    avg_cal, avg_prot = _avg_food(food_rows)
    today_rows = [r for r in food_rows if str(r.get("logged_at", ""))[:10] == today_str]
    today_cal, today_prot = _avg_food(today_rows)  # single-day list — _avg_food's per-day average is just the sum
    water_glasses = (water_row or {}).get("glasses", 0)
    planned_str = "; ".join(
        f"{r['food_name']} ({r['meal_type']}, {int(r.get('calories') or 0)}kcal)" for r in planned_rows
    ) or "nothing planned yet"
    trend            = _weight_trend(prog_rows)
    current_weight   = prog_rows[-1]["body_weight"] if prog_rows else (p.get("weight") or None)
    start_weight     = prog_rows[0]["body_weight"] if prog_rows else None
    current_streak   = (streak_row or {}).get("current_streak", 0)
    longest_streak   = (streak_row or {}).get("longest_streak", 0)
    pb_str           = "; ".join(f"{pb['exercise_name']}: {pb['weight_kg']}kg × {pb['reps']}" for pb in pb_rows) or "None"
    sessions_str     = "; ".join(recent_sessions) or "None this week"
    full_name        = (profile or {}).get("full_name", "Athlete")

    # Signup date, used below to scale the adherence expectation. Taken as the
    # UTC calendar date of created_at: a few hours of skew against the user's
    # local day is irrelevant across a 14-day window, and reading it this way
    # avoids the global-TZ trap documented in tz.py.
    account_created = None
    _created_raw = (profile or {}).get("created_at")
    if _created_raw:
        try:
            account_created = _date.fromisoformat(str(_created_raw)[:10])
        except Exception:
            account_created = None

    # Progressive overload stall detection (14-day window)
    # stalled_exs came from the fan-out above.
    stalled_block = ""
    if stalled_exs:
        lines = ["PROGRESSIVE OVERLOAD FLAGS (weight unchanged for 14+ days — suggest an increase):"]
        for s in stalled_exs:
            lines.append(
                f"  ⚠️ {s['exercise_name']}: stuck at {s['current_weight_kg']}kg "
                f"across {s['sessions_count']} sessions → try {s['suggested_weight_kg']}kg next time"
            )
        stalled_block = "\n" + "\n".join(lines)

    # Rep-range progression signals (increase / deload vs the program's prescription)
    progression_sigs = detect_progression_signals(user_id)
    progression_block = ""
    if progression_sigs:
        lines = ["PROGRESSION SIGNALS (rep performance vs prescription, last 2 sessions — state these proactively):"]
        for sig in progression_sigs:
            arrow = "⬆" if sig["action"] == "increase" else "⬇"
            lines.append(
                f"  {arrow} {sig['exercise_name']}: {sig['reason']} → "
                f"{sig['action']} {sig['current_weight_kg']}kg to {sig['new_weight_kg']}kg "
                f"(next target_weight: {sig['target_weight_str']})"
            )
        progression_block = "\n" + "\n".join(lines)

    # 14-day adherence: sessions completed vs. expected from the user's own
    # weekly frequency. This is what lets the coach open with "you've missed
    # your last 4 sessions" instead of only discussing the static plan.
    adherence_line = ""
    days_per_week = p.get("available_days_per_week")
    if days_per_week:
        try:
            # wl14_rows came from the fan-out above.
            adherence_line = adherence_status(
                int(days_per_week), account_created, len(wl14_rows), today_local()
            )
        except Exception:
            adherence_line = ""

    # Full current-week program structure (one line per day)
    program_structure_block = ""
    if user_program and user_program.get("program_id"):
        program_structure_block = _program_structure_block(
            user_program["program_id"], user_program.get("current_week") or 1
        )

    # Adaptive macro suggestion (deterministic plateau rule, suggest-only)
    adaptive_macro_line = _adaptive_macro_suggestion(p, macros, prog_rows, avg_cal)

    # Has today's session already been logged?
    #
    # user_programs.current_day is the NEXT session in the rotation, not the one
    # belonging to today. Handing the model a bare "Day 4" made it answer "today
    # is Day 4" while the app correctly showed "All done for today" for the Day 3
    # session just finished. Both were reading the same column and disagreeing
    # about what it meant. State it explicitly instead.
    trained_today = any(r.get("date") == today_str for r in wl_rows)
    todays_session_title = None
    if trained_today:
        for r in wl_rows:
            if r.get("date") != today_str:
                continue
            wid = r.get("workout_id")
            if wid:
                # wl_rows is date-desc, so today's session is inside the first
                # five the title batch already resolved. Fall back to a lookup
                # only if it somehow isn't, so behaviour is unchanged.
                todays_session_title = _titles.get(wid)
                if todays_session_title is None:
                    t = _maybe(
                        supabase.table("workouts").select("title").eq("id", wid).limit(1).execute()
                    )
                    if t:
                        todays_session_title = t.get("title")
            break

    # Program info
    program_str = "None"
    if user_program and user_program.get("programs"):
        prog = user_program["programs"]
        # The programs table has no days/week; use the user's preferred weekly
        # training frequency instead (programs only define duration + difficulty).
        days_per_week = (prefs or {}).get("available_days_per_week") or "?"
        next_day = user_program.get("current_day", 1)
        program_str = (
            f"{prog.get('title', 'Custom Plan')} "
            f"(Week {user_program.get('current_week', 1)}, "
            f"next session: Day {next_day}) "
            f"— {days_per_week} days/week, difficulty: {prog.get('difficulty', '?')}"
        )
        if trained_today:
            done = f" ({todays_session_title})" if todays_session_title else ""
            program_str += (
                f"\n  TODAY ({today_str}): ALREADY TRAINED{done}. "
                f"Day {next_day} is the user's NEXT session — do NOT call it today's. "
                f"They have nothing left to do in the gym today."
            )
        else:
            program_str += (
                f"\n  TODAY ({today_str}): no session logged yet. "
                f"Day {next_day} is today's session."
            )

    # Macro targets
    m = macros or {}
    macro_targets_str = (
        f"{int(m.get('calories', 0))} kcal | "
        f"{int(m.get('protein_g', 0))}g protein | "
        f"{int(m.get('carbs_g', 0))}g carbs | "
        f"{int(m.get('fat_g', 0))}g fat"
        if m.get("calories") else "Not calculated yet"
    )

    # Coaching directives (derived from real data)
    directives = _coaching_directives(p, macros, avg_cal, avg_prot, trend, len(wl_rows))

    # Persistent coach-memory facts (confirmed via chat card or the review screen)
    facts_block = _format_facts(fact_rows)

    # ── Base (always injected) ────────────────────────────────────────────────
    base = (
        f"ATHLETE PROFILE:\n"
        f"Name: {full_name} | Age: {p.get('age', '?')} | Gender: {p.get('gender', '?')} | Body type: {p.get('body_type', '?')}\n"
        f"Height: {p.get('height', '?')}cm | Current weight: {current_weight or '?'}kg\n"
        f"Goal: {p.get('primary_goal', 'general fitness')} | Fitness level: {p.get('fitness_level', 'unknown')}\n"
        f"Gym experience: {p.get('gym_experience_months', '?')} months | Consistency: {p.get('training_consistency', '?')}\n"
        f"INJURIES — NEVER stress these areas: {injuries_str}\n"
        + (f"\n{facts_block}\n" if facts_block else "\n")
        + f"\n{directives}"
    )

    # ── Nutrition slice ───────────────────────────────────────────────────────
    nutrition = (
        f"NUTRITION PROFILE:\n"
        f"Diet type: {p.get('diet_preference', '?')} | FOOD RESTRICTIONS — NEVER suggest: {allergies_str}\n"
        f"Cuisine preference: {p.get('cuisine_preference', 'any')} | Budget: {p.get('budget_tier', 'any')} | Cooking: {p.get('cooking_source', '?')}\n"
        f"Macro targets/day: {macro_targets_str}\n"
        f"7-day avg actual intake: {int(avg_cal)} kcal, {int(avg_prot)}g protein\n"
        f"Protein gap: {max(0, int((macros or {}).get('protein_g', 0)) - int(avg_prot))}g/day short of target\n"
        f"Water target: {m.get('water_liters', '?')}L/day | Fiber target: {m.get('fiber_g', '?')}g/day\n"
        + _today_intake_line(today_cal, today_prot, water_glasses, macros)
        + f"\nPlanned for tomorrow: {planned_str}"
        f"{adaptive_macro_line}"
    )

    # ── Training slice ────────────────────────────────────────────────────────
    training = (
        f"TRAINING PROFILE:\n"
        f"Equipment: {p.get('equipment', '?')} | Training days: {training_days} | {p.get('available_days_per_week', '?')} days/week{train_pref_block}\n"
        f"Active program: {program_str}{program_structure_block}\n"
        f"Experience: {p.get('gym_experience_months', '?')} months | Level: {p.get('fitness_level', '?')}\n"
        f"INJURIES (avoid loading these): {injuries_str}\n"
        f"This week: {len(wl_rows)} sessions completed | Recent: {sessions_str}{adherence_line}\n"
        f"Top PRs: {pb_str}"
        f"{stalled_block}"
        f"{progression_block}"
    )

    # ── Progress slice ────────────────────────────────────────────────────────
    weight_change = ""
    if current_weight and start_weight:
        diff = round(float(current_weight) - float(start_weight), 1)
        direction = "gained" if diff > 0 else "lost"
        weight_change = f" ({direction} {abs(diff)}kg over 30 days)"

    progress = (
        f"PROGRESS SNAPSHOT:\n"
        f"Current weight: {current_weight or '?'}kg{weight_change} | Trend: {trend} over 30 days\n"
        f"Streak: {current_streak} days active | Longest ever: {longest_streak} days\n"
        f"Workouts this week: {len(wl_rows)} / {p.get('available_days_per_week', '?')} planned{adherence_line}\n"
        f"Goal: {p.get('primary_goal', 'general fitness')}"
        f"{adaptive_macro_line}"
    )

    # Upsert to context_cache
    if supabase:
        supabase.table("context_cache").upsert({
            "user_id":           user_id,
            "context_text":      base,
            "nutrition_context": nutrition,
            "training_context":  training,
            "progress_context":  progress,
            "built_at":          utc_now_iso(),
            "invalidated_at":    None,
        }, on_conflict="user_id").execute()

    return {
        "base":      base,
        "nutrition": nutrition,
        "training":  training,
        "progress":  progress,
    }


def get_cached_context(user_id: str) -> Optional[dict]:
    """Returns cached context dict if valid (not invalidated), else None."""
    if not supabase:
        return None
    row = _maybe(
        supabase.table("context_cache")
        .select(
            "context_text, nutrition_context, training_context, progress_context, "
            "invalidated_at, built_at"
        )
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not row:
        return None
    if row.get("invalidated_at"):
        return None
    # Day-scoped by construction: the slices bake in "TODAY so far", water
    # today, and week-to-date counts. Nothing invalidates at midnight, so
    # without this a cache built yesterday reports yesterday's intake as
    # today's — the coach confidently quoting numbers from a day that ended.
    if is_before_local_today(row.get("built_at")):
        return None
    return {
        "base":      row["context_text"],
        "nutrition": row["nutrition_context"],
        "training":  row["training_context"],
        "progress":  row["progress_context"],
    }


def get_or_build_context(user_id: str) -> dict:
    """Returns cached context if valid, otherwise builds and caches fresh."""
    cached = get_cached_context(user_id)
    if cached:
        # A fresh build already has today's real numbers; a cached one is as old
        # as the last rebuild, which can be hours and several meals ago.
        cached["nutrition"] = refresh_today_intake(cached["nutrition"], user_id)
        return cached
    return build_user_context(user_id)


def build_prompt_context(slices: dict, topics: "Topic | list[Topic]") -> str:
    """
    Formats the final context block to inject into the system prompt.
    Accepts one topic or several (see detect_topics) — a message that talks
    about both food and training gets both slices.
    """
    if isinstance(topics, str):
        topics = [topics]
    base     = slices.get("base", "")
    progress = slices.get("progress", "")
    # Progress (weight trend + streak) is the anchor the coach's GROUND-in-real-
    # figures / NEVER-CONTRADICT rules lean on, so include it on EVERY topic. A
    # mis-routed classifier must never strip the weight data out from under an
    # "am I on track?" question — the one case those rules exist to protect.
    parts = ["=== USER CONTEXT ===", base]
    if progress:
        parts.append(progress)
    # "progress" is already in above; "general" has no slice of its own.
    for topic in topics:
        if topic in ("general", "progress"):
            continue
        extra = slices.get(topic, "")
        if extra:
            parts.append(extra)
    parts.append("=== END CONTEXT ===")
    return "\n".join(parts)


def invalidate_slice(user_id: str, slice_key: str) -> None:
    """
    Marks the cache as invalidated so the next chat request triggers a rebuild.
    slice_key: 'nutrition' | 'training' | 'progress' | 'all'
    """
    if not supabase:
        return
    supabase.table("context_cache").upsert({
        "user_id":        user_id,
        "invalidated_at": utc_now_iso(),
    }, on_conflict="user_id").execute()


async def rebuild_context_async(user_id: str) -> None:
    """Async wrapper for background rebuilds after invalidation."""
    try:
        await asyncio.to_thread(build_user_context, user_id)
        logger.info(f"[context_builder] rebuilt context for {user_id}")
    except Exception as e:
        logger.warning(f"[context_builder] rebuild failed for {user_id}: {e}")
