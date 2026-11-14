"""
RIPPD — AI Coach RAG Pipeline
Groq + Llama 3 with dynamic context injection from Supabase
"""

import contextvars
import json
import logging
import os
import re
import time
import traceback
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from datetime import datetime, timedelta

from groq import Groq
import groq as _groq_module
from openai import OpenAI  # used for DeepSeek (OpenAI-compatible API)
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from supabase import Client, create_client

from gym_knowledge_data import GYM_KNOWLEDGE
from context_builder import invalidate_slice, detect_stalled_exercises, get_facts_block, weight_delta
from diet_plan import generate_diet_plan
from tz import (
    is_training_day,
    local_date_str,
    local_midnight_utc_iso,
    now_local,
    today_local,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

# ─── CLIENTS ──────────────────────────────────────────────────────────────────
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
)

# ─── NOTIFICATION HELPER ──────────────────────────────────────────────────────

# Hard cap on how many PROACTIVE pings one user can get in a day, across every
# scheduled job. Reactive notifications (AURA received, PR hit, coach replied)
# are not capped — those answer something the user just did.
# Lives here, next to _notify, because every sender already imports from this
# module; cron_build_context imports both the cap and the type list from here.
DAILY_PUSH_CAP = 3
CAPPED_NOTIF_TYPES = [
    "morning_brief",
    "proactive_nudge",
    "morning_nudge",
    "lunch_reminder",
    "evening_reminder",
    "missed_workout_nudge",
]


def _notify(user_id: str, type_: str, title: str, body: str, data: dict | None = None) -> bool:
    """
    Insert a row into the notifications table. The Expo push is sent by the
    on_notification_push trigger on that table (supabase/notification_pipeline_fix.sql)
    — do NOT push from here as well, that double-sends.

    Push used to live in this function, which meant only backend-originated
    notifications ever reached a phone: every type written by a DB trigger, an
    RPC, or the client (friend_request, aura_received, coach_content,
    personal_best, ...) inserted a row and pushed nothing. Moving it onto the
    table covers all of them and leaves nobody to forget.

    Returns True only if the row actually persisted.

    The return value matters: this swallows insert errors so a failed notification
    can't crash a sweep, but callers were then counting CALLS as successes. A
    stale CHECK constraint rejected every 'missed_workout_nudge' insert and
    send_evening_nudges still reported {"sent": 26} while writing zero rows.
    Callers that report counts must count THIS, not their own loop.
    """
    try:
        supabase.table("notifications").insert({
            "user_id": user_id,
            "type": type_,
            "title": title,
            "body": body,
            "read": False,
            "data": data or {"screen": "home"},
            "created_at": utc_now_iso(),
        }).execute()
        return True
    except Exception:
        # Non-fatal, but this used to vanish entirely — now visible in logs AND
        # in the return value.
        logger.warning("_notify: failed to insert notification row for %s (%s)", user_id, type_, exc_info=True)
        return False


# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MODEL = "deepseek-v4-flash"          # DeepSeek — workout/diet plans, heavy reasoning (renamed from deepseek-chat 2026-07-25)
# Groq decommissioned the entire Llama family. llama-3.1-8b-instant and
# llama-3.3-70b-versatile both return 404 "model does not exist" as of
# 2026-08-26 — verified against the live models endpoint, which now lists only
# gpt-oss, qwen3.x, compound and the audio models. qwen leaks raw <think> into
# content, so gpt-oss is the only drop-in.
MODEL_FAST = "openai/gpt-oss-20b"   # Intent classifier & cheap extractors via Groq
MODEL_FAST_70B = "openai/gpt-oss-120b"  # Groq — fallback for conversational questions when DeepSeek returns nothing


def _groq_extra(model: str) -> dict:
    """Per-model kwargs for Groq calls.

    gpt-oss is a REASONING model: it emits thinking tokens before content, and
    they are billed against max_tokens. Left at the default effort it burns a
    small budget entirely on reasoning and returns content="" with
    finish_reason="length". "low" keeps these calls at ~0.4s, measured.
    """
    return {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
FREE_MESSAGE_LIMIT = 20    # Free tier daily limit
PREMIUM_MESSAGE_LIMIT = 100  # Premium tier daily limit
# A CEILING, not a target — the prompt controls length, this only caps it.
# It must clear the reasoning tokens gpt-oss burns before writing anything;
# at 150 the cheap path returned empty content on every call.
MAX_RESPONSE_TOKENS = 700


# Client singletons. Every `Groq(...)` builds its own httpx pool, so a chat turn
# that ran classify + reply + 2 extractors paid FOUR separate TLS handshakes to
# api.groq.com — roughly 150-250ms each from the Singapore dyno, for nothing.
# One cached client per process reuses the connection.
@lru_cache(maxsize=1)
def _groq() -> Groq:
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


@lru_cache(maxsize=1)
def _deepseek() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
    )

# Intents that earn the strongest model and a real token budget. These are the
# turns where a coach either sounds like a coach or sounds like a chatbot:
#   question  — goal-based asks needing a concrete plan
#   medical   — pain/injury: the moment trust is won or lost
#   check_in  — "am I on track?": the flagship personalisation answer
# medical and check_in used to fall through to the 8B model on a 150-token
# budget while a generic question got 70B and 800 — exactly backwards.
_RICH_REPLY_INTENTS = ("question", "medical", "check_in")

# The plan brief for the turn currently being handled, so save_exchange can tag
# the user's row with it wherever that row happens to get written.
#
# save_exchange has SEVENTEEN call sites. Passing the brief down to each one is
# how the card came to persist on exactly one of them — the create_program
# proposal branch — and vanish on every other outcome. Ask the coach to build a
# program, have it come back with a clarifying question instead, and the brief
# was gone: reload the thread and the body-map card was replaced by the raw
# sentence it had composed. A ContextVar is per-request (FastAPI copies the
# context into the threadpool), so this is safe under concurrency and cannot
# drift out of sync with a signature.
_BRIEF_CTX: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "plan_brief", default=None
)

MAX_CONVERSATION_HISTORY = 16  # Last N messages to include for context
MAX_STORED_MESSAGES = 200  # Hard cap on rows kept per user+creator in ai_conversations

# ─── GROQ TOOL DEFINITIONS ────────────────────────────────────────────────────
# Program building only. generate_diet used to have a propose_diet_plan tool
# here that made the LLM invent 7 days of meals and their macros — unverifiable
# and routinely off-target. It now runs through the same deterministic template
# engine as the Diet tab (_handle_diet_generation), so the schema is gone.
# All other intents continue to use the existing JSON-in-prompt approach.

GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "propose_workout_program",
            "description": (
                "Propose a complete workout program for the user to confirm before saving. "
                "Size every training day to fit the session length in the user's context: "
                "count ~8 min of warm-up, then per exercise sets x (working time + their "
                "default rest). Fewer exercises done properly beats a day that overruns. "
                "If the time is tight, cut exercises or drop a set rather than shortening rest."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Program name e.g. '4-Day Strength Split'"},
                    "difficulty": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
                    "goal": {"type": "string", "enum": ["lose", "gain", "maintain"]},
                    "duration_weeks": {"type": "integer", "description": "Number of weeks e.g. 8"},
                    "coaching_rationale": {"type": "string", "description": "1-2 sentence explanation of why this program fits the user"},
                    "days": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "day_number": {"type": "integer"},
                                "title": {"type": "string"},
                                "is_rest_day": {"type": "boolean"},
                                "exercises": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "sets": {"type": "integer"},
                                            "reps": {"type": "string", "description": "slash-separated reps per set e.g. '10/10/10' or '8/8/6'"},
                                            "rest_seconds": {"type": "integer"},
                                            "notes": {"type": "string"},
                                        },
                                        "required": ["name", "sets", "reps"],
                                    },
                                },
                            },
                            "required": ["day_number", "title", "is_rest_day"],
                        },
                    },
                },
                "required": ["title", "difficulty", "goal", "duration_weeks", "coaching_rationale", "days"],
            },
        },
    },
]

# ─── EXERCISE ALIAS MAP ───────────────────────────────────────────────────────
# Maps common shorthand / variations → canonical Title Case name stored in DB.
# Keys must be lowercase; _normalise_exercise() handles the lookup.
EXERCISE_ALIASES: dict[str, str] = {
    # Barbell Bench Press
    "bench": "Barbell Bench Press",
    "bench press": "Barbell Bench Press",
    "bp": "Barbell Bench Press",
    "barbell bench": "Barbell Bench Press",
    "barbell bench press": "Barbell Bench Press",
    "flat bench": "Barbell Bench Press",
    "flat bench press": "Barbell Bench Press",
    # Barbell Squat
    "squat": "Barbell Squat",
    "squats": "Barbell Squat",
    "back squat": "Barbell Squat",
    "barbell squat": "Barbell Squat",
    # Deadlift
    "deadlift": "Deadlift",
    "deadlifts": "Deadlift",
    "dl": "Deadlift",
    "conventional deadlift": "Deadlift",
    # Overhead Press
    "ohp": "Overhead Press",
    "overhead press": "Overhead Press",
    "shoulder press": "Overhead Press",
    "military press": "Overhead Press",
    "barbell ohp": "Overhead Press",
    "barbell overhead press": "Overhead Press",
    # Pull Up
    "pull up": "Pull Up",
    "pull ups": "Pull Up",
    "pullup": "Pull Up",
    "pullups": "Pull Up",
    "chin up": "Pull Up",
    "chin ups": "Pull Up",
    # Barbell Row
    "row": "Barbell Row",
    "barbell row": "Barbell Row",
    "bent over row": "Barbell Row",
    "bent-over row": "Barbell Row",
    "pendlay row": "Barbell Row",
    # Incline Bench Press
    "incline press": "Incline Bench Press",
    "incline bench": "Incline Bench Press",
    "incline bench press": "Incline Bench Press",
    # Romanian Deadlift
    "rdl": "Romanian Deadlift",
    "romanian deadlift": "Romanian Deadlift",
    "stiff leg deadlift": "Romanian Deadlift",
    # Dip
    "dip": "Dip",
    "dips": "Dip",
    "chest dip": "Dip",
    "tricep dip": "Dip",
    # Leg Press
    "leg press": "Leg Press",
}


def _normalise_exercise(name: str) -> str:
    """Canonical exercise name: alias lookup → title case fallback."""
    clean = name.lower().replace("_", " ").strip()
    return EXERCISE_ALIASES.get(clean, clean.title())


# Module-level cache: all names + equipment-filtered subsets.
# Key = equipment string (lowercased) or "" for the full list.
_EXERCISE_NAMES_CACHE: dict[str, list[str]] = {}

# Equipment keywords → exercises_library equipment values
_EQUIPMENT_MAP = {
    "barbell": ["barbell"],
    "dumbbell": ["dumbbell", "dumbbells"],
    "machine": ["machine", "cable", "cables"],
    "bodyweight": ["bodyweight", "none", "body weight"],
    "kettlebell": ["kettlebell", "kettlebells"],
    "resistance band": ["resistance band", "bands"],
    "full gym": ["barbell", "dumbbell", "machine", "cable", "bodyweight", "kettlebell"],
    "home gym": ["dumbbell", "bodyweight", "resistance band", "kettlebell"],
}


def _is_trivial_variant(candidate_name: str | None, original_name: str | None) -> bool:
    """
    True when the candidate is really the same lift wearing a hat —
    "Barbell Bench Press" vs "Wide-Grip Barbell Bench Press" / "Barbell Squat"
    vs "Barbell Full Squat".

    Embedding search ranks by SIMILARITY, so its top neighbour is reliably a
    trivial variant. That's the wrong objective for a swap: someone asking to
    replace bench because their shoulder hurts gains nothing from the same bar,
    same bench, different grip. A swap wants the same MUSCLE via a meaningfully
    different movement.

    Rule: if either name's words are a subset of the other's, it's a variant.
    Token-subset (not substring) so "Barbell Squat" vs "Barbell Full Squat" is
    caught, while "Barbell Bench Press" vs "Dumbbell Bench Press" is not.
    """
    c = set(re.findall(r"[a-z]+", (candidate_name or "").lower()))
    o = set(re.findall(r"[a-z]+", (original_name or "").lower()))
    if not c or not o:
        return False
    return c <= o or o <= c


def _equipment_allowed(exercise_equipment: str | None, user_equipment: str | None) -> bool:
    """
    Can this user actually perform an exercise needing `exercise_equipment`?
    Unknown/blank user equipment or "full gym" allows everything — the cost of a
    false 'no' (silently hiding a good swap) is worse than a false 'yes'.
    """
    key = (user_equipment or "").lower().strip()
    if not key or key == "full gym":
        return True
    allowed = _EQUIPMENT_MAP.get(key)
    if not allowed:
        return True  # unrecognised preference — don't filter on a guess
    return any(a in (exercise_equipment or "").lower() for a in allowed)


def _get_exercise_names(equipment: str = "") -> list[str]:
    """
    Return exercise names from exercises_library filtered by equipment.
    Results are cached per equipment string after the first call.
    """
    key = equipment.lower().strip()
    if key in _EXERCISE_NAMES_CACHE:
        return _EXERCISE_NAMES_CACHE[key]
    try:
        # Only recommend/snap to real, surfaceable exercises — never the hidden
        # bulk-import junk (stretches/drills/empty rows). get_exercise_info stays
        # unfiltered so it can still resolve a hidden name already in a user's plan.
        query = supabase.table("exercises_library").select("name, equipment").eq("is_searchable", True)
        res = query.execute()
        all_names = [r["name"] for r in (res.data or []) if r.get("name")]

        if key and key != "full gym":
            allowed = _EQUIPMENT_MAP.get(key, [])
            if allowed:
                filtered = [
                    r["name"] for r in (res.data or [])
                    if r.get("name") and any(
                        a in (r.get("equipment") or "").lower() for a in allowed
                    )
                ]
                # Fall back to full list if filter returns too few results
                _EXERCISE_NAMES_CACHE[key] = filtered if len(filtered) >= 20 else all_names
            else:
                _EXERCISE_NAMES_CACHE[key] = all_names
        else:
            _EXERCISE_NAMES_CACHE[key] = all_names
    except Exception:
        _EXERCISE_NAMES_CACHE[key] = []
    return _EXERCISE_NAMES_CACHE[key]


def _snap_to_db_exercise(name: str) -> str:
    """
    Given an LLM-generated exercise name, return the closest real name from
    the cached exercises_library using difflib — zero extra DB calls.
    """
    if not name:
        return name
    from difflib import get_close_matches
    all_names = _get_exercise_names()
    if not all_names:
        return name.title()
    # 1. Exact case-insensitive match
    name_lower = name.lower()
    for n in all_names:
        if n.lower() == name_lower:
            return n
    # 2. difflib close match
    matches = get_close_matches(name, all_names, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    # 3. Substring match on first two words
    words = name.strip().split()
    if len(words) >= 2:
        prefix = f"{words[0]} {words[1]}".lower()
        for n in all_names:
            if prefix in n.lower():
                return n
    # 4. Single keyword fallback
    if words:
        kw = words[0].lower()
        for n in all_names:
            if kw in n.lower():
                return n
    return name.title()


_SETS_BY_DIFFICULTY = {"beginner": 3, "intermediate": 4, "advanced": 5}

_DIFF_RANK = {"beginner": 0, "intermediate": 1, "advanced": 2}
# An untagged row counts as intermediate. Withholding is the safe direction for
# a beginner, and it matches src/lib/exerciseEligibility.
_UNKNOWN_DIFF_RANK = _DIFF_RANK["intermediate"]

_EXERCISE_META_CACHE: dict[str, dict] = {}
_EXERCISE_META_LOADED = False


def _ceiling_for_level(fitness_level: str | None) -> int | None:
    """
    Hardest difficulty this user may be GIVEN, or None for no ceiling.

    Deliberately mirrors ceilingFor in src/lib/exerciseEligibility.ts. They are
    two implementations because the client picks exercises for the starter plan
    and the swap sheet while this service picks them for coach proposals — if you
    change one, change the other.
    """
    level = (fitness_level or "").lower().strip()
    if not level:
        return None  # an unknown level is not a constraint
    if "complete beginner" in level:
        return _DIFF_RANK["beginner"]
    if "seriously training" in level or "advanced" in level:
        return None
    return _DIFF_RANK["intermediate"]


def _get_exercise_meta() -> dict[str, dict]:
    """
    name.lower() → the taxonomy columns needed to judge and substitute an
    exercise. Cached for the process, same as _get_exercise_names.
    """
    global _EXERCISE_META_LOADED
    if _EXERCISE_META_LOADED:
        return _EXERCISE_META_CACHE
    try:
        res = (
            supabase.table("exercises_library")
            .select("name, difficulty, muscle_group, subgroup, rank_gym_subgroup")
            .eq("is_searchable", True)
            .execute()
        )
        for r in res.data or []:
            if r.get("name"):
                _EXERCISE_META_CACHE[r["name"].lower()] = r
    except Exception:
        pass
    _EXERCISE_META_LOADED = True
    return _EXERCISE_META_CACHE


def _enforce_difficulty_ceiling(
    tool_args: dict,
    fitness_level: str | None,
    meta: dict[str, dict] | None = None,
) -> dict:
    """
    Replace any proposed exercise harder than the user's level allows.

    _enforce_program_rules re-derives sets and rest-day order but never looked at
    the movements themselves, so a complete beginner could still be handed
    Handstand Push-Ups here — the same bug the starter generator had, surviving in
    the one path that still lets an LLM choose exercises.

    Must run AFTER _snap_to_db_exercise: before that, names are LLM inventions
    and will not match the library at all.

    Substitutes the best-ranked eligible movement from the same subgroup, then
    the same muscle_group. Drops the exercise only when the library has nothing
    at that level — a slightly shorter day beats an unsuitable lift.
    """
    ceiling = _ceiling_for_level(fitness_level)
    if ceiling is None:
        return tool_args

    meta = _get_exercise_meta() if meta is None else meta
    if not meta:
        return tool_args  # no library data — don't guess

    def rank_of(row: dict) -> int:
        return _DIFF_RANK.get((row.get("difficulty") or "").lower(), _UNKNOWN_DIFF_RANK)

    def best_first(row: dict):
        r = row.get("rank_gym_subgroup")
        return (0, r) if isinstance(r, int) else (1, 0)

    def pool(row: dict, used: set, same_subgroup: bool) -> list[dict]:
        out = []
        for r in meta.values():
            if rank_of(r) > ceiling or r["name"].lower() in used:
                continue
            if same_subgroup:
                if row.get("subgroup") and r.get("subgroup") == row.get("subgroup"):
                    out.append(r)
            elif r.get("muscle_group") and r.get("muscle_group") == row.get("muscle_group"):
                out.append(r)
        return out

    for day in tool_args.get("days") or []:
        if day.get("is_rest_day"):
            continue
        exercises = day.get("exercises") or []
        used = {(e.get("name") or "").lower() for e in exercises}
        kept = []
        for ex in exercises:
            row = meta.get((ex.get("name") or "").lower())
            # Unknown to the library, or already within the ceiling.
            if row is None or rank_of(row) <= ceiling:
                kept.append(ex)
                continue
            candidates = pool(row, used, True) or pool(row, used, False)
            if not candidates:
                continue  # nothing suitable at this level — drop it
            pick = sorted(candidates, key=best_first)[0]
            used.add(pick["name"].lower())
            ex["name"] = pick["name"]
            kept.append(ex)
        day["exercises"] = kept

    return tool_args


# Goals whose rep prescription is a descending pyramid (see PROGRAM RULES 4).
# Fat loss and general health use straight sets by design, so they are excluded.
_PYRAMID_GOAL_HINTS = ("muscle", "strength", "stronger", "hypertroph")


def _pyramid_if_flat(reps: str, goal: str | None) -> str:
    """Turn '12/12/12/12' into '12/10/8/6' when the goal asks for a pyramid.

    The prompt tells the model to pyramid down for muscle and strength goals,
    and it routinely emits a flat number anyway, then writes a rationale
    describing the pyramid it did not build. Padding in the caller makes this
    worse: a single "12" against a 4-set target becomes 12/12/12/12.

    Only rewrites when EVERY set is identical. Anything already varied is the
    model's deliberate choice and is left alone.
    """
    g = (goal or "").lower()
    if not any(h in g for h in _PYRAMID_GOAL_HINTS):
        return reps
    parts = [p.strip() for p in str(reps).split("/") if p.strip()]
    if len(parts) < 3:
        return reps  # 2 sets cannot meaningfully pyramid
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return reps  # non-numeric (e.g. "AMRAP") — not ours to touch
    if len(set(nums)) != 1:
        return reps  # already varies
    top = nums[0]
    step, floor = (2, 5) if top >= 10 else (1, 3)
    return "/".join(str(max(floor, top - step * i)) for i in range(len(nums)))


def _enforce_program_rules(tool_args: dict) -> dict:
    """
    Server-side guardrail for propose_workout_program. The system prompt's
    PROGRAM RULES (sets-per-level, rest-day placement) are instructions only —
    nothing was verifying the LLM actually followed them, so drift like
    "intermediate" programs shipping with 3 sets or a rest day stuck mid-week
    on a bro split went straight to the user. Re-derive both here instead of
    trusting the generation.
    """
    days = tool_args.get("days") or []
    if not days:
        return tool_args

    difficulty = (tool_args.get("difficulty") or "intermediate").lower()
    target_sets = _SETS_BY_DIFFICULTY.get(difficulty, 4)

    for day in days:
        if day.get("is_rest_day"):
            continue
        for ex in day.get("exercises") or []:
            reps_list = [r.strip() for r in str(ex.get("reps", "")).split("/") if r.strip()] or ["10"]
            if len(reps_list) < target_sets:
                reps_list = reps_list + [reps_list[-1]] * (target_sets - len(reps_list))
            elif len(reps_list) > target_sets:
                reps_list = reps_list[:target_sets]
            ex["sets"] = target_sets
            ex["reps"] = _pyramid_if_flat("/".join(reps_list), tool_args.get("goal"))

    # Only a genuine Upper/Lower split conventionally needs a mid-week rest —
    # every other split should have rest days pushed to the end of the week.
    title_lower = (tool_args.get("title") or "").lower()
    is_upper_lower = "upper" in title_lower and "lower" in title_lower
    if is_upper_lower:
        tool_args["days"] = sorted(days, key=lambda d: d.get("day_number", 0))
    else:
        training = [d for d in days if not d.get("is_rest_day")]
        resting = [d for d in days if d.get("is_rest_day")]
        ordered = training + resting
        for i, day in enumerate(ordered, start=1):
            day["day_number"] = i
        tool_args["days"] = ordered

    return tool_args


# ─── EXERCISE SWAP HELPERS ────────────────────────────────────────────────────


def _first_muscle(muscles_val) -> str | None:
    """Extract first muscle name from array or comma-separated string."""
    if not muscles_val:
        return None
    if isinstance(muscles_val, list):
        return muscles_val[0].strip() if muscles_val else None
    parts = str(muscles_val).split(",")
    return parts[0].strip() if parts else None


def _format_muscles(muscles_val) -> str:
    """Format primary_muscles for display (handles array or string)."""
    if not muscles_val:
        return "Unknown"
    if isinstance(muscles_val, list):
        return ", ".join(str(m) for m in muscles_val)
    return str(muscles_val)


# Ranking signals used ONLY when the swap is driven by pain. Neither says
# anything about a healthy lifter's best option — a deficit deadlift is not a
# worse exercise, it is a worse exercise for an irritated joint.
#
# Friendly: the range is capped by the floor or the equipment, the torso is
# supported, or the bar path is fixed so the joint isn't stabilising under load.
_JOINT_FRIENDLY_RE = re.compile(
    r"\b(floor|machine|smith|landmine|cable|seated|supported|lying|"
    r"neutral[\s-]?grip|hammer|pin|board|assisted)\b",
    re.IGNORECASE,
)

# Aggravating: adds range past the normal end position, loads the joint where
# it is weakest, or makes the lift ballistic.
_JOINT_AGGRAVATING_RE = re.compile(
    r"\b(deficit|behind[\s-]?(the[\s-]?)?neck|upright[\s-]?row|wide[\s-]?grip|"
    r"plyo\w*|jump\w*|explosive|power|snatch|clean|jerk|kipping|ballistic|speed)\b",
    re.IGNORECASE,
)


def _pain_rank(name: str | None) -> tuple[bool, bool]:
    """Sort key fragment for a pain-driven swap: (aggravating, not_friendly).

    False sorts before True, so a capped/supported variant floats to the top and
    anything ballistic or range-extending sinks to the bottom.
    """
    # ponytail: name-based heuristic — exercises_library has no column describing
    # range of motion or joint stress, so the name is the only signal available.
    # It reads the ~90% of this library that names its own mechanics ("Floor
    # Press", "Machine Chest Press") and is blind to the rest. Upgrade path: add
    # a `joint_friendly` / `rom_capped` boolean to exercises_library and rank on
    # that instead, keeping this as the fallback for unlabelled rows.
    n = name or ""
    return (
        bool(_JOINT_AGGRAVATING_RE.search(n)),
        not _JOINT_FRIENDLY_RE.search(n),
    )


def _muscles_overlap(muscles_val, target: str) -> bool:
    """True if target muscle appears in muscles_val (array or string)."""
    if not muscles_val or not target:
        return False
    t = target.lower()
    if isinstance(muscles_val, list):
        return any(t in str(m).lower() for m in muscles_val)
    return t in str(muscles_val).lower()


def _extract_swap_exercise(message: str) -> str | None:
    """Extract the exercise name the user wants to swap out of the message."""
    patterns = [
        r"(?:swap|replace|change|sub(?:stitute)?)\s+(?:the\s+|my\s+)?(.+?)(?:\s+(?:for|with|to|out|exercise|movement)|\s*[.,!?]|\s*$)",
        r"don'?t\s+(?:want\s+to\s+do|like\s+doing)\s+(.+?)(?:[.,!?]|\s*$)",
        r"can'?t\s+do\s+(.+?)(?:\s+(?:exercise|movement)|\s*[.,!?]|\s*$)",
        r"skip\s+(?:the\s+)?(.+?)(?:\s+(?:exercise|today|this)|\s*[.,!?]|\s*$)",
        r"instead\s+of\s+(.+?)(?:\s*[.,!?]|\s*$)",
        r"alternative\s+(?:to|for)\s+(.+?)(?:\s*[.,!?]|\s*$)",
    ]
    text = message.lower().strip()
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) < 3 or candidate in {"it", "this", "that", "them"}:
                continue
            return _normalise_exercise(candidate)
    return None


# Implement words carry no meaning when identifying WHICH lift someone means:
# "bench press" should resolve whether they said barbell, dumbbell or machine.
_GENERIC_EXERCISE_WORDS = {
    "barbell", "dumbbell", "machine", "cable", "smith", "lever", "leverage",
    "kettlebell", "bodyweight", "band", "bands", "with", "the", "a", "and",
}


def _singular(words: list[str]) -> str:
    """Crude singularisation, applied to BOTH sides so they meet in the middle.

    Users type plurals ("sore after deadlifts", "knee pain during squats") and
    the library mixes both ("Barbell Deadlift" vs "Cable Deadlifts"). Without
    this, "deadlifts" scored a longer core phrase than "deadlift" and resolved to
    Cable Deadlifts — a real match on a wrong row, which is worse than no match.

    Keeps double-s words intact so "press" doesn't become "pres".
    """
    return " ".join(
        w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        for w in words
    )


@lru_cache(maxsize=1)
def _exercise_name_index() -> tuple[tuple[str, str], ...]:
    """(name, core phrase) for every searchable exercise. One query per process.

    The core phrase is the name with implement words stripped but ORDER kept, so
    "Smith Machine Bench Press" -> "bench press". That's what lets a message
    saying "bench press" resolve to a real library row.
    """
    try:
        rows = (
            supabase.table("exercises_library")
            .select("name")
            .eq("is_searchable", True)
            .limit(2000)
            .execute()
            .data
            or []
        )
    except Exception:
        return ()
    idx = []
    for r in rows:
        name = r.get("name") or ""
        core = _singular([
            w for w in re.findall(r"[a-z]+", name.lower())
            if w not in _GENERIC_EXERCISE_WORDS
        ])
        if core:
            idx.append((name, core))
    return tuple(idx)


# Gym shorthand. Nobody types "Barbell Bench Press" into a chat — they say "when
# i bench". The name index is built from library rows, so "bench press" matched
# and bare "bench" did not, and a medical turn phrased the normal way fell
# through to inventing a swap. Longest phrase wins, so "bench press" still beats
# "bench" and resolves the same either way.
_EXERCISE_ALIASES = {
    "bench": "Barbell Bench Press",
    "benching": "Barbell Bench Press",
    "squat": "Barbell Squat",
    "squatting": "Barbell Squat",
    "deadlift": "Barbell Deadlift",
    "deads": "Barbell Deadlift",
    "dl": "Barbell Deadlift",
    "ohp": "Barbell Shoulder Press",
    "overhead press": "Barbell Shoulder Press",
    "military press": "Standing Military Press",
    "pullup": "Pullups",
    "pull up": "Pullups",
    "chinup": "Chin-Up",
    "chin up": "Chin-Up",
    "row": "Barbell Row",
    "curls": "Barbell Curl",
    "dips": "Dips - Triceps Version",
}


def _find_exercise_in_text(text: str) -> str | None:
    """The exercise a free-text message refers to ("...hurts while doing bench press").

    Matches the core phrase as a CONTIGUOUS substring, not a bag of words: in
    "my shoulder is hurting while doing bench press", a word-set match would
    also hit "Shoulder Press" (both words present, far apart). Requiring
    adjacency picks "bench press" and rejects "shoulder press".
    """
    if not text:
        return None
    haystack = " " + _singular(re.findall(r"[a-z]+", text.lower())) + " "
    hits = [
        (name, core) for name, core in _exercise_name_index()
        if f" {core} " in haystack
    ]
    # Most specific phrase wins. Among names sharing a phrase ("Barbell/Dumbbell/
    # Machine Bench Press" all reduce to "bench press") take the shortest, which
    # is the unqualified canonical lift.
    # ponytail: the library has no popularity signal to break those ties with.
    # Upgrade path: prefer whichever variant is in the user's own program once
    # that's cheap to read here.
    if hits:
        hits.sort(key=lambda h: (-len(h[1]), len(h[0].split()), h[0]))
        return hits[0][0]

    # No library name matched — try gym shorthand. Longest alias first so
    # "overhead press" wins over "press" if both were ever listed.
    for alias in sorted(_EXERCISE_ALIASES, key=len, reverse=True):
        if f" {alias} " in haystack or f" {_singular(alias.split())} " in haystack:
            canonical = _EXERCISE_ALIASES[alias]
            if get_exercise_info(canonical):
                return canonical
    return None


def _mechanism_sentence(original: dict, replacement: dict) -> str | None:
    """Why this swap helps, stated as fact instead of generated from priors.

    This is the whole point of the mechanics columns. Asked "why does swapping
    barbell bench for a Smith machine help a sore shoulder?", the model answered
    from its priors and produced the exact opposite of the truth — that a Smith
    machine "lets the shoulder move more freely". The prompt then had to argue
    with it, which is a fight prompts lose eventually.

    Now the sentence is computed from the two fields the labelling pipeline
    validates against name rules (bar_path, rom_profile) and handed over as a
    fact. peak_load_position is deliberately NOT used: it's the weakest field
    and still unreviewed, and a confident sentence built on a bad label is the
    same bug wearing a database.

    Returns None when the data doesn't justify a claim — silence beats
    invention.
    """
    o_rom, r_rom = original.get("rom_profile"), replacement.get("rom_profile")
    o_path, r_path = original.get("bar_path"), replacement.get("bar_path")
    o_stab, r_stab = original.get("stabilizer_demand"), replacement.get("stabilizer_demand")
    parts = []

    # Range change is the strongest pain-relevant signal: end-of-range is where
    # an irritated joint actually complains.
    if r_rom == "capped" and o_rom in ("full", "extended"):
        parts.append(
            "it stops the movement short of the deep position where the joint "
            "is most stressed, instead of loading the joint at full stretch"
        )
    elif r_rom == "extended" and o_rom == "capped":
        parts.append("it adds range at the bottom, which is MORE demanding on the joint, not less")

    # Path change second.
    if o_path == "free" and r_path == "fixed":
        parts.append(
            "the machine holds the bar path, so the joint no longer has to "
            "stabilise the weight while it moves"
        )
    elif o_path == "free" and r_path == "guided":
        parts.append(
            "the cable keeps tension steady and lets the joint travel its own "
            "path instead of being locked to a bar"
        )
    elif o_path == "fixed" and r_path == "free":
        parts.append(
            "the joint controls its own path rather than being locked to the "
            "machine's, which suits some irritations and aggravates others"
        )

    # Only worth saying if neither of the above already covered it.
    if not parts and o_stab == "high" and r_stab == "low":
        parts.append("the load is supported for you, so you're not stabilising it with the sore joint")

    return "; ".join(parts) if parts else None


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


_IMPLEMENTS = ("barbell", "dumbbell", "kettlebell", "cable", "machine", "smith", "band")


def _same_implement(stated: str, candidate: str) -> bool:
    """False when the two names name DIFFERENT equipment.

    Silent on names that mention no implement at all — "Lateral Raises" ->
    "Dumbbell Lateral Raise" is a fine resolution, because the user never asked
    for a specific implement. Only an explicit mismatch is rejected.
    """
    s = {w for w in _IMPLEMENTS if w in (stated or "").lower()}
    c = {w for w in _IMPLEMENTS if w in (candidate or "").lower()}
    return not s or not c or bool(s & c)


def _sanitize_recommended_exercises(items: list) -> list:
    """Replace model-supplied exercise ids with real ones, or drop the entry.

    The prompt asks for ids from the injected catalog and says to omit anything
    not listed. The model ignores that often enough to matter — it produced
    "exercise_id": "LateralRaises" with a catalog of real UUIDs sitting in the
    same prompt. A fake id reaches the app as a card that opens nothing.

    Resolving by NAME rather than rejecting outright keeps the recommendation:
    lateral raises is the right answer for wider shoulders, it just arrived with
    a made-up identifier attached.
    """
    clean = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = (item.get("display_name") or "").strip()
        ex_id = str(item.get("exercise_id") or "").strip()

        if _UUID_RE.match(ex_id):
            clean.append(item)
            continue

        # Best-effort: a transient Supabase blip here must not take down a
        # reply that is otherwise finished. Losing a card beats losing the turn.
        try:
            row = get_exercise_info(name) if name else None
            if not row:
                resolved = _find_exercise_in_text(name) if name else None
                # _find_exercise_in_text strips implement words on purpose —
                # that's what lets "bench press" resolve when a user says
                # "swap this". For a CARD the implement is the whole point, and
                # that same stripping turned "Seated Dumbbell Shoulder Press"
                # into Barbell Shoulder Press: the user reads dumbbell, the plan
                # gets a barbell. Different job, stricter rule.
                if resolved and not _same_implement(name, resolved):
                    resolved = None
                row = get_exercise_info(resolved) if resolved else None
        except Exception:
            logger.warning("[metadata] id lookup failed for %r — dropping card", name)
            continue
        if row:
            clean.append({
                "exercise_id": row["id"],
                "display_name": row["name"],
                "slug": f"/exercise/{row['slug']}",
            })
        # else: unresolvable name — drop it rather than ship a dead card.
    return clean


def _resolve_swap_target(user_message: str, history: list | None) -> str | None:
    """Which exercise a swap actually refers to.

    Two ways this used to fall over, both ending in the same silent failure —
    swap_data=None, which drops the turn into the "no alternative found" branch
    where the coach invents a swap from memory instead of using the catalogue:

    1. The message names nothing: "can we swap this".
    2. The extraction regex returns junk. On the real message "can we swap and
       exercise for thsi" it captured the word "And" as the exercise name, which
       then missed the library.

    So: validate whatever we extract, and when it doesn't resolve, look back for
    the last exercise the conversation actually named.
    """
    name = _extract_swap_exercise(user_message)
    if name and get_exercise_info(name):
        return name

    # The exercise is often named in THIS message with no swap verb attached —
    # "my lower back is sore after deadlifts". Only history was being scanned,
    # which is right for "swap this" and useless for every medical turn.
    here = _find_exercise_in_text(user_message)
    if here:
        return here

    # Referent is in an earlier turn — most recent first.
    for msg in reversed((history or [])[-6:]):
        found = _find_exercise_in_text(msg.get("content") or "")
        if found:
            return found
    return None


def _subgroup_rank(candidate_subgroup, original_subgroup) -> int:
    """0 = same region as the original, 1 = unknown, 2 = a different region.

    A RANKING, never a filter. "Upper chest" and "lower chest" are different
    exercises to anyone who trains, and offering a decline press as the swap for
    an incline press is the complaint this exists to fix. But subgroup is only
    62% populated and the gaps are lopsided — 71% of shoulder rows have none —
    so filtering on it would return an EMPTY list for most shoulder swaps and
    leave ~13 candidates for upper chest before equipment is even considered.

    Unknown sorts between same and different on purpose: an unlabelled row might
    be the right region, so it should beat a row known to be the wrong one, and
    lose to a row known to be right.

    Uses `subgroup`, not `sub_muscle_group`. The library carries both and they
    disagree — "Cable Crossover" is lower_chest by one and chest_mid by the
    other — and subgroup is the better populated of the two.
    """
    if not original_subgroup:
        return 1  # nothing to match against; do not reorder on a guess
    c = (candidate_subgroup or "").strip().lower()
    if not c:
        return 1
    return 0 if c == str(original_subgroup).strip().lower() else 2


def find_exercise_alternative(
    original_name: str, user_id: str, pain_context: bool = False
) -> dict | None:
    """
    Looks up the exercises_library for an alternative that hits the same
    primary muscles as original_name. Returns a swap_data dict or None.

    pain_context: the swap was asked for because something hurts. This picks the
    candidate BEFORE the model sees the turn, so without the flag the ranking
    would hand the coach a full-range free-weight variant and the prompt would
    then be told to explain why a range-limited one is safer — the two
    disagreeing, with the pre-picked candidate winning.
    """
    original_info = get_exercise_info(original_name)
    if not original_info:
        return None

    # Excluded: replacements the user has already rejected, plus the original.
    try:
        overrides = (
            supabase.table("user_exercise_overrides")
            .select("replacement_exercise_name")
            .eq("user_id", user_id)
            .execute()
            .data
            or []
        )
        exclude = {o["replacement_exercise_name"].lower() for o in overrides}
    except Exception:
        exclude = set()
    exclude.add(original_name.lower())

    # What the user can actually train with. Needed by BOTH paths below: the
    # embedding RPC ranks by movement similarity and knows nothing about
    # equipment, so without this it will happily tell someone training at home
    # with dumbbells to use the cable crossover machine.
    user_equipment = ""
    try:
        prefs = _maybe_data(
            supabase.table("user_preferences")
            .select("equipment")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        user_equipment = (prefs or {}).get("equipment") or ""
    except Exception:
        pass

    # Primary: nearest neighbours by embedding (true movement similarity),
    # already ordered best-first and scoped to surfaceable rows by the RPC.
    try:
        candidates = (
            supabase.rpc(
                "match_similar_exercises",
                {"target_id": original_info["id"], "match_count": 10},
            )
            .execute()
            .data
            or []
        )
    except Exception:
        candidates = []

    orig_equipment = (original_info.get("equipment") or "").lower()
    orig_subgroup = original_info.get("subgroup")
    primary = _first_muscle(original_info.get("primary_muscles"))


    # HARD filter, not a ranking tier: match_similar_exercises ranks on embedded
    # TEXT similarity and carries no muscle constraint, so "Leg Press" (quads)
    # returns "Calf Press" (calves) as a near neighbour — the names read alike.
    # A swap onto the wrong muscle is actively wrong, not merely suboptimal, so
    # drop those outright. If nothing survives, the muscle-filtered scan below
    # takes over, which guarantees the muscle by construction.
    if candidates and primary:
        candidates = [
            ex for ex in candidates
            if _muscles_overlap(ex.get("primary_muscles"), primary)
        ]

    # Keep the RPC's similarity order within each tier, but sink (never drop):
    #   1. things they can't perform — no machines for a home-gym user
    #   2. same implement as the original — changing the implement is usually the
    #      whole point ("my shoulder hurts on BARBELL bench" -> dumbbell/machine)
    #   3. trivial variants — same lift, different grip
    # Similarity alone is the wrong objective here: this library is a bulk import
    # thick with near-duplicates, so the nearest neighbour to "Barbell Bench
    # Press" is "Wide-Grip Barbell Bench Press", which swaps nothing.
    # Sorting rather than filtering means that if EVERY neighbour is a machine
    # or a variant, we still answer instead of saying "no alternative found".
    if candidates:
        candidates = sorted(
            candidates,
            key=lambda ex: (
                not _equipment_allowed(ex.get("equipment"), user_equipment),
                # Same region of the muscle, above everything except "can they
                # actually do it". Swapping incline for decline is a wrong
                # answer, not a slightly worse one.
                _subgroup_rank(ex.get("subgroup"), orig_subgroup),
                # Prefer a candidate we can EXPLAIN. Handed a pair with no
                # recorded mechanical difference, the model invents one — told in
                # capitals to say nothing about the replacement, it still opened
                # "narrow stance squats take the barbell off your back" on three
                # runs out of three. That's unwinnable by prompting, so don't
                # create the situation: rank swaps we can justify above swaps we
                # can only assert.
                *((not _mechanism_sentence(original_info, ex),) if pain_context else ()),
                # Outranks "change the implement" when something hurts: taking
                # load off the joint matters more than swapping barbell for
                # dumbbell, and dumbbells ADD bottom-range on a press anyway.
                *(_pain_rank(ex.get("name")) if pain_context else ()),
                (ex.get("equipment") or "").lower() == orig_equipment,
                _is_trivial_variant(ex.get("name"), original_name),
            ),
        )

    # Fallback: the target row has no embedding yet (a newly added exercise), so
    # the RPC returned nothing. This used to fetch an UNORDERED .limit(60) out of
    # 824 rows and only then filter by muscle in Python — so the limit was spent
    # on irrelevant rows and a third of all muscle groups had no candidate in the
    # window at all, returning None. Filter in the DB so the limit applies to
    # rows that could actually be the answer.
    #
    # ALSO fires when the RPC returned candidates the user cannot perform. The
    # equipment check is a sort key rather than a filter — deliberately, so we
    # always answer instead of shrugging — but that only works while SOMETHING
    # in the list is performable. For a home-gym user, all ten embedding
    # neighbours of "Barbell Bench Press" are barbell or machine: every one ties
    # at can't-do-this, and the coach confidently recommends a Smith machine to
    # someone who owns dumbbells. The muscle scan below searches the whole
    # library instead of ten neighbours, so it can actually find them a dumbbell
    # or bodyweight option.
    if candidates and not any(
        _equipment_allowed(ex.get("equipment"), user_equipment) for ex in candidates
    ):
        candidates = []

    if not candidates:
        if not primary:
            return None
        try:
            rows = (
                supabase.table("exercises_library")
                .select("id, slug, name, primary_muscles, secondary_muscles, subgroup, equipment, "
                        "movement_type, bar_path, rom_profile, stabilizer_demand, contraindicated_for")
                .eq("is_searchable", True)
                .contains("primary_muscles", [primary])
                .limit(40)
                .execute()
                .data
                or []
            )
        except Exception:
            return None
        if not rows:
            return None

        # No similarity signal here, so equipment + movement type are the only
        # quality signals available. Same tiering as the embedding path.
        orig_movement = (original_info.get("movement_type") or "").lower()

        def _alt_rank(ex: dict) -> tuple:
            return (
                not _equipment_allowed(ex.get("equipment"), user_equipment),  # can they do it
                _subgroup_rank(ex.get("subgroup"), orig_subgroup),            # same region first
                *((not _mechanism_sentence(original_info, ex),)               # explainable first
                  if pain_context else ()),
                *(_pain_rank(ex.get("name")) if pain_context else ()),        # capped/supported first
                (ex.get("equipment") or "").lower() == orig_equipment,        # change the implement
                _is_trivial_variant(ex.get("name"), original_name),           # a real swap, not a re-grip
                (ex.get("movement_type") or "").lower() != orig_movement,     # push stays push
                (ex.get("name") or "").lower(),                               # stable tie-break
            )

        candidates = sorted(rows, key=_alt_rank)

    alt = next((ex for ex in candidates if ex["name"].lower() not in exclude), None)
    if not alt:
        return None

    return {
        "original_exercise": original_name,
        "replacement_exercise": alt["name"],
        "replacement_id": alt["id"],
        "replacement_slug": alt["slug"],
        "original_muscles": _format_muscles(original_info.get("primary_muscles")),
        "replacement_muscles": _format_muscles(alt.get("primary_muscles")),
        # None when the mechanics data doesn't support a claim — the prompt then
        # tells the model to skip the "why" rather than invent one.
        "mechanism": _mechanism_sentence(original_info, alt),
    }


# ─── FOOD QUESTION DETECTION ──────────────────────────────────────────────────
_FOOD_QUESTION_RE = re.compile(
    r"\b(replace|swap|instead of|can i eat|nutrition|macro(?:s)? in|"
    r"how much protein|calories? in|protein in|carbs? in|fat in|fiber in|"
    r"good source of|better than|vs\.?|versus|substitute|alternative|"
    r"what(?:\'?s| is) in|is \w+ good|how many calories)\b",
    re.IGNORECASE,
)

_FOOD_STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "do",
    "does",
    "will",
    "would",
    "could",
    "should",
    "can",
    "may",
    "i",
    "me",
    "my",
    "we",
    "you",
    "your",
    "he",
    "she",
    "it",
    "they",
    "this",
    "that",
    "what",
    "which",
    "how",
    "much",
    "many",
    "some",
    "any",
    "more",
    "most",
    "no",
    "not",
    "just",
    "so",
    "than",
    "too",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "from",
    "in",
    "on",
    "or",
    "and",
    "but",
    "if",
    "vs",
    "eat",
    "get",
    "instead",
    "replace",
    "swap",
    "versus",
    "good",
    "better",
    "bad",
    "ok",
    "okay",
    "yes",
    "no",
    "also",
    "nutrition",
    "macros",
    "macro",
    "calories",
    "calorie",
    "protein",
    "carbs",
    "carb",
    "fat",
    "fiber",
    "per",
    "grams",
    "gram",
}


_STATED_MACRO_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*g\s+(?:of\s+)?(protein|carbs?|carbohydrates?|fat|fiber|fibre)",
    re.IGNORECASE,
)
_STATED_CALORIE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:kcal|cal(?:ories?)?)\b",
    re.IGNORECASE,
)


def _extract_stated_macros(message: str) -> dict | None:
    """
    Returns {protein_g, carbs_g, fat_g, fiber_g, calories} for any macro values
    the user explicitly states in their message (e.g. 'it has 50g protein').
    Returns None if no stated values are found.
    """
    macros: dict = {}
    for m in _STATED_MACRO_RE.finditer(message):
        val = float(m.group(1))
        nutrient = m.group(2).lower()
        if "protein" in nutrient:
            macros["protein_g"] = val
        elif "carb" in nutrient:
            macros["carbs_g"] = val
        elif "fat" in nutrient:
            macros["fat_g"] = val
        elif "fib" in nutrient:
            macros["fiber_g"] = val
    cal_m = _STATED_CALORIE_RE.search(message)
    if cal_m:
        macros["calories"] = float(cal_m.group(1))
    return macros if macros else None


def _is_food_question(message: str) -> bool:
    return bool(_FOOD_QUESTION_RE.search(message))


def _extract_food_candidates(message: str) -> list[str]:
    """
    Returns unigram and bigram candidates after stripping stop words.
    Tries bigrams first so "chicken breast" beats "chicken" and "breast" separately.
    """
    cleaned = re.sub(r"[^\w\s\-]", " ", message.lower())
    tokens = [t for t in cleaned.split() if t not in _FOOD_STOP_WORDS and len(t) > 2]
    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return (
        bigrams + tokens
    )  # bigrams first so they match before their constituent words


# ─── INTENT CLASSIFIER ────────────────────────────────────────────────────────
# True medical red flags — symptoms no rep-range tweak should ever answer.
# Deliberately NARROW: this must catch emergencies without turning every gym
# ache into a liability disclaimer (the coach's whole value is not doing that).
# Ordinary joint tweaks and soreness fall through to the coachable "medical"
# intent below. Regex, not the LLM classifier, for the same reason crisis is:
# a false positive costs one over-cautious reply, a false negative can cost far
# more.
_MEDICAL_RED_FLAG_RE = re.compile(
    r"\b(chest\s*(pain|tightness|pressure)|heart\s*(attack|pain)|"
    r"short(ness)?\s*of\s*breath|can'?t\s+breathe|"
    r"numb(ness)?|tingling|pins\s+and\s+needles|"
    r"(radiat|shoot)\w*\s+(pain\s+)?(down|into|through)\s+(my\s+)?(leg|arm|back)|"
    r"dizzy|dizziness|faint(ed|ing)?|pass(ed)?\s+out|black(ed)?\s+out|"
    r"vomit\w*\s+blood|coughing\s+up\s+blood|"
    r"lost?\s+(control\s+of\s+)?(my\s+)?(bladder|bowel)|"
    r"deformed|deformity|dislocat(ed|ion)|deloc|"
    r"deep\s+vein|blood\s+clot)\b",
    re.IGNORECASE,
)

# Suspected fracture / structural failure — phrased separately because these
# only count as red flags in an injury context, not as loose words.
_MEDICAL_STRUCTURAL_RE = re.compile(
    r"\b(broke|broken|fractur\w+)\s+(my\s+)?(arm|leg|wrist|ankle|foot|hand|"
    r"finger|toe|rib|collarbone|clavicle|bone|hip|knee)\b"
    r"|\b(heard|felt)\s+(a\s+)?(pop|snap|crack)\b"
    r"|\bcan'?t\s+(bear\s+weight|put\s+weight|walk|stand)\b"
    r"|\b(severe|unbearable|excruciating|worst)\s+pain\b",
    re.IGNORECASE,
)


def _is_medical_red_flag(text: str) -> bool:
    return bool(_MEDICAL_RED_FLAG_RE.search(text) or _MEDICAL_STRUCTURAL_RE.search(text))


# Coachable pain/injury mentions — the tier below a red flag. Module-level and
# separate from intent classification because an exercise swap asked for BECAUSE
# something hurts is still a medical turn: "swap my bench" matches the swap regex
# and used to drop every injury guardrail on the floor.
_MEDICAL_COACHABLE_RE = re.compile(
    # "hurts?" missed the most common phrasing there is — "my shoulder is
    # hurting" — so it fell through to the LLM classifier instead of the
    # guaranteed medical branch. Same for "sore/soreness", which branch A of the
    # medical prompt exists specifically to answer.
    r"\b(pain|painful|hurt(s|ing)?|ach(e|es|ing|y)|sore(ness)?"
    r"|injur(ed|y|ies)|sprain(ed)?|strain(ed)?|torn|tore|fracture(d)?|dislocate(d)?|broken|popped"
    r"|doctor|physician|hospital|clinic|surgery|operated?|prescription|medication|medicine|diagnos(e|is|ed)"
    r"|tendon|ligament|cartilage|meniscus|rotator\s*cuff|hernia|arthritis|inflammation|swollen|swelling"
    r"|tweak(ed)?|niggle|impinge\w*|bother(s|ing|ed)?|flare[\s-]?up)\b"
    r"|\bcan'?t\s+(move|lift|bend|straighten)\b"
    r"|\bpulled\s+(?:\w+\s+){0,3}?(muscle|hamstring|groin|quad|calf|back|shoulder|neck|hip)\b",
    re.IGNORECASE,
)


def _mentions_pain(text: str) -> bool:
    return bool(_MEDICAL_COACHABLE_RE.search(text or ""))


# An explicit ask to BUILD something. Verb and object must both be present, so
# "my knee hurts during my program" (a report about an existing program) does
# not match, while "build me a program" does.
_PROGRAM_REQUEST_RE = re.compile(
    r"\b(build|make|create|design|write|give|generate)\b[^.!?]{0,40}?"
    r"\b(program|programme|split|routine|workout\s+plan|training\s+plan)\b",
    re.I,
)


def _is_program_request(text: str) -> bool:
    """Does this message ASK for a program to be built?

    Used to stop the coachable-pain branch from swallowing program requests.
    The plan brief asks "Anything bothering you right now?" and then composes
    the answer into the message it sends. Every answer except "Nothing" —
    Shoulder, Lower back, Knee, or anything typed — made _classify_intent_regex
    return "medical", which is a rich-TEXT intent, so the turn never reached
    _handle_tool_intent and no saveable program was ever built. The brief asked
    a question whose every meaningful answer disabled the feature it belonged
    to, and it looked like success: a well-written plan arrived, just as prose
    with no save button.
    """
    return bool(_PROGRAM_REQUEST_RE.search(text or ""))


# Movements people report by a bare count rather than as sets x reps.
_BODYWEIGHT_MOVES = (
    r"(?:push|press)[\s-]?ups?|(?:pull|chin)[\s-]?ups?|sit[\s-]?ups?|squats?|"
    r"lunges?|burpees?|crunches|dips?|planks?|curls?|rows?|deadlifts?"
)

# "<did> ... <n> <movement>", or "<n> <movement> ... <done>".
BODYWEIGHT_REPORT_RE = re.compile(
    r"(?:\b(?:did|done|finished|completed|knocked\s+out|banged\s+out|smashed|"
    r"crushed|got)\b.{0,20}?\d+\s*(?:" + _BODYWEIGHT_MOVES + r")"
    r"|\d+\s*(?:" + _BODYWEIGHT_MOVES + r")\b.{0,20}?\b(?:done|finished|completed))",
    re.IGNORECASE,
)

_QUESTION_RE = re.compile(
    r"^\s*(should|shall|can|could|would|will|do\s+i|does|is|are|am\s+i|how|"
    r"what|why|when|which|who)\b",
    re.IGNORECASE,
)


def _looks_like_question(text: str) -> bool:
    """A question about training is not a report of training."""
    return text.strip().endswith("?") or bool(_QUESTION_RE.search(text))


def _classify_intent_regex(message: str) -> str | None:
    """
    Safety-critical regex checks only. Returns an intent if matched, else None.
    Called before the LLM classifier — these must never be missed.
    """
    text = message.lower().strip()

    crisis_patterns = [
        r"\b(kill\s*(my|him|her)?\s*self|suicide|suicidal|end\s*(it|my\s*life|everything)|end(?:ing)?\s+it\s+all)\b",
        r"\b(wanna|want\s+to|going\s+to|gonna)\s+(die|kill\s+myself|end\s+(it|my\s*life))\b",
        r"\b(don\'?t\s+want\s+to\s+(live|be\s+alive|exist)|no\s+reason\s+to\s+live)\b",
        r"\b(self[\s-]?harm|cut(ting)?\s+myself|hurt(ing)?\s+myself)\b",
        r"\b(better\s+off\s+dead|wish\s+i\s+(was|were)\s+dead|rather\s+be\s+dead)\b",
        r"\b(life\s+is(n\'?t|\s+not)\s+worth|can\'?t\s+(go|do\s+this|take\s+it)\s+(on|anymore))\b",
        r"\b(overdose|jump\s+off|hang\s+myself)\b",
    ]
    if any(re.search(p, text) for p in crisis_patterns):
        return "crisis"

    # Red flags outrank the coachable medical branch below — that branch is
    # explicitly told never to say "see a doctor", which is the right call for a
    # cranky shoulder and the wrong one for chest pain or numbness.
    if _is_medical_red_flag(text):
        return "medical_urgent"

    if _mentions_pain(text):
        return "medical"

    oos_patterns = [
        r"\b(politic(s|al)?|government|election|president|prime\s*minister|parliament)\b",
        r"\b(stock(s|market)?|crypto(currency)?|bitcoin|ethereum|invest(ment|ing)?|trading)\b",
        r"\b(mov(ie|ies)|film|netflix|web\s*series|tv\s*show|celebrity|actor|actress)\b",
        r"\b(codi?ng|programming|software|javascript|python\s*lang|html|css|database)\b",
        r"\b(dating|boyfriend|girlfriend|marriage|divorce|break\s*up)\b",
        r"\b(cricket|football\s*score|ipl|nba\s*score|match\s*result|who\s*won)\b",
    ]
    if any(re.search(p, text) for p in oos_patterns):
        return "out_of_scope"

    return None


# ── Fast path: unambiguous whole-message turns ───────────────────────────────
# Anchored to the ENTIRE message on purpose. "hi" is a greeting; "hi can you
# build me a PPL split" is a create_program, and a substring match would answer
# it with a two-word hello. Anything needing conversational context to resolve
# ("ok i ate it", "yeah go ahead") is deliberately NOT here — that ambiguity is
# exactly what the LLM classifier is for.
_FAST_GREETING_RE = re.compile(
    r"^(hi+|hey+|hello+|yo+|sup|hola|namaste|namaskar|"
    r"good\s+(morning|afternoon|evening)|"
    r"h(ey|i|ello)\s+(coach|there|bro|man|buddy)|"
    r"thanks?|thank\s+you|thx|ty|"
    r"cool|nice|awesome|great|perfect)"
    r"[\s!.,?]*$",
    re.IGNORECASE,
)

_FAST_CONFIRM_RE = re.compile(
    r"^(y|ya|yes+|yeah+|yep|yup|ok|okay|okey|k|sure|"
    r"go\s+ahead|do\s+it|sounds?\s+good|please\s+do|"
    r"add\s+it|log\s+it|save\s+it|confirm)"
    r"[\s!.,]*$",
    re.IGNORECASE,
)

_FAST_CANCEL_RE = re.compile(
    r"^(n|no+|nope|nah|cancel|don'?t|skip\s+it|leave\s+it|"
    r"never\s*mind|nvm|not\s+now|no\s+thanks?)"
    r"[\s!.,]*$",
    re.IGNORECASE,
)


def _fast_intent(message: str, has_pending: bool | None = None) -> str | None:
    """Resolve dead-obvious turns without spending a Groq call on them.

    The LLM classifier is a network round trip on Groq's FREE tier, which
    throttles hard near its daily cap (see the routing note in
    complete_for_intent) — and it was being spent to discover that "ok" means
    ok. Every message pays it, so the cheapest win available is not making the
    call at all for the turns that need no thought.

    `has_pending` gates confirm/cancel because a bare "yes" is meaningless
    without something to say yes TO — the same rule the LLM prompt states. Pass
    None when it isn't known yet and those two are skipped rather than guessed.

    Returns None for anything uncertain. Falling through to the classifier is
    always the safe answer; a wrong fast intent is a wrong reply.
    """
    text = (message or "").strip()
    # A real greeting or confirmation is short. This cap is the cheapest
    # possible guard against a long message that merely opens with "ok".
    if not text or len(text) > 40:
        return None
    if _FAST_GREETING_RE.match(text):
        return "greeting"
    if has_pending:
        if _FAST_CONFIRM_RE.match(text):
            return "confirm"
        if _FAST_CANCEL_RE.match(text):
            return "cancel"
    return None


def classify_intent_smart(
    message: str,
    conversation_history: list,
    has_pending: bool,
    groq_client,
) -> str:
    """
    LLM-based intent classifier on MODEL_FAST (fast, free tier).
    Uses the last 3 messages as context so it understands references like
    'ok i ate it' or 'yeah go ahead' correctly.
    Safety intents (crisis, medical, out_of_scope) are handled by regex before
    this function is called and will never reach here.
    """
    recent = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history
    history_text = ""
    for msg in recent:
        role = "Coach" if msg["role"] == "assistant" else "User"
        history_text += f"{role}: {msg['content'][:300]}\n"

    prompt = f"""You are an intent classifier for a fitness coaching app. Classify the user's latest message into exactly one intent from the list below.

Recent conversation:
{history_text if history_text else "(no prior messages)"}

User's latest message: "{message}"

Has a pending unconfirmed action (food/workout/program/diet): {"yes" if has_pending else "no"}

INTENT LIST:
- greeting: user is just saying hello
- create_program: user wants a brand new workout program BUILT and SAVED (e.g. "make me a 5 day split", "build me a PPL", "i want a new program", "design a training schedule"). NOT for exercise suggestions, single-session questions, or "how do I get X" goal questions ("how do i get wider shoulders", "how do i build a bigger chest") — those are `question`
- modify_program: user wants to CHANGE or EDIT their EXISTING program — adding/removing days, changing sets/reps, swapping exercises within the program, making it harder/easier. ONLY when a structural change is requested. NOT for read-only questions/queries about the program ("how many sets do I have", "show me my plan", "what exercises are in day 3" are question intent, not modify_program)
- generate_diet: user wants a full meal plan created and saved for them. NOT for single food questions
- workout_log: user is reporting exercises they just did (sets, reps, weights)
- weight_log: user is reporting their body weight
- pending_food_log: user ate something or is about to eat something and wants it logged. Also use this if user says "ok i ate it", "i had that", "i ate what you suggested" — when recent context shows a meal was suggested. ALSO use this when the coach's last message asked for a quantity or clarification ("how much?", "grams or pieces?") and the user replies with a number, measurement, or quantity ("100gms", "200g", "3 pieces", "two chicken breasts")
- exercise_swap: user wants to replace or find an alternative for an exercise
- correct_log: user wants to REMOVE, DELETE, UNDO or CORRECT a food item they ALREADY logged ("remove the rice I logged", "delete my last meal", "undo that", "that was 300g not 200g", "the paneer entry is wrong"). For body-weight corrections use weight_log instead — re-stating the weight overwrites it
- confirm: user is saying yes/okay. ONLY use this if has_pending=yes. If has_pending=no, classify as question or check_in instead
- cancel: user is saying no/cancel. ONLY use this if has_pending=yes. If has_pending=no, classify as question or check_in instead
- skip_workout: user is saying they can't train today or are skipping
- check_in: user is sharing how they feel or asking about their progress
- increase_weight: user wants to increase the weights in their workout program (e.g. "increase my weights", "add more weight", "make my program heavier", "increase weight on bench press", "progressive overload", "bump up the weights")
- question: anything else — including questions/queries about the user's current program ("show me my plan", "how many sets do I have", "what exercises are in my program"), exercise suggestions, nutrition questions, general fitness advice, workout tips

Reply with ONLY the intent name, nothing else. No explanation."""

    try:
        response = groq_client.chat.completions.create(
            model=MODEL_FAST,
            messages=[{"role": "user", "content": prompt}],
            # This budget has to cover REASONING tokens, not just the single word
            # we want back. At the old max_tokens=10 gpt-oss hit
            # finish_reason="length" mid-thought and returned content="".
            max_tokens=300,
            temperature=0,
            **_groq_extra(MODEL_FAST),
        )
        parts = (response.choices[0].message.content or "").strip().lower().split()
        if not parts:
            raise ValueError(
                f"empty classification (finish={response.choices[0].finish_reason})"
            )
        result = parts[0]
        valid_intents = {
            "greeting", "create_program", "modify_program", "generate_diet",
            "workout_log", "weight_log", "pending_food_log", "exercise_swap",
            "confirm", "cancel", "skip_workout", "check_in", "question",
            "increase_weight", "correct_log",
        }
        return result if result in valid_intents else "question"
    except Exception:
        # The regex fallback keeps the coach replying, so this degrades in the
        # most deceptive way possible: every message still gets an answer, they
        # are just the WRONG KIND. The regex cannot see conversation context and
        # routes "build me a program ... prioritise chest" to `question`, so the
        # coach chats back instead of building anything. That is exactly what a
        # dead model name looked like from the outside. Never silent again.
        logger.warning(
            "[intent] LLM classifier failed — falling back to regex", exc_info=True
        )
        return classify_intent(message)


# ── Rest-timer adjustment ("increase my rest", "set rest to 2 min") ──────────
# Deterministic — never routed through the LLM classifier. Verbs are required
# so "I need more rest today" (a recovery statement) and "rest day" (skip
# intent) never trip it.
_REST_WORD = r"\brest(?:\s+(?:time|timer|period|duration|interval))?\b"
_REST_UP_RE = re.compile(
    rf"\b(increase|lengthen|extend|raise|bump(?:\s+up)?)\b.{{0,25}}{_REST_WORD}|{_REST_WORD}.{{0,15}}\b(longer|higher)\b",
    re.I,
)
_REST_DOWN_RE = re.compile(
    rf"\b(decrease|shorten|reduce|cut|lower)\b.{{0,25}}{_REST_WORD}|{_REST_WORD}.{{0,15}}\bshorter\b",
    re.I,
)
_REST_SET_RE = re.compile(
    rf"\b(set|change|make|update)\b.{{0,20}}{_REST_WORD}.{{0,15}}?\b(\d+(?:\.\d+)?)\s*(min(?:ute)?s?|sec(?:ond)?s?|s\b|m\b)?",
    re.I,
)


def _detect_rest_adjust(message: str) -> bool:
    text = message.lower()
    if "rest day" in text:
        return False
    return bool(
        _REST_SET_RE.search(text) or _REST_UP_RE.search(text) or _REST_DOWN_RE.search(text)
    )


def _compute_rest_adjust(message: str, current_seconds: int) -> int:
    """New rest value from the message: explicit value wins, else ±30s."""
    text = message.lower()
    m = _REST_SET_RE.search(text)
    if m:
        value = float(m.group(2))
        unit = (m.group(3) or "").strip()
        if unit.startswith("m") or (not unit and value <= 10):
            value *= 60  # "set rest to 2" almost certainly means minutes
        return int(value)
    if _REST_UP_RE.search(text):
        return current_seconds + 30
    return current_seconds - 30


# ── Cheat meal / night out being PLANNED ─────────────────────────────────────
# Deterministic for the same reason rest_adjust is. A forward-looking
# indulgence reads exactly like a food log ("burgers and beers tonight"), so
# both the regex classifier (pending_food_patterns) and the 8B LLM classifier
# answer a "how do I plan around this" question with a log-this-food card.
# Nothing has been eaten yet — there is nothing to log, only something to plan.
_CHEAT_PLAN_RE = re.compile(
    r"\bcheat\s*(?:meal|day)?s?\b"
    r"|\b(?:party|parties|feast(?:ing)?|buffet|banquet|potluck)\b"
    r"|\b(?:eat(?:ing)?|din(?:e|ing)|drink(?:s|ing)?|night|dinner|lunch)\s+out\b"
    r"|\bout\s+(?:for|to)\s+(?:dinner|lunch|drinks|eat|a\s+meal)\b",
    re.I,
)
# Past tense means it already happened, so it genuinely IS a log —
# "I had a cheat meal, log it" must stay pending_food_log.
_CHEAT_ALREADY_ATE_RE = re.compile(
    r"\b(?:ate|had|just\s+(?:ate|had|finished)|log(?:ged)?\s+(?:it|this|that))\b", re.I
)


def _detect_cheat_plan(message: str) -> bool:
    return bool(_CHEAT_PLAN_RE.search(message)) and not _CHEAT_ALREADY_ATE_RE.search(message)


def classify_intent(message: str) -> str:
    """
    Legacy regex classifier. Still used as fallback inside classify_intent_smart.
    Do not call this directly from chat() — use classify_intent_smart instead.
    """
    text = message.lower().strip()

    # Crisis / self-harm — HIGHEST PRIORITY, checked before everything else.
    # Even a false positive here is acceptable; a false negative is not.
    crisis_patterns = [
        r"\b(kill\s*(my|him|her)?\s*self|suicide|suicidal|end\s*(it|my\s*life|everything)|end(?:ing)?\s+it\s+all)\b",
        r"\b(wanna|want\s+to|going\s+to|gonna)\s+(die|kill\s+myself|end\s+(it|my\s*life))\b",
        r"\b(don\'?t\s+want\s+to\s+(live|be\s+alive|exist)|no\s+reason\s+to\s+live)\b",
        r"\b(self[\s-]?harm|cut(ting)?\s+myself|hurt(ing)?\s+myself)\b",
        r"\b(better\s+off\s+dead|wish\s+i\s+(was|were)\s+dead|rather\s+be\s+dead)\b",
        r"\b(life\s+is(n\'?t|\s+not)\s+worth|can\'?t\s+(go|do\s+this|take\s+it)\s+(on|anymore))\b",
        r"\b(overdose|jump\s+off|hang\s+myself)\b",
    ]
    if any(re.search(p, text) for p in crisis_patterns):
        return "crisis"

    # Medical — safety-critical, checked before everything else so a message
    # like "hey my shoulder hurts" is never swallowed by the greeting check
    medical_patterns = [
        r"\b(pain|painful|hurts?|aching|aches?|sharp\s+pain)\b",
        r"\b(injur(ed|y|ies)|sprain(ed)?|strain(ed)?|torn|tore|fracture(d)?|dislocate(d)?|broken|popped)\b",
        r"\b(doctor|physician|hospital|clinic|surgery|operated?|prescription|medication|medicine|diagnos(e|is|ed))\b",
        r"\b(tendon|ligament|cartilage|meniscus|rotator\s*cuff|hernia|arthritis|inflammation|swollen|swelling)\b",
        r"\b(chest\s*pain|heart\s*(attack|pain)|shortness\s*of\s*breath|dizziness?|nausea|vomiting|bleed(ing)?)\b",
        r"\b(numb(ness)?|tingling)\b",
        r"\bcan\'?t\s+(move|lift|bend|straighten)\b",
        r"\bpulled\s+(?:\w+\s+){0,3}?(muscle|hamstring|groin|quad|calf|back|shoulder|neck|hip)\b",
    ]
    if any(re.search(p, text) for p in medical_patterns):
        return "medical"

    # Greeting — only when the whole message is a greeting, so longer
    # messages that merely start with "hey" fall through to real intents
    if re.fullmatch(
        r"(hi|hello|hey|howdy|sup|yo|what\'?s up|good\s+(morning|afternoon|evening|night))[\s!.,]*",
        text,
    ):
        return "greeting"

    # Skip workout — user saying they can't train today
    skip_patterns = [
        r"\b(can'?t|cannot|won'?t|not\s+able|unable)\s+(to\s+)?(go|train|workout|work\s*out|hit\s+the\s+gym|make\s+it)\b",
        r"\b(skipping|skip)\s+(today|gym|workout|training|my\s+workout|the\s+gym)\b",
        r"\brest\s+day\b",
        r"\btaking\s+a\s+day\s+off\b",
        r"\b(day\s+off|off\s+day)\s+(today|from\s+(gym|training))?\b",
        r"\bnot\s+(going|training|working\s*out)\s+today\b",
        r"\b(missed|missing|gonna\s+miss)\s+(today'?s?\s+)?(workout|training|session|gym)\b",
        r"\btoo\s+(tired|busy|sick|exhausted|sore)\s+to\s+(train|workout|go|hit|gym)\b",
        r"\bno\s+(gym|workout|training)\s+today\b",
        r"\b(won'?t\s+be\s+able\s+to|can'?t\s+make\s+it)\s+(train|workout|go|today)?\b",
    ]
    if any(re.search(p, text) for p in skip_patterns):
        return "skip_workout"

    # Out of scope — topics with no fitness relevance
    oos_patterns = [
        r"\b(politic(s|al)?|government|election|president|prime\s*minister|parliament)\b",
        r"\b(stock(s|market)?|crypto(currency)?|bitcoin|ethereum|invest(ment|ing)?|trading)\b",
        r"\b(mov(ie|ies)|film|netflix|web\s*series|tv\s*show|celebrity|actor|actress)\b",
        r"\b(codi?ng|programming|software|javascript|python\s*lang|html|css|database)\b",
        r"\b(dating|boyfriend|girlfriend|marriage|divorce|break\s*up)\b",
        r"\b(cricket|football\s*score|ipl|nba\s*score|match\s*result|who\s*won)\b",
    ]
    if any(re.search(p, text) for p in oos_patterns):
        return "out_of_scope"

    # Create program — user wants AI to build a new workout program
    create_program_patterns = [
        r"\b(create|build|make|design|give me|generate|write)\b.{0,40}\b(program|routine|schedule|workout\s+plan|training\s+plan)\b",
        r"\b(new|custom|personalised?|personalized?)\b.{0,30}\b(program|routine)\b",
        r"\b(push\s+pull|ppl|upper\s+lower|bro\s+split|full\s+body)\b.{0,20}\b(program|plan|split|routine|schedule)?\b",
        r"\bcan you (design|create|make|build|write).{0,30}(for me|my|a)\b",
    ]
    if any(re.search(p, text) for p in create_program_patterns):
        return "create_program"

    # Generate diet plan — user wants a full meal plan
    generate_diet_patterns = [
        r"\b(create|build|make|generate|give me|write|design)\b.{0,40}\b(diet plan|meal plan|eating plan|nutrition plan|food plan)\b",
        r"\bwhat should i eat\b.{0,30}\b(all day|today|for the week|daily)?\b",
        r"\b(full day|daily|weekly)\b.{0,20}\b(diet|meal plan|eating|food)\b",
        r"\bplan (my|a) (diet|meals?|eating|nutrition)\b",
    ]
    if any(re.search(p, text) for p in generate_diet_patterns):
        return "generate_diet"

    # Correct / remove a previously logged food entry — checked before the
    # food/weight logging patterns so "that was 300g not 200g" isn't swallowed
    # as a fresh food log.
    correct_log_patterns = [
        r"\b(remove|delete|undo|erase|get\s+rid\s+of|take\s+off)\b.{0,40}\b(log(ged)?|entry|meal|food|it|that)\b",
        r"\b(that|it|those|the\s+\w+)\s+(was|were|is|weren'?t|wasn'?t|should\s+be)\b.{0,30}\b(wrong|not|instead|actually|\d+\s*(g|grams?|gms?))\b",
        r"\b(actually|oops|my\s+bad|mistake)\b.{0,40}\b(not|wrong|meant|was|should)\b",
        r"\bwrong\s+(amount|quantity|entry|food|portion)\b",
    ]
    if any(re.search(p, text) for p in correct_log_patterns):
        return "correct_log"

    # Workout log — user reporting a completed session or exercise
    workout_log_patterns = [
        r"\b(just\s+(did|finished|completed|crushed|smashed|hit)|did\s+my|finished\s+my|completed\s+my)\b",
        r"\b(trained|lifted|worked\s*out|hit\s+the\s+gym|went\s+to\s+(the\s+)?gym)\b",
        r"\d+\s*(sets?|reps?|rounds?)",  # "3 sets", "10 reps"
        r"\d+\s*[xX×]\s*\d+",  # "3x10", "5×5"
        r"\b(bench(ed)?|squat(ted)?|deadlift(ed)?|pull[\s-]?up|push[\s-]?up|curl(ed)?|press(ed)?)\b.{0,40}\d+",
        # Count-first reporting: "I did 20 pushups", "knocked out 15
        # pullups", "20 squats done".
        #
        # The movement pattern above could not see any of these. It needed
        # the number to come AFTER the movement, and its trailing \b
        # rejected every plural, so "pushups" and "pullups" never matched at
        # all. The most natural way to report a set fell through to general
        # chat, and the user got a lecture instead of the log card.
        BODYWEIGHT_REPORT_RE.pattern,
    ]
    # A question about a movement is not a report of one: "should I do 20
    # pushups?" must stay a question.
    if not _looks_like_question(text) and any(
        re.search(p, text) for p in workout_log_patterns
    ):
        return "workout_log"

    # Weight log — user reporting a scale reading
    weight_log_patterns = [
        r"\b(weigh(ed)?|weight\s+(is|today|this\s+morning|was)|scale\s+(said|read|showed))\b",
        r"\b(i\'?m|i\s+am|im)\s+\d+(\.\d+)?\s*(kg|kgs|lbs?|pounds?|kilos?)\b",
        r"\d+(\.\d+)?\s*(kg|kgs|lbs?|pounds?|kilos?)\b.{0,30}\b(today|morning|weigh)",
        r"\b(body[\s-]?weight|bw)\s*(is|was|today)?\s*\d+",
    ]
    if any(re.search(p, text) for p in weight_log_patterns):
        return "weight_log"

    # Pending food log — user states they are going to eat / are eating / have eaten something
    pending_food_patterns = [
        r"\b(i'?m\s+going\s+to\s+(have|eat)|i'?ll\s+(have|eat)|gonna\s+(have|eat)|about\s+to\s+(eat|have))\b",
        r"\b(going\s+to\s+eat|planning\s+to\s+(eat|have)|will\s+(have|eat))\b",
        r"\bhaving\b.{1,60}\bfor\s+(breakfast|lunch|dinner|snack|pre[\s-]?workout|post[\s-]?workout)\b",
        r"\b(eat|have|eating|having|ate|had|log|logging|consume|consumed|consuming)\b.{0,40}\d+\s*(?:gms?|g|grams?|kg)\b",
        r"\d+\s*(?:gms?|g|grams?|kg)\b.{0,40}\b(eat|have|eating|having|ate|had|log|logging|consume|consumed|consuming)\b",
        # Any message that has a food word + a quantity anywhere (e.g. "i ate 100gm paneer")
        r"\b(ate|had|eat|have|eating|having|log|logging)\b.{0,60}\d+\s*(?:gms?|g|grams?|kg|ml|cups?|pieces?|slices?)\b",
        r"\d+\s*(?:gms?|g|grams?|kg|ml|cups?|pieces?|slices?)\b.{0,60}\b(ate|had|eat|have|eating|having|log|logging)\b",
        # Standalone quantity reply — "100gm", "200g", "3 pieces" —
        # when user answers a coach clarification question about quantity.
        r"^\d+\s*(?:gms?|g|grams?|kg|ml|cups?|pieces?|slices?|bowls?|plates?|scoops?|servings?|tbsp|tsp|handful)s?\b",
    ]
    if any(re.search(p, text) for p in pending_food_patterns):
        return "pending_food_log"

    # Exercise swap — user wants to change or replace an exercise
    exercise_swap_patterns = [
        r"\bswap\b.{0,50}\b(exercise|movement|squat|bench|deadlift|row|curl|press|lunge|pull|push|dip|fly|crunch|it|this|out)\b",
        r"\b(squat|bench|deadlift|row|curl|press|lunge|pull|push|dip|fly|crunch)\b.{0,40}\bswap\b",
        r"\breplace\b.{0,40}\b(exercise|movement|squat|bench|deadlift|row|curl|press|lunge|pull|push|dip|this|it)\b",
        r"\bsubstitute\b.{0,40}\b(exercise|squat|bench|deadlift|row|curl|press|lunge|pull|push|dip|for)\b",
        r"\balternative\b.{0,40}\b(exercise|squat|bench|deadlift|row|curl|press|lunge|pull|push|dip|to|for)\b",
        r"don'?t\s+want\s+to\s+do\b",
        r"can'?t\s+do\b.{0,30}\b(squat|bench|deadlift|row|curl|press|lunge|pull|push|dip|this|them|it)\b",
        r"\bskip\b.{0,30}\b(squat|bench|deadlift|row|curl|press|lunge|pull|push|dip|this\s+exercise|the\s+exercise)\b",
        r"\bwhat\s+(?:can|should)\s+i\s+(?:do|use)\s+instead\b",
        r"\bchange\b.{0,30}\b(exercise|movement|squat|bench|deadlift|row|curl|press|this)\b",
    ]
    if any(re.search(p, text) for p in exercise_swap_patterns):
        return "exercise_swap"

    # Confirm — user approving a pending log
    if re.search(
        r"^(yes|yep|yeah|yup|yh|ok|okay|sure|correct|confirm|log it|log that|do it|go ahead|sounds good|looks good|that\'?s right)\.?$",
        text,
    ):
        return "confirm"

    # Cancel — user rejecting a pending log
    if re.search(
        r"^(no|nope|nah|cancel|wrong|nevermind|never mind|don\'?t|stop|abort|discard)\.?$",
        text,
    ):
        return "cancel"

    # Check-in — progress update or emotional state share
    check_in_patterns = [
        r"\b(how\s+am\s+i\s+doing|how\'?s\s+my\s+(progress|journey|streak)|checking?\s+in)\b",
        r"\b(progress\s+update|am\s+i\s+on\s+track|on\s+track)\b",
        r"\b(feeling\s+(good|great|tired|sore|strong|weak|motivated|unmotivated|amazing|terrible))\b",
        r"\b(wanted\s+to\s+(share|tell|let\s+you\s+know)|update\s+(you|on\s+my))\b",
    ]
    if any(re.search(p, text) for p in check_in_patterns):
        return "check_in"

    # Question — default for anything fitness/nutrition related
    if re.search(
        r"\b(how|what|why|when|should|can\s+i|which|is\s+it|best\s+way|help|advice|suggest|recommend|tell\s+me)\b",
        text,
    ):
        return "question"

    # Fallback — treat as a general question rather than silently failing
    return "question"


# ─── GUARDRAIL RESPONSES ───────────────────────────────────────────────────────
_CRISIS_RESPONSE = (
    "Hey — I hear you, and I want you to know that what you're feeling matters. "
    "I'm just a fitness coach and I'm not equipped to help with this, "
    "but please reach out to someone who is:\n\n"
    "🇮🇳 iCall: 9152987821\n"
    "🇮🇳 Vandrevala Foundation: 1860-2662-345 (24/7)\n"
    "🇮🇳 AASRA: 9820466726\n"
    "🌍 Crisis Text Line: Text HOME to 741741\n\n"
    "You're not alone in this. Please talk to someone today — "
    "a friend, a family member, or one of these helplines. "
    "I'll be right here whenever you're ready to train again. ❤️"
)

# Hard-coded, never model-generated — same reasoning as _CRISIS_RESPONSE.
# Reached only via _is_medical_red_flag, so it fires for genuine warning signs
# (chest pain, numbness, suspected fracture) and not for a sore shoulder.
_MEDICAL_URGENT_RESPONSE = (
    "Stop training and get that looked at — I'm serious.\n\n"
    "What you're describing isn't something I should be coaching you through. "
    "Symptoms like chest pain, breathlessness, numbness or tingling, dizziness, "
    "or a joint that won't take weight need an actual clinician, not a "
    "programming tweak.\n\n"
    "If it's severe, sudden, or getting worse — chest pain, trouble breathing, "
    "or you can't move or feel a limb — treat it as an emergency and get to "
    "urgent care now (dial 112 in India). Otherwise book a doctor or physio "
    "this week and rest the area until then.\n\n"
    "This isn't me brushing you off. Get cleared, tell me what they say, and "
    "I'll rebuild your training around whatever they find. That's a promise."
)

_OOS_RESPONSE = (
    "I'm built to help with your training, nutrition, and recovery — "
    "that's where I can actually move the needle for you. "
    "Got a fitness question? Let's get into it."
)

_RATE_LIMIT_RESPONSE = (
    "I've hit my daily message limit — I process a lot of requests and the free tier caps out. "
    "Try again in an hour and I'll be back at full capacity."
)


# ─── 1. FETCH USER CONTEXT FROM SUPABASE ─────────────────────────────────────
def _maybe_data(result):
    """maybe_single().execute() can return None (no row) in some postgrest
    versions, or a response whose .data is None. Normalise both to None."""
    return result.data if result else None


# ─── ACTIVITY LEVEL / GOAL MAPPING ────────────────────────────────────────────
# Maps the app's user_preferences values → macro calculator inputs

_FITNESS_TO_ACTIVITY = {
    "complete beginner": "sedentary",
    "trained before": "light",
    "somewhat active": "moderate",
    "seriously training": "active",
}

_GOAL_TO_MACRO_GOAL = {
    "lose fat": "lose",
    "build muscle": "gain",
    # Strength = maintenance on purpose (progress comes from the bar, not a
    # surplus) — the onboarding education screen and coaching directives both
    # tell the user this explicitly, so this must stay in sync with
    # GOAL_TO_MACRO in onboardingService.ts or the app contradicts itself.
    "get stronger": "maintain",
    "increase strength": "maintain",  # legacy stored value
    "stay healthy": "maintain",
    "body recomposition": "maintain",
    "improve athletic performance": "maintain",
}


def _auto_calculate_macros(user_id: str, prefs: dict) -> dict | None:
    """
    If the user has weight, height, age, gender, and goal in their preferences
    but no macros saved, calculate and save them automatically.
    Returns the saved macros dict (matching user_macros columns) or None.
    """
    weight = prefs.get("weight")
    height = prefs.get("height")
    age = prefs.get("age")
    gender = (prefs.get("gender") or "").lower().strip()
    goal_raw = (prefs.get("primary_goal") or "").lower().strip()
    fitness_raw = (prefs.get("fitness_level") or "").lower().strip()

    # Need at minimum weight, height, age, and a valid gender
    if not all([weight, height, age]) or gender not in ("male", "female"):
        return None

    activity = _FITNESS_TO_ACTIVITY.get(fitness_raw, "moderate")
    goal = _GOAL_TO_MACRO_GOAL.get(goal_raw, "maintain")

    try:
        from macros_v4_final import calculate_macros_v4, save_macros_to_supabase

        result = calculate_macros_v4(
            weight_kg=float(weight),
            height_cm=float(height),
            age_years=int(age),
            gender=gender,
            activity_level=activity,
            goal=goal,
        )
        save_macros_to_supabase(user_id, result)

        # Return in the same shape as a user_macros row
        t = result["targets"]
        h = result["health_metrics"]
        return {
            "calories": t["calories"],
            "protein_g": t["protein_g"],
            "carbs_g": t["carbs_g"],
            "fat_g": t["fat_g"],
            "water_liters": h["water_liters"],
            "fiber_g": h["fiber_g"],
        }
    except Exception:
        traceback.print_exc()
        return None


def get_user_context(user_id: str) -> dict:
    """
    Pulls everything we know about the user from Supabase.
    This is the RAG part — retrieving relevant context before generation.
    Any single-row lookup that finds nothing yields None instead of raising.
    """

    # Eight independent reads. None of them looks at another's result, but they
    # used to run strictly one after the next — and every one is its own
    # Singapore->Mumbai round trip, so in series they were the largest single
    # block of dead time in a coach turn (the `[chat timing] classify+context`
    # number). Fanning them out makes the whole set cost about one round trip.
    #
    # Exceptions are deliberately NOT swallowed here: .result() re-raises, which
    # is exactly what the serial version did. A failed context read has to break
    # the turn loudly — answering with half the user's data silently is how a
    # coach confidently tells someone the wrong thing.
    today = local_date_str()

    _reads = {
        # User profile
        "user": lambda: _maybe_data(
            supabase.table("profiles")
            .select("*")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        ),
        # Current program and progress
        "program": lambda: _maybe_data(
            supabase.table("user_programs")
            .select("*, programs(*, creators(*))")
            .eq("user_id", user_id)
            .eq("status", "active")
            .maybe_single()
            .execute()
        ),
        # Last 7 weight logs
        "weight_logs": lambda: supabase.table("progress_logs")
        .select("body_weight, logged_at, created_at")
        .eq("user_id", user_id)
        .order("logged_at", desc=True)
        .order("created_at", desc=True)
        .limit(7)
        .execute()
        .data,
        # Personal bests
        "personal_bests": lambda: supabase.table("personal_bests")
        .select("exercise_name, weight_kg, reps")
        .eq("user_id", user_id)
        .order("logged_at", desc=True)
        .limit(5)
        .execute()
        .data,
        # Today's workout completion
        "todays_workout": lambda: supabase.table("workout_logs")
        .select("completed, exercise_data")
        .eq("user_id", user_id)
        .eq("date", today)
        .execute()
        .data,
        # Streak
        "streak": lambda: _maybe_data(
            supabase.table("user_streaks")
            .select("current_streak")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        ),
        # Macro targets (from macro calculator output stored at onboarding)
        "macros": lambda: _maybe_data(
            supabase.table("user_macros")
            .select("*")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        ),
        # User preferences (dislikes, dietary restrictions)
        "preferences": lambda: _maybe_data(
            supabase.table("user_preferences")
            .select("*")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        ),
    }

    with ThreadPoolExecutor(max_workers=len(_reads)) as _pool:
        _futures = {k: _pool.submit(fn) for k, fn in _reads.items()}
        _out = {k: f.result() for k, f in _futures.items()}

    user = _out["user"]
    user_program = _out["program"]
    weight_logs = _out["weight_logs"]
    personal_bests = _out["personal_bests"]
    todays_workout = _out["todays_workout"]
    streak = _out["streak"]
    macros = _out["macros"]
    preferences = _out["preferences"]

    # Auto-calculate macros if profile is complete but macros haven't been set.
    # Stays sequential on purpose — it reads both results above.
    if not macros and preferences:
        macros = _auto_calculate_macros(user_id, preferences)

    return {
        "user": user,
        "program": user_program,
        "weight_logs": weight_logs,
        "personal_bests": personal_bests,
        "todays_workout": todays_workout,
        "streak": streak.get("current_streak", 0) if streak else 0,
        "macros": macros,
        "preferences": preferences,
    }


# ─── 2. GET CREATOR COACHING STYLE ───────────────────────────────────────────
def get_creator_context(creator_id: str) -> dict | None:
    """
    Fetches the creator's coaching philosophy and persona.
    This shapes how the AI coach responds — their tone and style.
    Returns None if the creator row doesn't exist.

    Also merges the structured ideology captured in the creator questionnaire
    (creator_profiles.ideology) so the coach reasons from the creator's actual
    personality, scenarios and non-negotiables, not just the free-text prose.
    """
    row = _maybe_data(
        supabase.table("creators")
        .select("name, coaching_philosophy, diet_philosophy, tone, story")
        .eq("id", creator_id)
        .maybe_single()
        .execute()
    )
    if not row:
        return None
    try:
        prof = _maybe_data(
            supabase.table("creator_profiles")
            .select("ideology, movement_bias, food_style, meal_structure")
            .eq("creator_id", creator_id)
            .maybe_single()
            .execute()
        )
        if prof:
            row["ideology"] = prof.get("ideology") or {}
            row["movement_bias"] = prof.get("movement_bias")
            row["food_style"] = prof.get("food_style")
            row["meal_structure"] = prof.get("meal_structure")
    except Exception:
        pass
    return row


# Renders the structured creator ideology into a compact prompt block the coach
# reasons from. Returns "" when nothing is set, so an unconfigured creator's
# prompt is unchanged. Keeps only fields that carry real signal.
def render_creator_ideology(creator: dict) -> str:
    ideo = creator.get("ideology") or {}
    if not ideo and not creator.get("movement_bias"):
        return ""
    lines: list[str] = []

    def _join(v):
        return ", ".join(v) if isinstance(v, list) else str(v)

    labelled = [
        ("personality",         "Personality"),
        ("training_school",     "Training school"),
        ("signature_phrases",   "Signature phrases"),
        ("foods_push_avoid",    "Foods pushed / avoided"),
        ("beginner_vs_advanced","Beginner vs advanced"),
        ("plateau",             "On a plateau"),
        ("fast_fat_loss",       "On fast fat loss"),
        ("non_negotiables",     "Always (non-negotiable)"),
        ("never_do",            "Never program/recommend"),
        ("defer_to_pro",        "Defers to a professional on"),
    ]
    for key, label in labelled:
        val = ideo.get(key)
        if val:
            lines.append(f"- {label}: {_join(val)}")
    style_bits = [b for b in [
        creator.get("movement_bias") and f"exercise: {creator['movement_bias']}",
        creator.get("food_style") and f"food: {creator['food_style']}",
        creator.get("meal_structure") and creator["meal_structure"] != "auto" and f"meals: {creator['meal_structure']}",
    ] if b]
    if style_bits:
        lines.append(f"- Programming lean: {', '.join(style_bits)}")
    if not lines:
        return ""
    return "COACH IDEOLOGY (reason and advise the way this coach would):\n" + "\n".join(lines)


# ─── 3. GET EXERCISE INFO FROM LIBRARY ───────────────────────────────────────
def get_exercise_info(exercise_name: str) -> dict | None:
    """
    Looks up a single exercise in the exercises_library table.
    Uses ilike so capitalisation differences don't cause misses.
    Returns the row dict or None if not found.
    """
    result = (
        supabase.table("exercises_library")
        .select(
            "id, slug, name, primary_muscles, secondary_muscles, subgroup, equipment, "
            "movement_type, bar_path, rom_profile, stabilizer_demand, contraindicated_for"
        )
        .ilike("name", exercise_name)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


# ─── 4. FOOD NUTRITION LOOKUP ─────────────────────────────────────────────────
def get_food_nutrition(food_name: str) -> dict | None:
    """
    Looks up a food by name in the `foods` table (USDA + Indian foods, ~11.7k rows).

    The table stores macros as `protein_g` / `carbs_g` / `fat_g` / `fiber_g`
    (all per 100g). Callers expect the legacy `*_per_100g` key names, so we
    normalise on the way out — that keeps every consumer working unchanged.

    `basis` says what those per-100g numbers describe: 'cooked' (as eaten),
    'dry' (as sold), 'as_is', or None if the row hasn't been reviewed. It is
    surfaced so the coach can SAY which form it used — a user who sees
    "Rice, cooked" can catch a mistake that a bare number hides.

    Returns: {name, calories_per_100g, protein_per_100g, carbs_per_100g,
               fat_per_100g, fiber_per_100g, serving_size_g, basis, source}
             or None.
    """
    cleaned = food_name.strip()
    if not cleaned:
        return None

    # This used to be ilike('%x%').limit(1) with NO ordering — the row you got
    # was whatever Postgres returned first, so "rice" resolved to "Flour, rice,
    # white, unenriched" (359 kcal/100g) and logged 718 kcal for a 200g plate of
    # rice. Wrong food matched = wrong macros logged, and that number then drives
    # today's intake, the protein gap, and the adaptive-macro rule.
    #
    # Ordering the fetch by is_priority is load-bearing, not a nicety: the LIMIT
    # is applied BEFORE ranking, so an unordered fetch can leave the correct row
    # outside the candidate window entirely and no ranking can recover it.
    result = (
        supabase.table("foods")
        .select(
            "name, calories_per_100g, protein_g, carbs_g, fat_g, fiber_g, "
            "serving_size_g, basis, is_priority, image_url"
        )
        .ilike("name", f"%{cleaned}%")
        .order("is_priority", desc=True)
        .limit(40)
        .execute()
    )
    if not result.data:
        return None

    lowered = cleaned.lower()

    def _rank(row: dict) -> tuple:
        name = (row.get("name") or "").lower()
        # What FOLLOWS the match is the signal. In USDA-style naming a comma
        # means the same food, qualified ("Rice, white, cooked"), while another
        # word means a DIFFERENT food ("Rice bran oil"). Ranking on name length
        # alone gets this exactly backwards and picks the oil.
        rest = name[len(lowered):] if name.startswith(lowered) else None
        qualified = rest is not None and (rest == "" or rest[0] == ",")
        return (
            name != lowered,                # 1. exact match wins outright
            not qualified,                  # 2. "Rice, white, cooked" over "Rice bran oil"
            rest is None,                   # 3. any prefix match over a mid-string one
            not row.get("is_priority"),     # 4. curated rows over bulk-imported ones
            len(name),                      # 5. tie-break: the least-qualified name
        )

    row = min(result.data, key=_rank)
    return {
        "name": row.get("name"),
        "calories_per_100g": row.get("calories_per_100g"),
        "protein_per_100g": row.get("protein_g"),
        "carbs_per_100g": row.get("carbs_g"),
        "fat_per_100g": row.get("fat_g"),
        "fiber_per_100g": row.get("fiber_g"),
        "serving_size_g": row.get("serving_size_g"),
        "basis": row.get("basis"),
        "image_url": row.get("image_url"),
        "source": "foods",
    }


# How a food's per-100g row should be described to the user. Silent basis is
# what made "200g rice" log 718 kcal; naming it makes the mistake catchable.
_BASIS_LABEL = {
    "cooked": "cooked / as eaten",
    "dry": "dry / uncooked weight",
    "as_is": "as eaten",
}


def _compile_tag(tag: str) -> re.Pattern:
    """
    Matcher for one knowledge tag: anchored at a word boundary on BOTH sides,
    but tolerating a common inflection so "plateau" still matches "plateaued"
    and "squat" matches "squats".

    The suffix list is deliberately closed rather than \\w* — \\w* would let
    "fat" match "fatigue" and put fat-loss facts under a fatigue question.
    """
    return re.compile(rf"\b{re.escape(tag.lower())}(?:s|es|ed|ing|d)?\b")


# Precompiled once at import — the knowledge base is static, and this runs on
# every question/medical turn.
#
# Word boundaries are not cosmetic here. Plain substring matching (the previous
# behaviour) meant short tags matched INSIDE unrelated words: "lat" hit
# "p-lat-eaued", "row" hit "g-row" — so a question about a plateau or about
# growing a muscle silently retrieved back anatomy and shipped it to the model
# labelled "VERIFIED BIOMECHANICS". Lowercasing the tag matters too: the query
# is already lowered, so a tag with a capital could never match at all.
_KNOWLEDGE_INDEX: list[tuple[dict, list[re.Pattern], re.Pattern]] = [
    (
        chunk,
        [_compile_tag(t) for t in chunk["tags"]],
        _compile_tag(chunk["category"].replace("anatomy_", "").replace("_", " ")),
    )
    for chunk in GYM_KNOWLEDGE
]


def search_gym_knowledge(query: str, top_k: int = 3) -> list[str]:
    """
    In-memory keyword-scored retrieval from GYM_KNOWLEDGE.
    Scores each chunk by counting how many of its tags appear in the query,
    plus a bonus if the category keyword matches. Zero external API calls.
    Returns content strings for the top_k most relevant chunks (score > 0).
    """
    q = query.lower()
    scored: list[tuple[int, str]] = []
    for chunk, tag_res, cat_re in _KNOWLEDGE_INDEX:
        score = sum(1 for pattern in tag_res if pattern.search(q))
        # Category bonus: e.g. "anatomy_chest" → "chest" → +2 if "chest" in query
        if cat_re.search(q):
            score += 2
        if score > 0:
            scored.append((score, chunk["content"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [content for _, content in scored[:top_k]]


def _build_nutrition_block(message: str, force_lookup: bool = False) -> str:
    """
    Returns a nutrition context block to inject into the system prompt.

    Priority:
      1. If the user explicitly stated macro values ("it has 50g protein") →
         return a USER-STATED block; never contradict with DB values.
      2. If the message is a food question OR force_lookup is True →
         look up DB and return FOOD NUTRITION DATA block.
      3. Otherwise → return empty string.

    force_lookup should be True for pending_food_log intent so that
    "I ate 200g rice and 150g rajma" still fetches real nutrition data
    even though it contains no question keywords.
    """
    stated = _extract_stated_macros(message)
    if stated:
        parts = []
        if "calories" in stated:
            parts.append(f"{stated['calories']} kcal")
        if "protein_g" in stated:
            parts.append(f"{stated['protein_g']}g protein")
        if "carbs_g" in stated:
            parts.append(f"{stated['carbs_g']}g carbs")
        if "fat_g" in stated:
            parts.append(f"{stated['fat_g']}g fat")
        if "fiber_g" in stated:
            parts.append(f"{stated['fiber_g']}g fiber")

        # Also look up the base food so we can fill in any missing macros.
        # e.g. user says "high protein paneer 50g protein" → look up "paneer" for carbs/fat/calories.
        # Sort by length descending (longer = more specific) and skip short filler words
        # to avoid false matches like "ate" → pomegranate juice, "high" → chicken thigh
        candidates = sorted(_extract_food_candidates(message), key=len, reverse=True)
        base_food = None
        for candidate in candidates:
            clean = candidate.strip()
            if len(clean) < 4 or clean.replace(".", "").isdigit():
                continue
            base_food = get_food_nutrition(clean)
            if base_food:
                break

        base_block = ""
        if base_food:
            kcal = base_food.get("calories_per_100g") or "?"
            prot = base_food.get("protein_per_100g") or "?"
            carbs = base_food.get("carbs_per_100g") or "?"
            fat = base_food.get("fat_per_100g") or "?"
            base_block = (
                f"\nBASE FOOD DATA for fallback (per 100g from database): "
                f"{base_food['name']}: {kcal} kcal | {prot}g protein | {carbs}g carbs | {fat}g fat"
            )

        base_name = base_food["name"] if base_food else "equivalent food"
        return (
            "USER-STATED NUTRITIONAL VALUES (specialty or custom product):\n"
            f"  User stated: {', '.join(parts)}\n"
            f"{base_block}\n"
            "INSTRUCTIONS: Use the user-stated values for any macros they specified. "
            "For any macros NOT stated by the user (e.g. carbs, fat, calories), "
            "use the BASE FOOD DATA above scaled to the portion size if available - "
            "do NOT log them as unknown or null. "
            f"Acknowledge this is a specialty product and that non-stated values are based on standard {base_name}."
        )

    if not _is_food_question(message) and not force_lookup:
        return ""

    candidates = _extract_food_candidates(message)
    found: dict[str, dict] = {}  # keyed by DB name to deduplicate

    for candidate in candidates:
        if len(found) >= 5:
            break
        # Skip if we already have a result whose name contains this candidate
        already_covered = any(candidate in key.lower() for key in found)
        if already_covered:
            continue
        result = get_food_nutrition(candidate)
        if result:
            found[result["name"].lower()] = result

    if not found:
        return ""

    lines = ["FOOD NUTRITION DATA (per 100g, verified from USDA/Indian food database):"]
    any_dry = False
    for food in found.values():
        kcal = food.get("calories_per_100g") or "?"
        prot = food.get("protein_per_100g") or "?"
        carbs = food.get("carbs_per_100g") or "?"
        fat = food.get("fat_per_100g") or "?"
        fiber = food.get("fiber_per_100g")
        fiber_str = f", {fiber}g fiber" if fiber else ""
        basis = food.get("basis")
        if basis == "dry":
            any_dry = True
        basis_str = f" [{_BASIS_LABEL[basis]}]" if basis in _BASIS_LABEL else ""
        lines.append(
            f"  - {food['name']}{basis_str}: {kcal} kcal | {prot}g protein | "
            f"{carbs}g carbs | {fat}g fat{fiber_str}"
        )
    lines.append(
        "MANDATORY: Use the exact numbers above in your answer. "
        "Do not estimate or use generic values when verified data is available."
    )
    if any_dry:
        # Cooking absorbs water: 100g of dry rice becomes ~250-300g cooked. If the
        # user means the cooked plate and we scale a dry row, we overstate by ~3x.
        lines.append(
            "BASIS WARNING: an entry above is marked [dry / uncooked weight]. A quantity "
            "the user gives for a cooked portion must NOT be scaled against it — say which "
            "form you used (e.g. \"assuming 60g dry oats, not cooked weight\") and ask them "
            "to confirm if it's ambiguous."
        )
    return "\n".join(lines)


def _join(val) -> str:
    """Join list values to comma-separated string, or return the string as-is."""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return val or ""


# The eight values muscle_group actually holds — every searchable row has one,
# unlike primary_muscles which is granular (front_delt / side_delt / rear_delt)
# and made a `cs.{"shoulders"}` filter miss the very rows it was aiming at.
_MUSCLE_GROUPS = {"chest", "back", "shoulders", "legs", "arms", "core", "cardio", "full_body"}

# How people ask, mapped to those eight.
_MUSCLE_SYNONYMS = {
    "shoulder": "shoulders", "delt": "shoulders", "delts": "shoulders", "traps": "shoulders",
    "chest": "chest", "pec": "chest", "pecs": "chest",
    "back": "back", "lat": "back", "lats": "back", "vtaper": "back", "v-taper": "back",
    "arm": "arms", "arms": "arms", "bicep": "arms", "biceps": "arms",
    "tricep": "arms", "triceps": "arms", "forearm": "arms",
    "leg": "legs", "legs": "legs", "quad": "legs", "quads": "legs", "glute": "legs",
    "glutes": "legs", "hamstring": "legs", "hamstrings": "legs", "calf": "legs",
    "calves": "legs", "squat": "legs",
    "core": "core", "abs": "core", "ab": "core", "oblique": "core", "obliques": "core",
    "six pack": "core", "cardio": "cardio", "conditioning": "cardio",
    "full body": "full_body",
}


def _build_exercise_catalog(message: str, user_equipment: str = "") -> str:
    """Real exercise ids for the muscle the user asked about.

    This used to run `.limit(5)` with NO ordering, so Postgres returned whatever
    five rows it felt like: "how do i get wider shoulders" produced One-Arm
    Kettlebell Para Press and Handstand Push-Ups out of 126 shoulder exercises.
    The model — correctly — ignored that and recommended lateral raises from its
    own knowledge, which is precisely how it ended up inventing ids for
    exercises the catalog never listed.

    Three things were wrong and all three mattered:
      * no ORDER BY, so the five rows were arbitrary
      * no is_searchable filter, so hidden bulk-import rows could surface
      * primary_muscles.cs.{"shoulders"} never matched, because that column
        stores front_delt / side_delt / rear_delt

    Ranking is best-effort by necessity: rank_gym is populated on 24 of 828
    rows. Where it exists it is excellent (Overhead Press, Dumbbell Shoulder
    Press, Arnold Press). Everything else falls back to compound-first, then
    shortest name.
    """
    # Singularise both sides, exactly like _find_exercise_in_text. A plain
    # \bshoulder\b does NOT match "shoulders", which is how the most common
    # phrasing of all — "how do i get wider shoulders" — matched nothing and
    # returned an empty catalog.
    hay = " " + _singular(re.findall(r"[a-z]+", (message or "").lower())) + " "
    target = next(
        (grp for word, grp in sorted(_MUSCLE_SYNONYMS.items(), key=lambda kv: -len(kv[0]))
         if f" {_singular(word.split())} " in hay),
        None,
    )
    if not target or target not in _MUSCLE_GROUPS:
        return ""

    try:
        res = (
            supabase.table("exercises_library")
            .select("id, slug, name, primary_muscles, equipment, is_compound, "
                    "rank_gym, rank_gym_subgroup, rank_home, rank_home_subgroup")
            .eq("is_searchable", True)
            .eq("muscle_group", target)
            .limit(120)
            .execute()
        )
        rows = res.data or []
    except Exception:
        logger.warning("[catalog] lookup failed for %r", target, exc_info=True)
        return ""

    if user_equipment:
        usable = [r for r in rows if _equipment_allowed(r.get("equipment"), user_equipment)]
        # Sorting rather than hard-filtering everywhere else in this file, but
        # here an unusable id is worse than none: the model would recommend it
        # and the user cannot do it. Only fall back if NOTHING is usable.
        rows = usable or rows

    # ponytail: name length as a proxy for how canonical an exercise is —
    # "Overhead Press" beats "Standing Dumbbell Straight-Arm Front Delt Raise
    # Above Head". Crude, but it correlates well in this library and needs no
    # new data. Upgrade path: backfill rank_gym past its current 24 rows and
    # this tier stops mattering.
    # rank_gym is populated on 24 of 828 rows; rank_gym_subgroup on 429 — 18x
    # denser and per-subgroup, which is exactly the granularity this needs. The
    # first version sorted on rank_gym then NAME LENGTH, so with rank almost
    # always null it degenerated to shortest-name and offered "Crucifix",
    # "Circus Bell" and "Arm Circles" for shoulders while Dumbbell Lateral Raise
    # — rank 1 in side_delt — sat unread in a column nothing queried.
    home = "home" in (user_equipment or "").lower()

    def _quality(r):
        primary = r.get("rank_home") if home else r.get("rank_gym")
        sub = r.get("rank_home_subgroup") if home else r.get("rank_gym_subgroup")
        return (
            primary is None, primary or 0,
            sub is None, sub or 0,
            # ponytail: name length is the last resort for rows with no rank at
            # all. It is a weak proxy and it is why the isolation picks were
            # junk before the subgroup rank went in — it only decides ties among
            # rows nobody has ranked. Backfill rank_*_subgroup to retire it.
            len(r.get("name") or ""),
        )

    # Compounds and isolation are quota'd separately rather than sorted together.
    # Sorting compound-first gave all 12 shoulder slots to presses — there are 60
    # compound shoulder exercises — so a question about shoulder WIDTH came back
    # without a single lateral raise, the one movement that actually builds it.
    # The model needs both shapes available to answer well.
    compound = sorted([r for r in rows if r.get("is_compound")], key=_quality)[:8]
    isolation = sorted([r for r in rows if not r.get("is_compound")], key=_quality)[:6]
    rows = (compound + isolation)[:14]
    if not rows:
        return ""

    lines = [
        f"EXERCISE CATALOG — {target} (these are the ONLY valid ids; "
        "recommend from this list and quote the id exactly):"
    ]
    for ex in rows:
        lines.append(
            f"  - {ex['name']} | {ex['id']} | {ex['slug']} | "
            f"{_join(ex.get('primary_muscles'))} | {ex.get('equipment', '?')}"
        )
    return "\n".join(lines)


# ─── 5. LOGGING HELPERS ───────────────────────────────────────────────────────
def _parse_structured_response(raw: str) -> tuple[str, dict | None, dict | None]:
    """
    Extracts structured data from the LLM response.
    Returns: (clean_text, action_json, metadata_json)
    """
    clean_text = raw
    metadata_json = None
    action_json = None

    # 1. Safely slice out and process METADATA blocks
    if "<METADATA_START>" in clean_text:
        try:
            parts = clean_text.split("<METADATA_START>", 1)
            before = parts[0]
            if "<METADATA_END>" in parts[1]:
                sub_parts = parts[1].split("<METADATA_END>", 1)
                json_str = sub_parts[0].strip()
                after = sub_parts[1]
                clean_text = (before + after).strip()
                try:
                    metadata_json = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            else:
                clean_text = before.strip()
        except Exception:
            clean_text = re.sub(
                r"<METADATA_START>[\s\S]*?(?:<METADATA_END>|$)", "", clean_text
            ).strip()
    else:
        # Failsafe regex
        clean_text = re.sub(
            r"<METADATA_START>[\s\S]*?(?:<METADATA_END>|$)", "", clean_text
        ).strip()

    # 2. Extract standard JSON actions (for logging intents)
    cleaned_for_action = re.sub(r"```(?:json)?", "", clean_text).strip()
    cleaned_for_action = cleaned_for_action.replace("{{", "{").replace("}}", "}")

    match = re.search(r"\{[\s\S]+\}", cleaned_for_action)
    if match:
        try:
            action_json = json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return clean_text, action_json, metadata_json


def check_and_update_pb(
    user_id: str, exercise_name: str, weight: float, reps: int
) -> dict:
    """
    Checks whether weight is a new PB for this exercise.
    Upserts personal_bests if so. Returns {"is_new_pb": bool}.
    """
    canonical = _normalise_exercise(exercise_name)
    result = (
        supabase.table("personal_bests")
        .select("id, weight_kg, reps")
        .eq("user_id", user_id)
        .ilike("exercise_name", canonical)
        .limit(1)
        .execute()
    )

    existing = result.data[0] if result.data else None
    now = utc_now_iso()
    existing_weight = float(existing["weight_kg"]) if existing else None

    # Only a new PB when weight is strictly greater than the current record
    is_new = existing_weight is None or float(weight) > existing_weight
    if is_new:
        if existing:
            supabase.table("personal_bests").update(
                {"weight_kg": weight, "reps": reps, "logged_at": now}
            ).eq("id", existing["id"]).execute()
        else:
            supabase.table("personal_bests").insert(
                {
                    "user_id": user_id,
                    "exercise_name": canonical,
                    "weight_kg": weight,
                    "reps": reps,
                    "logged_at": now,
                }
            ).execute()
        return {"is_new_pb": True}

    return {"is_new_pb": False}


def _to_float(val) -> float | None:
    """Coerce model output to float. Handles strings like '260.0 kcal' or full calculation text.
    Takes the last number found — models put the final result at the end of calculation strings."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    matches = re.findall(r"\d+(?:\.\d+)?", str(val))
    return float(matches[-1]) if matches else None


def _ground_food_macros(actions: list, user_message: str) -> list:
    """
    Recomputes log_food macros in code from the foods table rather than trusting
    the model's in-prompt arithmetic (quantity_g / 100 * per_100g).

    A wrong number here doesn't just show once — it lands in food_logs and then
    silently drives today's intake, the protein gap, and the adaptive-macro
    plateau rule. Deterministic maths where we can do it.

    Deliberately conservative — it only overrides when BOTH hold:
      • the DB row is confidently the same food (exact or prefix name match), and
      • the user didn't state that macro themselves.
    Anything less confident keeps the model's estimate, which is the right call
    for compound dishes ("egg curry") that have no clean row to scale from.
    """
    stated = _extract_stated_macros(user_message) or {}
    for a in actions:
        if a.get("type") != "log_food":
            continue
        qty = _to_float(a.get("quantity_g"))
        if not qty or qty <= 0:
            continue
        asked = (a.get("food_name") or "").strip().lower()
        if not asked:
            continue
        food = get_food_nutrition(asked)
        if not food:
            continue
        name = (food.get("name") or "").lower()
        if name != asked and not name.startswith(asked):
            continue  # not confidently the same food — leave the estimate alone

        factor = qty / 100.0
        for key, per100_key in (
            ("calories", "calories_per_100g"),
            ("protein_g", "protein_per_100g"),
            ("carbs_g", "carbs_per_100g"),
            ("fat_g", "fat_per_100g"),
        ):
            if key in stated:
                continue  # the user's own numbers win — specialty products etc.
            per100 = food.get(per100_key)
            if per100 is None:
                continue
            try:
                a[key] = round(float(per100) * factor, 1)
            except (TypeError, ValueError):
                continue
    return actions


def _resolve_food_log_removal(user_id: str, food_name: str) -> dict | None:
    """The food_logs row a delete_food_log action targets: the most recent entry
    today matching food_name (ILIKE), or the single most recent if no name given
    ("undo that"). Shared by the text-confirm path and the correction card so
    both act on the same row."""
    q = (
        supabase.table("food_logs")
        .select("id, food_name, quantity_g, calories, meal_type")
        .eq("user_id", user_id)
        .eq("logged_at", local_date_str())
    )
    name = (food_name or "").strip()
    if name:
        q = q.ilike("food_name", f"%{name}%")
    rows = (q.order("created_at", desc=True).limit(1).execute().data) or []
    return rows[0] if rows else None


def execute_actions(user_id: str, actions: list) -> dict:
    """
    Executes structured actions returned by the model.
    Returns {"new_pbs": [list of exercise names that hit a new PB]}.
    """
    results = {"new_pbs": []}
    today = local_date_str()
    now = utc_now_iso()

    for action in actions:
        atype = action.get("type")

        if atype == "log_workout":
            exercise = _normalise_exercise(action.get("exercise", ""))
            sets = action.get("sets", [])
            existing = (
                supabase.table("workout_logs")
                .select("id, exercise_data, exercises_done")
                .eq("user_id", user_id)
                .eq("date", today)
                .limit(1)
                .execute()
            )
            new_entry = {"name": exercise, "sets": sets}
            if existing.data:
                row = existing.data[0]
                merged = (row.get("exercise_data") or []) + [new_entry]
                supabase.table("workout_logs").update(
                    {
                        "exercise_data": merged,
                        "exercises_done": (row.get("exercises_done") or 0) + 1,
                    }
                ).eq("id", row["id"]).execute()
            else:
                supabase.table("workout_logs").insert(
                    {
                        "user_id": user_id,
                        "date": today,
                        "completed": True,
                        "exercises_done": 1,
                        "exercise_data": [new_entry],
                        "created_at": now,
                    }
                ).execute()
            if sets:
                heaviest = max(sets, key=lambda s: s.get("weight", 0))
                pb = check_and_update_pb(
                    user_id,
                    exercise,
                    heaviest.get("weight", 0),
                    heaviest.get("reps", 0),
                )
                if pb["is_new_pb"] and exercise not in results["new_pbs"]:
                    results["new_pbs"].append(exercise)

        elif atype == "check_pb":
            exercise = _normalise_exercise(action.get("exercise", ""))
            pb = check_and_update_pb(
                user_id, exercise, action.get("weight", 0), action.get("reps", 0)
            )
            if pb["is_new_pb"] and exercise not in results["new_pbs"]:
                results["new_pbs"].append(exercise)

        elif atype == "update_target_weights":
            # Progressive-overload bump — values were computed deterministically
            # before the user was asked to confirm (see the increase_weight path).
            execute_target_weight_updates(user_id, action.get("updates") or [])

        elif atype == "log_weight":
            # upsert handles the unique (user_id, logged_at) constraint —
            # logging weight twice on the same day updates rather than crashes
            supabase.table("progress_logs").upsert(
                {
                    "user_id": user_id,
                    "body_weight": action.get("value", 0),
                    "logged_at": today,
                    "created_at": now,
                },
                on_conflict="user_id,logged_at",
            ).execute()

        elif atype == "log_food":
            print(
                f"\n[log_food] raw action from model: {json.dumps(action, default=str)}"
            )
            _valid_meal_types = {"breakfast", "morning_snack", "lunch", "evening_snack", "dinner"}
            _meal_type = action.get("meal_type")
            payload = {
                "user_id": user_id,
                "food_name": action.get("food_name") or "",
                "quantity_g": _to_float(action.get("quantity_g")),
                "calories": _to_float(action.get("calories")),
                "protein_g": _to_float(action.get("protein_g")),
                "carbs_g": _to_float(action.get("carbs_g")),
                "fat_g": _to_float(action.get("fat_g")),
                "meal_type": _meal_type if _meal_type in _valid_meal_types else None,
                "logged_at": today,
                "created_at": now,
            }
            print(f"[log_food] inserting: {json.dumps(payload, default=str)}")
            try:
                insert_result = supabase.table("food_logs").insert(payload).execute()
                print(
                    f"[log_food] SUCCESS — inserted {len(insert_result.data or [])} row(s): {insert_result.data}"
                )
            except Exception as insert_err:
                print(f"[log_food] FAILED — error: {insert_err}")
                raise

        elif atype == "delete_food_log":
            # Remove the row the action targets (most recent match today). Shared
            # resolver so the text-confirm path and the card path agree on which
            # row that is.
            row = _resolve_food_log_removal(user_id, action.get("food_name") or "")
            if row:
                supabase.table("food_logs").delete().eq("id", row["id"]).execute()
                results.setdefault("deleted", []).append(row.get("food_name") or "")

    # Everything above changes numbers the coach quotes back ("today so far",
    # PBs, weight trend) — and it quotes them from the cached context slices.
    # The tap-to-confirm card path invalidates via /invalidate-context, but this
    # (the "say yes" text path) never did: the very next message answered from
    # pre-log data, i.e. the coach forgetting the meal it had just logged.
    if actions:
        try:
            invalidate_slice(user_id, "all")
        except Exception:
            logger.warning(
                "execute_actions: context invalidation failed for %s", user_id, exc_info=True
            )

    return results


# ─── TOOL CALL EXECUTION (writes to Supabase after user confirms) ─────────────

def _fetch_current_program_structure(user_id: str) -> dict | None:
    """
    Returns the user's active program with all workouts and exercises,
    formatted as a dict matching the propose_workout_program tool schema.
    Returns None if the user has no active program.
    """
    try:
        up = _maybe_data(
            supabase.table("user_programs")
            .select("program_id, programs(id, title, difficulty, goal, duration_weeks)")
            .eq("user_id", user_id)
            .eq("status", "active")
            .maybe_single()
            .execute()
        )
        if not up or not up.get("programs"):
            return None

        prog = up["programs"]
        prog_id = prog["id"]

        workouts = (
            supabase.table("workouts")
            .select("id, day_number, title, is_rest_day")
            .eq("program_id", prog_id)
            .order("day_number")
            .execute()
            .data or []
        )

        days = []
        for w in workouts:
            exercises = []
            if not w.get("is_rest_day"):
                ex_rows = (
                    supabase.table("exercises")
                    .select("name, sets, reps, rest_seconds, notes")
                    .eq("workout_id", w["id"])
                    .order("order_index")
                    .execute()
                    .data or []
                )
                exercises = [
                    {
                        "name": e["name"],
                        "sets": e.get("sets", 3),
                        "reps": e.get("reps", "10/10/10"),
                        "rest_seconds": e.get("rest_seconds", 90),
                        "notes": e.get("notes", ""),
                    }
                    for e in ex_rows
                ]
            days.append({
                "day_number": w["day_number"],
                "title": w["title"],
                "is_rest_day": w.get("is_rest_day", False),
                "exercises": exercises,
            })

        return {
            "title": prog.get("title", "Current Program"),
            "difficulty": prog.get("difficulty", "intermediate"),
            "goal": prog.get("goal", "maintain"),
            "duration_weeks": prog.get("duration_weeks", 8),
            "coaching_rationale": "",
            "days": days,
        }
    except Exception:
        traceback.print_exc()
        return None


def execute_target_weight_updates(user_id: str, updates: list) -> dict:
    """
    Writes new target_weight values onto the user's active program.

    updates: [{"exercise_name": str, "new_weight_kg": float}]

    This exists because "increase my weights" used to be answered by handing the
    already-computed numbers to the LLM and asking it to re-emit the ENTIRE
    program — slow, and free to drift on exercises it was told not to touch. We
    know the exact rows and the exact values; there is nothing here for a model
    to decide.

    Returns {"updated": [names], "missed": [names]}.
    """
    result = {"updated": [], "missed": []}
    if not updates:
        return result

    up = _maybe_data(
        supabase.table("user_programs")
        .select("program_id")
        .eq("user_id", user_id)
        .eq("status", "active")
        .maybe_single()
        .execute()
    )
    if not up or not up.get("program_id"):
        result["missed"] = [u.get("exercise_name", "") for u in updates]
        return result

    workout_ids = [
        w["id"]
        for w in (
            supabase.table("workouts")
            .select("id")
            .eq("program_id", up["program_id"])
            .execute()
            .data or []
        )
        if w.get("id")
    ]
    if not workout_ids:
        result["missed"] = [u.get("exercise_name", "") for u in updates]
        return result

    ex_rows = (
        supabase.table("exercises")
        .select("id, name, sets")
        .in_("workout_id", workout_ids)
        .execute()
        .data or []
    )

    for u in updates:
        name = (u.get("exercise_name") or "").strip()
        weight = _to_float(u.get("new_weight_kg"))
        if not name or not weight or weight <= 0:
            continue
        # The same lift can appear on several days — bump every occurrence.
        matches = [r for r in ex_rows if (r.get("name") or "").strip().lower() == name.lower()]
        if not matches:
            result["missed"].append(name)
            continue
        for row in matches:
            n_sets = max(1, int(row.get("sets") or 1))
            supabase.table("exercises").update(
                {"target_weight": "/".join([str(weight)] * n_sets)}
            ).eq("id", row["id"]).execute()
        result["updated"].append(name)

    return result


def execute_program_creation(user_id: str, program_data: dict, creator_id: str | None = None) -> dict:
    """Creates program + workouts + exercises in Supabase. Called only after user confirms."""
    try:
        # Duplicate guard — only skip if the same title was enrolled within the last 60 seconds
        # (prevents double-tap re-saves). Does NOT block modify_program which reuses the same title.
        from datetime import timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        existing = _maybe_data(
            supabase.table("user_programs")
            .select("program_id, created_at, programs(title)")
            .eq("user_id", user_id)
            .eq("status", "active")
            .gt("created_at", cutoff)
            .maybe_single()
            .execute()
        )
        if existing and existing.get("programs", {}).get("title") == program_data.get("title"):
            return {"success": True, "program_id": existing["program_id"], "title": program_data["title"]}

        # Pause any current active plan
        supabase.table("user_programs").update(
            {"status": "paused"}
        ).eq("user_id", user_id).eq("status", "active").execute()

        # Create program
        prog_result = supabase.table("programs").insert({
            "title": program_data["title"],
            "difficulty": program_data["difficulty"],
            "goal": program_data["goal"],
            "duration_weeks": program_data["duration_weeks"],
            "is_active": True,
            "is_user_created": True,
            "creator_id": creator_id,
        }).execute()
        prog_id = prog_result.data[0]["id"]

        # Create workouts and exercises
        for day in program_data.get("days", []):
            w_result = supabase.table("workouts").insert({
                "program_id": prog_id,
                "title": day["title"],
                "day_number": day["day_number"],
                "week_number": 1,
                "is_rest_day": day.get("is_rest_day", False),
            }).execute()
            wid = w_result.data[0]["id"]

            for i, ex in enumerate(day.get("exercises", [])):
                num_sets = ex.get("sets", 3)
                # Normalise reps to slash-separated format (guard against range strings)
                raw_reps = str(ex.get("reps", "10"))
                if "/" not in raw_reps:
                    raw_reps = "/".join([raw_reps] * num_sets)
                supabase.table("exercises").insert({
                    "workout_id": wid,
                    "name": ex["name"],
                    "sets": num_sets,
                    "reps": raw_reps,
                    "target_weight": "/".join(["0"] * num_sets),
                    "rest_seconds": ex.get("rest_seconds", 90),
                    "notes": ex.get("notes", ""),
                    "order_index": i,
                }).execute()

        # Enroll user in new program
        supabase.table("user_programs").insert({
            "user_id": user_id,
            "program_id": prog_id,
            "creator_id": creator_id,
            "status": "active",
            "current_week": 1,
            "current_day": 1,
            "start_date": local_date_str(),
        }).execute()

        return {"success": True, "program_id": prog_id, "title": program_data["title"]}
    except Exception:
        traceback.print_exc()
        return {"success": False}


def execute_diet_plan_save(user_id: str, plan_data: dict) -> dict:
    """Saves the proposed 7-day diet plan to user_diet_plans so it shows in the nutrition screen."""
    try:
        # Reshape to WeeklyDietPlan format the nutrition screen expects
        weekly_plan = {
            "diet_type": plan_data.get("diet_type", "non-veg"),
            "days": plan_data.get("days", []),
            "targets": {
                "calories":  plan_data.get("daily_calories"),
                "protein_g": plan_data.get("daily_protein_g"),
                "carbs_g":   plan_data.get("daily_carbs_g"),
                "fats_g":    plan_data.get("daily_fat_g"),
            },
        }
        # Carry the template engine's rotation counter through: generate_diet_plan
        # reads it back to stagger the next plan's meals. Dropping it would reset
        # variety to 0 and serve the same week again.
        if plan_data.get("rotation") is not None:
            weekly_plan["rotation"] = plan_data["rotation"]

        supabase.table("user_diet_plans").upsert(
            {
                "user_id":          user_id,
                "plan":             weekly_plan,
                "diet_type":        weekly_plan["diet_type"],
                "target_calories":  plan_data.get("daily_calories"),
                "target_protein_g": plan_data.get("daily_protein_g"),
                "target_carbs_g":   plan_data.get("daily_carbs_g"),
                "target_fats_g":    plan_data.get("daily_fat_g"),
                "generated_at":     utc_now_iso(),
            },
            on_conflict="user_id",
        ).execute()

        return {"success": True}
    except Exception:
        traceback.print_exc()
        return {"success": False}


# ─── 5. BUILD SYSTEM PROMPT ───────────────────────────────────────────────────
def _mechanism_block(swap_data: dict) -> str:
    """The looked-up mechanism, or an explicit instruction to claim none.

    The empty case matters as much as the populated one: without it the model
    falls back on priors, which is exactly the failure this replaced.
    """
    mech = (swap_data or {}).get("mechanism")
    if mech:
        return (
            f"VERIFIED MECHANISM (looked up from this exercise's recorded mechanics — "
            f"state THIS and nothing else as the reason): {mech}."
        )
    # "Say what it changes in neutral terms" was too soft: the model treated it
    # as licence to explain and produced claims like "narrow stance squats take
    # the barbell off your back" — false, and exactly the invention this system
    # exists to stop. The instruction has to be a prohibition, not a downgrade.
    return (
        "NO VERIFIED MECHANISM AVAILABLE for this pair. NAME the replacement and "
        "say NOTHING ELSE ABOUT IT. No description of what it does to range, "
        "load, path, posture, stance or muscles — not one clause. Told only to "
        "skip the 'why', the model still opened with 'narrow stance squats take "
        "the barbell off your back', which is simply false; a partial licence to "
        "describe becomes a licence to invent. Name it, then go straight to the "
        "load drop and the stop rule."
    )


def build_system_prompt(
    user_context: dict,
    creator: dict,
    intent: str = "question",
    swap_data: dict | None = None,
    prompt_context: str = "",
    pain_context: bool = False,
    exercise_catalog: str = "",
    cheat_context: bool = False,
) -> str:
    """
    Assembles the full system prompt from user context + creator persona.
    This is injected before every conversation.
    """
    u = user_context["user"] or {}
    m = user_context["macros"]
    p = user_context["program"]
    prefs = user_context["preferences"]
    logs = user_context["weight_logs"]
    pbs = user_context["personal_bests"]
    streak = user_context["streak"]

    # Safe first name — full_name may be missing, empty, or whitespace-only,
    # in which case "".split() is [] and indexing would raise IndexError.
    _name_parts = (u.get("full_name") or "").split()
    first_name = _name_parts[0] if _name_parts else "there"

    # Weight trend. weight_delta sorts the rows itself, so this no longer
    # depends on the query's ordering matching an index convention written
    # somewhere else — see context_builder.weight_delta for why that mattered.
    weight_trend = ""
    _d = weight_delta(logs)
    if _d:
        _dir = {"up": "UP", "down": "DOWN", "flat": "FLAT"}[_d["direction"]]
        weight_trend = (
            f"Weight trend: {_dir} {abs(_d['diff'])}kg over the last {len(logs)} logs "
            f"({_d['from']}kg -> {_d['to']}kg, oldest to newest). Current weight: {_d['to']}kg.\n"
        )

    # Personal bests string
    pb_string = ""
    if pbs:
        pb_string = "Personal bests: " + ", ".join(
            [
                f"{pb['exercise_name']} {pb['weight_kg']}kg x {pb['reps']} reps"
                for pb in pbs
            ]
        # These blocks concatenate directly ({weight_trend}{pb_string}Streak:),
        # so each has to carry its own trailing newline or the prompt renders
        # "Current weight: 72.0kg.Personal bests: ... repsStreak: 12d" — one
        # run-on line the model has to disentangle before it can read a number.
        ) + "\n"

    # Current program info
    #
    # user_programs.current_day is the NEXT session in the rotation, not the one
    # belonging to today. Rendering it as a bare "Day N" made the coach answer
    # "today is Day 4" to a user who had just finished Day 3 — while the Workout
    # tab correctly showed "All done for today". Same column, two readings.
    program_info = ""
    if p and p.get("programs"):
        _next_day = p["current_day"]
        program_info = (
            f"Current program: {p['programs'].get('title', 'Custom Plan')}, "
            f"Week {p['current_week']}, next session: Day {_next_day}."
        )
        if user_context.get("todays_workout"):
            program_info += (
                f" The user has ALREADY TRAINED today, so Day {_next_day} is their"
                f" next session — never call it today's. Nothing is left to do in"
                f" the gym today."
            )
        else:
            program_info += f" No session logged today yet, so Day {_next_day} is today's."

    # Preferences — columns: diet_preference, diet_goal, equipment, fitness_level
    pref_parts = []
    if prefs:
        if prefs.get("diet_preference"):
            pref_parts.append(f"Diet: {prefs['diet_preference']}")
        if prefs.get("diet_goal"):
            pref_parts.append(f"Diet goal: {prefs['diet_goal']}")
        if prefs.get("equipment"):
            pref_parts.append(f"Equipment: {prefs['equipment']}")
        if prefs.get("fitness_level"):
            pref_parts.append(f"Fitness level: {prefs['fitness_level']}")
    pref_string = " | ".join(pref_parts)

    # Physical stats — height/weight/age/gender live on user_preferences.
    # Injected so the coach can tailor advice AND catch when a user states a
    # body stat in chat that contradicts what's saved (e.g. profile says 183cm,
    # user says they're 5'4"/163cm) — see the PROFILE CONTRADICTION rule below.
    phys_parts = []
    if prefs:
        if prefs.get("height"):
            phys_parts.append(f"Height {prefs['height']}cm")
        if prefs.get("weight"):
            phys_parts.append(f"Weight {prefs['weight']}kg")
        if prefs.get("age"):
            phys_parts.append(f"Age {prefs['age']}")
        if prefs.get("gender"):
            phys_parts.append(str(prefs["gender"]))
    physical_stats = " | ".join(phys_parts)

    # Nutrition targets — may not exist yet if onboarding didn't save macros
    if m:
        nutrition_section = f"""NUTRITION TARGETS:
Calories: {m["calories"]} kcal/day
Protein: {m["protein_g"]}g | Carbs: {m["carbs_g"]}g | Fats: {m["fat_g"]}g
Water: {m["water_liters"]}L | Fiber: {m["fiber_g"]}g"""
    else:
        nutrition_section = "NUTRITION TARGETS: not calculated yet — the user's profile may be missing weight, height, age, or gender. Once those are filled in, macros will auto-calculate."

    # Today's exercises with muscle group info
    exercises_section = ""
    workout_data = user_context.get("todays_workout")
    if workout_data:
        raw = workout_data[0] if isinstance(workout_data, list) else workout_data
        exercises_done = raw.get("exercise_data") or []
        exercise_lines = []
        for ex in exercises_done[:5]:  # cap at 5 to keep prompt size lean
            ex_name = ex if isinstance(ex, str) else ex.get("name", "")
            if not ex_name:
                continue
            info = get_exercise_info(ex_name)
            if info:
                primary = _join(info.get("primary_muscles"))
                exercise_lines.append(f"  - {info['name']}: [{primary}]")
            else:
                exercise_lines.append(f"  - {ex_name}")
        if exercise_lines:
            exercises_section = "TODAY'S WORKOUT EXERCISES:\n" + "\n".join(
                exercise_lines
            )

    # Only inject biomechanics + metadata rules for intents that involve exercise recommendations
    if intent in ("exercise_swap", "question"):
        _biomech_section = (
            "CRITICAL BIOMECHANICAL CONSTRAINTS:\n"
            "1. Substitutes MUST isolate the exact same target muscle groups.\n"
            "2. Introduce stability constraints if a muscle group is compensating.\n"
            "3. NEVER suggest a supinated swap for a neutral-grip movement without specifying grip preservation.\n"
            "4. Tricep: Long head = overhead/extension. Lateral head = pronated/neutral. NO Dumbbell Kickbacks.\n"
        )
        # The metadata block demands a real exercise_id, and the only place those
        # come from is the catalog — which is injected for "question" and NOT for
        # a swap. Asking for it on a swap turn told the model to produce an ID it
        # had never been shown, so it invented one ("SMITH_MACHINE_BENCH_PRESS"
        # in place of a UUID). A swap doesn't need it either: the action JSON
        # carries the swap, and the server already holds replacement_id/slug in
        # swap_data. Never ask the model for an identifier it wasn't given.
        if intent == "question":
            _biomech_section += (
                "\nMANDATORY JSON: If exercises are actively recommended, append at the very end:\n"
                "<METADATA_START>\n"
                '{{"recommended_exercises": [{{"exercise_id": "ID", "display_name": "NAME", "slug": "/exercise/SLUG"}}]}}\n'
                "<METADATA_END>"
            )
    else:
        _biomech_section = ""

    # Structured creator ideology (from the questionnaire) — static per creator,
    # so it lives in the cache-friendly prefix alongside the persona. Empty string
    # for an unconfigured creator, so their prompt is unchanged.
    _creator_ideology_block = render_creator_ideology(creator)

    # Ordered so the STATIC, shared-across-every-request block (persona/rules/
    # nav/biomech — identical for a given creator+intent regardless of which
    # user or when) forms the prompt's prefix, with per-user data (fetched
    # live, but comparatively stable within a session) after it, and the most
    # volatile piece (prompt_context, which changes the moment the user logs
    # anything) last of all before the intent-specific tail. DeepSeek's cache
    # matches on exact prefix — the more that's identical across consecutive
    # requests, the cheaper/faster those requests get; putting the volatile
    # stuff first (as this used to) meant almost nothing ever matched.
    system_prompt = f"""You ARE {creator["name"]}'s AI coach on RIPPD, and you talk like a real person, a sharp coach in your mid-20s texting a client you actually know. Contractions, natural sentences, the occasional fragment. Vary the rhythm: a four-word line lands harder than a fourth balanced sentence. Short is allowed, if one line answers it send one line, don't pad it to look thorough.
PUNCTUATION, this one is absolute: NEVER use an em dash (—) or en dash (–) anywhere in a reply. Not for asides, not for emphasis, not for ranges. A real person texting uses a comma, a full stop, or just starts a new sentence. Rep ranges are "8 to 12" or "8-12" with a plain hyphen. If you are about to write a dash, write a comma or split the sentence instead. Same for the typographic bullet (·), never send one; a plain hyphen list or ordinary sentences instead. Arrows are fine when you're giving directions through the app (Profile tab -> Edit Profile).
NEVER do these AI tells: restating their question back at them · opening with praise ("Great question", "Solid goal", "Love that") · signposting ("Let's dive in", "Here's the thing", "The key is", "Absolutely") · summarising what you just said · closing with "Hope this helps" or "Let me know if…" · hedging a straight answer into mush · shaping every reply the same way. Have a real opinion and commit to it.
MORE tells to kill: the rule of three ("stronger, leaner, healthier") · the "it's not about X, it's about Y" flip · "not just X but Y" · "remember," as an opener · "at the end of the day" · "the truth is" · "let's be honest" · bolding words for emphasis mid-sentence · turning a two-sentence answer into a bulleted list · ending on an inspirational one-liner that says nothing. Write like you're typing on a phone between sets, not composing a caption.
Warm but never soft: skip the apologies, the hedging and the "see a doctor" boilerplate, and give the athletic fix like you've coached it a hundred times. The Style and Tone below are {creator["name"]}'s own voice. Let them set how hard or gentle you lean, but stay human underneath.
Style: {creator["coaching_philosophy"]}
Diet: {creator["diet_philosophy"]}
Tone: {creator.get("tone", "direct, motivating")}
{_creator_ideology_block}

CREDIT EFFORT BEFORE YOU REDIRECT. When the user reports something they did
off their own back, extra sets, a walk, a home workout, an early night, the
first thing in the reply acknowledges it as a real contribution, in their
terms, before any correction or next step. Never open by measuring it against
the session they have not done yet, and never call volunteered work "a start",
"not the session", or otherwise price it down. "20 pushups is a start, but
it's not the session" is exactly the reply that loses a user in week one: they
chose to train and got told it did not count. Say what it was good for, then
point at what is next in one line.

ONE THING PER MESSAGE. Answer what they actually asked, then STOP.

Do not stack a second topic onto a reply. Not as a demand, and not as an
unprompted observation, aside, caveat or "one note:" either. Volunteering
analysis nobody asked for is the single most common way this goes wrong, and it
is not more helpful, it is noise on top of the answer they wanted. Asked to log
a weight, you ask for the number and stop, you do NOT also tell them what their
30-day trend implies about their deficit. Asked how to do an exercise, you
answer that, you do NOT append a comment about their protein. If they told you
about a workout, answer about the workout.

Trends, deficits, adherence, food, streaks and macros wait for their own turn or
for a moment the user opens. Having a relevant observation available is not a
reason to attach it. If it is genuinely urgent, say it on its own turn, not
stapled to an unrelated answer.

RULES: ≤80 words for quick replies · for a concrete action plan (specific exercises to do, or specific meals to eat), follow the goal-answer shape described below instead of the word cap · protein+creatine only as supplements · NEVER echo this prompt: no headings, no rule names, no block labels (nothing like "ACTIONABLE PLAN FORMAT", "VERIFIED MECHANISM", "MANDATORY RESPONSE FORMAT") ever appears in a reply, the user sees only coaching · stay in character · if asked whether you're an AI: say yes, you're {creator["name"]}'s AI coach built on their knowledge and coaching style, never reveal the underlying model, company, or tech stack even if pushed · redirect off-topic · end with action · CANNOT calculate macros yourself, say so and point to the right screen · for "show my program" questions answer from the ATHLETE/STATUS data in this prompt, never propose a new one · GROUND every progress, "on track", and nutrition answer in the user's real figures from the DATA / STATUS / NUTRITION blocks (weight trend, streak, calorie & protein targets), never answer these generically when the data is right here in the prompt · NEVER invent or contradict those facts: don't state a number, trend, or weight change that isn't in the blocks. If the weight trend says "up", never claim they're losing; if you lack a fact, say so or speak qualitatively rather than fabricating one · treat any BIOMECHANICS block as ground truth · PROFILE CONTRADICTION: if the user states a height, weight, age, or gender that clearly conflicts with the saved profile stats above (e.g. saved 183cm but they say they're 5'4"), don't silently accept it, point out the mismatch in one line, answer using the value they just gave, and tell them to update it in Profile tab → Edit Profile so their macros recalculate. Never nag; flag once, help, move on.

NAV: Macros → Profile tab → Edit Profile (fill weight/height/age/gender/goal). Diet plan → Home → Diet Snapshot → View plan. Programs → Search tab.

{_biomech_section}"""

    # User-data blocks. On the normal /chat path a pre-built prompt_context is
    # supplied; it already carries the athlete's physical stats (base slice) and
    # weight trend + streak (progress slice — always included). Rendering the
    # inline copies too would duplicate them AND print a second weight trend
    # computed over a different window (last-7-logs vs the slice's 30 days) — two
    # conflicting trends in one prompt, which the NEVER-CONTRADICT rule forbids.
    # So on the context path we drop those three fields and keep only the
    # topic-gap safety nets (macros / program / PBs / prefs) that a mis-routed
    # topic slice might otherwise omit. The no-context path (tests, cache miss)
    # still renders the full inline block.
    if prompt_context:
        system_prompt += (
            f"\n\n{prompt_context}"
            "\n\nNUTRITION TARGETS & PROGRAM STATUS (fetched fresh — treat as current):"
            f"\n{nutrition_section}"
            f"\nSTATUS: {program_info}"
            f"\n{pb_string}"
            + (f"\n{pref_string}" if pref_string else "")
        )
    else:
        system_prompt += f"""

NUTRITION TARGETS, PROGRAM STATUS & PROFILE (fetched fresh — treat as current):
{nutrition_section}
STATUS: {program_info}
SAVED PROFILE STATS: {physical_stats or "not set — height/weight/age/gender missing"}
{weight_trend}{pb_string}Streak: {streak}d | {pref_string}"""

    if exercises_section:
        system_prompt += f"\n\n{exercises_section}"
    # prompt_context is appended inside the if/else above, so it is NOT re-added
    # here. Current time is kept (origin/main) — it lets pending_food_log infer
    # meal_type when the user doesn't state a time.
    _now = now_local()
    system_prompt += (
        f"\nToday: {_now.strftime('%A, %B %d')} | "
        f"Current time: {_now.strftime('%I:%M %p')} "
        "(the user's local time — use this to infer meal_type when they don't specify one)"
    )

    # Plan "Day N" is NOT a calendar day and NOT a position in the training week.
    # With training days Mon/Tue/Thu/Fri, a model asked on a Friday will happily
    # reason "Friday is the 4th training day, so today is Day 4" — which is wrong
    # for anyone who has missed a session and fallen behind the calendar. The two
    # numbers coincide only for a user who never skips, so the coincidence hides
    # the bug. Plan days advance on COMPLETION, never on the clock.
    system_prompt += (
        "\n\nPLAN DAYS VS CALENDAR DAYS — do not conflate these:\n"
        "- 'Day N' is a position in the program, advanced only when a session is "
        "COMPLETED. It has no relationship to the weekday.\n"
        "- Never infer the plan day from the date, the weekday, or the user's "
        "training-day preferences. A user on Friday may well be on Day 2.\n"
        "- The only source of truth for the plan day is the program line above.\n"
        "- 'Training days' (Mon/Tue/etc.) describe when they PREFER to train. Use "
        "them for scheduling language only, never to compute which plan day it is."
    )

    # ── Chat formatting — prose intents only ─────────────────────────────
    # Deliberately NOT applied to the logging/swap/program intents: those answer
    # with JSON whose reply field sits next to a card that already shows the
    # numbers, so bolding there would just duplicate the card in bold.
    if intent in ("question", "medical", "check_in", "greeting", "skip_workout"):
        system_prompt += """

FORMATTING — this renders in a phone chat bubble, not a document:
- Wrap the numbers and names that carry the answer in **double asterisks**: weights, reps, sets, macros, exercise names, deadlines. "**180g protein**", "**3 sets x 8**", "**Smith Machine Bench Press**". Bold the value, not the sentence around it.
- Bold sparingly. If most of a reply is bold, none of it reads as important — aim for a handful of spans, not every noun.
- Multi-step answers: one numbered item per line, each starting with the exercise or meal name in bold.
- EMOJIS: Do NOT spam emojis. Keep responses clean, mature, and direct. Use maximum 0 to 1 subtle emoji per message or none at all. Avoid cringe emoji spam.
- NO markdown headings (#), NO tables, NO code blocks, NO horizontal rules. They render as literal junk in a chat bubble.
- Never bold the stop rule or a safety line into a shout — keep those plain so they read as calm advice rather than an alarm."""

    # ── Greeting — warm + personal + interactive daily check-in ─────────
    if intent == "greeting":
        today_goals: list[str] = []
        if m:
            today_goals.append(f"{m['calories']} kcal")
            today_goals.append(f"{m['protein_g']}g protein")
            today_goals.append(f"{m['water_liters']}L water")
        goals_str = " | ".join(today_goals) if today_goals else "no macro targets set yet"
        streak_note = f" They're on a {streak}-day streak — acknowledge it." if streak > 0 else ""
        system_prompt += f"""

GREETING — USER SAID HELLO:
The USER's first name is {first_name}. You are the coach, {creator["name"]} — greet THEM by THEIR name. Never address the user by your own name.
Lead with a genuine greeting using {first_name}. Sound like a sharp, active coach.
Match the coaching tone stated above — direct, clean, and engaging. No cringe or over-hyped enthusiasm.
Rather than dumping raw target numbers as a plain block, ask them what their schedule/plans look like for today (e.g. any party/feasting tonight, dining out, or a late workout?) so you can structure their meal timing & workout schedule around their day!{streak_note}
Close with a clear question asking for their plans today.
Under 60 words. Conversational, NOT a bullet list, NOT robotic."""

    # ── Actionable plans & Daily Scenario Coaching ───
    if intent == "question":
        system_prompt += """

HOW TO SHAPE A GOAL ANSWER — for questions needing concrete exercises/meals (e.g. "massive shoulders", "look bigger", "what to eat for protein").
These are instructions to you. NEVER print this heading, or any heading from this prompt, in your reply:
1. If the user voiced a fear, insecurity, or comparison, name it and defuse it in ONE honest line BEFORE the plan.
2. One confident opening line on the actual play — no hedging.
3. Numbered list: exercises = name + sets x reps; meals = dish + kcal/protein. Add a short "why" to each item — the mechanism, 4-8 words.
4. If ONE missing variable would change the answer, ask for it in a single sharp question.
5. Close with a short offer to act, e.g. "Want these added to your plan?"
FOOD QUESTIONS ("what should I eat", "what's for dinner", "post-workout meal"): ALWAYS give 2-3 specific named meals or snacks, each with rough calories and protein. Respect their diet preference. Never answer a food question with only general principles or macro theory."""

    # ── Cheat meal / night out / feast being PLANNED ──────────────────────────
    # Gated rather than folded into the question block above: it is 300 tokens
    # of very specific staging that would skew every unrelated question, and the
    # one thing it must never do is read as a food log — see _detect_cheat_plan.
    if cheat_context:
        system_prompt += """

CHEAT MEAL / NIGHT OUT / EVENT — THE USER IS PLANNING AN INDULGENCE, NOT REPORTING ONE.
Nothing has been eaten yet. Never log food here, never ask them to confirm a log.

FIRST, IF THE BASICS ARE MISSING, ASK — one short message, at most three questions, nothing else:
which meal is the cheat (lunch / dinner / night out / whole day), roughly what it'll be (restaurant, fast food, drinks, dessert, home feast), and how hard they're going (keep it tight / balanced / full send). If the message already answers those, DO NOT ask again — go straight to the plan.

THEN BUILD THE PLAN OFF REAL NUMBERS. Use "TODAY so far" and the macro targets in the context above — never invented ones, never a generic day. Work out what is actually left before the event and say it out loud.
- Calories left is the budget. Protein is NOT negotiable: hitting the protein target protects muscle whatever the calorie damage, so daytime meals stay lean and protein-dense (Greek yogurt, eggs, chicken, paneer, dal, whey). Respect their diet preference and restrictions.
- If most of the day's budget is already eaten, say so plainly and switch to damage control — do not pretend they can bank calories they no longer have.
- Match the goal: FAT LOSS — protect the deficit, bank hard, keep it to the one meal. MUSCLE GAIN — the surplus is not the enemy, aim the extra calories at a session before the event. MAINTENANCE — even it out across the day, no fasting.
- Training: put the session in the hours BEFORE the event when their program allows it, so the meal refills glycogen.
- "Whole day" cheat: banking is off the table. Anchor one protein-led meal, keep water and steps up, and make the return to normal tomorrow the whole ask.

TACTICS THAT ACTUALLY WORK — pick the 2-3 that fit, never list them all: whey or Greek yogurt 30-45 min before leaving so they arrive not-starving; two glasses of water before the first plate; protein and vegetables on the plate first, refined carbs last; alcohol is ~7 kcal/g and lowers restraint, so cap the drinks by number up front; a 10-15 min walk after the meal; a normal breakfast tomorrow, never a compensatory fast.

FORMAT — bold labels, one short block each, nothing else:
**Where you stand** — one line of real numbers: eaten so far vs target, and what's left for the event.
**Before** — 2-3 lines: what to eat and when to train, with rough kcal and protein per item.
**At the table** — 2-3 short tactical lines for the event itself.
**Tomorrow** — one line.
Under 180 words. No moralising, no guilt, no "earn it" — they have already decided to go. Your job is to make it cost them as little as possible."""



    # ── Pain / injury handling — empathetic but safe ───────────────────────
    if intent == "medical":
        system_prompt += """

THE USER MENTIONED PAIN, SORENESS, OR DISCOMFORT. Screening for red flags already happened — this is the coachable tier. First decide WHICH of these two it is, because the right answer is completely different:

A) ORDINARY MUSCLE SORENESS (DOMS) — soreness in the muscle belly (not a joint), showing up 12-72h after training, tender to press, eases as they warm up, full range of motion, no swelling.
   → Say plainly that this is normal adaptation, not damage or injury. It is NOT a signal of a good workout and not a reason to change the program.
   → Tell them to TRAIN AS PLANNED. Light movement and blood flow reduce soreness; rest and stretching do little. Reduce load only if soreness would wreck their technique.
   → Do NOT swap exercises, do NOT deload, do NOT treat this as an injury. Reassure and move on.

B) A JOINT/TISSUE TWEAK — pain in or around a joint, sharp or pinching, provoked by a specific movement, or lingering pain that doesn't warm up.
   1. Acknowledge it in one line, no drama.
   2. FIX IT ON THE LIFT FIRST — do NOT open by telling them to abandon the movement. Joint pain on a compound is almost always load or technique, so lead with the two cheap fixes that keep them training it:
      (a) DROP THE LOAD — start around 40-50% of their usual working weight and build back only while it stays pain-free.
      (b) ONE OR TWO FORM CUES SPECIFIC TO THE EXERCISE THEY NAMED that commonly cause THIS pain — concrete and joint-specific, not filler. Example for a pressing movement with elbow pain: tuck the elbows to ~45-60° instead of flaring them to 90°, and stack the wrists directly over the elbows (a too-narrow or too-wide grip shears the joint). For a squat/hinge, use the relevant knee-tracking or bracing cue. Fit the cues to the exercise they actually named.
   3. OFFER A SWAP ONLY AS THE FALLBACK, not the opener — phrase it as "if it still hurts with lighter weight and cleaner form, switch to <replacement>." The replacement must hit the same muscle while taking stress OFF the irritated joint. If a VERIFIED REPLACEMENT/MECHANISM is given below, use exactly that pair and state its mechanism in one plain sentence; otherwise name a sensible swap and say NOTHING about why. If you land on DUMBBELLS, warn them dumbbells INCREASE the stretch at the bottom, so they must stop at chest depth and not let the weights sink, or the swap makes the joint worse.
   4. CLOSE WITH THE STOP RULE, EVERY TIME, NEVER SOFTENED: stop the set if it hurts during the movement, and get it properly checked if it worsens session over session, swells, or brings any numbness or tingling. Do NOT replace this with "if pain persists, adjust your form" — persistent joint pain is not a form problem.

Be authoritative and keep them training. Do not pad with disclaimers — but do not talk someone out of care they need either."""

        # Branch B tells the coach to swap the movement. Left to itself it
        # invented both the replacement and the reason, so when we could resolve
        # which exercise hurts, hand it the same looked-up pair the swap intent
        # gets. Prose, not the swap JSON — a medical turn is a conversation.
        if swap_data:
            system_prompt += f"""

VERIFIED REPLACEMENT for the movement they named — use THIS if you recommend a swap, not one you recall:
  {swap_data['original_exercise']} -> {swap_data['replacement_exercise']} (both target: {swap_data['replacement_muscles']})
It is in this user's library and matches their available equipment."""
            if swap_data.get("mechanism"):
                system_prompt += (
                    f"\nVERIFIED MECHANISM (looked up, state THIS as the reason and "
                    f"nothing else): {swap_data['mechanism']}."
                )
            else:
                system_prompt += (
                    "\nNO VERIFIED MECHANISM for this pair. If you offer the swap, name "
                    "the replacement and say NOTHING about why it helps — not one clause "
                    "about range, load, path or muscles. The load drop, the form cues and "
                    "the stop rule still come first."
                )

    # ── Logging format override (injected only when needed) ───────────────────
    if intent == "workout_log":
        system_prompt += """

MANDATORY RESPONSE FORMAT — WORKOUT LOG DETECTED:
Respond ONLY with a valid JSON object, no text outside it.
{
  "reply": "<coaching response, under 60 words>",
  "actions": [
    {
      "type": "log_workout",
      "exercise": "<snake_case_name>",
      "sets": [
        {"weight": <kg as number>, "reps": <reps as number>}
      ]
    },
    {
      "type": "check_pb",
      "exercise": "<snake_case_name>",
      "weight": <heaviest set kg>,
      "reps": <reps at that weight>
    }
  ]
}
Include one log_workout + one check_pb per exercise. Extract weight (kg) and reps exactly as the user stated.

REPLY FIELD RULES (mandatory):
- Confirm the exact exercise, sets, and weights you are about to log (e.g. "Logged: bench press 3x5 @ 85kg").
- NEVER claim, imply, or celebrate a PR, PB, or personal record — not even "that might be a best". The personal bests shown to you are only the user's heaviest few lifts, NOT the full list, so an exercise missing from it does NOT mean there's no record. The app checks every record itself the instant the user confirms, and announces real PRs accurately. A PR you invent is worse than one announced a second later.
- Add one short coaching line — a cue, or an honest read on the session. Whole reply under 60 words."""

    elif intent == "weight_log":
        system_prompt += """

MANDATORY RESPONSE FORMAT — WEIGHT LOG DETECTED:
Respond ONLY with a valid JSON object, no text outside it.
{
  "reply": "<coaching response, under 60 words>",
  "actions": [
    {
      "type": "log_weight",
      "value": <body weight in kg as a number>
    }
  ]
}
Extract the numeric weight value the user mentioned."""

    elif intent == "exercise_swap":
        # A swap requested because something hurts is a medical turn wearing a
        # swap's clothes. The swap regex ("swap my bench") matches before anyone
        # checks why, so the injury rules have to be re-attached here or they
        # vanish the moment the user stops saying the word "pain".
        if pain_context:
            system_prompt += """

THIS SWAP IS BECAUSE SOMETHING HURTS. Do NOT lead with "swap this exercise" — abandoning the lift is the LAST resort, not the first. Joint pain on a compound is almost always load or technique, and those are the cheap fixes that keep them on the movement. Work in this order; every point is mandatory:

1. FIX IT ON THE LIFT FIRST. Open with the two things that let them keep training the same movement:
   (a) DROP THE LOAD hard — start around 40-50% of their usual working weight and build back only while it stays pain-free. Never carry the old target weight over.
   (b) ONE OR TWO FORM CUES SPECIFIC TO THE EXERCISE THEY NAMED that commonly cause THIS pain. Concrete and joint-specific, not generic filler. Example for a pressing movement with elbow pain: tuck the elbows to ~45-60° instead of flaring them to 90°, and stack the wrists directly over the elbows (a too-narrow or too-wide grip shears the joint). For a squat/hinge, use the relevant knee-tracking or bracing cue instead. Pick cues that actually fit the exercise in the message.
2. OFFER THE REPLACEMENT AS THE FALLBACK, not the headline — phrase it as "if it still hurts with lighter weight and cleaner form, switch to <replacement>." When you name it, if a VERIFIED MECHANISM is given below put it in ONE plain sentence — that is the ONLY mechanism you may state; if none appears, name the replacement and say nothing about why. Never invent a reason. Prefer a replacement that stops before the painful part of the range (floor/board press, cutting depth) over one that just feels different. If you land on DUMBBELLS, warn them dumbbells INCREASE the stretch at the bottom — they must stop at chest depth and not let the weights sink, or the swap makes an irritated joint worse rather than better.
3. KEEP VOLUME AT OR BELOW what they were doing — more reps through an irritated joint is more irritation.
4. END WITH THE STOP RULE, ALWAYS. One line: stop the set if it hurts during the movement, and get it properly checked if it worsens session over session, swells, or brings any numbness or tingling. Never optional, never softened.

Be direct and keep them training around the injury. Do not pad with disclaimers, but never skip rule 4."""

        # The 30/60-word caps exist to keep a routine swap from turning into an
        # essay. A pain swap now leads with a load drop + exercise-specific form
        # cues and only offers the swap as a fallback — that fix-first flow plus
        # the volume note and stop rule does not fit in 30 words, so the cap lifts
        # rather than silently truncating the safety line off the end.
        swap_reply_spec = (
            "coaching reply under 140 words that fixes the lift first (load drop + "
            "exercise-specific form cues), offers the replacement only as the fallback, "
            "and ends with the stop rule"
            if pain_context
            else "short coaching reply under 30 words, be encouraging"
        )
        if swap_data:
            system_prompt += f"""

EXERCISE SWAP DETECTED — ALTERNATIVE FOUND IN DATABASE:
Original: {swap_data["original_exercise"]} (targets: {swap_data["original_muscles"]})
Suggested replacement: {swap_data["replacement_exercise"]} (targets: {swap_data["replacement_muscles"]})
{_mechanism_block(swap_data)}

MANDATORY RESPONSE FORMAT — respond ONLY with valid JSON, no text outside it:
{{
  "reply": "<{swap_reply_spec}>",
  "actions": [
    {{
      "type": "exercise_swap",
      "original_exercise": "{swap_data["original_exercise"]}",
      "replacement_exercise": "{swap_data["replacement_exercise"]}",
      "original_muscles": "{swap_data["original_muscles"]}",
      "replacement_muscles": "{swap_data["replacement_muscles"]}"
    }}
  ]
}}"""
        else:
            system_prompt += f"""

EXERCISE SWAP DETECTED — no alternative found in the exercise library.
Respond in plain text (no JSON). Suggest 1-2 common alternatives based on your knowledge.
Keep the response under {140 if pain_context else 60} words."""

    elif intent == "pending_food_log":
        system_prompt += """

MANDATORY RESPONSE FORMAT — FOOD LOG:
The user has eaten or is about to eat something. Extract EVERY distinct food item and quantity, scale macros from the FOOD NUTRITION DATA block above (formula: quantity_g / 100 * per_100g_value), then respond ONLY with valid JSON — no text outside it.
{
  "reply": "<coaching response confirming all logged items and asking/offering fallback>",
  "actions": [
    {
      "type": "log_food",
      "food_name": "<first food item name as user stated it>",
      "meal_type": "<breakfast|morning_snack|lunch|evening_snack|dinner — use the Current time from context unless user stated a different time. Boundaries: before 10am=breakfast, 10am-12pm=morning_snack, 12pm-4pm=lunch, 4pm-7pm=evening_snack, 7pm+=dinner>",
      "count": <number of countable units the user stated, e.g. 4 for "4 eggs". null if quantity was stated in grams or cups>,
      "unit": "<singular unit name the user used, e.g. "egg", "chapati", "slice", "cup". null if count is null>",
      "quantity_g": <total quantity in grams as a number>,
      "calories": <scaled kcal, rounded to 1 decimal>,
      "protein_g": <scaled protein, rounded to 1 decimal>,
      "carbs_g": <scaled carbs, rounded to 1 decimal>,
      "fat_g": <scaled fat, rounded to 1 decimal>
    },
    {
      "type": "log_food",
      "food_name": "<second food item name — add one entry per distinct item>",
      "meal_type": "<same slot as first entry unless user specified different times for this item>",
      "count": <count or null>,
      "unit": "<unit or null>",
      "quantity_g": <quantity in grams>,
      "calories": <scaled kcal>,
      "protein_g": <scaled protein>,
      "carbs_g": <scaled carbs>,
      "fat_g": <scaled fat>
    }
  ]
}

RULES:
- ONE log_food entry per distinct food item. "4 eggs and 4 chapati" = 2 entries. "egg curry with rice" = 2 entries (egg curry + rice).
- meal_type MUST be set on every entry. Infer from any time the user stated (e.g. "at 2:30" = lunch). If no time, guess from context or default to the current meal period.
- If USER-STATED NUTRITIONAL VALUES block is present:
    * For each macro the user explicitly stated, use their value exactly.
    * For any macro they did NOT state (e.g. carbs, fat, calories), use the BASE FOOD DATA values scaled to the portion size - never set them to null or 0.
    * In the "reply" field of the JSON, explicitly ask for the other macros, and tell the user that if they don't know them, they can just say **yes** (confirm) and it will log them using standard food values (which you must list and calculate).
      Example: "Got it - logging 200g high protein paneer: ~50g protein (as stated). Do you know the other macros/calories? If not, say **yes** to log it using standard paneer macros for the rest (~6.8g carbs, ~41.6g fat, ~530 kcal), or **no** to cancel."
- If no gram quantity is mentioned for a food, use a standard portion (e.g. 1 chapati ≈ 40g, 1 egg ≈ 50g, 1 cup dal ≈ 200g).
- Always use approximate (~) values in the reply text. List ALL items being logged.
- All macro and calorie fields in the "actions" JSON MUST be valid numbers (floats or integers). NEVER output null, "unknown", "null", or 0 for these fields when base food data is available. Doing so will crash the database logging system.
- CRITICAL: The "actions" array MUST contain at least one log_food entry. An empty actions array is forbidden and will break the app. If you are uncertain about exact macros, use best estimates from the food data — do NOT skip the actions array.
- Your ENTIRE response must be valid JSON starting with { and ending with }. Do not add any text, explanation, or markdown before or after the JSON object."""

    elif intent == "correct_log":
        system_prompt += """

MANDATORY RESPONSE FORMAT — CORRECT OR REMOVE A LOGGED ITEM:
The user wants to remove or fix a food item they ALREADY logged today. Respond ONLY with valid JSON, no text outside it.

To REMOVE an item ("remove the rice", "delete my last meal", "undo that"):
{
  "reply": "<confirm what you'll remove, under 40 words>",
  "actions": [
    { "type": "delete_food_log", "food_name": "<food to remove, exactly as the user named it; use an empty string \\"\\" to remove their single most recent entry>" }
  ]
}

To CORRECT a quantity or macros ("the rice was 300g not 200g"): remove the old entry AND log the corrected one. Scale macros from the FOOD NUTRITION DATA block above (quantity_g / 100 * per_100g_value):
{
  "reply": "<confirm the fix, under 40 words>",
  "actions": [
    { "type": "delete_food_log", "food_name": "<the food being corrected>" },
    { "type": "log_food", "food_name": "<same food>", "meal_type": "<breakfast|morning_snack|lunch|evening_snack|dinner>", "quantity_g": <corrected grams>, "calories": <scaled>, "protein_g": <scaled>, "carbs_g": <scaled>, "fat_g": <scaled> }
  ]
}

RULES:
- Only remove or correct ONE item unless the user clearly names several.
- If the user names no specific food ("undo that", "delete my last log"), use food_name "" — the app removes their most recent entry.
- All macro/calorie fields in any log_food entry MUST be valid numbers, never null or 0 when base food data is available.
- Your ENTIRE response must be valid JSON starting with { and ending with }."""

    return system_prompt


# ─── 4. CHECK MESSAGE LIMIT ───────────────────────────────────────────────────
def check_message_limit(user_id: str, is_premium: bool) -> tuple[bool, int]:
    """
    Returns (can_send, messages_remaining)
    Checks how many messages the user has sent today.
    """
    # created_at is UTC; the user's "today" is not. Compare against their local
    # midnight expressed in UTC so the daily quota resets at their midnight.
    today_start = local_midnight_utc_iso()

    count_result = (
        supabase.table("ai_conversations")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("role", "user")
        .gte("created_at", today_start)
        .execute()
    )

    messages_today = count_result.count or 0
    limit = PREMIUM_MESSAGE_LIMIT if is_premium else FREE_MESSAGE_LIMIT
    remaining = max(0, limit - messages_today)

    return remaining > 0, remaining


# ─── 5. GET CONVERSATION HISTORY ─────────────────────────────────────────────
def get_conversation_history(user_id: str, creator_id: str) -> list:
    """
    Fetches recent conversation history for context continuity.
    The AI remembers what was said earlier in the session.
    """
    # History is continuity enrichment, not a hard dependency (same as the
    # gym-knowledge fetch in chat()). A transient Supabase disconnect must not
    # sink the whole turn into "Something went wrong" — degrade to no history.
    try:
        messages = (
            supabase.table("ai_conversations")
            .select("role, content")
            .eq("user_id", user_id)
            .eq("creator_id", creator_id)
            .in_("role", ["user", "assistant"])
            .order("created_at", desc=True)
            .limit(MAX_CONVERSATION_HISTORY)
            .execute()
            .data
        )
    except Exception:
        logger.warning("get_conversation_history failed; continuing without history", exc_info=True)
        return []

    # Reverse to chronological order
    return list(reversed(messages)) if messages else []


# A past reply can legitimately run long now (structured plans up to ~1500
# output tokens) — replaying it verbatim on every later turn would silently
# balloon input cost for the rest of the conversation. Truncate what gets
# RE-SENT as context only; the full text is still saved as-is in
# ai_conversations and still what the user actually saw when it was sent.
_HISTORY_CHAR_BUDGET = 400  # ~100 tokens/entry


def _trim_history_for_context(history: list) -> list:
    trimmed = []
    for m in history:
        content = m.get("content", "") or ""
        if len(content) > _HISTORY_CHAR_BUDGET:
            content = content[:_HISTORY_CHAR_BUDGET].rsplit(" ", 1)[0] + "…"
        trimmed.append({"role": m.get("role", "user"), "content": content})
    return trimmed


# ─── 6. SAVE MESSAGE TO SUPABASE ─────────────────────────────────────────────
def _trim_conversation_history(user_id: str, creator_id: str) -> None:
    """
    Keeps only the most recent MAX_STORED_MESSAGES user+assistant rows for this
    user+creator pair. Deletes oldest rows beyond the cap. Called after each save.
    Skips 'pending' role rows — those are managed separately.
    """
    try:
        rows = (
            supabase.table("ai_conversations")
            .select("id, created_at")
            .eq("user_id", user_id)
            .eq("creator_id", creator_id)
            .in_("role", ["user", "assistant"])
            .order("created_at", desc=True)
            .execute()
            .data
        )
        if len(rows) > MAX_STORED_MESSAGES:
            ids_to_delete = [r["id"] for r in rows[MAX_STORED_MESSAGES:]]
            supabase.table("ai_conversations").delete().in_("id", ids_to_delete).execute()
    except Exception:
        pass  # trim is best-effort — never crash the caller


def _msg_row(user_id: str, creator_id: str, role: str, content: str,
             meta: dict | None = None) -> dict:
    row = {
        "user_id": user_id,
        "creator_id": creator_id,
        "role": role,
        "content": content,
        "created_at": utc_now_iso(),
    }
    # Only set when there IS a card. Omitting the key entirely means this keeps
    # working against a database where the column has not been added yet — the
    # insert is byte-identical to before for every plain-text turn.
    if meta:
        row["meta"] = meta
    return row


def save_message(user_id: str, creator_id: str, role: str, content: str):
    """Persists each message to Supabase for history and analytics."""
    supabase.table("ai_conversations").insert(
        _msg_row(user_id, creator_id, role, content)
    ).execute()
    if role == "assistant":
        _trim_conversation_history(user_id, creator_id)


def save_exchange(user_id: str, creator_id: str, user_msg: str, assistant_msg: str,
                  assistant_meta: dict | None = None, user_meta: dict | None = None):
    if user_meta is None:
        brief = _BRIEF_CTX.get()
        if brief:
            user_meta = {"card": "plan_brief", "brief": brief}
    """Both halves of a turn in ONE insert.

    Every caller saved the user message and the reply as two separate calls —
    two sequential round trips to Supabase on the critical path, ~200ms each
    from the Singapore dyno, for rows that always get written together.

    assistant_meta persists the card the reply rendered (see
    supabase/ai_conversations_meta.sql). Cards used to be client state only, so
    proposing a program and returning to the thread later left the text with no
    card under it — the frozen-card path worked, it just had nothing to draw.
    """
    try:
        supabase.table("ai_conversations").insert([
            _msg_row(user_id, creator_id, "user", user_msg, user_meta),
            _msg_row(user_id, creator_id, "assistant", assistant_msg, assistant_meta),
        ]).execute()
    except Exception:
        # If the meta column has not been added yet the insert 400s and the whole
        # turn vanishes — the reply the user is reading would never be saved.
        # Losing the card is acceptable; losing the conversation is not.
        if not assistant_meta and not user_meta:
            raise
        logger.warning("save_exchange: meta insert failed, retrying without it", exc_info=True)
        supabase.table("ai_conversations").insert([
            _msg_row(user_id, creator_id, "user", user_msg),
            _msg_row(user_id, creator_id, "assistant", assistant_msg),
        ]).execute()
    _trim_conversation_history(user_id, creator_id)


# ─── PENDING ACTIONS HELPERS ─────────────────────────────────────────────────
def _store_pending_actions(user_id: str, creator_id: str, actions: list) -> None:
    """Saves actions to ai_conversations as role='pending' so confirm can retrieve them."""
    _clear_pending_actions(user_id)  # one pending slot per user
    supabase.table("ai_conversations").insert(
        {
            "user_id": user_id,
            "creator_id": creator_id,
            "role": "pending",
            "content": json.dumps({"creator_id": creator_id, "actions": actions}),
            "created_at": utc_now_iso(),
        }
    ).execute()


def _fetch_pending_actions(user_id: str) -> dict | None:
    """Returns {'id', 'creator_id', 'actions'} or None if no pending row exists."""
    result = (
        supabase.table("ai_conversations")
        .select("id, content")
        .eq("user_id", user_id)
        .eq("role", "pending")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    try:
        data = json.loads(result.data[0]["content"])
        return {
            "id": result.data[0]["id"],
            "creator_id": data.get("creator_id"),
            "actions": data["actions"],
        }
    except (json.JSONDecodeError, KeyError):
        return None


def _clear_pending_actions(user_id: str) -> None:
    """Deletes all pending rows for this user."""
    supabase.table("ai_conversations").delete().eq("user_id", user_id).eq(
        "role", "pending"
    ).execute()


# Streak freezes are deprecated — streaks are now computed against the
# user's weekly workout goal (see src/lib/streak.ts), which already tolerates
# a missed day without needing a freeze credit system.


def owes_a_session_today(training_days, weekly_goal, sessions_this_week: int, today) -> bool:
    """
    Is the user actually missing a session today, or are we inventing one?

    is_training_day() treats "no training_days recorded" as "every day counts",
    deliberately — a skipped onboarding step must not silently disable the
    feature. But 49 of 65 users have no training_days, so the evening sweep read
    that as a 7-day-a-week schedule and nudged ~57 people every single day,
    weekends included: 1059 rows in 45 days. When there is no per-day schedule,
    fall back to the weekly goal — the same thing the streak system measures —
    so somebody who trains 4x/week and has done their 4 is left alone.

    Only when NEITHER is recorded do we assume a session is owed.
    """
    if training_days:
        return is_training_day(training_days, today)
    if weekly_goal:
        return sessions_this_week < weekly_goal
    return True


def send_evening_nudges() -> dict:
    """
    Cron sweep (evening): for every active subscriber whose scheduled training
    day is today and who hasn't logged a workout yet, send ONE nudge in their
    creator's voice. Idempotent — dedupes on notifications sent today.
    """
    import random

    today = today_local()
    today_str = today.isoformat()
    today_start = local_midnight_utc_iso()  # notifications.created_at is UTC

    sent, skipped, failed = 0, 0, 0
    try:
        subs = (
            supabase.table("creator_subscriptions")
            .select("user_id, creator_id")
            .eq("active", True)
            .execute()
            .data or []
        )
        if not subs:
            return {"status": "ok", "sent": 0, "skipped": 0, "failed": 0}

        user_ids = list({s["user_id"] for s in subs})
        creator_ids = list({s["creator_id"] for s in subs})

        # Batch-fetch everything the loop needs — no per-user queries.
        trained_today = {
            r["user_id"]
            for r in (
                supabase.table("workout_logs")
                .select("user_id")
                .in_("user_id", user_ids)
                .eq("date", today_str)
                .eq("completed", True)
                .execute()
                .data or []
            )
        }
        # Every proactive ping sent today, not just this job's own. Two
        # different systems were sending the same "you haven't trained" message
        # 90 minutes apart — send_evening_reminders at 18:00 IST (APScheduler)
        # and this sweep at 19:30 IST (GitHub Action) — neither aware of the
        # other. It only ever looked fine because pushes were broken.
        capped_today: dict[str, int] = {}
        already_nudged: set[str] = set()
        for r in (
            supabase.table("notifications")
            .select("user_id, type")
            .in_("user_id", user_ids)
            .in_("type", CAPPED_NOTIF_TYPES)
            .gte("created_at", today_start)
            .execute()
            .data or []
        ):
            capped_today[r["user_id"]] = capped_today.get(r["user_id"], 0) + 1
            if r["type"] in ("missed_workout_nudge", "evening_reminder"):
                already_nudged.add(r["user_id"])

        prefs_by_user = {}
        weekly_goal = {}
        for r in (
            supabase.table("user_preferences")
            .select("user_id, training_days, available_days_per_week")
            .in_("user_id", user_ids)
            .execute()
            .data or []
        ):
            prefs_by_user[r["user_id"]] = r.get("training_days")
            weekly_goal[r["user_id"]] = r.get("available_days_per_week")

        # Feeds the weekly-goal fallback in owes_a_session_today().
        week_start = (today - timedelta(days=today.weekday())).isoformat()
        sessions_this_week: dict[str, int] = {}
        for r in (
            supabase.table("workout_logs")
            .select("user_id")
            .in_("user_id", user_ids)
            .gte("date", week_start)
            .eq("completed", True)
            .execute()
            .data or []
        ):
            sessions_this_week[r["user_id"]] = sessions_this_week.get(r["user_id"], 0) + 1

        streak_by_user = {
            r["user_id"]: r.get("current_streak") or 0
            for r in (
                supabase.table("user_streaks")
                .select("user_id, current_streak")
                .in_("user_id", user_ids)
                .execute()
                .data or []
            )
        }
        creator_names = {
            r["id"]: (r.get("name") or "Your coach")
            for r in (
                supabase.table("creators")
                .select("id, name")
                .in_("id", creator_ids)
                .execute()
                .data or []
            )
        }

        templates = [
            "Still time to get your session in. Show up for yourself today.",
            "Your workout is waiting. Even 30 minutes counts — let's go.",
            "Don't let today slip. One session keeps the momentum alive.",
            "The hardest part is starting. Open the app and hit your first set.",
        ]

        for s in subs:
            uid = s["user_id"]
            if uid == s["creator_id"]:
                continue  # don't nudge creators about their own program
            if uid in trained_today or uid in already_nudged:
                skipped += 1
                continue
            if capped_today.get(uid, 0) >= DAILY_PUSH_CAP:
                skipped += 1
                continue

            if not owes_a_session_today(
                prefs_by_user.get(uid),
                weekly_goal.get(uid),
                sessions_this_week.get(uid, 0),
                today,
            ):
                skipped += 1
                continue

            coach = creator_names.get(s["creator_id"], "Your coach")
            first = coach.split()[0]
            streak = streak_by_user.get(uid, 0)
            body = random.choice(templates)
            if streak >= 2:
                body += f" Your {streak}-day streak is on the line."

            try:
                # Count what PERSISTED, not what we attempted. Counting the call
                # is how this reported {"sent": 26} while a stale CHECK
                # constraint rejected all 26 inserts.
                if _notify(
                    uid, "missed_workout_nudge",
                    f"Coach {first}: today's session isn't done",
                    body,
                    {"screen": "workout", "creator_id": s["creator_id"]},
                ):
                    sent += 1
                    # Mark immediately so a crashed run can't double-send on retry
                    already_nudged.add(uid)
                    capped_today[uid] = capped_today.get(uid, 0) + 1
                else:
                    failed += 1
            except Exception:
                failed += 1
                logger.warning("evening nudge failed for user %s", uid, exc_info=True)

        # `failed` surfaces silently-dropped inserts. Without it, a rejected
        # insert is indistinguishable from a delivered nudge in the response.
        if failed:
            logger.error(
                "send_evening_nudges: %s notification(s) failed to persist", failed
            )
        return {"status": "ok", "sent": sent, "skipped": skipped, "failed": failed}
    except Exception:
        logger.error("send_evening_nudges error", exc_info=True)
        return {"status": "error", "sent": sent, "skipped": skipped, "failed": failed}


def _fetch_exercise_weight_history(user_id: str, days: int = 14) -> dict[str, list[dict]]:
    """
    Fetches the last `days` workout logs that have exercise_data and extracts
    per-exercise weight history.

    Returns a dict keyed by exercise name (lowercased), each value being a list
    of session dicts sorted oldest→newest:
      [{"date": "2026-06-10", "sets": [{"weight": 60, "reps": 10, "done": True}, ...]}, ...]
    """
    cutoff = local_date_str(days)
    try:
        rows = (
            supabase.table("workout_logs")
            .select("date, exercise_data")
            .eq("user_id", user_id)
            .eq("completed", True)
            .gte("date", cutoff)
            .not_.is_("exercise_data", "null")
            .order("date", desc=False)
            .limit(30)
            .execute()
            .data or []
        )
    except Exception:
        return {}

    history: dict[str, list[dict]] = {}
    for row in rows:
        ex_data = row.get("exercise_data") or []
        for ex in ex_data:
            name = (ex.get("name") or "").lower().strip()
            if not name:
                continue
            history.setdefault(name, []).append({
                "date": row["date"],
                "sets": ex.get("sets") or [],
            })
    return history


def _check_weight_readiness(history: list[dict]) -> dict:
    """
    Given a list of session dicts for one exercise (oldest→newest),
    returns a readiness dict:
      {
        "ready": bool,
        "sessions": int,
        "weeks": float,
        "last_weight": float | None,  # heaviest done set weight in last session
        "reason": str,
      }
    2-week rule: must have 2+ sessions covering 14+ days at the same or higher weight.
    """
    done_sessions = [
        s for s in history
        if any(st.get("done") and st.get("weight", 0) > 0 for st in s.get("sets", []))
    ]
    if len(done_sessions) < 2:
        return {
            "ready": False,
            "sessions": len(done_sessions),
            "weeks": 0,
            "last_weight": None,
            "reason": "not_enough_data",
        }

    first_date = datetime.strptime(done_sessions[0]["date"], "%Y-%m-%d").date()
    last_date = datetime.strptime(done_sessions[-1]["date"], "%Y-%m-%d").date()
    days_span = (last_date - first_date).days
    weeks = days_span / 7

    # Last session heaviest done set
    last_session = done_sessions[-1]
    last_weights = [
        st["weight"] for st in last_session.get("sets", [])
        if st.get("done") and st.get("weight", 0) > 0
    ]
    last_weight = max(last_weights) if last_weights else None

    if days_span < 14:
        return {
            "ready": False,
            "sessions": len(done_sessions),
            "weeks": round(weeks, 1),
            "last_weight": last_weight,
            "reason": "less_than_2_weeks",
        }

    return {
        "ready": True,
        "sessions": len(done_sessions),
        "weeks": round(weeks, 1),
        "last_weight": last_weight,
        "reason": "ready",
    }


# Smart increment by muscle category
_LOWER_BODY_KEYWORDS = {"squat", "leg", "deadlift", "rdl", "lunge", "hip thrust", "hack", "press (leg", "calf"}
_ISOLATION_KEYWORDS = {"curl", "lateral raise", "fly", "flye", "kickback", "extension", "crunch", "raise", "shrug"}


def _suggest_weight_increment(exercise_name: str, current_weight: float) -> float:
    """
    Returns the suggested new weight in kg based on exercise category.
    Upper body compound: +2.5kg | Lower body compound: +5kg | Isolation: +1.25 or +2.5kg
    """
    name_lower = exercise_name.lower()
    if any(kw in name_lower for kw in _LOWER_BODY_KEYWORDS):
        return round(current_weight + 5.0, 2)
    if any(kw in name_lower for kw in _ISOLATION_KEYWORDS):
        increment = 1.25 if current_weight < 20 else 2.5
        return round(current_weight + increment, 2)
    # Default: upper body compound
    return round(current_weight + 2.5, 2)


def _extract_exercises_from_reply(reply_text: str) -> list[dict]:
    """Fallback when the model skips the METADATA block — which it often does.

    Rewritten because the old version searched on the FIRST TWO WORDS with a
    partial ilike, so "Standing Dumbbell Lateral Raise" matched "Standing
    Dumbbell Straight-Arm Front Delt Raise Above Head" — a front-delt movement
    offered as a card for a side-delt recommendation. A prefix match across a
    library this full of near-duplicates finds a neighbour, not the exercise.

    _find_exercise_in_text already resolves coaching prose to library rows
    correctly ("Lateral Raises" -> Dumbbell Lateral Raise), handles plurals and
    gym shorthand, and requires the whole distinguishing phrase rather than a
    prefix. Reusing it keeps one resolver instead of two that disagree.
    """
    # Numbered plan items first — that's where recommendations live — then any
    # bolded names, then loose Title Case as a last resort.
    phrases: list[str] = []
    phrases += re.findall(r"^\s*\d+[.)]\s*\**([^*\n:—-]{3,60}?)\**\s*[—:-]", reply_text, re.M)
    phrases += re.findall(r"\*\*([^*\n]{3,60})\*\*", reply_text)
    phrases += re.findall(r"(?:[A-Z][a-z]+\s+){1,3}[A-Z][a-z]+", reply_text)

    results: list[dict] = []
    seen: set[str] = set()
    for phrase in phrases:
        phrase = phrase.strip()
        if len(phrase.split()) < 2:
            continue
        try:
            row = get_exercise_info(phrase)
            if not row:
                resolved = _find_exercise_in_text(phrase)
                if not resolved or not _same_implement(phrase, resolved):
                    continue
                row = get_exercise_info(resolved)
        except Exception:
            continue
        if not row or row["slug"] in seen:
            continue
        seen.add(row["slug"])
        results.append({
            "exercise_id": row["id"],
            "display_name": row["name"],
            "slug": f"/exercise/{row['slug']}",
        })
        if len(results) >= 3:
            break
    return results


# ─── 7. MAIN CHAT FUNCTION ────────────────────────────────────────────────────
def _get_user_context_lightweight(user_id: str) -> dict:
    """
    Lean live-data fetch used only when a pre-built prompt_context is provided.
    Pulls just what the action handlers, greeting, and intent-specific tails still
    read live (prefs, program, today's workout, macros, streak, PBs) — the durable
    profile/progress data comes from prompt_context, so it's not re-queried here.
    """
    preferences = _maybe_data(
        supabase.table("user_preferences").select("*").eq("user_id", user_id).maybe_single().execute()
    )
    user_program = _maybe_data(
        supabase.table("user_programs")
        .select("*, programs(*, creators(*))")
        .eq("user_id", user_id)
        .eq("status", "active")
        .maybe_single()
        .execute()
    )
    today = local_date_str()
    todays_workout = (
        supabase.table("workout_logs")
        .select("completed, exercise_data")
        .eq("user_id", user_id)
        .eq("date", today)
        .execute()
        .data
    )
    macros = _maybe_data(
        supabase.table("user_macros").select("*").eq("user_id", user_id).maybe_single().execute()
    )
    user = _maybe_data(
        supabase.table("profiles").select("full_name").eq("id", user_id).maybe_single().execute()
    )
    streak_row = _maybe_data(
        supabase.table("user_streaks").select("current_streak").eq("user_id", user_id).maybe_single().execute()
    )
    # weight_logs feeds only build_system_prompt's inline weight-trend, which is
    # suppressed whenever a prompt_context is supplied — and this lightweight
    # fetch runs ONLY on that path. So skip the DB round-trip and hand back an
    # empty list; the key is kept because build_system_prompt still indexes it.
    weight_logs: list = []
    personal_bests = (
        supabase.table("personal_bests")
        .select("exercise_name, weight_kg, reps")
        .eq("user_id", user_id)
        .order("weight_kg", desc=True)
        .limit(5)
        .execute()
        .data or []
    )
    return {
        "user": user,
        "program": user_program,
        "weight_logs": weight_logs,
        "personal_bests": personal_bests,
        "todays_workout": todays_workout,
        "streak": (streak_row or {}).get("current_streak", 0),
        "macros": macros,
        "preferences": preferences,
    }


# ─── TOOL CALL HANDLER ────────────────────────────────────────────────────────

def _handle_tool_intent(
    user_id: str,
    creator_id: str,
    user_message: str,
    intent: str,
    user_context: dict,
    creator: dict,
    remaining: int,
    deepseek_client,
) -> dict:
    """
    Handles create_program / modify_program using DeepSeek tool calling.
    Proposes the change and stores it as a pending action for user confirmation.
    Does NOT write to Supabase until the user confirms.
    """
    u = user_context.get("user") or {}
    p = user_context.get("preferences") or {}
    m = user_context.get("macros") or {}
    program = user_context.get("program")

    program_info = "None"
    if program and program.get("programs"):
        prog = program["programs"]
        # current_day is the NEXT session, not today's. Say so, or the model
        # reports "today is Day N" to someone who already trained today.
        next_day = program.get("current_day", 1)
        program_info = (
            f"{prog.get('title', 'Custom Plan')} "
            f"Week {program.get('current_week', 1)}, "
            f"next session: Day {next_day}"
        )
        if user_context.get("todays_workout"):
            program_info += " (already trained today — Day {} is NOT today's session)".format(next_day)

    # For modify_program, fetch the full current program structure
    current_program_section = ""
    if intent == "modify_program":
        current = _fetch_current_program_structure(user_id)
        if current:
            day_lines = []
            for d in current["days"]:
                if d["is_rest_day"]:
                    day_lines.append(f"  Day {d['day_number']}: {d['title']} (REST)")
                else:
                    ex_list = ", ".join(e["name"] for e in d.get("exercises", []))
                    day_lines.append(f"  Day {d['day_number']}: {d['title']} — {ex_list}")
            current_program_section = (
                f"\n\nCURRENT PROGRAM TO MODIFY:\n"
                f"Title: {current['title']}\n"
                f"Difficulty: {current['difficulty']} | Goal: {current['goal']} | Weeks: {current['duration_weeks']}\n"
                + "\n".join(day_lines)
                + "\n\nApply the user's requested change to this program. Rules:"
                "\n- Keep all exercise names exactly as they appear above unless the user asks to swap one."
                "\n- Return the COMPLETE updated program (all days) via propose_workout_program — not just the changed days."
                "\n- Do not add or remove days unless the user explicitly asks."
            )
        else:
            current_program_section = "\n\nNote: No active program found. Build a new one based on the user's preferences."

    # Build exercise list for the prompt (only when creating or modifying a program)
    exercise_list_section = ""
    # For modify_program the LLM edits existing exercises — it doesn't need the full library.
    # For create_program, include the list but cap at 200 names to stay inside token budget.
    if intent == "create_program":
        ex_names = _get_exercise_names(p.get("equipment", ""))
        if ex_names:
            exercise_list_section = (
                "\n\nAVAILABLE EXERCISES — you MUST only use names from this list, spelled exactly as shown:\n"
                + ", ".join(ex_names[:200])
            )

    # ── Onboarding preferences (muscle priorities / rest / routine) ───────────
    # Only surfaced when the user actually set them, so the prompt stays clean
    # for anyone who skipped these steps.
    _grow_m    = [x for x in (p.get("target_muscles_grow") or []) if x]
    _define_m  = [x for x in (p.get("target_muscles_define") or []) if x]
    _exclude_m = [x for x in (p.get("target_muscles_exclude") or []) if x]
    muscle_priority_line = ""
    if _grow_m or _define_m or _exclude_m:
        muscle_priority_line = (
            "\nMuscle priorities — "
            f"GROW (bias more volume here): {', '.join(_grow_m) or 'none'} | "
            f"DEFINE (maintain, don't grow): {', '.join(_define_m) or 'none'} | "
            f"MINIMIZE (little/no direct work): {', '.join(_exclude_m) or 'none'}"
        )
    _rest_pref = p.get("rest_time_seconds")
    rest_line = f"\nPreferred rest between sets: {_rest_pref}s (use as the default)" if _rest_pref else ""
    _routine_pref = (p.get("routine_structure") or "").strip()
    _ROUTINE_HINT = {
        "Plan":     "trains a fixed split — respect a clean, repeatable structure",
        "Flexible": "trains with a rough plan — keep structure but allow easy swaps",
        "Wing it":  "decides in the gym — favour simple, flexible day templates over a rigid split",
    }
    routine_line = (
        f"\nTraining approach: {_routine_pref} ({_ROUTINE_HINT.get(_routine_pref, '')})"
        if _routine_pref else ""
    )

    # Durable facts the user confirmed in chat ("no overhead pressing, shoulder
    # impingement"). This prompt is built from scratch off user_preferences and
    # never saw them — so the coach would accept the limitation, then hand back
    # a program that violated it on day 1. Nothing erodes trust faster than the
    # app forgetting something you just told it.
    _facts = get_facts_block(user_id)
    facts_section = (
        f"\n\n{_facts}\n"
        "Treat any [Limitation] above exactly like an entry in the injuries list — "
        "never program a movement that violates it, and honour [Preference] and "
        "[Exception] entries in your selections."
        if _facts else ""
    )

    # The questionnaire ideology — personality, training school, signature
    # phrases, non-negotiables and "never program" — was fetched by
    # get_creator_context and then dropped here, so the coach reasoned like the
    # creator in conversation and reverted to a generic engine for the one
    # artefact the user keeps. A creator who said "never program behind-the-neck
    # press" was honoured in chat and ignored in their own program.
    # Same renderer chat uses; returns "" for an unconfigured creator, so their
    # prompt is unchanged.
    _ideology_block = render_creator_ideology(creator)

    system = f"""You are {creator.get('name', 'an AI fitness coach')} on RIPPD.
You are direct, data-driven, and elite. You speak like a high-end coach, not a chatbot.

USER PROFILE:
Name: {(u.get('full_name') or '').split()[0] if (u.get('full_name') or '').split() else 'there'}
Age: {p.get('age', '?')} | Gender: {p.get('gender', '?')}
Weight: {p.get('weight', '?')}kg | Height: {p.get('height', '?')}cm
Goal: {p.get('primary_goal', 'general fitness')}
Fitness level: {p.get('fitness_level', 'intermediate')}
Equipment: {p.get('equipment', 'full gym')}
Available days/week: {p.get('available_days_per_week', 4)}
Training days: {', '.join(p.get('training_days') or ['Mon', 'Tue', 'Thu', 'Fri'])}
Injuries — NEVER load these: {p.get('injuries') or 'None'}{muscle_priority_line}{rest_line}{routine_line}
Diet preference: {p.get('diet_preference', 'any')} | Cuisine: {p.get('cuisine_preference', 'Indian')}
Allergies/dislikes — NEVER suggest: {p.get('allergies_or_dislikes') or 'None'}{facts_section}

MACRO TARGETS:
Calories: {m.get('calories', '?')} kcal | Protein: {m.get('protein_g', '?')}g
Carbs: {m.get('carbs_g', '?')}g | Fat: {m.get('fat_g', '?')}g

CURRENT PROGRAM: {program_info}

CREATOR STYLE: {creator.get('coaching_philosophy', 'Evidence-based, progressive overload')}
TONE: {creator.get('tone', 'direct, motivating')}
{_ideology_block}

CRITICAL RULES:
- Call the appropriate tool — do NOT respond in text for a program/diet request
- Be specific to THIS user — never give a generic template; use their goal, level, equipment

PROGRAM RULES (read carefully, every point is enforced):
0. Build the program NOW from the profile above. Goal, training days, session length,
   equipment and level are all known — never ask the user to repeat them, and never reply
   with questions instead of a tool call. If something is genuinely missing, choose a sane
   default and say which default you chose in coaching_rationale.
1. Training days: if the user explicitly states a number (e.g. "5 day split", "3 days a week"), use THAT number.
   Otherwise default to their profile: {p.get('available_days_per_week', 4)} days.
   The `days` array MUST contain exactly that many training days — no more, no fewer.
   Pad the 7-day week with rest days so day_number 1–7 are all present.
   Place rest days at the END of the week (e.g. day 6 and 7 for a 5-day plan) — never in the middle unless the split specifically requires it (e.g. Upper/Lower needs a mid-week rest on day 3).
2. For named splits, build the FULL structure:
   - PPL (Push/Pull/Legs): Push = chest + shoulders + triceps | Pull = back + biceps | Legs = quads + hams + glutes + calves
     If 6 days → PPL repeated twice (P1, P2, L1, P3, P4, L2). If 3 days → one cycle.
   - Upper/Lower: alternate Upper (chest/back/shoulders/arms) and Lower (quads/hams/glutes/calves) across available days
   - Upper/Lower + Full-body (5 days): Mon Upper, Tue Lower, Wed Full-body, Thu Upper, Fri Lower
   - Bro split (5 days): Chest | Back | Shoulders | Arms | Legs
   - Full-body: every session hits all major muscle groups
3. Each training day needs 5–8 exercises, UNLESS the session length cannot hold them.
   Their sessions are ~{p.get('session_duration_mins', 45)} min. Budget it: ~8 min warm-up, then per
   exercise sets x (working time + their rest). If 5 will not fit, build the number that does —
   drop a set before you shorten rest, and never pad a day past its time.
4. Sets/reps — use BOTH fitness level AND goal. Start with level baseline, then apply goal override:
   Fitness level baseline: beginner = 3 sets, intermediate = 4 sets, advanced = 5 sets.
   GOAL OVERRIDES (mandatory — these define the rep target for every exercise):
   • "Lose Fat" / "Stay Healthy": 12–20 reps per set. Use higher end (15–20) for isolation, lower end (12–15) for compounds. Keep volume high.
   • "Build Muscle": 8–12 reps per set. Pyramid down: e.g. "12/10/8" or "10/8/8". Mix compound + isolation.
   • "Get Stronger": 3–6 reps per set. Pyramid down: e.g. "5/4/3/3" or "6/5/4". Lead each day with 1–2 barbell compounds (squat, deadlift, bench, OHP, row).
   Example slash-separated reps by goal: Lose Fat → "15/15/12", Build Muscle → "12/10/8", Get Stronger → "5/4/3".
5. Reps MUST be slash-separated per set e.g. '10/10/10' or '8/8/6' — NEVER use ranges like '8-12'.
   If the goal calls for a pyramid, the numbers must ACTUALLY descend: "12/10/8" is a pyramid,
   "12/12/12/12" is not. Do not describe a pyramid you did not build.
6. Every DAY title must describe what that day actually contains. A day holding both
   pressing and pulling work is an Upper day — never call it 'Push'. If you name a day
   after a split pattern, its exercises must match the definition in rule 2.
   Name the program accurately: '6-Day PPL Split', '4-Day Upper/Lower', '5-Day Bro Split', etc.
   Set duration_weeks to 8 unless the user specifies otherwise.
7. coaching_rationale: 2 sentences max — state WHY this split fits the user's goal and schedule.
   It must describe the plan you actually built. Never claim a rep scheme, a split shape or a
   session length that the exercises above do not show — the user reads it as fact.
8. Never load injured areas: {p.get('injuries') or 'None'}
9. rest_seconds — set per exercise based on GOAL (user's preferred: {_rest_pref or 90}s, use as anchor):
   • "Lose Fat" / "Stay Healthy": 45–75s. Shorter rest = more calories burned. Superset isolation work where possible.
   • "Build Muscle": 60–120s. Heavier compounds get 90–120s, isolation 60–90s.
   • "Get Stronger": 180–300s on main lifts (squat/deadlift/bench/OHP), 90–120s on accessories.
   If user has a preferred rest set, stay within 20% of it unless goal strongly conflicts.
10. Muscle priorities (only if listed in the profile above): add 1-2 extra exercises or sets for GROW muscles, keep DEFINE muscles at maintenance volume (1 movement), and add NO direct isolation for MINIMIZE muscles — they may still appear only as secondary movers in compounds. Never drop a muscle a requested split structurally needs.

DIET RULES:
- Use Indian foods only, match macro targets, never include: {p.get('allergies_or_dislikes') or 'None'}{current_program_section}{exercise_list_section}"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    # Only program building reaches here — generate_diet is served by the
    # deterministic template engine (see _handle_diet_generation).
    tools = GROQ_TOOLS

    # Appended to the system prompt on the retry below. Kept blunt on purpose:
    # the first attempt already had every rule and still did not call the tool.
    _TOOL_NUDGE = (
        "\n\nCRITICAL: Reply ONLY by calling the propose_workout_program function. "
        "Do not write the program as prose, do not ask a question, do not explain "
        "first. A prose answer is unusable — the app cannot save it, so the user "
        "gets a plan they cannot start. Call the function now with the complete "
        "program."
    )

    def _propose(nudge: str = "") -> object:
        return deepseek_client.chat.completions.create(
            model=MODEL,
            messages=(
                [{"role": "system", "content": system + nudge}] + messages[1:]
                if nudge else messages
            ),
            tools=tools,
            # NOT "required". MODEL is a thinking model since the deepseek-chat →
            # deepseek-v4-flash rename, and DeepSeek rejects tool_choice="required"
            # in thinking mode with a 400 ("Thinking mode does not support this
            # tool_choice"). That 400 was caught by the blanket except below and
            # surfaced as "Something went wrong generating your plan", so EVERY
            # coach program build failed while ordinary chat (Groq 70B) kept
            # working and hid it. Re-verified 2026-08-26: BOTH "required" and a
            # named function still 400. "auto" is the only option, so the tool
            # call cannot be forced — only asked for again, below.
            tool_choice="auto",
            max_tokens=8000,
            temperature=0.3,
        )

    try:
        response = _propose()
        choice = response.choices[0]

        # No tool call means an unusable answer: at best a program the user can
        # read but not save, at worst an empty bubble. One more attempt costs
        # ~20s on a path that has already failed, which beats handing back a
        # plan that cannot be started. Only ever retried once.
        if not (choice.finish_reason == "tool_calls" and choice.message.tool_calls):
            logger.warning(
                "[program] %s did not call the tool (intent=%s, finish=%s, chars=%s)"
                " — retrying once with an explicit instruction",
                MODEL, intent, choice.finish_reason,
                len(choice.message.content or ""),
            )
            response = _propose(_TOOL_NUDGE)
            choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            if tool_name == "propose_workout_program" and tool_args.get("days"):
                # Fresh builds get sets/rest-day rules enforced in code — a
                # modify only touches what the user asked, so leave its
                # structure alone.
                if intent == "create_program":
                    tool_args = _enforce_program_rules(tool_args)
                # Snap exercise names to real DB entries (catches LLM hallucinations)
                for day in tool_args["days"]:
                    for ex in day.get("exercises") or []:
                        ex["name"] = _snap_to_db_exercise(ex.get("name", ""))
                # Only now are the names real library rows, so difficulty can be
                # looked up. Applies to modify_program too: the user editing one
                # day is no reason to leave an over-ceiling lift in another.
                tool_args = _enforce_difficulty_ceiling(tool_args, p.get("fitness_level"))

            confirm_reply = _build_confirmation_message(tool_name, tool_args, u)
            pending_payload = {"tool": tool_name, "args": tool_args, "intent": intent}
            _store_pending_actions(user_id, creator_id, [pending_payload])

            # Persist the card with the reply so the thread can redraw it after a
            # reload, and so it survives the pending row being cleared on
            # confirm/cancel — which is exactly when the frozen card takes over.
            save_exchange(
                user_id, creator_id, user_message, confirm_reply,
                assistant_meta={"card": "program_proposal", "args": tool_args},
            )

            return {
                "response": confirm_reply,
                "pending_tool": tool_name,
                "pending_args": tool_args,
                "messages_remaining": remaining - 1,
                "upgrade_required": False,
                "intent": intent,
            }

        # Model replied in prose instead of calling the tool. This is the
        # quietest failure in the whole pipeline: the user gets a well-written
        # program they cannot save, and the logs record a successful turn. It
        # cannot be forced away — DeepSeek 400s on both tool_choice="required"
        # and a named function in thinking mode (see the comment above) — so the
        # least we can do is say it happened.
        logger.error(
            "[program] %s still did not call the tool AFTER the retry "
            "(intent=%s, finish=%s, chars=%s) — user gets an unsaveable plan",
            MODEL, intent, choice.finish_reason,
            len(choice.message.content or ""),
        )
        reply = choice.message.content or "Tell me more about what you're looking for and I'll build it for you."
        save_exchange(user_id, creator_id, user_message, reply)
        return {
            "response": reply,
            "messages_remaining": remaining - 1,
            "upgrade_required": False,
            "intent": intent,
        }

    # No catch-all: anything else raises so /chat returns a 503. An apology
    # returned here is how a 400 on every program build reached users as HTTP 200.
    except _groq_module.RateLimitError:
        return {
            "response": _RATE_LIMIT_RESPONSE,
            "messages_remaining": remaining,
            "upgrade_required": False,
            "intent": intent,
        }


def _handle_diet_generation(
    user_id: str,
    creator_id: str,
    user_message: str,
    remaining: int,
) -> dict:
    """
    Builds the 7-day plan with the deterministic template engine (the same one
    behind /diet-plan and the Diet tab) instead of having the LLM invent meals.

    An LLM asked to emit 7 days of Indian meals will hallucinate foods, invent
    macros that don't match the foods, and produce days that miss the calorie
    target outright — none of which is checkable after the fact. The template
    engine draws from curated meal_templates, hits the user's real macro
    targets, and reports its own accuracy. Same chat confirm card either way;
    only the source of truth changes.
    """
    try:
        plan = generate_diet_plan(user_id, save=False)  # persisted on confirm
    except ValueError as e:
        # No macros or no templates for this diet type — both are actionable.
        reply = (
            f"{e} Fill in your weight, height, age and goal in Profile → Edit Profile "
            "and I'll build the plan straight after."
            if "macro" in str(e).lower()
            else str(e)
        )
        save_exchange(user_id, creator_id, user_message, reply)
        return {
            "response": reply,
            "messages_remaining": remaining,
            "upgrade_required": False,
            "intent": "generate_diet",
        }
    # Anything else raises so /chat returns a 503 — see _handle_tool_intent.

    targets = plan.get("targets") or {}
    days = plan.get("days") or []

    # Rationale grounded in what the engine actually produced — the weakest
    # protein day is the honest trust signal, not a generic promise.
    protein_pcts = [
        d["accuracy"]["protein_g"]
        for d in days
        if (d.get("accuracy") or {}).get("protein_g") is not None
    ]
    if protein_pcts:
        rationale = (
            f"Built from your real targets — {int(targets.get('calories') or 0)} kcal and "
            f"{int(targets.get('protein_g') or 0)}g protein a day. Every day lands at "
            f"{min(protein_pcts)}-{max(protein_pcts)}% of your protein target, and the meals "
            f"respect your diet preference and anything you've told me you won't eat."
        )
    else:
        rationale = (
            f"Built from your real targets — {int(targets.get('calories') or 0)} kcal and "
            f"{int(targets.get('protein_g') or 0)}g protein a day, matched to your diet preference."
        )

    args = {
        "daily_calories":  targets.get("calories"),
        "daily_protein_g": targets.get("protein_g"),
        "daily_carbs_g":   targets.get("carbs_g"),
        "daily_fat_g":     targets.get("fats_g"),
        "diet_type":       plan.get("diet_type"),
        "rotation":        plan.get("rotation"),
        "coaching_rationale": rationale,
        "days":            days,
    }

    confirm_reply = _build_confirmation_message("propose_diet_plan", args, {})
    _store_pending_actions(
        user_id, creator_id,
        [{"tool": "propose_diet_plan", "args": args, "intent": "generate_diet"}],
    )
    save_exchange(user_id, creator_id, user_message, confirm_reply)

    return {
        "response": confirm_reply,
        "pending_tool": "propose_diet_plan",
        "pending_args": args,
        "messages_remaining": remaining - 1,
        "upgrade_required": False,
        "intent": "generate_diet",
    }


def _build_confirmation_message(tool_name: str, args: dict, user: dict) -> str:
    """Builds a human-readable confirmation card asking the user to say yes/no."""
    if tool_name == "propose_workout_program":
        days = args.get("days", [])
        training_days = [d for d in days if not d.get("is_rest_day")]
        rest_days = [d for d in days if d.get("is_rest_day")]

        day_lines = []
        for day in days:
            if day.get("is_rest_day"):
                day_lines.append(f"  Day {day['day_number']}: {day['title']} — Rest")
            else:
                exercises = day.get("exercises", [])
                ex_names = ", ".join(ex["name"] for ex in exercises[:4])
                if len(exercises) > 4:
                    ex_names += f" +{len(exercises) - 4} more"
                day_lines.append(f"  Day {day['day_number']}: {day['title']} — {ex_names}")

        return (
            f"{args.get('coaching_rationale', '')}\n\n"
            f"Here's what I've built for you:\n\n"
            f"**{args['title']}**\n"
            f"{args['duration_weeks']} weeks | {args['difficulty']} | "
            f"{len(training_days)} training days, {len(rest_days)} rest days\n\n"
            + "\n".join(day_lines)
            + "\n\nThis will replace your current active program.\n"
            "Say **yes** to activate it or **no** to cancel."
        )

    elif tool_name == "propose_diet_plan":
        days = args.get("days", [])
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_lines = []
        for i, day in enumerate(days[:7]):
            meals = day.get("meals", [])
            # The template engine names meals "name"; the old LLM tool used
            # "meal_name". Accept either — diet/plan.tsx already normalises both.
            meal_names = " · ".join(
                (m.get("name") or m.get("meal_name") or "Meal") for m in meals[:4]
            )
            if len(meals) > 4:
                meal_names += f" +{len(meals) - 4} more"
            totals = day.get("totals", {})
            day_lines.append(
                f"  **{day_names[i] if i < len(day_names) else f'Day {i+1}'}**: "
                f"{totals.get('calories', '?')} kcal · {totals.get('protein_g', '?')}g protein — {meal_names}"
            )

        return (
            f"{args.get('coaching_rationale', '')}\n\n"
            f"Your 7-day plan ({args['daily_calories']} kcal/day | "
            f"{args['daily_protein_g']}g protein/day):\n\n"
            + "\n".join(day_lines)
            + "\n\nSay **yes** to save this plan or **no** to cancel."
        )

    return "Here's what I've prepared. Say **yes** to confirm or **no** to cancel."


# ─── Meal suggestion extraction (for in-chat meal cards) ─────────────────────

_MEAL_EXTRACT_PROMPT = """Extract the specific meals or foods the coach is SUGGESTING the user eat, from the coach message below.
Rules:
- Only meals proposed as something to eat (now or later). If the message is reviewing past intake, giving general advice, or proposing a workout, return {"meals":[]}.
- Max 3 meals. Use the macros stated in the message; estimate briefly only when missing.
- meal_type is one of breakfast|morning_snack|lunch|evening_snack|dinner if the message implies one, else null.
Respond ONLY with JSON: {"meals":[{"name":str,"calories":int,"protein_g":float,"carbs_g":float,"fat_g":float,"meal_type":str|null}]}"""

_MEAL_HINT_RE = re.compile(r"kcal|calorie", re.I)


def _extract_meal_suggestions(reply_text: str) -> list | None:
    """Best-effort fast-model pass over the coach reply. Returns None when the
    reply isn't suggesting food, so the app renders a plain text bubble."""
    if not reply_text or len(reply_text) < 60:
        return None
    # Cheap gate: only run the extractor when the reply talks macros.
    if not _MEAL_HINT_RE.search(reply_text) or "protein" not in reply_text.lower():
        return None

    client = _groq()
    response = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[
            {"role": "system", "content": _MEAL_EXTRACT_PROMPT},
            {"role": "user", "content": reply_text[:2000]},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=900,  # headroom for gpt-oss reasoning tokens
        **_groq_extra(MODEL_FAST),
    )
    data = json.loads(response.choices[0].message.content)

    valid_slots = {"breakfast", "morning_snack", "lunch", "evening_snack", "dinner"}
    meals = []
    for m in data.get("meals", [])[:3]:
        try:
            calories = int(round(float(m.get("calories", 0))))
            if not m.get("name") or calories <= 0:
                continue
            slot = m.get("meal_type")
            meals.append(
                {
                    "name": str(m["name"])[:80],
                    "calories": calories,
                    "protein_g": round(max(0.0, float(m.get("protein_g", 0))), 1),
                    "carbs_g": round(max(0.0, float(m.get("carbs_g", 0))), 1),
                    "fat_g": round(max(0.0, float(m.get("fat_g", 0))), 1),
                    "meal_type": slot if slot in valid_slots else None,
                }
            )
        except (TypeError, ValueError):
            continue
    return meals or None


# ─── Exercise recommendation extraction (for in-chat "add to plan" cards) ────

_EXERCISE_EXTRACT_PROMPT = """Extract the specific exercises the coach is RECOMMENDING the user do, from the coach message below.
Rules:
- Only exercises proposed as something to add to training (not exercises being discussed generally or swapped out). If the message doesn't recommend concrete exercises to do, return {"exercises":[]}.
- Max 6 exercises, in the order given.
- sets is an integer. reps is a slash-separated string with one value per set (e.g. sets=3 → reps="10/10/8"). If the message gives a rep range like "8-10", pick the top of the range for every set.
- rest_seconds: a sensible default (60 for isolation, 90 for compound) if not stated.
- notes: a short cue if the message gives one (e.g. "control the negative"), else empty string.
Respond ONLY with JSON: {"exercises":[{"name":str,"sets":int,"reps":str,"rest_seconds":int,"notes":str}]}"""

_EXERCISE_HINT_RE = re.compile(r"\d+\s*[x×]\s*\d+|\bsets?\b.{0,20}\breps?\b|\breps?\b.{0,20}\bsets?\b", re.I)


def _extract_exercise_recommendations(reply_text: str) -> list | None:
    """Best-effort fast-model pass over the coach reply. Returns None when the
    reply isn't recommending concrete exercises, so the app renders plain text."""
    if not reply_text or len(reply_text) < 60:
        return None
    if not _EXERCISE_HINT_RE.search(reply_text):
        return None

    client = _groq()
    response = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[
            {"role": "system", "content": _EXERCISE_EXTRACT_PROMPT},
            {"role": "user", "content": reply_text[:2500]},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=1200,  # headroom for gpt-oss reasoning tokens
        **_groq_extra(MODEL_FAST),
    )
    data = json.loads(response.choices[0].message.content)

    exercises = []
    for ex in data.get("exercises", [])[:6]:
        try:
            sets = max(1, int(ex.get("sets", 3)))
            reps = str(ex.get("reps", "")).strip()
            if not ex.get("name") or not reps:
                continue
            exercises.append(
                {
                    "name": str(ex["name"])[:80],
                    "sets": sets,
                    "reps": reps,
                    "rest_seconds": max(15, int(ex.get("rest_seconds", 60))),
                    "notes": str(ex.get("notes", ""))[:150],
                }
            )
        except (TypeError, ValueError):
            continue
    return exercises or None


# ─── Coach memory fact extraction (for in-chat "remember this" cards) ────────
# Durable facts the USER states about themselves — injuries/limitations, standing
# preferences, negotiated exceptions — extracted from the user's own message
# (not the coach's reply, unlike meal/exercise extraction above).

_FACT_EXTRACT_PROMPT = """Extract any DURABLE facts about the user that a coach should remember for future conversations: physical injuries/limitations, standing food or exercise preferences/aversions, or exceptions the user has explicitly stated (e.g. "I don't do overhead pressing because of my shoulder", "I'm allergic to peanuts", "I always take Sundays as a cheat day").
Rules:
- Only extract facts stated as ongoing/durable about the user themselves — not a one-off log entry, a question, or something about a single workout/meal.
- Max 2 facts per message.
- fact_type is one of injury_limitation | preference | override.
- content is a short, self-contained statement under 140 characters, written in third person (e.g. "Avoids overhead pressing due to shoulder impingement").
If nothing durable is stated, return {"facts":[]}.
Respond ONLY with JSON: {"facts":[{"fact_type":str,"content":str}]}"""

_FACT_HINT_RE = re.compile(
    r"\b(injur\w*|hurt|pain(ful)?|sore|shoulder|knee|elbow|wrist|hip|ankle|"
    r"my back|avoid|allerg\w*|can'?t (do|eat)|never (do|eat)|instead of|"
    r"always (do|eat|take)|prefer not|don'?t (do|eat)|no longer|used to)\b",
    re.I,
)


def _extract_coach_memory_facts(user_message: str) -> list | None:
    """Best-effort fast-model pass over the USER's message. Returns None when
    nothing durable is stated, so no memory card is shown."""
    if not user_message or len(user_message) < 15:
        return None
    # Cheap gate: only run the extractor when the message hints at something durable.
    if not _FACT_HINT_RE.search(user_message):
        return None

    client = _groq()
    response = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[
            {"role": "system", "content": _FACT_EXTRACT_PROMPT},
            {"role": "user", "content": user_message[:1000]},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=700,  # headroom for gpt-oss reasoning tokens
        **_groq_extra(MODEL_FAST),
    )
    data = json.loads(response.choices[0].message.content)

    valid_types = {"injury_limitation", "preference", "override"}
    facts = []
    for f in data.get("facts", [])[:2]:
        try:
            fact_type = f.get("fact_type")
            content = str(f.get("content", "")).strip()
            if fact_type not in valid_types or not content:
                continue
            facts.append({"fact_type": fact_type, "content": content[:140]})
        except (TypeError, ValueError):
            continue
    return facts or None


def _finish_reason(resp) -> str | None:
    try:
        return resp.choices[0].finish_reason
    except Exception:
        return None


def _has_content(resp) -> bool:
    """True when the completion actually carries user-visible text.

    A thinking model can return a well-formed 200 whose message.content is ""
    because reasoning consumed the token budget. Callers treat that as a reply
    and render an empty bubble, so it has to be detected here.
    """
    try:
        return bool((resp.choices[0].message.content or "").strip())
    except Exception:
        return False


def tokens_and_temp_for_intent(intent: str) -> tuple[int, float]:
    """Production token budget and temperature per intent.

    These budgets must account for REASONING tokens. Since the deepseek-chat →
    deepseek-v4-flash rename, MODEL is a thinking model and its reasoning is
    billed against max_tokens before a single character of content is emitted.
    At the old 500 the model hit finish_reason="length" mid-reasoning and
    returned content="" — the user typed "i had 2 rotis and dal for lunch" and
    got a completely empty chat bubble. Measured against the live API:

        max_tokens=500  -> finish_reason=length, content empty
        max_tokens=1500 -> finish_reason=stop,   content present
    """
    if intent in ("workout_log", "weight_log", "pending_food_log", "exercise_swap"):
        return 2000, 0.3
    if intent in _RICH_REPLY_INTENTS:
        # A ceiling, not a target — cost/latency track actual output, and the
        # prompt targets ~200 words (~260-300 tokens) even for a full plan. The
        # extra headroom over that is for reasoning tokens, per the note above.
        return 2500, 0.6
    if intent == "greeting":
        return 700, 0.75  # headroom for gpt-oss reasoning tokens
    return MAX_RESPONSE_TOKENS, 0.7


def complete_for_intent(intent: str, messages: list, force_model: str | None = None) -> tuple:
    """Run the completion exactly the way production routes it.

    force_model overrides the routing table for A/B evaluation only — production
    never passes it. It exists so the golden suite can run identical prompts
    through both providers and answer "is Groq actually worse here?" with data
    instead of a hunch.

    Extracted so the golden eval suite exercises the SAME routing as a live
    turn. When this lived inline in chat(), any eval had to re-declare the model
    and temperature, and would silently keep testing the old model the day the
    routing changed — an eval that drifts from production is worse than none.

    Returns (response, model_used).
    """
    max_tokens, temp = tokens_and_temp_for_intent(intent)

    if force_model:
        client = _deepseek() if force_model == MODEL else _groq()
        return client.chat.completions.create(
            model=force_model, messages=messages,
            max_tokens=max_tokens, temperature=temp, top_p=0.9,
        ), force_model

    # Rich conversational replies go to DeepSeek, with Groq 70b as the FALLBACK.
    # This is reversed from the original wiring, which put Groq first "because
    # DeepSeek is ~20s". Measured 2026-07-19: DeepSeek is 2.35s median, and Groq's
    # FREE tier throttles to ~10s once it nears its 100k/day cap — which it hits
    # at ~54 rich messages/day, well under launch traffic. Both scored 39/39 on
    # the golden assertions, so correctness didn't decide it; the daily ceiling
    # did. DeepSeek is also cheaper per token ($0.14/$0.28 vs $0.59/$0.79).
    # Groq stays as the fallback so a DeepSeek outage still gets answered, and
    # the 8b classifier below stays on Groq free (different model, own quota).
    heavy = intent in ("exercise_swap", "workout_log", "weight_log", "pending_food_log")

    if intent in _RICH_REPLY_INTENTS or heavy:
        try:
            resp = _deepseek().chat.completions.create(
                model=MODEL, messages=messages,
                max_tokens=max_tokens, temperature=temp, top_p=0.9,
            )
            if _has_content(resp):
                return resp, MODEL
            # A thinking model that spends its whole budget reasoning returns
            # content="" with finish_reason="length". The budgets above make that
            # unlikely, but an empty bubble is the worst possible failure — it
            # looks like the app is broken and leaves nothing to retry from — so
            # never return one. Groq 70b does not think, so it always emits text.
            logger.warning(
                "[chat] %s returned empty content (finish=%s) for intent=%s — retrying on Groq 70b",
                MODEL, _finish_reason(resp), intent,
            )
        except Exception as err:
            logger.warning("[chat] DeepSeek failed (%s) — falling back to Groq 70b", err)
        return _groq().chat.completions.create(
            model=MODEL_FAST_70B, messages=messages,
            max_tokens=max_tokens, temperature=temp, top_p=0.9,
            **_groq_extra(MODEL_FAST_70B),
        ), MODEL_FAST_70B

    return _groq().chat.completions.create(
        model=MODEL_FAST, messages=messages,
        max_tokens=max_tokens, temperature=temp, top_p=0.9,
        **_groq_extra(MODEL_FAST),
    ), MODEL_FAST


def chat(
    user_id: str,
    user_message: str,
    creator_id: Optional[str] = None,
    is_premium: bool = False,
    prompt_context: Optional[str] = None,
    brief: Optional[dict] = None,
) -> dict:
    """
    Main entry point. Called from your API endpoint.
    Returns the AI coach response or an upgrade prompt.

    Args:
        user_id: Supabase user ID
        user_message: What the user typed
        creator_id: Selected creator coach ID (optional)
        is_premium: Whether the user has an active subscription

    Returns:
        {
            "response": str,          # AI coach message
            "messages_remaining": int, # How many left today
            "upgrade_required": bool   # True if limit hit
        }
    """

    groq_client = _groq()
    deepseek_client = _deepseek()

    # Phase timing — logged so we can see where the 30-45s actually goes
    # (classify+context vs the model call). Grep Render logs for "[chat timing]".
    _t0 = time.perf_counter()

    # 0. Safety-critical regex check first (crisis / medical / out_of_scope)
    intent = _classify_intent_regex(user_message)

    # "Build me a program, work around my knee" NAMES a constraint; it does not
    # report an injury for advice. Hand those back to the classifier so they can
    # reach the tool path and produce a saveable program.
    #
    # Deliberately narrow: only the COACHABLE "medical" branch yields. crisis
    # and medical_urgent — chest pain, numbness, the red flags — still win
    # outright and are never reachable from here. The injury itself is not lost
    # either: it stays in the message text, and PROGRAM RULES rule 8 ("Never
    # load injured areas") is what acts on it.
    if intent == "medical" and _is_program_request(user_message):
        intent = None

    # 0a. Rest-timer adjustment is deterministic — a settings change, not a
    # conversation, so it never goes through the LLM classifier.
    if intent is None and _detect_rest_adjust(user_message):
        intent = "rest_adjust"

    # 0b. A cheat meal / night out the user is PLANNING is a coaching question,
    # not a food log. Left to the classifier it lands on pending_food_log and
    # the user gets a "log this?" card instead of a strategy. Only when no
    # safety intent fired — "my knee hurts, party tonight" stays medical.
    cheat_context = intent is None and _detect_cheat_plan(user_message)
    if cheat_context:
        intent = "question"

    # 0d. "hi" / "thanks" need no classifier. Checked here rather than inside the
    # block below so a greeting also skips the two lookups that only exist to
    # feed the classifier.
    if intent is None:
        intent = _fast_intent(user_message)

    # Which context function to call depends only on `prompt_context`, which is
    # an argument — NOT on the intent. So the context read has no reason to wait
    # for classification, yet it used to: the classifier is a Groq round trip
    # plus two Supabase reads, and every one of those finished before the first
    # context query was even issued. Start it here and collect it below, so the
    # classifier overlaps the context fetch instead of queueing ahead of it.
    _ctx_pool = ThreadPoolExecutor(max_workers=3)
    _f_context = _ctx_pool.submit(
        _get_user_context_lightweight if prompt_context else get_user_context, user_id
    )

    # 0c. If not a safety intent, use LLM classifier with conversation context
    early_history: list[dict] = []
    if intent is None:
        # These two are independent of each other as well.
        _f_history = _ctx_pool.submit(
            lambda: supabase.table("ai_conversations")
            .select("role, content")
            .eq("user_id", user_id)
            .in_("role", ["user", "assistant"])
            .order("created_at", desc=True)
            .limit(3)
            .execute()
            .data
        )
        _f_pending = _ctx_pool.submit(_fetch_pending_actions, user_id)
        early_history = _f_history.result()
        early_history = list(reversed(early_history)) if early_history else []
        has_pending = _f_pending.result() is not None
        # A bare "yes"/"no" answering a pending card is unambiguous now that we
        # know something IS pending — no reason to ask an LLM about it.
        intent = _fast_intent(user_message, has_pending) or classify_intent_smart(
            user_message, early_history, has_pending, groq_client
        )

    _ctx_pool.shutdown(wait=False)

    # Tag every row this turn writes, whichever branch ends up writing it.
    # Set unconditionally, including to None: asyncio.to_thread reuses worker
    # threads, and a turn WITHOUT a brief must never inherit the previous one.
    _BRIEF_CTX.set(brief)

    # 1. User context (or lightweight version when pre-built context is injected)
    user_context = _f_context.result()

    _t_ctx = time.perf_counter()
    logger.info("[chat timing] intent=%s classify+context=%.2fs", intent, _t_ctx - _t0)

    # Sticky pain flag: "my shoulder hurts" → "can we swap the bench?" is ONE
    # injury conversation, but only the first message mentions pain. Look back a
    # couple of user turns so the follow-up keeps its guardrails.
    pain_context = _mentions_pain(user_message) or any(
        h.get("role") == "user" and _mentions_pain(h.get("content", ""))
        for h in early_history[-4:]
    )

    if not creator_id:
        program = user_context.get("program")
        if (
            program
            and program.get("programs")
            and program["programs"].get("creators")
            and program["programs"]["creators"].get("id")
        ):
            creator_id = program["programs"]["creators"]["id"]
        else:
            # Fallback: get first followed coach
            subs = (
                supabase.table("creator_subscriptions")
                .select("creator_id")
                .eq("user_id", user_id)
                .eq("active", True)
                .limit(1)
                .execute()
                .data
            )
            if subs:
                creator_id = subs[0]["creator_id"]
            else:
                return {
                    "response": "Please follow a coach first before chatting with your AI coach.",
                    "messages_remaining": None,
                    "upgrade_required": False,
                    "intent": intent,
                }

    # 0. Bypass RAG on UI actions / system events
    msg_lower = user_message.lower().strip()
    if msg_lower == "clear chat":
        supabase.table("ai_conversations").delete().eq("user_id", user_id).eq(
            "creator_id", creator_id
        ).execute()
        return {
            "response": "Chat history cleared. What are we hitting next?",
            "messages_remaining": PREMIUM_MESSAGE_LIMIT
            if is_premium
            else FREE_MESSAGE_LIMIT,
            "upgrade_required": False,
            "intent": "system",
        }

    if intent == "crisis":
        save_exchange(user_id, creator_id, user_message, _CRISIS_RESPONSE)
        return {
            "response": _CRISIS_RESPONSE,
            "messages_remaining": None,
            "upgrade_required": False,
            "intent": intent,
        }

    if intent == "medical_urgent":
        # Deterministic, like crisis: a red flag must never depend on the model
        # deciding to be careful this time.
        save_exchange(user_id, creator_id, user_message, _MEDICAL_URGENT_RESPONSE)
        return {
            "response": _MEDICAL_URGENT_RESPONSE,
            "messages_remaining": None,
            "upgrade_required": False,
            "intent": intent,
        }

    if intent == "out_of_scope":
        save_exchange(user_id, creator_id, user_message, _OOS_RESPONSE)
        return {
            "response": _OOS_RESPONSE,
            "messages_remaining": None,
            "upgrade_required": False,
            "intent": intent,
        }

    # Confirm / cancel — no Groq call, no message charge
    if intent == "confirm":
        pending = _fetch_pending_actions(user_id)
        if not pending:
            return {
                "response": "Nothing to confirm — I don't have any pending logs for you. Send your workout or weight first.",
                "messages_remaining": None,
                "upgrade_required": False,
                "intent": intent,
            }
        try:
            actions = pending["actions"]

            # Clear pending BEFORE executing — prevents duplicate execution if
            # the user taps confirm twice or a network retry fires a second request.
            _clear_pending_actions(user_id)

            # Tool-call confirmation (create_program, generate_diet)
            if actions and isinstance(actions[0], dict) and "tool" in actions[0]:
                tool_action = actions[0]
                tool_name = tool_action["tool"]
                tool_args = tool_action["args"]

                if tool_name == "propose_workout_program":
                    result = execute_program_creation(user_id, tool_args, creator_id=pending.get("creator_id"))
                    if result.get("success"):
                        reply = f"Done! **{result['title']}** is now your active program. Head to the workout tab to start. Let's get after it. 💪"
                        try:
                            invalidate_slice(user_id, "training")
                        except Exception:
                            pass
                    else:
                        reply = "Something went wrong saving the program. Please try again."

                elif tool_name == "propose_diet_plan":
                    result = execute_diet_plan_save(user_id, tool_args)
                    if result.get("success"):
                        reply = "Diet plan saved. Check the Diet tab to see your full meal plan."
                        try:
                            invalidate_slice(user_id, "nutrition")
                        except Exception:
                            pass
                    else:
                        reply = "Something went wrong saving the diet plan. Please try again."

                else:
                    reply = "Unknown action. Please try again."

            else:
                # Legacy confirm flow — food log, workout log, weight log
                action_results = execute_actions(user_id, actions)
                new_pbs = action_results.get("new_pbs", [])
                deleted = [d for d in (action_results.get("deleted") or []) if d]
                _has_delete = any(a.get("type") == "delete_food_log" for a in actions)
                _has_relog = any(a.get("type") == "log_food" for a in actions)
                # A progressive-overload bump edits the program; "All logged"
                # would be the wrong word for it.
                bumped = [
                    u.get("exercise_name")
                    for a in actions if a.get("type") == "update_target_weights"
                    for u in (a.get("updates") or [])
                    if u.get("exercise_name")
                ]
                if _has_delete:
                    # Correction (delete + re-log) vs a plain removal.
                    _what = ", ".join(deleted) if deleted else "that entry"
                    reply = (
                        f"Fixed — updated {_what} in today's log."
                        if _has_relog
                        else f"Done — removed {_what} from today's log."
                    )
                elif bumped:
                    reply = (
                        f"Done — {', '.join(bumped)} updated in your program. "
                        "You'll see the new target weights next time you train them. "
                        "Earn every rep. 🔥"
                    )
                elif new_pbs:
                    pb_str = " and ".join(new_pbs)
                    reply = f"Logged! New PR on {pb_str}! Keep pushing. 🔥"
                else:
                    reply = "Done! All logged."

        except Exception:
            traceback.print_exc()
            reply = "Something went wrong saving your log. Please try again."

        save_exchange(user_id, creator_id, user_message, reply)
        return {
            "response": reply,
            "messages_remaining": None,
            "upgrade_required": False,
            "intent": intent,
        }

    if intent == "cancel":
        # Word it for whatever was actually pending. "Nothing logged" is
        # food-log copy; saying it after declining a workout program reads as a
        # failed save rather than a declined suggestion.
        _pending = _fetch_pending_actions(user_id)
        _tool = None
        if _pending:
            _acts = _pending.get("actions") or []
            if _acts and isinstance(_acts[0], dict):
                _tool = _acts[0].get("tool")
        _clear_pending_actions(user_id)
        if _tool == "propose_workout_program":
            _cancel_reply = "No problem — your current plan is unchanged."
        elif _tool == "propose_diet_plan":
            _cancel_reply = "No problem — your diet plan is unchanged."
        else:
            _cancel_reply = "No problem, nothing logged."
        save_exchange(user_id, creator_id, user_message, _cancel_reply)
        return {
            "response": _cancel_reply,
            "messages_remaining": None,
            "upgrade_required": False,
            "intent": intent,
        }

    if intent == "rest_adjust":
        # Deterministic settings change: update the default rest timer in
        # user_preferences. Clamped 30–300s. Applies to new programs and the
        # workout rest-timer default; existing programs keep their per-exercise
        # rest_seconds until regenerated.
        prefs_row = _maybe_data(
            supabase.table("user_preferences")
            .select("rest_time_seconds")
            .eq("user_id", user_id)
            .maybe_single().execute()
        )
        current_rest = int((prefs_row or {}).get("rest_time_seconds") or 90)
        new_rest = max(30, min(300, _compute_rest_adjust(user_message, current_rest)))

        if new_rest == current_rest:
            reply = f"Your rest timer is already at {current_rest}s — that's the floor/ceiling I'd recommend (30s–5min range)."
        else:
            supabase.table("user_preferences").update(
                {"rest_time_seconds": new_rest}
            ).eq("user_id", user_id).execute()
            invalidate_slice(user_id, "training")
            direction = "up" if new_rest > current_rest else "down"
            mins = f"{new_rest // 60}:{new_rest % 60:02d}" if new_rest >= 60 else f"{new_rest}s"
            reply = (
                f"Done — rest timer {direction} from {current_rest}s to {new_rest}s ({mins}). "
                "It applies to new programs and your default timer; say the word if you want your current program's rest updated too."
            )

        save_exchange(user_id, creator_id, user_message, reply)
        return {
            "response": reply,
            "messages_remaining": None,
            "upgrade_required": False,
            "intent": intent,
        }

    if intent == "skip_workout":
        # Streaks are now weekly-goal-based (see src/lib/streak.ts), so a single
        # missed day only matters if it's needed to still hit the week's goal —
        # no more freeze credits to spend.
        today = today_local()
        today_str = today.isoformat()
        monday_str = (today - timedelta(days=today.weekday())).isoformat()

        already_done = supabase.table("workout_logs") \
            .select("id") \
            .eq("user_id", user_id) \
            .eq("date", today_str) \
            .eq("completed", True) \
            .limit(1).execute().data
        if already_done:
            reply = "You already completed today's session — no need to skip anything. Rest up and I'll see you tomorrow."
        else:
            prefs = _maybe_data(
                supabase.table("user_preferences")
                .select("available_days_per_week")
                .eq("user_id", user_id)
                .maybe_single().execute()
            )
            weekly_goal = (prefs or {}).get("available_days_per_week") or 3

            week_count_result = supabase.table("workout_logs") \
                .select("id", count="exact") \
                .eq("user_id", user_id) \
                .eq("completed", True) \
                .gte("date", monday_str) \
                .execute()
            done_this_week = week_count_result.count or 0
            remaining = max(0, weekly_goal - done_this_week)

            if remaining == 0:
                reply = "No problem — you've already hit your workout goal for this week, so today's a free pass."
            elif remaining == 1:
                reply = (
                    "Totally fine to rest today, but heads up — you'll need one more session "
                    "before the week's out to keep your streak alive."
                )
            else:
                reply = f"No worries, rest up. You've got {remaining} more workouts to fit in this week to stay on track."

        save_exchange(user_id, creator_id, user_message, reply)
        return {
            "response": reply,
            "messages_remaining": None,
            "upgrade_required": False,
            "intent": intent,
        }

    # ── check_in after a meal suggestion → ask if they ate it ─────────────────
    # If the last assistant message contained a meal/food suggestion (detected by
    # gram quantities or meal keywords) and the user sent a short ambiguous reply
    # (e.g. "done with dinner", "finished it"), nudge them to log it.
    if intent == "check_in":
        last_assistant = next(
            (m["content"] for m in reversed(early_history) if m["role"] == "assistant"),
            "",
        )
        meal_suggestion = bool(re.search(
            r"\b(\d+\s*g|\d+\s*(kcal|cal)|cup|scoop|roti|rice|dal|paneer|egg|chicken|oats|whey|banana|sabzi)\b",
            last_assistant,
            re.IGNORECASE,
        ))
        short_reply = len(user_message.split()) <= 6

        if meal_suggestion and short_reply:
            # Extract the suggested meal snippet (first 120 chars after a number+g pattern)
            snippet_match = re.search(r"([A-Z][^.!?\n]{10,120}(?:\d+\s*g|\d+\s*kcal)[^.!?\n]{0,80})", last_assistant)
            snippet = f' ("{snippet_match.group(1).strip()}")' if snippet_match else ""
            follow_up = (
                f"Did you eat the meal I suggested{snippet}? "
                f"Say **yes** and I'll log it, or tell me what you actually had."
            )
            save_exchange(user_id, creator_id, user_message, follow_up)
            return {
                "response": follow_up,
                "messages_remaining": None,
                "upgrade_required": False,
                "intent": "check_in",
            }

    # ── increase_weight — progressive overload readiness check ────────────────
    if intent == "increase_weight":
        can_send, remaining = check_message_limit(user_id, is_premium)
        if not can_send:
            return {
                "response": None,
                "messages_remaining": 0,
                "upgrade_required": True,
                "upgrade_message": f"You've used your {FREE_MESSAGE_LIMIT} free messages for today.",
            }

        weight_history = _fetch_exercise_weight_history(user_id, days=14)

        if not weight_history:
            # No logged workout data yet — advise the user
            reply = (
                "I don't have any workout weight data for you yet — weights only get tracked once you complete a session through the app. "
                "Do a few workouts and I'll be able to tell you exactly when and how much to increase. "
                "In the meantime, the rule is simple: if you hit all your reps cleanly for 2 weeks at the same weight, bump it up."
            )
            save_exchange(user_id, creator_id, user_message, reply)
            return {
                "response": reply,
                "messages_remaining": remaining - 1,
                "upgrade_required": False,
                "intent": intent,
            }

        # Check readiness per exercise and build a response
        ready_exercises: list[dict] = []
        not_ready_exercises: list[dict] = []

        for name, sessions in weight_history.items():
            readiness = _check_weight_readiness(sessions)
            if readiness["ready"] and readiness["last_weight"]:
                suggested = _suggest_weight_increment(name, readiness["last_weight"])
                ready_exercises.append({
                    "name": name.title(),
                    "current": readiness["last_weight"],
                    "suggested": suggested,
                    "sessions": readiness["sessions"],
                    "weeks": readiness["weeks"],
                })
            else:
                not_ready_exercises.append({
                    "name": name.title(),
                    "reason": readiness["reason"],
                    "sessions": readiness["sessions"],
                    "weeks": readiness["weeks"],
                    "last_weight": readiness["last_weight"],
                })

        if not ready_exercises and not not_ready_exercises:
            reply = (
                "No workout data found in the last 2 weeks. Complete some sessions first and I'll run the readiness check for you."
            )
            save_exchange(user_id, creator_id, user_message, reply)
            return {
                "response": reply,
                "messages_remaining": remaining - 1,
                "upgrade_required": False,
                "intent": intent,
            }

        if ready_exercises:
            # The readiness numbers are already exact. This used to be handed to
            # DeepSeek with "re-emit the whole program with these applied" —
            # a slow round-trip that could quietly reword titles or drop
            # exercises it was told to leave alone. Propose the precise edit
            # instead and let confirm write it.
            updates = [
                {"exercise_name": e["name"], "new_weight_kg": e["suggested"]}
                for e in ready_exercises
            ]
            _store_pending_actions(
                user_id, creator_id,
                [{"type": "update_target_weights", "updates": updates}],
            )

            reply = (
                "You've earned it — here's what's going up:\n\n"
                + "\n".join(
                    f"  • **{e['name']}**: {e['current']}kg → **{e['suggested']}kg** "
                    f"({e['sessions']} sessions over {e['weeks']} weeks at the same load)"
                    for e in ready_exercises
                )
                + (
                    "\n\nHolding steady for now (need 2 weeks at a weight before I bump it):\n"
                    + "\n".join(
                        f"  • {e['name']}: {e['sessions']} session(s) logged"
                        + (f", {e['last_weight']}kg last time" if e["last_weight"] else "")
                        for e in not_ready_exercises
                    )
                    if not_ready_exercises else ""
                )
                + "\n\nThis updates the target weights in your program. "
                "Say **yes** to apply it or **no** to leave it as is."
            )
            save_exchange(user_id, creator_id, user_message, reply)
            return {
                "response": reply,
                "pending_actions": [{"type": "update_target_weights", "updates": updates}],
                "messages_remaining": remaining - 1,
                "upgrade_required": False,
                "intent": intent,
            }
        else:
            # None ready — give advice and explain why
            not_ready_lines = "\n".join(
                f"- {e['name']}: {e['sessions']} session(s) logged"
                + (f" over {e['weeks']} weeks" if e["weeks"] else "")
                + (f", {e['last_weight']}kg last used" if e["last_weight"] else "")
                for e in not_ready_exercises
            )
            reply = (
                f"Not quite ready to increase yet — my 2-week rule says you need at least 14 days of consistent reps at the same weight before bumping up.\n\n"
                f"Here's where you're at:\n{not_ready_lines}\n\n"
                f"Keep training at your current weights. If you're completing all your sets cleanly, you're building the base — "
                f"I'll flag you when it's time to add weight. If you still want to increase now, just say so and I'll update the program."
            )
            save_exchange(user_id, creator_id, user_message, reply)
            return {
                "response": reply,
                "messages_remaining": remaining - 1,
                "upgrade_required": False,
                "intent": intent,
            }

    # ── Tool-calling intents (create_program, generate_diet) ──────────────────
    if intent in ("create_program", "modify_program", "generate_diet"):
        can_send, remaining = check_message_limit(user_id, is_premium)
        if not can_send:
            return {
                "response": None,
                "messages_remaining": 0,
                "upgrade_required": True,
                "upgrade_message": f"You've used your {FREE_MESSAGE_LIMIT} free messages for today.",
            }

        # Meal plans are generated, not written: the template engine hits the
        # user's real macro targets from curated foods. Doesn't need the creator
        # persona — the plan is numbers, and the rationale is derived from them.
        if intent == "generate_diet":
            return _handle_diet_generation(user_id, creator_id, user_message, remaining)

        creator = get_creator_context(creator_id)
        if not creator:
            return {
                "response": "Your coach's profile isn't available right now. Please try again.",
                "messages_remaining": remaining,
                "upgrade_required": False,
            }
        return _handle_tool_intent(
            user_id=user_id,
            creator_id=creator_id,
            user_message=user_message,
            intent=intent,
            user_context=user_context,
            creator=creator,
            remaining=remaining,
            deepseek_client=deepseek_client,
        )

    try:
        # These four hit the network and none of them reads another's result, but
        # they ran strictly one after the next: two Supabase round trips, the
        # history fetch, and (on question/medical) a Gemini embedding call for
        # the knowledge search. ~1s of dead time before the model was even asked.
        # Start them together here and collect each where it's actually needed.
        _io = ThreadPoolExecutor(max_workers=4)
        _f_limit = _io.submit(check_message_limit, user_id, is_premium)
        _f_creator = _io.submit(get_creator_context, creator_id)
        _f_history = _io.submit(get_conversation_history, user_id, creator_id)
        _f_knowledge = (
            _io.submit(search_gym_knowledge, user_message, 2)
            if intent in ("question", "medical")
            else None
        )
        _io.shutdown(wait=False)

        # 1. Check message limit
        can_send, remaining = _f_limit.result()

        if not can_send:
            if not is_premium:
                return {
                    "response": None,
                    "messages_remaining": 0,
                    "upgrade_required": True,
                    "upgrade_message": f"You've used your {FREE_MESSAGE_LIMIT} free messages for today. Upgrade to Premium for {PREMIUM_MESSAGE_LIMIT} messages daily. 🔥",
                }
            else:
                return {
                    "response": None,
                    "messages_remaining": 0,
                    "upgrade_required": False,
                    "upgrade_message": f"You've reached your daily limit of {PREMIUM_MESSAGE_LIMIT} messages. Come back tomorrow!",
                }

        # 2. Fetch creator context
        creator = _f_creator.result()
        if not creator:
            return {
                "response": "Your coach's profile isn't available right now. Please try again later.",
                "messages_remaining": remaining,
                "upgrade_required": False,
            }

        # 3. Build system prompt — passes intent so logging format is injected
        # For exercise_swap, find the alternative first so it can be injected
        swap_data = None
        if intent in ("exercise_swap", "medical"):
            # History is already in flight above; a swap almost always refers to
            # something named a turn or two earlier ("swap this").
            #
            # medical is included because branch B of that prompt TELLS the coach
            # to swap the movement, and it was doing so with no lookup at all —
            # inventing both the replacement and the reason ("swap deadlifts for
            # Romanian deadlifts, they don't require full range of motion", which
            # is not true of RDLs). Same class of bug as the Smith machine, just
            # in the intent where being wrong matters most.
            original = _resolve_swap_target(user_message, _f_history.result())
            if original:
                # A medical turn is always pain-driven by definition.
                swap_data = find_exercise_alternative(
                    original, user_id, pain_context or intent == "medical"
                )
            system_prompt = build_system_prompt(
                user_context, creator, intent, swap_data, prompt_context,
                pain_context=pain_context,
            )
        else:
            system_prompt = build_system_prompt(
                user_context, creator, intent, prompt_context=prompt_context,
                pain_context=pain_context,
                # Equipment matters here: an id the user cannot perform is
                # worse than no id, because the model will recommend it.
                exercise_catalog=_build_exercise_catalog(
                    user_message,
                    (user_context.get("preferences") or {}).get("equipment") or "",
                ) if intent == "question" else "",
                cheat_context=cheat_context,
            )

        # 3a. Inject nutrition context for food questions and pending food logs
        if intent in ("question", "pending_food_log", "correct_log"):
            nutrition_block = _build_nutrition_block(
                user_message,
                force_lookup=(intent in ("pending_food_log", "correct_log")),
            )
            if nutrition_block:
                system_prompt += f"\n\n{nutrition_block}"

        # 3b. Inject progressive-overload stall warnings for workout logging and
        # check-ins — but only if the pre-built training slice hasn't already put
        # them in the prompt (topic=training path). Guards against both a duplicate
        # stall block and a redundant detect_stalled_exercises DB pass.
        if intent in ("workout_log", "check_in") and "PROGRESSIVE OVERLOAD" not in system_prompt:
            stalled = detect_stalled_exercises(user_id)
            if stalled:
                stall_lines = [
                    "PROGRESSIVE OVERLOAD ALERTS — these exercises have shown NO weight increase over 14+ days:"
                ]
                for s in stalled:
                    stall_lines.append(
                        f"  ⚠️ {s['exercise_name']}: {s['current_weight_kg']}kg for "
                        f"{s['sessions_count']} sessions → suggest {s['suggested_weight_kg']}kg"
                    )
                stall_lines.append(
                    "COACHING RULE: If the user logs or mentions any of these exercises, "
                    "proactively tell them their body has adapted and they should add weight. "
                    "Be direct and specific: name the exercise, state the exact suggested weight, "
                    "and give one execution cue. Do not wait for them to ask."
                )
                system_prompt += "\n\n" + "\n".join(stall_lines)

        # 3c. Inject exercise catalog for general questions
        if intent == "question":
            exercise_catalog = _build_exercise_catalog(user_message)
            if exercise_catalog:
                system_prompt += f"\n\n{exercise_catalog}"
                system_prompt += (
                    "\n\nRULE — METADATA IS MANDATORY: If your reply names any specific exercise, "
                    "you MUST append the <METADATA_START>…<METADATA_END> block at the very end "
                    "using the exact ID and slug from the catalog above. Never omit it."
                )

        # 3d. Inject gym knowledge for anatomy / training science / injury questions
        if _f_knowledge is not None:
            try:
                knowledge_chunks = _f_knowledge.result()
            except Exception:
                knowledge_chunks = None  # knowledge is enrichment, not a hard dep
            if knowledge_chunks:
                knowledge_block = (
                    "VERIFIED BIOMECHANICS & TRAINING SCIENCE "
                    "(use these exact facts — no substitutions):\n\n"
                    + "\n\n---\n\n".join(knowledge_chunks)
                    + "\n\nTEACH THE MECHANISM: don't just name exercises — give the user "
                    "the ONE reason each pick works, in plain words drawn from the facts above "
                    "(e.g. \"lateral raises — they build the side delt that widens your frame\"). "
                    "Spell out any jargon the first time you use it. The 'why' is what makes you "
                    "sound like an expert coach instead of a list."
                )
                system_prompt += f"\n\n{knowledge_block}"

        # 4. Get conversation history (started concurrently above)
        history = _f_history.result()

        # 5. Build messages array for Groq
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(_trim_history_for_context(history))
        messages.append({"role": "user", "content": user_message})

        # 6. Call Groq — model and token budget by intent
        # Use fast 8b model for text-only intents (saves 70b TPD quota for JSON/complex tasks)
        # "question" now routes to the heavier model too — this is the bucket for
        # goal-based coaching asks ("I want massive shoulders") that need a real
        # structured plan, not just fast/logging replies. Same providers already
        # in use (DeepSeek), no new model added.
        _t_llm = time.perf_counter()
        response, _model = complete_for_intent(intent, messages)
        logger.info(
            "[chat timing] intent=%s model=%s prep=%.2fs model_call=%.2fs",
            intent, _model, _t_llm - _t_ctx, time.perf_counter() - _t_llm,
        )

        # Ground-truth token/cache accounting — DeepSeek returns
        # prompt_cache_hit_tokens/prompt_cache_miss_tokens on its usage object;
        # Groq won't have those fields, hence getattr with a default. This is
        # the real number to watch (vs. any estimate) — check Render logs.
        try:
            usage = getattr(response, "usage", None)
            if usage:
                logger.info(
                    "[chat usage] model=%s intent=%s prompt=%s completion=%s total=%s cache_hit=%s cache_miss=%s",
                    _model, intent,
                    getattr(usage, "prompt_tokens", None),
                    getattr(usage, "completion_tokens", None),
                    getattr(usage, "total_tokens", None),
                    getattr(usage, "prompt_cache_hit_tokens", None),
                    getattr(usage, "prompt_cache_miss_tokens", None),
                )
        except Exception:
            pass

        raw_response = response.choices[0].message.content

        # 7. Parse structured response and metadata
        reply_text, parsed_actions, parsed_metadata = _parse_structured_response(
            raw_response
        )

        # Fallback: if the LLM named exercises but skipped the METADATA block, find them
        if intent == "question" and (
            not parsed_metadata or not parsed_metadata.get("recommended_exercises")
        ):
            fallback_exs = _extract_exercises_from_reply(reply_text)
            if fallback_exs:
                parsed_metadata = parsed_metadata or {}
                parsed_metadata["recommended_exercises"] = fallback_exs

        # Ids are validated here, not requested politely in the prompt. Told to
        # use only catalog ids and to omit anything not listed, the model still
        # emitted "exercise_id": "LateralRaises" on 2 runs out of 3 — a fake id
        # that the app would then try to open. Resolving names against the
        # library keeps the good recommendation (lateral raises IS the right
        # call) while replacing whatever it invented with the real row.
        if parsed_metadata and parsed_metadata.get("recommended_exercises"):
            try:
                parsed_metadata["recommended_exercises"] = _sanitize_recommended_exercises(
                    parsed_metadata["recommended_exercises"]
                )
            except Exception:
                # Never ship unvalidated ids as a fallback — an empty card list
                # is correct, a fake card is not.
                parsed_metadata["recommended_exercises"] = []

        # Logging intents emit a {"reply", "actions"} JSON that we stash as a
        # pending action for the user to confirm. The definition of is_logging was
        # dropped in the nutrition-redesign refactor, leaving this a live NameError
        # that broke every text-confirm log path — restored here.
        is_logging = intent in (
            "workout_log", "weight_log", "pending_food_log", "correct_log"
        )
        # exercise_swap emits the same {"reply", "actions"} JSON and MUST be parsed
        # here too. It wasn't, which caused two bugs at once: the raw JSON was shown
        # to the user as the chat reply, and pending_actions stayed empty so the swap
        # card silently fell back to the DB's own unrelated alternative (offering
        # "Good Morning -> Band Good Morning" while the reply said bench press).
        # It differs from logging only in that it's confirmed via the card's
        # buttons, so it gets no "say yes to confirm" suffix.
        is_structured = is_logging or intent == "exercise_swap"
        pending_actions: list = []
        if is_structured:
            if parsed_actions and "reply" in parsed_actions:
                reply_text = parsed_actions["reply"]
                actions = parsed_actions.get("actions") or []
                # Normalise: model sometimes wraps single action in a dict instead
                # of a list (e.g. "actions": {...} instead of "actions": [{...}]).
                if isinstance(actions, dict):
                    actions = [actions]
                if actions:
                    if intent in ("pending_food_log", "correct_log"):
                        actions = _ground_food_macros(actions, user_message)
                    _store_pending_actions(user_id, creator_id, actions)
                    pending_actions = actions
                    if is_logging:
                        _verb = "confirm" if intent == "correct_log" else "log this"
                        reply_text += f"\n\nSay **yes** to {_verb} or **no** to cancel."

        # 8. Save both messages — store the clean reply, not raw JSON
        save_exchange(user_id, creator_id, user_message, reply_text)

        # Extract structured food data for the frontend confirmation card.
        # Support multiple items (e.g. "egg curry + 4 chapati") — collect every
        # log_food action so the card shows all items and logs them individually.
        _valid_meal_types = {"breakfast", "morning_snack", "lunch", "evening_snack", "dinner"}
        pending_food_data = None
        if intent == "pending_food_log" and pending_actions:
            food_actions = [a for a in pending_actions if a.get("type") == "log_food"]
            if food_actions:
                pending_food_data = [
                    {
                        "food_name": a.get("food_name", ""),
                        "count": a.get("count"),
                        "unit": a.get("unit"),
                        "quantity_g": a.get("quantity_g"),
                        "calories": a.get("calories"),
                        "protein_g": a.get("protein_g"),
                        "carbs_g": a.get("carbs_g"),
                        "fat_g": a.get("fat_g"),
                        "meal_type": a.get("meal_type") if a.get("meal_type") in _valid_meal_types else None,
                    }
                    for a in food_actions
                ]

        # Fallback: if the model generated the right reply text but returned empty
        # or missing actions, parse macros directly from the reply text.
        # Pattern: "Logging N food (Xg total). ~Ycal | Zg protein | Wg carbs | Vg fat"
        if intent == "pending_food_log" and not pending_food_data and reply_text:
            try:
                _m = re.search(
                    r"[Ll]ogging\s+(.+?)\s*[\.\(].*?"
                    r"~?(\d+(?:\.\d+)?)\s*kcal.*?"
                    r"~?(\d+(?:\.\d+)?)\s*g\s*protein.*?"
                    r"~?(\d+(?:\.\d+)?)\s*g\s*carbs.*?"
                    r"~?(\d+(?:\.\d+)?)\s*g\s*fat",
                    reply_text, re.IGNORECASE | re.DOTALL,
                )
                if _m:
                    # Infer meal type from current hour (same logic as the prompt)
                    _hour = now_local().hour
                    if _hour < 10:
                        _meal = "breakfast"
                    elif _hour < 12:
                        _meal = "morning_snack"
                    elif _hour < 16:
                        _meal = "lunch"
                    elif _hour < 19:
                        _meal = "evening_snack"
                    else:
                        _meal = "dinner"
                    # Extract count from original user message (e.g. "4 eggs" → count=4, unit="egg")
                    _count_m = re.search(r"(\d+)\s+(\w+)", user_message)
                    _count = int(_count_m.group(1)) if _count_m else None
                    _unit  = _count_m.group(2).rstrip("s") if _count_m else None
                    pending_food_data = [{
                        "food_name": _m.group(1).strip(),
                        "count":     _count,
                        "unit":      _unit,
                        "quantity_g": None,
                        "calories":  float(_m.group(2)),
                        "protein_g": float(_m.group(3)),
                        "carbs_g":   float(_m.group(4)),
                        "fat_g":     float(_m.group(5)),
                        "meal_type": _meal,
                    }]
                    # Also store as pending action so "yes" confirm can log it
                    _fallback_action = [{
                        "type": "log_food",
                        "food_name": pending_food_data[0]["food_name"],
                        "meal_type": _meal,
                        "count":     _count,
                        "unit":      _unit,
                        "quantity_g": None,
                        "calories":  float(_m.group(2)),
                        "protein_g": float(_m.group(3)),
                        "carbs_g":   float(_m.group(4)),
                        "fat_g":     float(_m.group(5)),
                    }]
                    _store_pending_actions(user_id, creator_id, _fallback_action)
                    pending_actions = _fallback_action
            except Exception:
                pass

        # Extract structured workout-set data for the frontend confirmation card
        # (mirrors pending_food_data — the client logs directly on tap-confirm,
        # the "say yes"/"confirm" text path above still works as a fallback).
        pending_workout_data = None
        if intent == "workout_log" and pending_actions:
            workout_actions = [a for a in pending_actions if a.get("type") == "log_workout"]
            if workout_actions:
                pending_workout_data = {
                    "exercises": [
                        {
                            "exercise": _normalise_exercise(a.get("exercise", "")),
                            "sets": [
                                {
                                    "weight_kg": _to_float(s.get("weight")),
                                    "reps": int(_to_float(s.get("reps")) or 0),
                                }
                                for s in a.get("sets", [])
                            ],
                        }
                        for a in workout_actions
                    ]
                }

        # Extract swap data for the frontend exercise swap card
        pending_swap_data = None
        if intent == "exercise_swap" and pending_actions:
            swap_action = next(
                (a for a in pending_actions if a.get("type") == "exercise_swap"), None
            )
            if swap_action:
                pending_swap_data = {
                    "original_exercise": swap_action.get("original_exercise", ""),
                    "replacement_exercise": swap_action.get("replacement_exercise", ""),
                    "original_muscles": swap_action.get("original_muscles", ""),
                    "replacement_muscles": swap_action.get("replacement_muscles", ""),
                }
        # NO blind fallback to the DB's own alternative. `swap_data` is computed
        # independently of what the coach actually said, so using it when the
        # model's action is missing produced a card that contradicted the reply
        # (card: "Good Morning -> Band Good Morning", reply: bench press) — and
        # confirming "Always" rewrites the user's plan. Only offer the card when
        # it matches the advice; otherwise show none. Same rule as the exercise
        # ids above: an empty card is correct, a wrong card is not.
        if intent == "exercise_swap" and not pending_swap_data and swap_data:
            logger.warning(
                "exercise_swap: model returned no swap action; suppressing card "
                "rather than offering the unrelated DB alternative %s -> %s",
                swap_data.get("original_exercise"),
                swap_data.get("replacement_exercise"),
            )

        # Correction card: resolve each delete_food_log to the real food_logs row
        # (so the card can show "Remove 200g Rice") and pass any corrected re-log.
        # The client deletes by row id on confirm — same rows this resolver picks.
        pending_correction_data = None
        if intent == "correct_log" and pending_actions:
            items = [
                row for a in pending_actions
                if a.get("type") == "delete_food_log"
                for row in [_resolve_food_log_removal(user_id, a.get("food_name") or "")]
                if row
            ]
            corrected = next(
                (a for a in pending_actions if a.get("type") == "log_food"), None
            )
            if items:
                pending_correction_data = {
                    "mode": "correct" if corrected else "remove",
                    "items": items,
                    "corrected": {
                        "food_name": corrected.get("food_name"),
                        "quantity_g": corrected.get("quantity_g"),
                        "calories": corrected.get("calories"),
                        "protein_g": corrected.get("protein_g"),
                        "carbs_g": corrected.get("carbs_g"),
                        "fat_g": corrected.get("fat_g"),
                        "meal_type": corrected.get("meal_type"),
                    } if corrected else None,
                }

        # Extract concrete meal/exercise suggestions so the app can render them as
        # tappable cards (Log it / Plan it / Add to plan). Best-effort — never
        # blocks the reply. Skipped when a full program proposal already owns
        # this turn (that has its own confirm flow — avoid duplicate cards).
        suggested_meals = None
        suggested_exercises = None
        candidate_facts = None
        _has_program_proposal = intent in ("create_program", "modify_program")
        _is_food_log = intent == "pending_food_log"

        # These three read nothing from each other, so running them back-to-back
        # just stacked their latency onto a reply that was already finished —
        # up to ~3s of the user staring at typing dots. Fan them out instead;
        # the turn now waits for the slowest, not the sum. Each still has its own
        # cheap regex gate, so the common case is three no-ops.
        # Suppress meal suggestions when the food confirmation card owns the turn
        # — that card IS the log confirmation. Never show exercise cards on a
        # food-log turn either.
        _extractors = {
            "meals": (lambda: _extract_meal_suggestions(reply_text)) if not _is_food_log else None,
            "exercises": (lambda: _extract_exercise_recommendations(reply_text))
            if not _has_program_proposal and not _is_food_log
            else None,
            # Durable facts (injuries, preferences, exceptions) for the
            # "remember this" coach-memory card.
            "facts": lambda: _extract_coach_memory_facts(user_message),
        }
        _live = {k: fn for k, fn in _extractors.items() if fn}
        if _live:
            with ThreadPoolExecutor(max_workers=len(_live)) as _pool:
                _futures = {k: _pool.submit(fn) for k, fn in _live.items()}
                _out = {}
                for _k, _f in _futures.items():
                    try:
                        _out[_k] = _f.result()
                    except Exception:
                        _out[_k] = None  # best-effort: a dead card never kills the reply
            suggested_meals = _out.get("meals")
            suggested_exercises = _out.get("exercises")
            candidate_facts = _out.get("facts")

        # 9. Return response
        return {
            "response": reply_text,
            "suggested_meals": suggested_meals,
            "suggested_exercises": suggested_exercises,
            "candidate_facts": candidate_facts,
            "pending_actions": pending_actions or None,
            "pending_food_data": pending_food_data,
            "pending_workout_data": pending_workout_data,
            "pending_swap_data": pending_swap_data,
            "pending_correction_data": pending_correction_data,
            "recommended_exercises": parsed_metadata.get("recommended_exercises", [])
            if parsed_metadata
            else None,
            "messages_remaining": remaining - 1,
            "upgrade_required": False,
            "intent": intent,
        }

    # No catch-all: anything else raises so /chat returns a 503 (main.py). This
    # used to return "Something went wrong" as a normal reply, and the persona
    # eval logged 31 of 72 turns as successes while the judge graded the
    # apology as bad coaching.
    except _groq_module.RateLimitError:
        return {
            "response": _RATE_LIMIT_RESPONSE,
            "messages_remaining": None,
            "upgrade_required": False,
            "intent": intent,
            "error": True,
        }


# ─── PROACTIVE WORKOUT DEBRIEF ───────────────────────────────────────────────
# Called right after the user finishes (or partially finishes) a session.
# Generates a coach-initiated message in the chat: progressive-overload nudges
# when lifts have stalled, a check-in when exercises were skipped, or a short
# genuine congrats otherwise. The message is stored as a normal assistant turn
# so the user can reply and the model keeps full context.

def _resolve_coach_creator_id(user_id: str) -> Optional[str]:
    """Active program's creator, else first followed coach."""
    try:
        prog = (
            supabase.table("user_programs")
            .select("program:programs(creator_id)")
            .eq("user_id", user_id)
            .eq("status", "active")
            .maybe_single()
            .execute()
            .data
        )
        cid = (prog or {}).get("program", {}).get("creator_id") if prog else None
        if cid:
            return cid
    except Exception:
        pass
    try:
        subs = (
            supabase.table("creator_subscriptions")
            .select("creator_id")
            .eq("user_id", user_id)
            .eq("active", True)
            .limit(1)
            .execute()
            .data
        )
        return subs[0]["creator_id"] if subs else None
    except Exception:
        return None


def workout_debrief(
    user_id: str,
    workout_title: str,
    duration_mins: int,
    exercises: list,           # [{name, sets: [{weight, reps}]}] — completed sets only
    skipped_exercises: list,   # names with zero completed sets
    done_sets: int,
    total_sets: int,
    new_prs: list,             # exercise names that hit a PR this session
    creator_id: Optional[str] = None,
) -> dict:
    """Generates and stores a proactive post-workout coach message."""
    creator_id = creator_id or _resolve_coach_creator_id(user_id)
    if not creator_id:
        return {"message": None, "skipped": "no_coach"}

    # Coach display name for voice consistency
    coach_name = "Coach"
    try:
        c = supabase.table("creators").select("name").eq("id", creator_id).maybe_single().execute().data
        if c and c.get("name"):
            coach_name = c["name"]
    except Exception:
        pass

    # Lifts that have plateaued over the last 14+ days
    stalled = []
    try:
        session_names = {e.get("name", "").strip().lower() for e in exercises}
        stalled = [
            s for s in detect_stalled_exercises(user_id)
            if s.get("exercise_name", "").strip().lower() in session_names
        ]
    except Exception:
        stalled = []

    # Compact session summary for the prompt
    lines = []
    for e in exercises[:10]:
        sets = e.get("sets") or []
        if not sets:
            continue
        top = max(sets, key=lambda s: (s.get("weight") or 0))
        lines.append(
            f"- {e.get('name')}: {len(sets)} sets, top set {top.get('weight') or 'BW'}kg x {top.get('reps')}"
        )
    session_txt = "\n".join(lines) if lines else "(no sets logged)"
    stalled_txt = (
        "\n".join(
            f"- {s.get('exercise_name')}: stuck at {s.get('current_weight_kg')}kg for "
            f"{s.get('sessions_count')} sessions — suggest {s.get('suggested_weight_kg')}kg next"
            for s in stalled[:5]
        )
        or "none"
    )
    skipped_txt = ", ".join(skipped_exercises[:8]) or "none"
    prs_txt = ", ".join(new_prs[:5]) or "none"
    completion = f"{done_sets}/{total_sets} sets"

    prompt = f"""You are {coach_name}, this user's personal gym coach inside the Rippd app. The user JUST finished a workout seconds ago and you are messaging them first, like a real coach who watched their session.

Session: "{workout_title}" — {duration_mins} min, {completion} completed.
Exercises done:
{session_txt}
Skipped entirely: {skipped_txt}
New PRs this session: {prs_txt}
Lifts plateaued 2+ weeks: {stalled_txt}

Write ONE short chat message (2-4 sentences, no headings, no lists). Rules by priority:
1. If any lift is plateaued: congratulate briefly, then tell them specifically it's time to add weight on that lift next session (suggest +2.5kg for isolation, +5kg for squat/bench/deadlift/rows) or add a rep target.
2. Else if they skipped exercises or finished under 70% of sets: ask what happened in a supportive, curious way (fatigue? time? pain?) — like a coach who cares, not a guilt trip.
3. Else if they hit a PR: celebrate it by name — genuine hype, no invented next-time target.
4. Else: genuine specific praise referencing an actual number from the session, plus one forward-looking cue for next time.
End with a question or nudge that invites a reply. Sound like a real human coach texting — warm, direct, no emojis overload (max 1), no corporate tone.
Only reference numbers, exercises, or milestones that appear in the session data above — never invent a time, rep target, weight, or metric that isn't given."""

    try:
        groq_client = _groq()
        resp = groq_client.chat.completions.create(
            model=MODEL_FAST,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,  # headroom for gpt-oss reasoning tokens
            temperature=0.7,
            **_groq_extra(MODEL_FAST),
        )
        message = (resp.choices[0].message.content or "").strip()
    except Exception:
        logging.getLogger(__name__).error("debrief LLM failed for %s", user_id, exc_info=True)
        return {"message": None, "skipped": "llm_error"}

    if not message:
        return {"message": None, "skipped": "empty"}

    # Store as a normal assistant turn so the chat shows it and follow-ups have context
    save_message(user_id, creator_id, "assistant", message)

    # Light the AI Coach tab badge (in-app, not push)
    try:
        supabase.table("notifications").insert(
            {
                "user_id": user_id,
                "type": "coach_content",
                "title": f"{coach_name} sent you a message",
                "body": message[:140],
                "data": {"screen": "coach"},
                "read": False,
            }
        ).execute()
    except Exception:
        pass

    return {"message": message, "creator_id": creator_id}


# ─── CREATOR SUBSCRIBER INSIGHTS ─────────────────────────────────────────────

def gather_creator_subscriber_insights(creator_id: str) -> dict:
    """
    Aggregates anonymised subscriber behaviour data for a creator.
    All data is counts/percentages — no individual user info is exposed.
    """
    # 1. Subscribers
    subs = (
        supabase.table("creator_subscriptions")
        .select("user_id")
        .eq("creator_id", creator_id)
        .eq("active", True)
        .execute()
        .data or []
    )
    sub_ids = [s["user_id"] for s in subs]
    total_subs = len(sub_ids)

    if total_subs == 0:
        return {"total_subscribers": 0}

    ids = sub_ids[:200]

    # 2. Content performance
    content = (
        supabase.table("creator_content")
        .select("id, title, content_type, view_count")
        .eq("creator_id", creator_id)
        .order("view_count", desc=True)
        .limit(10)
        .execute()
        .data or []
    )
    for item in content:
        likes_res = (
            supabase.table("content_likes")
            .select("id", count="exact")
            .eq("content_id", item["id"])
            .execute()
        )
        item["likes"] = likes_res.count or 0

    # 3. Workout completion rates across creator programs
    programs = (
        supabase.table("programs")
        .select("id, title")
        .eq("creator_id", creator_id)
        .execute()
        .data or []
    )
    completion_data = []
    for prog in programs[:5]:
        workouts = (
            supabase.table("workouts")
            .select("id, title, day_number, is_rest_day")
            .eq("program_id", prog["id"])
            .execute()
            .data or []
        )
        for w in [x for x in workouts if not x.get("is_rest_day")]:
            logs = (
                supabase.table("workout_logs")
                .select("completed")
                .eq("workout_id", w["id"])
                .in_("user_id", ids)
                .execute()
                .data or []
            )
            if not logs:
                continue
            done = sum(1 for l in logs if l.get("completed"))
            completion_data.append({
                "program": prog["title"],
                "workout": w["title"],
                "day": w["day_number"],
                "attempts": len(logs),
                "completion_pct": round(done / len(logs) * 100),
            })
    completion_data.sort(key=lambda x: x["completion_pct"])

    # 4. Most swapped exercises (original_exercise_name = what subscribers wanted to replace)
    overrides = (
        supabase.table("user_exercise_overrides")
        .select("original_exercise_name, replacement_exercise_name")
        .in_("user_id", ids)
        .execute()
        .data or []
    )
    swap_counts: dict[str, int] = {}
    for o in overrides:
        n = (o.get("original_exercise_name") or "").strip()
        if n:
            swap_counts[n] = swap_counts.get(n, 0) + 1
    top_swapped = sorted(swap_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # 5. Coach conversation topic signals (anonymised — keyword counts only)
    thirty_ago = local_midnight_utc_iso(30)  # created_at is UTC
    msgs = (
        supabase.table("ai_conversations")
        .select("content")
        .eq("creator_id", creator_id)
        .eq("role", "user")
        .gte("created_at", thirty_ago)
        .limit(300)
        .execute()
        .data or []
    )
    swap_msgs = food_msgs = pain_msgs = diet_msgs = 0
    exercise_swap_targets: dict[str, int] = {}
    food_items: dict[str, int] = {}
    for m_row in msgs:
        text = (m_row.get("content") or "").lower()
        if re.search(r"\b(swap|replace|alternative|substitute|instead of)\b", text):
            swap_msgs += 1
            ex_m = re.search(r"(?:swap|replace|instead of)\s+([a-z ]{3,30}?)(?:\s+with|\s+for|\b)", text)
            if ex_m:
                t = ex_m.group(1).strip()
                exercise_swap_targets[t] = exercise_swap_targets.get(t, 0) + 1
        if re.search(r"\b(diet|eat|food|meal|protein|calorie|carb|macro)\b", text):
            food_msgs += 1
            food_m = re.search(r"(?:avoid|skip|can't eat|allergic to|don't like)\s+([a-z ]{2,20})", text)
            if food_m:
                item = food_m.group(1).strip()
                food_items[item] = food_items.get(item, 0) + 1
        if re.search(r"\b(pain|hurt|injury|sore|difficult|can't|struggle|hard)\b", text):
            pain_msgs += 1
        if re.search(r"\b(vegan|vegetarian|keto|no meat|gluten|dairy free)\b", text):
            diet_msgs += 1

    top_exercise_swaps = sorted(exercise_swap_targets.items(), key=lambda x: x[1], reverse=True)[:5]
    top_food_items = sorted(food_items.items(), key=lambda x: x[1], reverse=True)[:5]

    # 6. Subscriber diet preferences and allergies
    prefs = (
        supabase.table("user_preferences")
        .select("allergies_or_dislikes, diet_preference")
        .in_("user_id", ids)
        .execute()
        .data or []
    )
    allergy_counts: dict[str, int] = {}
    diet_counts: dict[str, int] = {}
    for p in prefs:
        for a in (p.get("allergies_or_dislikes") or []):
            if a:
                key = a.lower().strip()
                allergy_counts[key] = allergy_counts.get(key, 0) + 1
        d = p.get("diet_preference")
        if d:
            diet_counts[d] = diet_counts.get(d, 0) + 1
    top_allergies = sorted(allergy_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # 7. 7-day active subscriber rate
    seven_ago = local_date_str(7)
    recent = (
        supabase.table("workout_logs")
        .select("user_id")
        .in_("user_id", ids)
        .gte("date", seven_ago)
        .execute()
        .data or []
    )
    active_ids = {r["user_id"] for r in recent}
    active_pct = round(len(active_ids) / total_subs * 100) if total_subs else 0

    return {
        "total_subscribers": total_subs,
        "active_this_week_pct": active_pct,
        "active_this_week_count": len(active_ids),
        "content_performance": [
            {"title": c["title"], "type": c["content_type"], "views": c["view_count"], "likes": c["likes"]}
            for c in content
        ],
        "workout_completion": completion_data[:10],
        "most_swapped_exercises": [{"exercise": e, "swaps": c} for e, c in top_swapped],
        "top_exercise_swap_targets": [{"exercise": e, "mentions": c} for e, c in top_exercise_swaps],
        "top_food_topics": [{"item": f, "mentions": c} for f, c in top_food_items],
        "coach_messages_30d": len(msgs),
        "swap_topic_messages": swap_msgs,
        "food_topic_messages": food_msgs,
        "pain_or_difficulty_messages": pain_msgs,
        "diet_preference_messages": diet_msgs,
        "common_food_dislikes": [{"item": a, "subscribers": c} for a, c in top_allergies],
        "diet_distribution": diet_counts,
    }


# ─── Tool schema for program creation ────────────────────────────────────────
_CREATE_PROGRAM_TOOL = {
    "type": "function",
    "function": {
        "name": "create_program",
        "description": (
            "Create a full workout program and save it as a draft in the creator's studio. "
            "Call this when the creator explicitly asks to create, build, save, or draft a program. "
            "Generate every week, every workout day, and every exercise with proper sets/reps."
        ),
        "parameters": {
            "type": "object",
            "required": ["title", "description", "difficulty", "goal", "duration_weeks", "days_per_week", "weeks"],
            "properties": {
                "title":          {"type": "string", "description": "Short, compelling program name"},
                "description":    {"type": "string", "description": "2-3 sentence program overview"},
                "difficulty":     {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
                "goal":           {"type": "string", "description": "e.g. strength, hypertrophy, fat loss, endurance"},
                "duration_weeks": {"type": "integer", "minimum": 1, "maximum": 16},
                "days_per_week":  {"type": "integer", "minimum": 1, "maximum": 7},
                "weeks": {
                    "type": "array",
                    "description": "One entry per week",
                    "items": {
                        "type": "object",
                        "required": ["week_number", "days"],
                        "properties": {
                            "week_number": {"type": "integer"},
                            "days": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["day_number", "title", "is_rest_day"],
                                    "properties": {
                                        "day_number":             {"type": "integer"},
                                        "title":                  {"type": "string"},
                                        "is_rest_day":            {"type": "boolean"},
                                        "description":            {"type": "string"},
                                        "estimated_duration_mins":{"type": "integer"},
                                        "exercises": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "required": ["name", "sets", "reps", "target_weight", "order_index"],
                                                "properties": {
                                                    "name":          {"type": "string"},
                                                    "sets":          {"type": "integer", "minimum": 1},
                                                    "reps":          {"type": "string", "description": "slash-separated per set, e.g. '10/8/6'"},
                                                    "target_weight": {"type": "string", "description": "slash-separated kg per set, '0' if bodyweight, e.g. '60/65/70'"},
                                                    "rest_seconds":  {"type": "integer"},
                                                    "notes":         {"type": "string"},
                                                    "order_index":   {"type": "integer"},
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


def _execute_create_program(creator_id: str, args: dict) -> dict:
    """Insert the AI-generated program into Supabase. Returns {id, title}."""
    prog_row = {
        "creator_id":     creator_id,
        "title":          args["title"],
        "description":    args.get("description", ""),
        "difficulty":     args.get("difficulty", "intermediate"),
        "goal":           args.get("goal", ""),
        "duration_weeks": args["duration_weeks"],
        "is_active":      False,   # draft — not published yet
        "is_user_created": False,
    }
    prog_res = supabase.table("programs").insert(prog_row).execute()
    program_id = prog_res.data[0]["id"]

    for week in args.get("weeks", []):
        week_num = week["week_number"]
        for day in week.get("days", []):
            workout_row = {
                "program_id":              program_id,
                "week_number":             week_num,
                "day_number":              day["day_number"],
                "title":                   day["title"],
                "description":             day.get("description", ""),
                "estimated_duration_mins": day.get("estimated_duration_mins"),
                "is_rest_day":             day.get("is_rest_day", False),
            }
            w_res = supabase.table("workouts").insert(workout_row).execute()
            workout_id = w_res.data[0]["id"]

            if not day.get("is_rest_day"):
                exercise_rows = [
                    {
                        "workout_id":    workout_id,
                        "name":          ex["name"],
                        "sets":          ex["sets"],
                        "reps":          ex["reps"],
                        "target_weight": ex["target_weight"],
                        "rest_seconds":  ex.get("rest_seconds", 60),
                        "notes":         ex.get("notes", ""),
                        "order_index":   ex["order_index"],
                    }
                    for ex in day.get("exercises", [])
                ]
                if exercise_rows:
                    supabase.table("exercises").insert(exercise_rows).execute()

    return {"id": program_id, "title": args["title"]}


def creator_insights_chat(creator_id: str, message: str, history: list[dict]) -> dict:
    """
    AI strategy advisor for creators with program-creation tool calling.
    Returns {"reply": str, "program_created": {"id": str, "title": str} | None}
    """
    creator = get_creator_context(creator_id) or {}
    insights = gather_creator_subscriber_insights(creator_id)

    system_prompt = f"""You are a strategic content and fitness advisor on the Rippd platform.
You advise creators on what programs to build, what content to post, and what to coach subscribers on — backed by real aggregate subscriber behaviour data.

CREATOR: {creator.get("name", "Unknown")}
COACHING PHILOSOPHY: {creator.get("coaching_philosophy", "N/A")}

SUBSCRIBER ANALYTICS (anonymised — counts and percentages only, no individual user data):
{json.dumps(insights, indent=2, default=str)}

Your job:
- Give specific, data-driven recommendations referencing exact numbers from the analytics.
- Suggest program ideas based on low workout completion rates and exercise swap patterns.
- Suggest content topics based on what subscribers ask the AI coach about most.
- Highlight pain/difficulty patterns as content opportunities.
- Identify diet and food restriction patterns and suggest targeted nutrition content.
- Be direct, strategic, and actionable. Format with bullet points or numbered lists where helpful.
- Never reveal individual user identities — only aggregate patterns.

IMPORTANT: When the creator asks you to CREATE, BUILD, MAKE, SAVE, or DRAFT a program — call the create_program tool immediately with a complete, fully detailed program. Do not ask for confirmation; just generate and save it. Include every week, every day, and every exercise with realistic sets, reps, and weights."""

    messages_payload = [{"role": "system", "content": system_prompt}]
    messages_payload.extend(history[-12:])
    messages_payload.append({"role": "user", "content": message})

    client = _groq()

    resp = client.chat.completions.create(
        model=MODEL_FAST_70B,
        messages=messages_payload,
        tools=[_CREATE_PROGRAM_TOOL],
        tool_choice="auto",
        max_tokens=4000,
        temperature=0.6,
        **_groq_extra(MODEL_FAST_70B),
    )

    msg = resp.choices[0].message
    program_created = None

    # Handle tool call
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        program_created = _execute_create_program(creator_id, args)

        # Send tool result back so the model can produce a natural reply
        messages_payload.append({"role": "assistant", "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
            }
        ]})
        messages_payload.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps({"status": "created", "program_id": program_created["id"]}),
        })

        follow_up = client.chat.completions.create(
            model=MODEL_FAST_70B,
            messages=messages_payload,
            max_tokens=900,  # headroom for gpt-oss reasoning tokens
            temperature=0.6,
            **_groq_extra(MODEL_FAST_70B),
        )
        reply = follow_up.choices[0].message.content or "Program saved as draft — open it in your Studio to review and publish."
    else:
        reply = msg.content or ""

    return {"reply": reply, "program_created": program_created}
