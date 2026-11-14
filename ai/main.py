"""
RIPPD — FastAPI Backend
/chat               → AI coach via RAG pipeline (uses context_cache for 1 DB read)
/invalidate-context → Mark user context as stale + async rebuild
/macros             → Macro calculator + save to Supabase user_macros
GET /diet-plan      → Return stored Indian meal plan for a user
POST /diet-plan     → Generate (or regenerate) a 7-day meal plan from meal_templates

Run with: uvicorn main:app --reload  (from the ai/ directory)
"""

import asyncio
import logging
import os
import re
import sys
import traceback
from datetime import datetime

# Windows consoles default to cp1252, which raises UnicodeEncodeError on the
# ✅/❌/⚠️ emoji in the startup banner and kills boot ("Application startup
# failed"). Render (Linux/UTF-8) is unaffected; this only bites local Windows
# dev. Force UTF-8 stdout so `uvicorn main:app` boots on any console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from contextlib import asynccontextmanager
from typing import Literal, Optional

from dotenv import find_dotenv, load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from supabase import create_client

# Load .env before importing anything that reads env vars
_dotenv_path = find_dotenv(usecwd=True)
_dotenv_loaded = load_dotenv(_dotenv_path)

from context_builder import (
    build_prompt_context,
    detect_topics,
    get_or_build_context,
    invalidate_slice,
    rebuild_context_async,
)
from cron_build_context import start_scheduler
from diet_plan import (
    _record_feedback,
    generate_diet_plan,
    get_current_plan,
    regenerate_plan_slot,
)
from discord_bot import start_bot
from exercise_search import semantic_exercise_search
from food_photo import analyze_food_photo, parse_food_text
from food_search import semantic_food_search
from macros_v4_final import (
    calculate_macros_v4,
    get_saved_macros,
    save_macros_to_supabase,
)
from rag_pipeline import (
    _notify,
    chat,
    creator_insights_chat,
    send_evening_nudges,
    workout_debrief,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
if not _SUPABASE_URL or not _SUPABASE_KEY:
    # Fail loudly at boot rather than serving requests with a client that can
    # never authenticate anyone — a crashed deploy is far easier to notice
    # and diagnose than a "running" one where every request 401s.
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_KEY must be set — check Render dashboard → Environment."
    )

# Supabase admin client used for token verification and other lightweight
# admin queries (e.g. /ping) — reused everywhere instead of recreated per call.
_supabase_auth = create_client(_SUPABASE_URL, _SUPABASE_KEY)


async def verify_user(authorization: str = Header(...)) -> str:
    """Extract and verify the Supabase JWT; return the authenticated user_id."""
    token = authorization.removeprefix("Bearer ").strip()
    try:
        res = _supabase_auth.auth.get_user(token)
    except Exception as e:
        # gotrue raises for both a genuinely bad/expired token (HTTP 4xx from
        # Supabase's auth server) and infra issues (timeout, DNS, outage).
        # Telling the client "invalid token" for the latter triggers a bogus
        # client-side logout during what's really our outage — so only treat
        # it as 401 when the exception actually carries a 4xx status.
        status = getattr(e, "status", None)
        if status and 400 <= status < 500:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        logger.error(
            "verify_user: auth check failed (possible infra issue)", exc_info=True
        )
        raise HTTPException(
            status_code=503,
            detail="Auth service temporarily unavailable — please retry",
        )
    if not res or not res.user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return res.user.id


def assert_owns(caller_id: str, user_id: str) -> None:
    """Raise 403 if the verified caller is trying to act on another user's data."""
    if caller_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")


_scheduler = None

# Single source of truth for the allowed origin(s) — referenced by both the
# CORS middleware (normal responses) and the exception handlers below.
# Exception handlers registered via @app.exception_handler run outside the
# normal middleware response path, so CORSMiddleware never gets a chance to
# stamp its headers onto an error response — these have to be set explicitly
# here, which is why this dict exists at all. Keep both reads from
# _ALLOWED_ORIGINS so they can't silently drift apart again.
_ALLOWED_ORIGINS = ["*"]
_CORS_HEADERS = {
    "Access-Control-Allow-Origin": _ALLOWED_ORIGINS[0],
    "Access-Control-Allow-Methods": "*",
    "Access-Control-Allow-Headers": "*",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler
    print("\n== RIPPD startup check ==========================")
    # A .env file is only used for LOCAL dev. On Render/production the vars are
    # injected by the platform, so "found: False" here is EXPECTED and harmless.
    print(
        f"  .env file    : {bool(_dotenv_path)} ({_dotenv_path or 'none — normal in production'})"
    )

    # The values below are what actually matter. These three must be present
    # (set them in the Render dashboard → Environment; render.yaml marks them
    # sync:false so they are NOT auto-populated).
    required = ("GROQ_API_KEY", "SUPABASE_URL", "SUPABASE_KEY")
    missing = []
    for name in required:
        val = os.environ.get(name, "")
        if not val:
            missing.append(name)
            print(f"  {name:<13}: ❌ NOT SET")
        elif name == "SUPABASE_URL":
            print(f"  {name:<13}: ✅ {val}")
        else:
            print(f"  {name:<13}: ✅ {val[:6]}…({len(val)} chars)")

    if missing:
        print("  -----------------------------------------------")
        print(f"  ⚠️  MISSING REQUIRED ENV VARS: {', '.join(missing)}")
        print("  /chat, food logging and macros WILL FAIL until these are set.")
        print("  Fix: Render dashboard → rippd-backend → Environment → add them.")
        logger.error("Startup: missing required env vars: %s", ", ".join(missing))
    print("=================================================\n")

    _scheduler = start_scheduler()
    asyncio.create_task(start_bot())
    yield
    if _scheduler:
        _scheduler.shutdown(wait=False)


app = FastAPI(title="RIPPD API", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    logger.error(f"Request body: {exc.body}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
        headers=_CORS_HEADERS,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers=_CORS_HEADERS,
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── SCHEMAS ──────────────────────────────────────────────────────────────────

SliceKey = Literal["nutrition", "training", "progress", "all"]


class ChatRequest(BaseModel):
    user_id: str
    # The app's TextInput already caps at 500 chars — this is the real boundary,
    # enforced server-side so a direct API call can't paste a huge wall of text
    # to burn tokens (client-side limits are UX only, never a security boundary).
    message: str = Field(..., min_length=1, max_length=1000)
    creator_id: Optional[str] = None
    is_premium: bool = False
    # user_context from the client is intentionally ignored — backend builds it
    user_context: Optional[str] = None
    # What the user picked in the muscle-map plan brief. Stored as the user
    # message's meta so the recap card can be redrawn on reload; never fed to
    # the model, which reads the same choices from the message text itself.
    brief: Optional[dict] = None


class InvalidateContextRequest(BaseModel):
    user_id: str
    slice: SliceKey = "all"


class MacroRequest(BaseModel):
    user_id: str
    weight_kg: float
    height_cm: float
    age_years: int
    gender: str
    activity_level: str
    goal: str
    body_fat_pct: Optional[float] = None
    aggressive_cut: bool = False
    loss_rate_kg_per_week: Optional[float] = None


class FoodSearchRequest(BaseModel):
    query: str
    limit: int = 25


class ExerciseSearchRequest(BaseModel):
    query: str
    muscle: Optional[str] = None
    equipment: Optional[str] = None
    limit: int = 20


class AnalyzeFoodRequest(BaseModel):
    image_base64: str
    hint: str = ""


class ParseFoodTextRequest(BaseModel):
    text: str = Field(..., max_length=500)


class DietPlanRequest(BaseModel):
    user_id: str


class RegenerateSlotRequest(BaseModel):
    user_id: str
    slot_type: str
    day_index: int


class TemplateFeedbackRequest(BaseModel):
    user_id: str
    template_id: str
    action: str  # "logged" | "skipped"


class DailyCheckRequest(BaseModel):
    user_id: str


class DebriefSet(BaseModel):
    weight: Optional[float] = None
    reps: Optional[int] = None


class DebriefExercise(BaseModel):
    name: str = Field(..., max_length=120)
    sets: list[DebriefSet] = []


class WorkoutDebriefRequest(BaseModel):
    user_id: str
    workout_title: str = Field("Workout", max_length=120)
    duration_mins: int = 0
    exercises: list[DebriefExercise] = []
    skipped_exercises: list[str] = []
    done_sets: int = 0
    total_sets: int = 0
    new_prs: list[str] = []
    creator_id: Optional[str] = None


class NewSubscriberRequest(BaseModel):
    user_id: str  # the subscriber (caller must own this)
    creator_id: str  # the creator being followed


class CreatorPublishRequest(BaseModel):
    creator_id: str  # the creator (caller must own this)
    title: str
    publish_type: str  # "content" | "program"
    content_id: Optional[str] = (
        None  # set for publish_type="content" so tap deep-links to the post
    )


class CreatorTrainedRequest(BaseModel):
    creator_id: str  # the creator (caller must own this)
    workout_title: Optional[str] = None
    duration_mins: Optional[int] = None


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────


@app.get("/")
async def health_check():
    return {"status": "ok"}


@app.get("/version")
async def version():
    """Which commit is actually running.

    Added after a long stretch of guessing. Whether a fix had reached
    production was unanswerable without opening the Render dashboard, so
    "is this deployed?" got assumed rather than checked — and the
    assumption was wrong more than once.

    RENDER_GIT_COMMIT is injected by Render automatically; the fallback
    keeps this honest when running locally rather than reporting a
    misleading value.
    """
    return {
        "commit": os.environ.get("RENDER_GIT_COMMIT", "local")[:12],
        "branch": os.environ.get("RENDER_GIT_BRANCH", "local"),
        "service": os.environ.get("RENDER_SERVICE_NAME", "local"),
    }


@app.get("/ping")
async def ping():
    """Keep-alive endpoint. Hit every 5 min to prevent Render sleep and Supabase inactivity pause."""
    try:
        _supabase_auth.table("profiles").select("id").limit(1).execute()
        return {"status": "alive", "supabase": "ok"}
    except Exception:
        return {"status": "alive", "supabase": "error"}


@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    creator_id: str = Form(...),
    title: str = Form(...),
    caller_id: str = Depends(verify_user),
):
    """
    Receives a video file from the app, proxies it to Bunny Stream.
    Bunny API key never leaves the backend.
    """
    import httpx

    assert_owns(caller_id, creator_id)

    library_id = os.environ.get("BUNNY_LIBRARY_ID", "")
    api_key = os.environ.get("BUNNY_API_KEY", "")
    cdn_host = os.environ.get("BUNNY_CDN_HOSTNAME", "")

    if not library_id or not api_key:
        raise HTTPException(status_code=500, detail="Bunny not configured")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Step 1: Create video slot on Bunny
            create_resp = await client.post(
                f"https://video.bunnycdn.com/library/{library_id}/videos",
                headers={"AccessKey": api_key, "Content-Type": "application/json"},
                json={"title": title},
            )
        if create_resp.status_code not in (200, 201):
            raise HTTPException(status_code=502, detail="Bunny video creation failed")

        video_id = create_resp.json()["guid"]

        # Step 2: stream file bytes to Bunny in chunks — a multi-hundred-MB
        # video no longer has to sit fully buffered in worker memory at once.
        async def _stream_upload():
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk

        async with httpx.AsyncClient(timeout=300) as client:
            upload_resp = await client.put(
                f"https://video.bunnycdn.com/library/{library_id}/videos/{video_id}",
                headers={
                    "AccessKey": api_key,
                    "Content-Type": "application/octet-stream",
                },
                content=_stream_upload(),
            )
        if upload_resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=502,
                detail=f"Bunny upload failed: {upload_resp.status_code} {upload_resp.text}",
            )

        return {
            "video_id": video_id,
            "playback_url": f"https://{cdn_host}/{video_id}/playlist.m3u8",
            "thumbnail_url": f"https://{cdn_host}/{video_id}/thumbnail.jpg",
        }
    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to upload video")


@app.post("/notify/new-subscriber")
async def notify_new_subscriber(
    req: NewSubscriberRequest, caller_id: str = Depends(verify_user)
):
    """Called by the frontend after a user follows a creator. Sends a push to the creator."""
    assert_owns(caller_id, req.user_id)
    try:
        sub = (
            _supabase_auth.table("profiles")
            .select("full_name")
            .eq("id", req.user_id)
            .maybe_single()
            .execute()
            .data
        )
        _name = ((sub or {}).get("full_name") or "").strip()
        first_name = _name.split()[0] if _name else "Someone"
        await asyncio.to_thread(
            _notify,
            req.creator_id,
            "new_subscriber",
            "New follower! 🎉",
            f"{first_name} just started following you.",
            {"screen": "studio"},
        )
        return {"status": "ok"}
    except Exception:
        logger.error("notify_new_subscriber error", exc_info=True)
        return {"status": "error"}


@app.post("/notify/creator-publish")
async def notify_creator_publish(
    req: CreatorPublishRequest, caller_id: str = Depends(verify_user)
):
    """
    Called by the frontend when a creator publishes new content or a program.
    Fans out a push + in-app notification to every active subscriber.
    Returns immediately — fan-out runs in the background.
    """
    assert_owns(caller_id, req.creator_id)
    try:
        creator = (
            _supabase_auth.table("creators")
            .select("name")
            .eq("id", req.creator_id)
            .maybe_single()
            .execute()
            .data
        )
        creator_name = (creator or {}).get("name", "Your coach")

        subs = (
            _supabase_auth.table("creator_subscriptions")
            .select("user_id")
            .eq("creator_id", req.creator_id)
            .eq("active", True)
            .execute()
            .data
            or []
        )

        if req.publish_type == "program":
            title = f"New program from {creator_name} 💪"
            body = f'"{req.title}" is now available — enroll from the Search tab.'
            notif_data = {"screen": "home"}
        else:
            title = f"New post from {creator_name} 🔥"
            body = f'"{req.title}" — tap to check it out.'
            notif_data = (
                {"screen": "content", "id": req.content_id}
                if req.content_id
                else {"screen": "home"}
            )

        user_ids = [s["user_id"] for s in subs]

        # Bounded concurrency — a creator with a large subscriber list should
        # never turn into thousands of simultaneous blocking notify() threads.
        _fan_out_sem = asyncio.Semaphore(20)

        async def _notify_one(uid: str):
            async with _fan_out_sem:
                try:
                    await asyncio.to_thread(
                        _notify, uid, "coach_content", title, body, notif_data
                    )
                except Exception:
                    logger.warning(
                        "creator-publish notify failed for user %s", uid, exc_info=True
                    )

        async def _fan_out():
            await asyncio.gather(*(_notify_one(uid) for uid in user_ids))

        asyncio.create_task(_fan_out())
        return {"status": "ok", "notified": len(user_ids)}
    except Exception:
        logger.error("notify_creator_publish error", exc_info=True)
        return {"status": "error"}


@app.post("/notify/creator-trained")
async def notify_creator_trained(
    req: CreatorTrainedRequest, caller_id: str = Depends(verify_user)
):
    """
    Called by the frontend when a CREATOR finishes a workout.
    Fans out a motivational push to every active subscriber so they feel
    connected to their coach. Deduped to max 1 per creator per day.
    Returns immediately — fan-out runs in the background.
    """
    assert_owns(caller_id, req.creator_id)
    try:
        # Dedupe: only the first completed workout of the day notifies subscribers.
        today_start = (
            datetime.now()
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .isoformat()
        )
        already = (
            _supabase_auth.table("notifications")
            .select("id")
            .eq("type", "creator_trained")
            .eq("data->>creator_id", req.creator_id)
            .gte("created_at", today_start)
            .limit(1)
            .execute()
            .data
        )
        if already:
            return {"status": "ok", "notified": 0, "reason": "already_sent_today"}

        creator = (
            _supabase_auth.table("creators")
            .select("name")
            .eq("id", req.creator_id)
            .maybe_single()
            .execute()
            .data
        )
        creator_name = (creator or {}).get("name") or "Your coach"
        first_name = creator_name.split()[0]

        subs = (
            _supabase_auth.table("creator_subscriptions")
            .select("user_id")
            .eq("creator_id", req.creator_id)
            .eq("active", True)
            .execute()
            .data
            or []
        )
        user_ids = [s["user_id"] for s in subs if s["user_id"] != req.creator_id]

        title = f"{first_name} just finished training 💪"
        if req.workout_title and req.duration_mins:
            body = f"{first_name} crushed {req.workout_title} in {req.duration_mins} min. Your turn — keep up with your coach."
        elif req.workout_title:
            body = f"{first_name} crushed {req.workout_title}. Your turn — keep up with your coach."
        else:
            body = f"{first_name} just got a session in. Your turn — keep up with your coach."
        notif_data = {"screen": "workout", "creator_id": req.creator_id}

        _fan_out_sem = asyncio.Semaphore(20)

        async def _notify_one(uid: str):
            async with _fan_out_sem:
                try:
                    await asyncio.to_thread(
                        _notify, uid, "creator_trained", title, body, notif_data
                    )
                except Exception:
                    logger.warning(
                        "creator-trained notify failed for user %s", uid, exc_info=True
                    )

        async def _fan_out():
            await asyncio.gather(*(_notify_one(uid) for uid in user_ids))

        asyncio.create_task(_fan_out())
        return {"status": "ok", "notified": len(user_ids)}
    except Exception:
        logger.error("notify_creator_trained error", exc_info=True)
        return {"status": "error"}


@app.post("/cron/evening-nudges")
async def cron_evening_nudges(request: Request):
    """
    Server-side sweep, meant to be hit by a scheduler (Render cron / cron-job.org)
    once in the evening. For every user who had a scheduled training day today but
    hasn't logged a workout, sends ONE motivational nudge in their creator's voice.
    Protected by CRON_SECRET header — not a user-facing endpoint.
    """
    secret = os.environ.get("CRON_SECRET")
    if not secret or request.headers.get("x-cron-secret") != secret:
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        result = await asyncio.to_thread(send_evening_nudges)
    except Exception:
        # 500, not 200-with-an-error-body. A scheduler only sees the status
        # code: returning 200 here meant a crashed sweep looked identical to a
        # successful one, and this function crashed on a NameError for its
        # entire life without anything noticing.
        logger.error("cron_evening_nudges error", exc_info=True)
        raise HTTPException(status_code=500, detail="nudge sweep failed")
    if result.get("status") != "ok":
        logger.error("cron_evening_nudges returned non-ok: %s", result)
        raise HTTPException(status_code=500, detail=str(result))
    return result


@app.post("/daily-check")
async def daily_check_endpoint(
    req: DailyCheckRequest, caller_id: str = Depends(verify_user)
):
    # Streak freezes are deprecated — the mobile app now computes streaks against
    # the user's weekly workout goal, which already tolerates a missed day.
    # Endpoint kept as a harmless no-op for any client build still calling it.
    assert_owns(caller_id, req.user_id)
    return {"action": "none", "reason": "deprecated"}


@app.post("/workout-debrief")
async def workout_debrief_endpoint(
    req: WorkoutDebriefRequest, caller_id: str = Depends(verify_user)
):
    """
    Fired by the app right after a session ends. Generates a coach-initiated
    chat message (progressive-overload nudge, skipped-workout check-in, or
    PR celebration) and stores it in ai_conversations so the user can reply.
    """
    assert_owns(caller_id, req.user_id)
    try:
        return await asyncio.to_thread(
            workout_debrief,
            user_id=req.user_id,
            workout_title=req.workout_title,
            duration_mins=req.duration_mins,
            exercises=[e.model_dump() for e in req.exercises],
            skipped_exercises=req.skipped_exercises,
            done_sets=req.done_sets,
            total_sets=req.total_sets,
            new_prs=req.new_prs,
            creator_id=req.creator_id,
        )
    except Exception:
        logger.error("workout-debrief error for %s", req.user_id, exc_info=True)
        return {"message": None, "error": True}


@app.post("/invalidate-context")
async def invalidate_context_endpoint(
    req: InvalidateContextRequest, caller_id: str = Depends(verify_user)
):
    """
    Marks the user's context_cache row as invalidated and triggers an async
    background rebuild. Returns 200 immediately — the rebuild is non-blocking.
    """
    assert_owns(caller_id, req.user_id)
    try:
        await asyncio.to_thread(invalidate_slice, req.user_id, req.slice)
        asyncio.create_task(rebuild_context_async(req.user_id))
    except Exception:
        logger.error("invalidate-context failed for %s", req.user_id, exc_info=True)
        # Non-fatal: context will rebuild on next chat request
    return {"status": "ok", "user_id": req.user_id, "slice": req.slice}


# Em dashes are the single loudest "this was written by an AI" tell, and the
# prompt rule against them leaks often enough to matter. This is the belt to
# that braces: applied to the whole payload on the way out, so it also covers
# the canned strings and tool-intent paths that never see the system prompt.
# A dash with whitespace on either side becomes a comma; a tight one (8—12)
# becomes a hyphen, which is what a range wanted in the first place. The
# spaced pattern needs \s+ on at least one side, not \s* on both: \s* matches
# zero characters, so it would claim tight dashes too and render "8—12" as
# "8, 12".
_DASH_SPACED = re.compile(r"\s*[—–]\s+|\s+[—–]\s*")
_DASH_TIGHT = re.compile(r"[—–]")


def _dedash(text: str) -> str:
    return _DASH_TIGHT.sub("-", _DASH_SPACED.sub(", ", text))


def _dedash_payload(value):
    """Walks the response payload, cleaning every string it contains."""
    if isinstance(value, str):
        return _dedash(value)
    if isinstance(value, dict):
        return {k: _dedash_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_dedash_payload(v) for v in value]
    return value


@app.post("/chat")
async def chat_endpoint(req: ChatRequest, caller_id: str = Depends(verify_user)):
    """
    1. Read context_cache (1 DB read). Build inline on miss.
    2. Detect every topic the message touches.
    3. Build context = base + progress + those topic slices.
    4. Pass to chat() as prompt_context — chat() then does a lean live-data fetch
       instead of rebuilding every context slice per message.
    """
    assert_owns(caller_id, req.user_id)
    try:
        slices = await asyncio.to_thread(get_or_build_context, req.user_id)
        topics = detect_topics(req.message)
        prompt_context = build_prompt_context(slices, topics)
        result = await asyncio.to_thread(
            chat,
            brief=req.brief,
            user_id=req.user_id,
            user_message=req.message,
            creator_id=req.creator_id,
            is_premium=req.is_premium,
            prompt_context=prompt_context,
        )
        return _dedash_payload(result)
    except HTTPException:
        raise
    except Exception:
        # 503, NOT 200. This returned a 200 with an apology in the body, which
        # made every backend failure invisible: uptime checks saw success, and a
        # QA run of 75 messages was "100% HTTP 200" while ~80% of the bodies were
        # error text. Two production outages (program builds 400ing on
        # tool_choice, and empty replies from the thinking model) survived
        # precisely because nothing external could see them.
        #
        # coach.tsx retries a 503 once, then shows "Something went wrong. Please
        # try again." The retry cannot double-log: confirmed actions are cleared
        # before they run. This only works if nothing below swallows — chat() and
        # its handlers must raise, not return an apology (fixed_evals.py checks).
        logger.error("chat error for %s", req.user_id, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="I'm having trouble right now. Please try again in a moment.",
        )


@app.get("/macros")
async def get_macros_endpoint(user_id: str, caller_id: str = Depends(verify_user)):
    assert_owns(caller_id, user_id)
    macros = await asyncio.to_thread(get_saved_macros, user_id)
    if not macros:
        raise HTTPException(
            status_code=404,
            detail="No macro targets found for this user. POST /macros to calculate them.",
        )
    return macros


@app.post("/macros")
async def macros_endpoint(req: MacroRequest, caller_id: str = Depends(verify_user)):
    assert_owns(caller_id, req.user_id)
    try:
        result = calculate_macros_v4(
            weight_kg=req.weight_kg,
            height_cm=req.height_cm,
            age_years=req.age_years,
            gender=req.gender,
            activity_level=req.activity_level,
            goal=req.goal,
            body_fat_pct=req.body_fat_pct,
            aggressive_cut=req.aggressive_cut,
            loss_rate_kg_per_week=req.loss_rate_kg_per_week,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        save_macros_to_supabase(req.user_id, result)
    except Exception:
        # Non-fatal: still return the calculated macros so the UI isn't
        # blocked, but the failure is now actually visible in logs.
        logger.error(
            "save_macros_to_supabase failed for %s", req.user_id, exc_info=True
        )

    return result


@app.post("/food-search")
async def food_search_endpoint(
    req: FoodSearchRequest, caller_id: str = Depends(verify_user)
):
    results = await asyncio.to_thread(semantic_food_search, req.query, req.limit)
    return {"results": results}


@app.post("/exercise-search")
async def exercise_search_endpoint(
    req: ExerciseSearchRequest, caller_id: str = Depends(verify_user)
):
    # ILIKE-first + semantic top-up; degrades to ILIKE-only if Gemini/embeddings
    # aren't available, so this is safe to ship before embeddings are generated.
    results = await asyncio.to_thread(
        semantic_exercise_search, req.query, req.muscle, req.equipment, req.limit
    )
    return {"results": results}


@app.post("/analyze-food")
async def analyze_food_endpoint(
    req: AnalyzeFoodRequest, caller_id: str = Depends(verify_user)
):
    # ~6 MB of base64 ≈ 4.5 MB image; Groq rejects larger payloads anyway.
    if len(req.image_base64) > 6_000_000:
        raise HTTPException(
            status_code=413, detail="Image too large — retake at lower quality."
        )
    try:
        result = await asyncio.to_thread(analyze_food_photo, req.image_base64, req.hint)
        return result
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=502, detail="Food analysis failed. Please try again."
        )


@app.post("/parse-food-text")
async def parse_food_text_endpoint(
    req: ParseFoodTextRequest, caller_id: str = Depends(verify_user)
):
    text = req.text.strip()
    if not text:
        return {"items": [], "notes": ""}
    try:
        result = await asyncio.to_thread(parse_food_text, text)
        return result
    except Exception:
        traceback.print_exc()
        # 502 so the client can fall back to its offline parser.
        raise HTTPException(
            status_code=502, detail="Couldn't read that meal. Please try again."
        )


@app.get("/diet-plan")
async def get_diet_plan_endpoint(user_id: str, caller_id: str = Depends(verify_user)):
    assert_owns(caller_id, user_id)
    plan = await asyncio.to_thread(get_current_plan, user_id)
    if not plan:
        raise HTTPException(
            status_code=404,
            detail="No diet plan found for this user. POST /diet-plan to generate one.",
        )
    return plan


@app.post("/diet-plan")
async def generate_diet_plan_endpoint(
    req: DietPlanRequest, caller_id: str = Depends(verify_user)
):
    assert_owns(caller_id, req.user_id)
    try:
        plan = await asyncio.to_thread(generate_diet_plan, req.user_id)
        return plan
    except ValueError as e:
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail="Failed to generate diet plan. Please try again."
        )


@app.post("/diet-plan/regenerate-slot")
async def regenerate_slot_endpoint(
    req: RegenerateSlotRequest, caller_id: str = Depends(verify_user)
):
    assert_owns(caller_id, req.user_id)
    try:
        day = await asyncio.to_thread(
            regenerate_plan_slot, req.user_id, req.slot_type, req.day_index
        )
        return day
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to regenerate slot.")


@app.post("/diet-plan/feedback")
async def diet_feedback_endpoint(
    req: TemplateFeedbackRequest, caller_id: str = Depends(verify_user)
):
    assert_owns(caller_id, req.user_id)
    # Best-effort — never surface errors to the client, but log them.
    try:
        await asyncio.to_thread(
            _record_feedback, req.user_id, req.template_id, req.action
        )
    except Exception:
        logger.error("_record_feedback failed for %s", req.user_id, exc_info=True)
    return {"status": "ok"}


# ─── CREATOR INSIGHTS ────────────────────────────────────────────────────────


class CreatorInsightsRequest(BaseModel):
    creator_id: str
    message: str
    history: list[dict] = []


@app.post("/creator-insights")
async def creator_insights_endpoint(
    req: CreatorInsightsRequest, caller_id: str = Depends(verify_user)
):
    # Only the creator themselves can access their own insights
    if caller_id != req.creator_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        result = await asyncio.to_thread(
            creator_insights_chat, req.creator_id, req.message, req.history
        )
        return result  # {"reply": str, "program_created": {...} | None}
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to generate insights.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
