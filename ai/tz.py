"""
tz.py — Local-time helpers for the coach pipeline.

The backend runs on Render in UTC. Every "today", "now", and meal-time
decision the coach makes is about the USER's day, not the server's — so a
naked datetime.now() put the coach 5.5h behind an Indian user: dinner logged
as lunch, "today's macros" reset at 5:30am, adherence windows off by a day.

Two distinct clocks, kept deliberately separate:
  • now_local() / today_local() / local_date_str() — user-facing day logic
    (meal slots, "today so far", date columns, streak/adherence windows).
  • utc_now_iso() — storage. created_at columns are timestamptz; a naive
    local string would be read back as UTC and jump forward 5.5h. Anything
    written to a timestamp column must use this, never now_local().

APP_TZ is a single app-wide zone (the user base is India). Per-user zones
would live on user_preferences and turn these into user_id-taking calls.
# ponytail: single app-wide timezone; go per-user only when users are.
"""

import os
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

APP_TZ = ZoneInfo(os.environ.get("APP_TZ", "Asia/Kolkata"))


def now_local() -> datetime:
    """Current time in the app's timezone (tz-aware)."""
    return datetime.now(APP_TZ)


def today_local() -> date:
    """The user's current calendar date."""
    return now_local().date()


def local_date_str(days_ago: int = 0) -> str:
    """ISO date string N days before the user's today. Negative = future."""
    return (today_local() - timedelta(days=days_ago)).isoformat()


def local_midnight_utc_iso(days_ago: int = 0) -> str:
    """
    Start of the user's local day, expressed in UTC — for comparing against
    timestamptz columns (created_at) that are stored in UTC. Using a local
    naive midnight here would silently shift the window by the UTC offset.
    """
    local_midnight = datetime.combine(
        today_local() - timedelta(days=days_ago), datetime.min.time(), tzinfo=APP_TZ
    )
    return local_midnight.astimezone(timezone.utc).isoformat()


def utc_now_iso() -> str:
    """Now, in UTC — for writing to created_at / logged_at timestamp columns."""
    return datetime.now(timezone.utc).isoformat()


# user_preferences.training_days holds whatever the UI writes, and the UI writes
# ABBREVIATIONS: ["M","Tu","W","Th","F","Sa","Su"] (src/app/profile/details.tsx,
# src/app/onboarding/about.tsx). Three separate cron paths were comparing those
# against full lowercase weekday names ("friday"), which never matches — so the
# evening reminder skipped every user with training days set, and the morning
# brief told everyone it was a rest day, every day. One mapping, one helper, all
# callers. Legacy rows also contain 3-letter and full names, hence the aliases.
_DAY_ALIASES: dict[int, set[str]] = {
    0: {"m", "mo", "mon", "monday"},
    # Bare "t" and "s" are genuinely ambiguous and exist in legacy rows
    # (e.g. ["T","T","M","Tu","W","Th","F"]). Match them on both candidate days:
    # over-matching risks one extra nudge, under-matching silently drops the
    # feature — which is the failure we're fixing.
    1: {"tu", "tue", "tues", "tuesday", "t"},
    2: {"w", "we", "wed", "weds", "wednesday"},
    3: {"th", "thu", "thur", "thurs", "thursday", "t"},
    4: {"f", "fr", "fri", "friday"},
    5: {"sa", "sat", "saturday", "s"},
    6: {"su", "sun", "sunday", "s"},
}


def is_training_day(training_days, day: date | None = None) -> bool:
    """
    Is `day` (default: the user's today) one of their scheduled training days?

    Empty / None / unparseable training_days => True: no restriction is recorded,
    so don't invent one. Callers that treat "no data" as "rest day" would silently
    disable themselves for every user who skipped that onboarding step.
    """
    if not training_days:
        return True
    target = (day or today_local()).weekday()
    aliases = _DAY_ALIASES[target]
    return any(
        str(d).strip().lower().rstrip(".") in aliases
        for d in training_days
        if d is not None
    )


def is_before_local_today(iso_ts: str | None) -> bool:
    """
    True when a stored UTC timestamp falls on an earlier local day than now.
    Used to expire day-scoped caches. Unparseable/missing → True (treat as
    stale): rebuilding costs a query, serving yesterday's numbers as today's
    costs the user's trust.
    """
    if not iso_ts:
        return True
    try:
        parsed = datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00"))
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(APP_TZ).date() < today_local()


if __name__ == "__main__":
    # Self-check: the whole point of this module is the UTC/local boundary.
    assert now_local().tzinfo is not None
    assert local_date_str(0) == today_local().isoformat()
    assert local_date_str(1) == (today_local() - timedelta(days=1)).isoformat()
    assert local_date_str(-1) == (today_local() + timedelta(days=1)).isoformat()

    # Local midnight in UTC must round-trip back to local midnight.
    midnight = datetime.fromisoformat(local_midnight_utc_iso()).astimezone(APP_TZ)
    assert (midnight.hour, midnight.minute) == (0, 0), midnight
    assert midnight.date() == today_local()

    # A UTC instant that is still "yesterday" locally must not read as today.
    # 22:00 UTC yesterday = 03:30 IST today → same local day, NOT stale.
    yday_late_utc = datetime.combine(
        today_local() - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    ) + timedelta(hours=22)
    assert is_before_local_today(yday_late_utc.isoformat()) is False

    # 10:00 UTC yesterday = 15:30 IST yesterday → genuinely a past local day.
    yday_utc = yday_late_utc - timedelta(hours=12)
    assert is_before_local_today(yday_utc.isoformat()) is True

    assert is_before_local_today(None) is True
    assert is_before_local_today("not-a-date") is True
    assert is_before_local_today(utc_now_iso()) is False

    # ── is_training_day ────────────────────────────────────────────────────
    # THE BUG THIS EXISTS FOR: the UI writes abbreviations, three cron paths
    # compared them against "friday". Real row from production:
    friday = date(2026, 7, 17)   # a Friday
    saturday = date(2026, 7, 18)
    assert is_training_day(["M", "Tu", "W", "Th", "F", "Sa", "Su"], friday) is True
    assert is_training_day(["M", "Tu", "W", "Th", "F"], friday) is True
    assert is_training_day(["M", "Tu", "W", "Th", "F"], saturday) is False   # real rest day
    assert is_training_day(["M", "Tu", "Th", "F"], date(2026, 7, 15)) is False  # Wed off

    # Legacy formats that also live in the table.
    assert is_training_day(["friday"], friday) is True
    assert is_training_day(["Fri"], friday) is True
    assert is_training_day(["FRIDAY"], friday) is True
    assert is_training_day(["T", "T", "M", "Tu", "W", "Th", "F"], friday) is True
    # Ambiguous bare "T" matches Tue AND Thu — over-match beats silently
    # disabling the feature, which is the failure being fixed.
    assert is_training_day(["T"], date(2026, 7, 14)) is True   # Tuesday
    assert is_training_day(["T"], date(2026, 7, 16)) is True   # Thursday
    assert is_training_day(["T"], friday) is False

    # No restriction recorded => don't invent one.
    assert is_training_day(None, friday) is True
    assert is_training_day([], friday) is True
    # Junk must not crash or silently match.
    assert is_training_day(["zzz"], friday) is False
    assert is_training_day([None, "F"], friday) is True

    print("tz.py self-check passed")
