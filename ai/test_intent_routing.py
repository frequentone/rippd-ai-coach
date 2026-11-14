"""
Self-check for the safety-critical intent tiers in rag_pipeline._classify_intent_regex.

These are the branches where a miss is expensive: a red flag answered with a
grip change, or a crisis message answered with macros. Run: python test_intent_routing.py
"""

import os

# _classify_intent_regex is pure regex, but importing rag_pipeline builds a
# Supabase client at module scope. Feed it throwaway values so the import works
# without real credentials.
os.environ.setdefault("SUPABASE_URL", "http://localhost")
os.environ.setdefault("SUPABASE_KEY", "test-key")

from rag_pipeline import _classify_intent_regex, _detect_cheat_plan  # noqa: E402

CASES = [
    # ── Red flags: must never reach the coachable medical tier ──────────────
    ("i get chest pain when i bench", "medical_urgent"),
    ("shortness of breath during cardio", "medical_urgent"),
    ("my left arm is going numb on deadlifts", "medical_urgent"),
    ("tingling in my fingers after pull ups", "medical_urgent"),
    ("i felt a pop in my hamstring", "medical_urgent"),
    ("i think i broke my wrist", "medical_urgent"),
    ("i can't bear weight on my ankle", "medical_urgent"),
    ("got dizzy and passed out at the gym", "medical_urgent"),
    ("severe pain in my lower back", "medical_urgent"),
    ("pain shooting down my leg", "medical_urgent"),
    # ── Coachable: a tweak the coach SHOULD handle, not escalate ─────────────
    ("my shoulder hurts on overhead press", "medical"),
    ("my knee aches when i squat", "medical"),
    ("i pulled my hamstring a bit last week", "medical"),
    ("my elbow is a little sore and swollen", "medical"),
    # ── Crisis outranks everything ──────────────────────────────────────────
    ("i want to kill myself", "crisis"),
    ("i don't want to live anymore", "crisis"),
    # ── Must NOT be hijacked into a medical tier ────────────────────────────
    ("what should i eat after training", None),
    ("build me a 5 day split", None),
    ("i benched 100kg today", None),
    ("hey", None),
    # ── Out of scope ────────────────────────────────────────────────────────
    ("who won the ipl final", "out_of_scope"),
]


# A cheat meal being PLANNED must never be swallowed as a food log — the whole
# point of _detect_cheat_plan. One that already HAPPENED must still log.
CHEAT_CASES = [
    # ── Planning: coach it ──────────────────────────────────────────────────
    ("Planning a cheat meal, do not log any food. Cheating at: Dinner.", True),
    ("i have a party tonight", True),
    ("cheat day tomorrow, how do i handle it", True),
    ("going out for dinner with friends", True),
    ("eating out tonight, what's the plan", True),
    ("wedding buffet this weekend", True),
    ("drinks out after work", True),
    # ── Already eaten: this really is a log ─────────────────────────────────
    ("i had a cheat meal, log it", False),
    ("just ate a whole pizza at the party", False),
    ("ate out at dominos, log that", False),
    # ── Unrelated: must not steal the intent ────────────────────────────────
    ("how much protein should i eat", False),
    ("build me a push pull legs program", False),
    ("what should i eat for dinner", False),
]


def main() -> None:
    failures = []
    for message, expected in CASES:
        actual = _classify_intent_regex(message)
        if actual != expected:
            failures.append(f"  {message!r}\n    expected {expected!r}, got {actual!r}")

    for message, expected_cheat in CHEAT_CASES:
        actual_cheat = _detect_cheat_plan(message)
        if actual_cheat != expected_cheat:
            failures.append(
                f"  {message!r}\n    _detect_cheat_plan expected "
                f"{expected_cheat!r}, got {actual_cheat!r}"
            )

    total = len(CASES) + len(CHEAT_CASES)
    if failures:
        raise AssertionError(
            f"{len(failures)}/{total} intent routing cases failed:\n"
            + "\n".join(failures)
        )
    print(f"intent routing self-check passed ({total} cases)")


if __name__ == "__main__":
    main()
