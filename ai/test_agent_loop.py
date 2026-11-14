"""
Self-check for the coach's agent loop (agent_loop.py). No network, no model:
a scripted fake client stands in for DeepSeek and the data reads are stubbed.

Run: python test_agent_loop.py
"""

import json
import os
import sys
from types import SimpleNamespace as NS
from unittest import mock

os.environ.setdefault("SUPABASE_URL", "http://localhost")
os.environ.setdefault("SUPABASE_KEY", "test-key")

import agent_loop  # noqa: E402

rp = agent_loop.rp


def call(call_id, name, **args):
    return NS(id=call_id, function=NS(name=name, arguments=json.dumps(args)))


def say(text=None, calls=None):
    return NS(content=text, tool_calls=calls)


class FakeClient:
    """Plays back scripted model replies and records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return NS(choices=[NS(message=self.replies.pop(0))])


class NoDatabase:
    def table(self, name):
        raise AssertionError(f"the agent touched the database ({name})")


SLICES = {"base": "ATHLETE PROFILE: goal muscle gain | INJURIES: none",
          "training": "TRAINING PROFILE: PPL, bench stalled"}
BENCH = {"bench press": [
    {"date": "2026-10-20", "sets": [{"weight": 60, "reps": 8, "done": True}]},
    {"date": "2026-11-10", "sets": [{"weight": 60, "reps": 8, "done": True}]},
]}
RICE = {"name": "rice", "calories_per_100g": 130, "protein_per_100g": 2.7,
        "carbs_per_100g": 28, "fat_per_100g": 0.3}


def run(replies, message="is my bench ready to go up?"):
    client = FakeClient(replies)
    with mock.patch.object(agent_loop, "get_or_build_context", return_value=SLICES), \
         mock.patch.object(rp, "_fetch_exercise_weight_history", return_value=BENCH), \
         mock.patch.object(rp, "search_gym_knowledge", return_value=["progressive overload chunk"]), \
         mock.patch.object(rp, "get_food_nutrition", return_value=RICE), \
         mock.patch.object(rp, "supabase", NoDatabase()):
        return agent_loop.run_agent("u1", message, client=client), client


def test_safety_runs_before_the_model():
    client = FakeClient([])
    out = agent_loop.run_agent("u1", "I want to end it all", client=client)
    assert out["reply"] == rp._CRISIS_RESPONSE
    assert client.requests == [], "a crisis message reached the model"


def test_fetches_data_over_several_steps():
    out, client = run([
        say(calls=[call("c1", "get_user_data", section="training")]),
        say(calls=[call("c2", "get_lift_history", exercise="bench")]),
        say("Bench is ready: go to 62.5 kg."),
    ])
    assert out["reply"] == "Bench is ready: go to 62.5 kg."
    assert [s["tool"] for s in out["steps"]] == ["get_user_data", "get_lift_history"]
    assert "PPL" in out["steps"][0]["result"]
    bench = json.loads(out["steps"][1]["result"])["exercises"]["bench press"]["readiness"]
    assert bench["ready"] and bench["suggested_next_weight_kg"] == 62.5

    final = client.requests[-1]["messages"]
    assert "ATHLETE PROFILE" in final[0]["content"], "profile must always be in the prompt"
    assert [m["tool_call_id"] for m in final if m["role"] == "tool"] == ["c1", "c2"]
    assert all(r["tool_choice"] == "auto" for r in client.requests)


def test_stops_after_max_steps():
    loop = [say(calls=[call(f"c{i}", "search_knowledge", query="bench")])
            for i in range(agent_loop.MAX_STEPS)]
    out, client = run(loop + [say("Here's what I found.")])
    assert len(client.requests) == agent_loop.MAX_STEPS + 1
    assert "tools" not in client.requests[-1], "the last call must not offer tools"
    assert out["reply"] == "Here's what I found."


def test_wrong_exercise_name_gets_a_way_to_retry():
    out, _ = run([say(calls=[call("c1", "get_lift_history", exercise="squat")]), say("No squats logged.")])
    result = json.loads(out["steps"][0]["result"])
    assert result == {"found": False, "exercises_logged": ["bench press"]}


def test_proposals_are_grounded_and_never_written():
    out, _ = run([
        say(calls=[call("c1", "propose_change", type="log_food", summary="Log 200 g rice",
                        details={"food_name": "rice", "quantity_g": 200, "calories": 999})]),
        say("Want me to log it?"),
    ])
    [proposal] = out["proposals"]
    assert proposal["details"]["calories"] == 260.0, "macros must come from the foods table"
    assert "Nothing has changed" in out["steps"][0]["result"]


def test_bad_tool_calls_go_back_to_the_model():
    out, _ = run([
        say(calls=[call("c1", "get_lift_history")]),       # missing argument
        say(calls=[call("c2", "delete_everything")]),       # not a tool
        say("Sorry, let me answer directly."),
    ])
    assert out["steps"][0]["result"].startswith("Tool error: TypeError")
    assert out["steps"][1]["result"] == "Unknown tool: delete_everything"


def test_empty_answer_raises():
    try:
        run([say("")])
    except RuntimeError:
        return
    raise AssertionError("an empty answer must raise, not become a reply")


def main() -> int:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as err:
            failed += 1
            print(f"  FAIL  {name}: {type(err).__name__}: {err}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
