"""
Agent loop for the AI coach. A prototype: it is NOT wired into /chat.

The production coach (rag_pipeline.chat) is a routed workflow. Code picks the
intent, keyword matching picks which slices of the user's data go into the
prompt, and the model gets one call. This module hands that choice to the
model: it calls tools to fetch what it needs, reads the results, and decides
what to fetch next, up to MAX_STEPS rounds, before it answers.

What carries over from production, on purpose:
  1. Safety runs first, in code. Crisis messages, medical red flags and
     out-of-scope messages get the same fixed replies and never reach the model.
  2. The user's profile (goals, injuries, standing facts) is always in the
     prompt. Anything safety-critical never depends on the model asking for it.
  3. Tools only read. A change to the user's data comes back as a proposal for
     the user to confirm, never as a write.
  4. It is bounded. After MAX_STEPS rounds the model must answer with what it
     has, and an empty answer raises instead of turning into an apology.

Try it (read-only against your real data):
    python agent_loop.py <user_id> "is my bench ready to go up?"
"""

import json
import logging
import sys

import rag_pipeline as rp
from context_builder import get_or_build_context

logger = logging.getLogger(__name__)

MAX_STEPS = 4            # tool rounds before the model must answer
MAX_TOOL_CHARS = 2500    # cap on what one tool result feeds back to the model

SYSTEM_PROMPT = """You are RIPPD's fitness coach, answering one user about their own training, nutrition and progress.

Their profile is below. Use the tools to fetch the rest of their real data before you answer. Fetch only what the question needs, and stop once you can answer.
Never invent numbers. If the data isn't there, say so plainly.
You cannot change anything yourself. If a change would help, call propose_change and tell the user they can confirm it.
Answer in under 150 words: specific to their numbers, with one clear next step."""

MEDICAL_NOTE = (
    "\n\nThe user mentions pain or an injury. Do not diagnose. Suggest training around it "
    "(lighter load, a pain-free alternative) and say to see a physio if it persists."
)

# Safety replies stay deterministic, exactly as in production.
FIXED_REPLIES = {
    "crisis": rp._CRISIS_RESPONSE,
    "medical_urgent": rp._MEDICAL_URGENT_RESPONSE,
    "out_of_scope": rp._OOS_RESPONSE,
}


# ── Tools ────────────────────────────────────────────────────────────────────
# Each takes the user id (bound by the loop, never chosen by the model) plus the
# model's arguments, and returns something JSON-serialisable. All of them read.

def get_user_data(user_id: str, section: str):
    """One cached context slice: the same text production picks by keyword."""
    text = get_or_build_context(user_id).get(section)
    return text or f"No {section} data for this user."


def get_lift_history(user_id: str, exercise: str, days: int = 28):
    """Recent sessions for one exercise, plus production's 2-week readiness rule."""
    days = max(7, min(int(days), 90))
    history = rp._fetch_exercise_weight_history(user_id, days)
    wanted = exercise.lower().strip()
    names = [n for n in history if n == wanted] or [n for n in history if wanted in n or n in wanted]
    if not names:
        # Hand back what IS logged so the model can retry with the right name.
        return {"found": False, "exercises_logged": sorted(history)[:20]}

    found = {}
    for name in names[:3]:
        sessions = history[name]
        readiness = rp._check_weight_readiness(sessions)
        if readiness["ready"] and readiness["last_weight"]:
            readiness["suggested_next_weight_kg"] = rp._suggest_weight_increment(name, readiness["last_weight"])
        found[name] = {"recent_sessions": sessions[-6:], "readiness": readiness}
    return {"found": True, "days": days, "exercises": found}


def search_knowledge(user_id: str, query: str):
    """The coach's curated knowledge base, with production's retrieval."""
    return rp.search_gym_knowledge(query, top_k=3) or "No matching knowledge."


def lookup_food(user_id: str, name: str):
    """Per-100 g macros from the foods table."""
    return rp.get_food_nutrition(name) or f"'{name}' is not in the food database."


READ_TOOLS = {
    "get_user_data": get_user_data,
    "get_lift_history": get_lift_history,
    "search_knowledge": search_knowledge,
    "lookup_food": lookup_food,
}


def _tool(name: str, description: str, properties: dict, required: list) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


TOOLS = [
    _tool("get_user_data",
          "Fetch one section of the user's data. training: equipment, program, sessions this week, "
          "PRs, stalled lifts. nutrition: diet, food restrictions, macro targets, 7-day average, "
          "today's intake. progress: weight trend, streak, workouts vs plan.",
          {"section": {"type": "string", "enum": ["training", "nutrition", "progress"]}},
          ["section"]),
    _tool("get_lift_history",
          "Logged sets for one exercise over recent days, and whether it is ready for a weight "
          "increase (2+ sessions over 14+ days). If the name isn't found, returns the exercise "
          "names the user has logged so you can retry.",
          {"exercise": {"type": "string"}, "days": {"type": "integer", "minimum": 7, "maximum": 90}},
          ["exercise"]),
    _tool("search_knowledge",
          "Search the coach's curated training and nutrition knowledge base.",
          {"query": {"type": "string"}},
          ["query"]),
    _tool("lookup_food",
          "Calories and macros per 100 g for one food.",
          {"name": {"type": "string"}},
          ["name"]),
    _tool("propose_change",
          "Propose a change for the user to confirm. Nothing is saved until they do.",
          {"type": {"type": "string",
                    "enum": ["log_food", "log_weight", "update_target_weight", "swap_exercise"]},
           "summary": {"type": "string", "description": "One line the user will see."},
           "details": {"type": "object", "description":
                       "log_food: food_name, quantity_g, meal_type, calories, protein_g, carbs_g, "
                       "fat_g. log_weight: weight_kg. update_target_weight: exercise, "
                       "new_weight_kg. swap_exercise: from, to."}},
          ["type", "summary"]),
]


# ── Loop ─────────────────────────────────────────────────────────────────────

def run_agent(user_id: str, message: str, history: list | None = None,
              client=None, model: str = rp.MODEL) -> dict:
    """Answer one message. Returns {"reply", "steps", "proposals"}."""
    intent = rp._classify_intent_regex(message)
    if intent in FIXED_REPLIES:
        return {"reply": FIXED_REPLIES[intent], "steps": [], "proposals": []}

    profile = get_or_build_context(user_id).get("base", "")
    system = SYSTEM_PROMPT + (MEDICAL_NOTE if intent == "medical" else "") + "\n\n" + profile
    messages = [{"role": "system", "content": system}, *(history or []),
                {"role": "user", "content": message}]
    client = client or rp._deepseek()
    steps: list[dict] = []
    proposals: list[dict] = []

    for _ in range(MAX_STEPS):
        reply = _complete(client, model, messages, tools=TOOLS)
        if not reply.tool_calls:
            return _finish(reply, steps, proposals)
        messages.append(_assistant_turn(reply))
        for call in reply.tool_calls:
            result = _call_tool(call, user_id, message, proposals, steps)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # Out of rounds. No tools on this call, so it has to answer with what it has.
    return _finish(_complete(client, model, messages), steps, proposals)


def _complete(client, model: str, messages: list, tools: list | None = None):
    kwargs = {"model": model, "messages": messages, "max_tokens": 2500, "temperature": 0.3}
    if tools:
        # "auto", never "required": DeepSeek's thinking models reject "required"
        # with a 400, which once broke every program build in production.
        kwargs.update(tools=tools, tool_choice="auto")
    return client.chat.completions.create(**kwargs).choices[0].message


def _assistant_turn(reply) -> dict:
    turn = {
        "role": "assistant",
        "content": reply.content or "",
        "tool_calls": [{"id": c.id, "type": "function",
                        "function": {"name": c.function.name, "arguments": c.function.arguments}}
                       for c in reply.tool_calls],
    }
    # Thinking models need their reasoning handed back between tool rounds.
    reasoning = getattr(reply, "reasoning_content", None)
    if reasoning:
        turn["reasoning_content"] = reasoning
    return turn


def _call_tool(call, user_id: str, message: str, proposals: list, steps: list) -> str:
    """Run one tool call. Errors go back to the model as text, and into the trace."""
    name = call.function.name
    args: dict = {}
    try:
        args = json.loads(call.function.arguments or "{}")
        if name == "propose_change":
            result = _propose(args, message, proposals)
        elif name in READ_TOOLS:
            result = READ_TOOLS[name](user_id, **args)
        else:
            result = f"Unknown tool: {name}"
    except Exception as err:  # bad arguments from the model, or a failed read
        logger.warning("[agent] tool %s failed: %s", name, err)
        result = f"Tool error: {type(err).__name__}: {err}"

    text = result if isinstance(result, str) else json.dumps(result, default=str)
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + " ...[truncated]"
    steps.append({"tool": name, "args": args, "result": text})
    return text


def _propose(args: dict, message: str, proposals: list) -> str:
    proposal = {"type": args["type"], "summary": args["summary"], "details": dict(args.get("details") or {})}
    if proposal["type"] == "log_food":
        # Same grounding as production: when the food is in the database, the
        # macros are recomputed from it rather than taken from the model.
        grounded = rp._ground_food_macros([{"type": "log_food", **proposal["details"]}], message)[0]
        grounded.pop("type")
        proposal["details"] = grounded
    proposals.append(proposal)
    return "Proposed. Nothing has changed yet; the user will be asked to confirm."


def _finish(reply, steps: list, proposals: list) -> dict:
    text = (reply.content or "").strip()
    if not text:
        # Loud on purpose. An apology here would hide the failure, the same
        # mistake that made /chat return HTTP 200 on errors.
        raise RuntimeError("agent: model returned an empty answer")
    return {"reply": text, "steps": steps, "proposals": proposals}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 3:
        sys.exit('usage: python agent_loop.py <user_id> "<message>"')
    out = run_agent(sys.argv[1], sys.argv[2])
    for i, step in enumerate(out["steps"], 1):
        print(f"\n[{i}] {step['tool']}({json.dumps(step['args'])})")
        print("    -> " + step["result"][:300].replace("\n", " ") + ("..." if len(step["result"]) > 300 else ""))
    for p in out["proposals"]:
        print(f"\nPROPOSED (needs confirmation): {p['summary']}  {json.dumps(p['details'])}")
    print("\n" + out["reply"])
