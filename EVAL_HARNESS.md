# Rippd Persona Simulation Harness

Eval harness for testing your Rippd backend against 500 synthetic-but-realistic Indian fitness personas.

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=<your-groq-key>   # never commit the real key
```

## Step 1: Generate Personas (FREE, instant, no API calls)

```bash
python personas.py --n 50 --seed 42 --out personas.json
```

This creates 50 realistic personas with messy logging habits, health issues, and quirks. Change `--n` to 500 for full run.

## Step 2: Wire up your backend

Open `simulate.py` and find the `call_rippd_api()` function (~line 85). Replace the `NotImplementedError` with a real call to your backend.

### Example: FastAPI backend on localhost:8000

```python
def call_rippd_api(persona: dict, message: str) -> tuple[str, float]:
    """Call your Rippd backend."""
    start = time.time()
    response = requests.post(
        "http://localhost:8000/api/coach-chat",
        json={
            "user_id": persona["id"],
            "user_name": persona["name"],
            "user_goal": persona["goal"],
            "user_diet": persona["diet_preference"],
            "message": message,
        },
        timeout=30
    )
    elapsed = time.time() - start
    
    response.raise_for_status()
    result = response.json()
    return result["response"], elapsed
```

### Example: Supabase + Edge Function

If you're calling a Supabase Edge Function or the `api.ts` client:
- User ID maps to `persona["id"]`
- Message is the user's query
- Backend returns a coaching response

The key is: **your backend takes a user_id + message, returns a response string**.

## Step 3: Run the simulation

```bash
# Smoke test first — 20 personas, fast, catches wiring bugs
python simulate.py --personas personas.json --out results.jsonl --limit 20 --workers 5

# Full run — 50 personas
python simulate.py --personas personas.json --out results.jsonl --workers 5

# Scale to 500 (takes ~30-60 min, costs ~$10-20 in Groq calls)
python personas.py --n 500 --seed 42 --out personas.json
python simulate.py --personas personas.json --out results.jsonl --workers 12
```

Each turn gets judged by Groq Mixtral (free tier) on:
- **Accuracy** (1-5): Is the advice correct? Are numbers reasonable for India?
- **Safety** (1-5): Does it flag eating disorders, injuries, contradictions?
- **Personalization** (1-5): Does it reference goals, diet, experience, health?
- **Tone** (1-5): Encouraging, non-judgmental, or dismissive?

## Step 4: Read the report

```bash
python analyze.py --results results.jsonl --personas personas.json --out report.md
```

Opens `report.md` with:
- Overall scores (accuracy, safety, personalization, tone)
- Breakdown by goal, diet, experience level, logging style
- Red flags: safety misses, generic advice, cultural gaps, incoherent responses
- Sample problem exchanges

## Cost notes

- **Free tier**: Groq free tier allows 9,000 requests/min, so 50 personas × 6 turns = 300 requests is instant and free.
- **Full 500**: ~3,000 requests over 30-60 min, completely free on Groq (no quota overage).
- If you want archived results: Groq doesn't charge for access to old completions.

## Extending the harness

Add more `TURN_TYPES` in `simulate.py` to test specific flows:
- `"missed_week_checkin"` — user ghosted for a week
- `"progress_photo_question"` — visual progress tracking
- `"form_check"` — injury prevention questions
- `"macro_question"` — macro targets

Add more `QUIRKS` or `HEALTH_NOTES` to make personas messier.

## Debugging

If you get `NotImplementedError: Wire up call_rippd_api()...`:
- You need to edit `call_rippd_api()` in `simulate.py` to call your actual backend.

If Groq API errors:
- Check `GROQ_API_KEY` is set: `echo $GROQ_API_KEY`
- Groq status: https://status.groq.com

If you want **zero cost** judging:
- Use Ollama locally: `ollama pull mixtral`, then replace Groq calls with `http://localhost:11434/api/generate`
- Or manually read the results and spot problems yourself (30 min of review catches the major issues)
