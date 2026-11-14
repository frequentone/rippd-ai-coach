#!/usr/bin/env python3
"""
Generate 500 synthetic Indian fitness personas with realistic quirks.
No API calls — pure local generation.
"""

import json
import random
import argparse
import uuid
from typing import TypedDict

class Persona(TypedDict):
    id: str
    label: str
    name: str
    age: int
    gender: str
    experience_level: str  # beginner, intermediate, advanced
    goal: str  # weight_loss, muscle_gain, maintenance, endurance
    diet_preference: str  # vegetarian, non_veg, vegan, jain
    logging_style: str  # consistent, sporadic, typo_prone, contradictory
    quirks: list[str]  # realistic messiness
    stats: dict
    health_notes: list[str]

FIRST_NAMES_M = ["Arjun", "Raj", "Amit", "Vikram", "Rohan", "Aditya", "Manish", "Nikhil", "Sanjay", "Ravi"]
FIRST_NAMES_F = ["Priya", "Anjali", "Kavya", "Divya", "Isha", "Neha", "Pooja", "Riya", "Sneha", "Tanya"]
LAST_NAMES = ["Singh", "Patel", "Sharma", "Kumar", "Gupta", "Reddy", "Nair", "Desai", "Bhat", "Verma"]

GOALS = ["weight_loss", "muscle_gain", "maintenance", "endurance", "strength"]
DIET_PREFS = ["vegetarian", "non_veg", "vegan", "jain", "pescatarian"]
EXPERIENCE = ["beginner", "intermediate", "advanced"]
LOGGING_STYLES = ["consistent", "sporadic", "typo_prone", "contradictory"]

QUIRKS = [
    "forgets to log lunch",
    "logs in inconsistent units (kg vs lbs)",
    "mentions injury but never explains it",
    "contradicts own stats week-to-week",
    "mentions 'cheat day' but logs 5000 cal",
    "vague on meal composition",
    "logs time inconsistently",
    "diet swap mid-week unexplained",
    "describes 'feeling bloated' frequently",
    "compares self to influencers unrealistically",
    "mentions gym attendance but no workout logs",
    "typos in food names (e.g., 'roti' → 'roti', 'daal' → 'dal')",
    "skips logging for days then logs bulk",
    "macro targets wildly inconsistent",
    "mentions thyroid/hormonal issues vaguely",
    "logs 'stress eating' category",
    "inconsistent sleep notes",
    "mentions 'period bloat' but no pattern",
    "asks same question twice in different ways",
    "logs in regional language transliteration",
]

HEALTH_NOTES = [
    "PCOS",
    "thyroid issues",
    "lower back pain",
    "knee injury (2022)",
    "high cholesterol",
    "prediabetic",
    "recovering from COVID",
    "sedentary job",
    "shift worker",
    "mother of 2",
    "recent breakup",
    "disordered eating history",
    "takes thyroid meds",
    "vegetarian for 3 years",
]

def generate_persona(user_id: int, rng: random.Random) -> Persona:
    gender = rng.choice(["M", "F"])
    first = rng.choice(FIRST_NAMES_M if gender == "M" else FIRST_NAMES_F)
    last = rng.choice(LAST_NAMES)

    # Supabase user_id columns are typed uuid — a fake string like "user_0018"
    # gets rejected at the DB layer. Generate a valid-format v4 UUID instead,
    # deterministically from the seeded rng so reruns are reproducible.
    persona_uuid = str(uuid.UUID(int=rng.getrandbits(128), version=4))

    return {
        "id": persona_uuid,
        "label": f"user_{user_id:04d}",
        "name": f"{first} {last}",
        "age": rng.randint(18, 65),
        "gender": gender,
        "experience_level": rng.choice(EXPERIENCE),
        "goal": rng.choice(GOALS),
        "diet_preference": rng.choice(DIET_PREFS),
        "logging_style": rng.choice(LOGGING_STYLES),
        "quirks": rng.sample(QUIRKS, k=rng.randint(2, 4)),
        "stats": {
            "height_cm": rng.randint(150, 195),
            "weight_kg": rng.randint(45, 120),
            "target_weight_kg": rng.randint(45, 120),
            "experience_months": rng.randint(0, 120),
        },
        "health_notes": rng.sample(HEALTH_NOTES, k=rng.randint(0, 3)) if rng.random() > 0.6 else [],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of personas to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", default="personas.json", help="Output file")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    personas = [generate_persona(i, rng) for i in range(args.n)]

    with open(args.out, "w") as f:
        json.dump(personas, f, indent=2)

    print(f"Generated {args.n} personas -> {args.out}")

if __name__ == "__main__":
    main()
