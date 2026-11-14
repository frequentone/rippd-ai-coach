#!/usr/bin/env python3
"""
Analyze simulation results and generate report with breakdowns.
"""

import json
import argparse
from collections import defaultdict
from statistics import mean, stdev
from typing import TypedDict

class Analysis(TypedDict):
    metric: str
    avg_accuracy: float
    avg_safety: float
    avg_personalization: float
    avg_tone: float
    count: int

def load_results(path: str):
    """Load JSONL results."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results

def load_personas(path: str):
    """Load personas for context."""
    with open(path) as f:
        return {p["id"]: p for p in json.load(f)}

def extract_scores(results):
    """Extract all scores from results."""
    scores = []
    errors = []
    for result in results:
        for turn in result["turns"]:
            if turn.get("error"):
                errors.append(f"{result['persona_id']}: {turn['error']}")
                continue

            judge = turn.get("judge", {})
            if "error" in judge:
                errors.append(f"{result['persona_id']} turn {turn['turn_num']}: {judge['error']}")
                continue

            if "accuracy" in judge:
                scores.append({
                    "persona_id": result["persona_id"],
                    "turn_type": turn["turn_type"],
                    "accuracy": judge["accuracy"],
                    "safety": judge["safety"],
                    "personalization": judge["personalization"],
                    "tone": judge["tone"],
                    "flags": judge.get("flags", []),
                    "error": turn.get("error"),
                })

    if errors:
        print(f"WARNING: {len(errors)} errors during evaluation:")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    return scores

def group_by(scores, persona_map, group_key: str):
    """Group scores by a persona attribute."""
    groups = defaultdict(list)
    for score in scores:
        persona = persona_map.get(score["persona_id"])
        if not persona:
            continue

        if group_key == "goal":
            key = persona["goal"]
        elif group_key == "diet":
            key = persona["diet_preference"]
        elif group_key == "experience":
            key = persona["experience_level"]
        elif group_key == "logging_style":
            key = persona["logging_style"]
        elif group_key == "turn_type":
            key = score["turn_type"]
        else:
            continue

        groups[key].append(score)

    return groups

def compute_stats(score_list):
    """Compute stats for a group."""
    if not score_list:
        return None

    accuracy = [s["accuracy"] for s in score_list]
    safety = [s["safety"] for s in score_list]
    personalization = [s["personalization"] for s in score_list]
    tone = [s["tone"] for s in score_list]

    return {
        "accuracy": mean(accuracy),
        "safety": mean(safety),
        "personalization": mean(personalization),
        "tone": mean(tone),
        "count": len(score_list),
    }

def find_red_flags(scores):
    """Find concerning patterns."""
    flags = defaultdict(int)
    low_safety = []
    low_personalization = []

    for score in scores:
        for flag in score.get("flags", []):
            flags[flag] += 1

        if score["safety"] <= 2:
            low_safety.append(score)
        if score["personalization"] <= 2:
            low_personalization.append(score)

    return {
        "flag_counts": dict(flags),
        "low_safety_count": len(low_safety),
        "low_personalization_count": len(low_personalization),
        "sample_low_safety": low_safety[:5],
        "sample_low_personalization": low_personalization[:5],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results.jsonl")
    parser.add_argument("--personas", required=True, help="Path to personas.json")
    parser.add_argument("--out", default="report.md", help="Output markdown file")
    args = parser.parse_args()

    results = load_results(args.results)
    personas = load_personas(args.personas)
    scores = extract_scores(results)

    if not scores:
        print("ERROR: No valid scores extracted. Check that your backend is running and returning valid responses.")
        print(f"   Loaded {len(results)} personas with {sum(len(r['turns']) for r in results)} total turns.")
        return

    # Overall stats
    overall = compute_stats(scores)

    # Breakdown by dimension
    breakdowns = {
        "goal": group_by(scores, personas, "goal"),
        "diet": group_by(scores, personas, "diet"),
        "experience": group_by(scores, personas, "experience"),
        "logging_style": group_by(scores, personas, "logging_style"),
        "turn_type": group_by(scores, personas, "turn_type"),
    }

    # Red flags
    red_flags = find_red_flags(scores)

    # Generate report
    report = f"""# Rippd Simulation Report

**Run**: {len(results)} personas × {len(scores) // len(results) if results else 0} turns ≈ {len(scores)} exchanges

## Overall Performance

| Metric | Score |
|--------|-------|
| Accuracy | {overall['accuracy']:.2f}/5 |
| Safety | {overall['safety']:.2f}/5 |
| Personalization | {overall['personalization']:.2f}/5 |
| Tone | {overall['tone']:.2f}/5 |

## Breakdown by Goal

"""

    for goal, group_scores in sorted(breakdowns["goal"].items()):
        stats = compute_stats(group_scores)
        if stats:
            report += f"**{goal.replace('_', ' ').title()}** (n={stats['count']}): "
            report += f"Accuracy {stats['accuracy']:.1f} | Safety {stats['safety']:.1f} | "
            report += f"Personalization {stats['personalization']:.1f} | Tone {stats['tone']:.1f}\n\n"

    report += "## Breakdown by Diet Preference\n\n"
    for diet, group_scores in sorted(breakdowns["diet"].items()):
        stats = compute_stats(group_scores)
        if stats:
            report += f"**{diet.replace('_', ' ').title()}** (n={stats['count']}): "
            report += f"Accuracy {stats['accuracy']:.1f} | Safety {stats['safety']:.1f} | "
            report += f"Personalization {stats['personalization']:.1f} | Tone {stats['tone']:.1f}\n\n"

    report += "## Breakdown by Experience Level\n\n"
    for exp, group_scores in sorted(breakdowns["experience"].items()):
        stats = compute_stats(group_scores)
        if stats:
            report += f"**{exp.replace('_', ' ').title()}** (n={stats['count']}): "
            report += f"Accuracy {stats['accuracy']:.1f} | Safety {stats['safety']:.1f} | "
            report += f"Personalization {stats['personalization']:.1f} | Tone {stats['tone']:.1f}\n\n"

    report += "## Red Flags\n\n"
    if red_flags["flag_counts"]:
        report += "### Flag Counts\n"
        for flag, count in sorted(red_flags["flag_counts"].items(), key=lambda x: -x[1]):
            report += f"- {flag}: {count}\n"
        report += "\n"

    report += f"### Safety Concerns\n"
    report += f"Low safety responses (≤2/5): {red_flags['low_safety_count']}\n\n"

    report += f"### Personalization Gaps\n"
    report += f"Low personalization (≤2/5): {red_flags['low_personalization_count']}\n\n"

    # Write report
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to {args.out}")
    print(f"\nSummary:")
    print(f"  Overall Accuracy: {overall['accuracy']:.2f}/5")
    print(f"  Overall Safety: {overall['safety']:.2f}/5")
    print(f"  Overall Personalization: {overall['personalization']:.2f}/5")
    print(f"  Overall Tone: {overall['tone']:.2f}/5")

if __name__ == "__main__":
    main()
