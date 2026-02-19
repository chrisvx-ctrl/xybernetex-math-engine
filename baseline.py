"""
baseline.py
Cold single-pass GPT-3.5-turbo call for comparison against MCTS output.

Reads the same scenario.txt file as run.py.
Makes ONE API call with no phases, no MCTS, no directives.
Writes output to baseline_output.txt and baseline_output.json.

Usage:
    python baseline.py
    python baseline.py --scenario scenario.txt
    OPENAI_API_KEY=sk-... python baseline.py
"""

import argparse
import json
import os
import sys
from datetime import datetime
from openai import OpenAI

from scenario_loader import parse_scenario_file, print_loaded_scenario


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def run_baseline(framework: dict, client: OpenAI) -> dict:
    prompt = (
        f"You are a senior strategic analyst. Analyze the following decision scenario "
        f"and provide a complete, rigorous recommendation.\n\n"
        f"SCENARIO:\n{framework['scenario']}\n\n"
        f"Your analysis should address the following dimensions:\n"
        + "\n".join(f"- {d}" for d in framework["dimensions"])
        + f"\n\nAlso address the following constraints:\n"
        + "\n".join(f"- {c}" for c in framework["constraints"])
        + f"\n\nProvide: your assessment of the situation, the key tradeoffs, "
        f"a clear recommendation, and an execution plan."
    )

    print(f"\nSending single cold call to gpt-3.5-turbo...")
    print(f"Prompt length: {len(prompt.split())} words\n")

    start = datetime.now()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior strategic analyst. "
                    "Be specific, direct, and analytically rigorous. "
                    "Make a clear recommendation. Do not hedge excessively."
                )
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.7,
        logprobs=True,
        top_logprobs=1
    )

    elapsed = round((datetime.now() - start).total_seconds(), 2)
    content = response.choices[0].message.content or ""

    # Compute avg logprob across full response
    lp_data = response.choices[0].logprobs
    if lp_data and lp_data.content:
        lps = [t.logprob for t in lp_data.content]
        avg_lp = round(sum(lps) / len(lps), 4)
        token_count = len(lps)
    else:
        avg_lp = -2.0
        token_count = 0

    print(f"Response received in {elapsed}s")
    print(f"Tokens generated: {token_count}")
    print(f"Avg logprob: {avg_lp:.4f}")
    print(f"\n{'='*70}")
    print(f"BASELINE OUTPUT")
    print(f"{'='*70}\n")
    print(content)

    return {
        "scenario": framework["name"],
        "model": "gpt-3.5-turbo",
        "call_type": "single_cold_pass",
        "elapsed_seconds": elapsed,
        "token_count": token_count,
        "avg_logprob": avg_lp,
        "content": content,
        "prompt_word_count": len(prompt.split()),
        "timestamp": datetime.now().isoformat()
    }


def main():
    parser = argparse.ArgumentParser(description="Cold baseline single-pass LLM call")
    parser.add_argument("--scenario", default="scenario.txt")
    parser.add_argument("--output_txt", default="baseline_output.txt")
    parser.add_argument("--output_json", default="baseline_output.json")
    args = parser.parse_args()

    client = get_client()
    framework = parse_scenario_file(args.scenario)
    print_loaded_scenario(framework)

    result = run_baseline(framework, client)

    # Write plain text output
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write(f"BASELINE — Single Cold Pass\n")
        f.write(f"Scenario: {result['scenario']}\n")
        f.write(f"Model: {result['model']}\n")
        f.write(f"Timestamp: {result['timestamp']}\n")
        f.write(f"Tokens: {result['token_count']}  |  Avg logprob: {result['avg_logprob']:.4f}\n")
        f.write(f"{'='*70}\n\n")
        f.write(result["content"])

    # Write JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Baseline output → {args.output_txt}")
    print(f"Baseline JSON   → {args.output_json}")
    print(f"\nNow compare baseline_output.txt against audit/BEST_PATH.txt manually.")
    print(f"Read both blind if possible — cover which is which and judge quality first.")


if __name__ == "__main__":
    main()
