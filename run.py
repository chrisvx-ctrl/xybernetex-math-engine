"""
run.py
Entry point for Xybernetex MCTS Cognitive State Machine.

Reads scenario from scenario.txt in the same directory.
To change the scenario, edit scenario.txt and re-run.

Usage:
    python run.py                          # 25 sims, reads scenario.txt
    python run.py --sims 10                # fewer sims for cheap test run
    python run.py --scenario other.txt     # use a different scenario file
    python run.py --audit_dir my_audit     # custom audit directory
    OPENAI_API_KEY=sk-... python run.py

Output:
    - stdout: tick-by-tick MCTS trace with reward components
    - audit/<scenario_name>/sim_NNN.txt: full reasoning for every simulation
    - audit/<scenario_name>/BEST_PATH.txt: final best path full text
    - results.json: complete structured output

Requirements:
    pip install openai
"""

import argparse
import json
import os
import sys
from datetime import datetime
from openai import OpenAI

from scenario_loader import parse_scenario_file, print_loaded_scenario
from mcts_engine import mcts_search


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        print("  Windows:  $env:OPENAI_API_KEY='sk-...'")
        print("  Linux/Mac: export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def estimate_cost(n_simulations: int, n_phases: int, n_expand: int) -> float:
    """
    Rough cost estimate for gpt-3.5-turbo at $0.002/1K tokens.
    Each simulation: n_phases generations (~500 tok each) + 1 scoring call (~120 tok)
    Plus expansion calls: n_expand * 500 tok per expanded node
    """
    gen_tokens_per_sim = n_phases * 500
    scoring_tokens_per_sim = 120
    expansion_tokens = n_expand * 500 * (n_simulations // 3)  # rough: expand every ~3 sims
    total_tokens = n_simulations * (gen_tokens_per_sim + scoring_tokens_per_sim) + expansion_tokens
    cost = (total_tokens / 1000) * 0.002
    return round(cost, 2)


def print_summary(result: dict):
    print(f"\n{'#'*70}")
    print(f"RUN SUMMARY")
    print(f"{'#'*70}\n")
    print(f"Scenario        : {result['scenario']}")
    print(f"Phases          : {len(result['phases'])}")
    print(f"Simulations run : {result['n_simulations']}")
    print(f"Best path depth : {len(result['best_path'])} phases")

    scored = [n for n in result["best_path"] if n["reward_hybrid"] > 0]
    if scored:
        peak = max(scored, key=lambda n: n["reward_hybrid"])
        print(f"\nPeak reward node : {peak['phase']}")
        print(f"  Rhybrid        : {peak['reward_hybrid']:.4f}")
        print(f"  C1 constraints : {peak['reward_constraints']:.4f}")
        print(f"  C2 alignment   : {peak['reward_alignment']:.4f}")
        print(f"  C3 consistency : {peak['reward_consistency']:.4f}")
        print(f"  logprob        : {peak['avg_logprob']:.4f}  ({peak['token_count']} tokens)")

    rec = next((n for n in result["best_path"] if n["phase"] == "DECISION_RECOMMENDATION"), None)
    if rec:
        print(f"\nDECISION_RECOMMENDATION excerpt:")
        print(f"{'─'*60}")
        print(rec["content"][:800] + ("..." if len(rec["content"]) > 800 else ""))

    plan = next((n for n in result["best_path"] if n["phase"] == "EXECUTION_PLAN"), None)
    if plan:
        print(f"\nEXECUTION_PLAN excerpt:")
        print(f"{'─'*60}")
        print(plan["content"][:600] + ("..." if len(plan["content"]) > 600 else ""))

    print(f"\nReward distribution across simulations:")
    rewards = [s["reward"] for s in result["simulation_trace"]]
    if rewards:
        print(f"  Min   : {min(rewards):.4f}")
        print(f"  Max   : {max(rewards):.4f}")
        print(f"  Mean  : {sum(rewards)/len(rewards):.4f}")
        spread = max(rewards) - min(rewards)
        print(f"  Spread: {spread:.4f}  {'[GOOD: reward gradient exists]' if spread > 0.05 else '[LOW: gradient weak, consider more sims]'}")


def main():
    parser = argparse.ArgumentParser(description="Xybernetex MCTS Cognitive State Machine")
    parser.add_argument("--scenario", default="scenario.txt", help="Path to scenario text file (default: scenario.txt)")
    parser.add_argument("--sims", type=int, default=25, help="MCTS simulations (default: 25)")
    parser.add_argument("--expand", type=int, default=2, help="Expansions per node (default: 2)")
    parser.add_argument("--audit_dir", default="audit", help="Audit output directory (default: audit/)")
    parser.add_argument("--output", default="results.json", help="JSON output file (default: results.json)")
    args = parser.parse_args()

    client = get_client()
    framework = parse_scenario_file(args.scenario)
    print_loaded_scenario(framework)

    est_cost = estimate_cost(args.sims, len(framework["phases"]), args.expand)
    print(f"Estimated API cost: ~${est_cost} (gpt-3.5-turbo)")
    print(f"Audit files will be written to: {args.audit_dir}/{framework['name'].replace(' ', '_')[:60]}/\n")

    start_time = datetime.now()
    print(f"Run started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    result = mcts_search(
        framework=framework,
        client=client,
        n_simulations=args.sims,
        n_expand=args.expand,
        audit_dir=args.audit_dir
    )

    print_summary(result)

    output = {
        "run_timestamp": start_time.isoformat(),
        "duration_seconds": round((datetime.now() - start_time).total_seconds(), 1),
        "config": {
            "scenario_file": args.scenario,
            "n_simulations": args.sims,
            "n_expand": args.expand,
            "model": "gpt-3.5-turbo",
            "audit_dir": args.audit_dir
        },
        "result": result
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results → {args.output}")
    print(f"Audit files  → {args.audit_dir}/")
    print(f"Run complete : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
