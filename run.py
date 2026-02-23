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
    - audit/<scenario_name>/RUN_REPORT_*.txt: human-readable run report

Requirements:
    pip install openai
"""

import argparse
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
    Phase generation: n_phases * ~500 tok per sim
    Scoring (C3+C5): 2 extra calls * ~200 tok per sim
    Expansion: n_expand * 500 tok every ~3 sims
    """
    gen_tokens = n_simulations * n_phases * 500
    scoring_tokens = n_simulations * 2 * 200
    expand_tokens = n_expand * 500 * (n_simulations // 3)
    total = gen_tokens + scoring_tokens + expand_tokens
    return round((total / 1000) * 0.002, 2)


def print_summary(result: dict):
    print(f"\n{'#'*70}")
    print(f"RUN SUMMARY")
    print(f"{'#'*70}\n")

    print(f"Scenario        : {result['scenario']}")
    print(f"Simulations run : {result['n_simulations']}")
    print(f"Best sim        : #{result['best_sim_idx']:02d}")
    print(f"Best reward     : {result['best_reward']:.4f}")

    # Timing block
    t = result.get("timing", {})
    if t:
        print(f"\nTIMING")
        print(f"  Wall time      : {t.get('run_wall_s', 0):.1f}s  ({t.get('run_wall_s', 0)/60:.1f} min)")
        print(f"  LLM time       : {t.get('total_llm_s', 0):.1f}s  "
              f"({round(t.get('total_llm_s',0)/t.get('run_wall_s',1)*100,1)}% of wall)")
        print(f"  Overhead       : {t.get('total_overhead_s', 0):.1f}s  (scoring + audit + C6)")
        print(f"  Avg sim time   : {t.get('avg_sim_wall_ms', 0):.0f}ms")
        print(f"  Avg tick       : {t.get('avg_tick_ms', 0):.0f}ms")
        print(f"  Total tokens   : {t.get('total_tokens', 0):,}")

    # C6 convergence
    c6 = result.get("c6_convergence", {})
    if c6:
        print(f"\nC6 CONVERGENCE")
        print(f"  Score   : {c6.get('score', 0):.4f}")
        print(f"  Label   : {c6.get('label', 'N/A')}")
        print(f"  Summary : {c6.get('summary', 'N/A')}")

    # Best path — now a list of TickNode objects
    best_path = result.get("best_path", [])

    # Find recommendation and plan phases
    rec = next((n for n in best_path if n.phase == "DECISION_RECOMMENDATION"), None)
    plan = next((n for n in best_path if n.phase == "EXECUTION_PLAN"), None)

    # Peak reward node
    scored = [n for n in best_path if n.reward_hybrid > 0]
    if scored:
        peak = max(scored, key=lambda n: n.reward_hybrid)
        print(f"\nPEAK REWARD NODE : {peak.phase}")
        print(f"  Rhybrid        : {peak.reward_hybrid:.4f}")
        print(f"  C1 constraints : {peak.reward_constraints:.4f}")
        print(f"  C2 alignment   : {peak.reward_alignment:.4f}")
        print(f"  C3 consistency : {peak.reward_consistency:.4f}")
        print(f"  C5 defensibility: {peak.reward_defensibility:.4f}")
        print(f"  logprob        : {peak.avg_logprob:.4f}  ({peak.token_count} tokens)")

    if rec:
        print(f"\nDECISION_RECOMMENDATION excerpt:")
        print(f"{'─'*60}")
        print(rec.content[:800] + ("..." if len(rec.content) > 800 else ""))

    if plan:
        print(f"\nEXECUTION_PLAN excerpt:")
        print(f"{'─'*60}")
        print(plan.content[:600] + ("..." if len(plan.content) > 600 else ""))

    # Persona distribution
    persona_dist = result.get("persona_distribution", {})
    if persona_dist:
        print(f"\nPERSONA DISTRIBUTION")
        for p, count in sorted(persona_dist.items(), key=lambda x: -x[1]):
            print(f"  {p:<22} {count} sims")

    print(f"\nRun report → {result.get('report_path', 'see audit dir')}")


def main():
    parser = argparse.ArgumentParser(description="Xybernetex MCTS Cognitive State Machine")
    parser.add_argument("--scenario", default="scenario.txt",
                        help="Path to scenario text file (default: scenario.txt)")
    parser.add_argument("--sims", type=int, default=25,
                        help="MCTS simulations (default: 25)")
    parser.add_argument("--expand", type=int, default=2,
                        help="Expansions per node (default: 2)")
    parser.add_argument("--audit_dir", default="audit",
                        help="Audit output directory (default: audit/)")
    args = parser.parse_args()

    client = get_client()
    framework = parse_scenario_file(args.scenario)
    print_loaded_scenario(framework)

    est_cost = estimate_cost(args.sims, len(framework["phases"]), args.expand)
    n_phases = len(framework["phases"])
    print(f"Estimated API cost : ~${est_cost} (gpt-3.5-turbo)")
    print(f"Phases             : {n_phases}  →  {' → '.join(framework['phases'])}")
    print(f"Audit dir          : {args.audit_dir}/\n")

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

    duration = round((datetime.now() - start_time).total_seconds(), 1)
    print(f"\nRun complete : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ({duration}s)")


if __name__ == "__main__":
    main()
