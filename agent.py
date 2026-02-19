"""
agent.py
Minimal Viable Autonomous Agent for the Xybernetex MCTS Decision Engine.

The agent wraps mcts_search with a self-evaluation and mutation loop.
After each run it calls the LLM to reason visibly about the run quality,
then issues a verdict and mutates parameters if needed.

The reasoning step is printed to stdout in full and written to AGENT_LOG.txt
so every autonomous decision is legible and auditable.

Verdicts:
  ACCEPT          — adequate quality, commit result
  RERUN_TIGHTEN   — too many ambiguous outcomes, inject commitment forcing
  RERUN_DIVERSIFY — simulations converged, inject adversarial pressure
  RERUN_DEEPEN    — C5 defensibility weak, force traceability

Writes:
  agent_audit/AGENT_LOG.txt       — full decision trace per pass
  agent_audit/pass_NN/            — sim audit files per pass
  agent_audit/agent_result.json   — structured output across all passes

Usage:
    python agent.py
    python agent.py --scenario scenario.txt --max_passes 3 --sims 25
    python agent.py --dry_run
"""

import argparse
import json
import os
import sys
import re
from datetime import datetime
from collections import Counter
from openai import OpenAI

from scenario_loader import parse_scenario_file, print_loaded_scenario
from mcts_engine import mcts_search, PERSONAS


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "min_reward_spread":     0.05,
    "min_c5_mean":           0.55,
    "max_ambiguous_ratio":   0.50,
    "min_outcome_diversity": 2,
    "accept_reward_mean":    0.90,
}


# ---------------------------------------------------------------------------
# Metric Extraction
# ---------------------------------------------------------------------------

def detect_commitment(audit_file: str) -> bool:
    """
    Returns True if the DECISION_RECOMMENDATION contains a specific named commitment.
    Uses both scenario-specific and generic commitment patterns.
    """
    if not audit_file or not os.path.exists(audit_file):
        return False

    try:
        with open(audit_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return False

    rec_match = re.search(
        r'PHASE \d+/\d+: DECISION_RECOMMENDATION\n.*?(?=PHASE \d+/\d+:|$)',
        content, re.DOTALL
    )
    if not rec_match:
        return False

    rec_text = rec_match.group(0).lower()

    commitment_patterns = [
        # --- Pharma: named candidate cuts ---
        r'\bcut candidate [abc]\b',
        r'\beliminate candidate [abc]\b',
        r'\bdiscontinue candidate [abc]\b',
        r'\bdrop candidate [abc]\b',
        r'\bcandidate [abc] should be (cut|eliminated|discontinued|dropped)\b',
        r'\bcandidate [abc] must be (cut|eliminated|discontinued|dropped)\b',
        r'\brecommend (cutting|eliminating|discontinuing) candidate [abc]\b',
        r'\bprotect candidate [abc]\b',
        r'\bretain candidate [abc]\b',

        # --- Infrastructure: bridge-specific ---
        r'\bdefer (the )?(commerce|eastside|lakeside|pedestrian)\b',
        r'\bpatch (the )?(commerce|eastside|lakeside|pedestrian)\b',
        r'\breplace (the )?(commerce|eastside|lakeside|pedestrian)\b',
        r'\b(commerce|eastside|lakeside|pedestrian) (viaduct|connector|bridge) (should|must) (be )?(deferred|patched|replaced)\b',
        r'\bdefer bridge [123]\b',
        r'\bpatch bridge [123]\b',
        r'\breplace bridge [123]\b',
        r'\boption [123] —\b',
        r'\boption [123]:',

        # --- Generic strong commitment signals (any domain) ---
        r'\bthe (recommendation|decision) is to\b',
        r'\bwe (recommend|recommend that|should|must)\b.{0,40}\b(cut|eliminate|defer|replace|patch|select|choose|adopt|reject|discontinue|retain|protect)\b',
        r'\bour recommendation:',
        r'\bfinal recommendation:',
        r'\brecommended (decision|option|choice|action):',
        r'\bthe optimal (choice|option|decision|path) is\b',
        r'\bthe recommended (course|option|decision|action) is\b',
        r'\bi recommend\b',
        r'\bwe recommend\b',

        # --- Bold/header commitment signals (markdown formatted output) ---
        r'\*\*(recommendation|decision|recommended option|final decision)\*\*',
        r'\*\*option [123]\b',
        r'\*\*defer\b',
        r'\*\*replace\b',
        r'\*\*cut\b',
        r'\*\*eliminate\b',
    ]

    return any(re.search(p, rec_text) for p in commitment_patterns)


def extract_run_metrics(result: dict, audit_base_dir: str = "") -> dict:
    trace = result["simulation_trace"]
    if not trace:
        return {}

    rewards       = [s["reward"] for s in trace]
    c5_scores     = [s["components"].get("c5", 0.5) for s in trace]
    c3_scores     = [s["components"].get("c3", 0.5) for s in trace]
    c1_scores     = [s["components"].get("c1", 0.5) for s in trace]
    personas_used = Counter(s["persona"] for s in trace)
    low_c5_count  = sum(1 for c in c5_scores if c < 0.4)
    reward_tiers  = Counter(round(r, 1) for r in rewards)

    # Real ambiguity detection — parse audit files for named commitments
    committed = 0
    non_committed = 0
    for s in trace:
        audit_file = s.get("audit_file", "")
        if detect_commitment(audit_file):
            committed += 1
        else:
            non_committed += 1

    n = len(trace)
    ambiguous_ratio = round(non_committed / n, 4) if n > 0 else 1.0

    return {
        "n_sims":                len(trace),
        "spread":                round(max(rewards) - min(rewards), 4),
        "min_reward":            round(min(rewards), 4),
        "max_reward":            round(max(rewards), 4),
        "mean_reward":           round(sum(rewards) / len(rewards), 4),
        "mean_c5":               round(sum(c5_scores) / len(c5_scores), 4),
        "mean_c3":               round(sum(c3_scores) / len(c3_scores), 4),
        "mean_c1":               round(sum(c1_scores) / len(c1_scores), 4),
        "low_c5_count":          low_c5_count,
        "committed_count":       committed,
        "non_committed_count":   non_committed,
        "ambiguous_ratio":       ambiguous_ratio,
        "outcome_diversity":     len(reward_tiers),
        "personas_distribution": dict(personas_used),
    }


# ---------------------------------------------------------------------------
# LLM Reasoning Step — the agent thinks out loud before deciding
# ---------------------------------------------------------------------------

def agent_reason(
    pass_number: int,
    metrics: dict,
    scenario_name: str,
    prior_verdicts: list,
    client: OpenAI
) -> tuple[str, str]:
    """
    Calls the LLM to reason about the run quality and recommend a verdict.
    Returns (reasoning_text, recommended_verdict).

    The reasoning is printed in full to stdout and written to the log.
    This is the agent's visible thinking step — not a black box decision.
    """

    prior_context = ""
    if prior_verdicts:
        prior_context = (
            f"Prior pass verdicts: {', '.join(prior_verdicts)}\n"
            f"This is pass {pass_number}. Prior passes did not meet acceptance criteria.\n\n"
        )

    prompt = (
        f"You are the autonomous evaluation agent for the Xybernetex MCTS decision engine.\n"
        f"You have just completed pass {pass_number} of MCTS simulation on the scenario: "
        f"'{scenario_name}'.\n\n"
        f"{prior_context}"
        f"Here are the quantitative metrics from this pass:\n\n"
        f"  Simulations run      : {metrics.get('n_sims')}\n"
        f"  Reward spread        : {metrics.get('spread')}  (threshold: >{THRESHOLDS['min_reward_spread']})\n"
        f"  Min / Max reward     : {metrics.get('min_reward')} / {metrics.get('max_reward')}\n"
        f"  Mean reward          : {metrics.get('mean_reward')}  (threshold: >{THRESHOLDS['accept_reward_mean']})\n"
        f"  Mean C5 defensibility: {metrics.get('mean_c5')}  (threshold: >{THRESHOLDS['min_c5_mean']})\n"
        f"  Mean C3 consistency  : {metrics.get('mean_c3')}\n"
        f"  Mean C1 constraints  : {metrics.get('mean_c1')}\n"
        f"  Low-C5 sim count     : {metrics.get('low_c5_count')} / {metrics.get('n_sims')}\n"
        f"  Ambiguous ratio      : {metrics.get('ambiguous_ratio')}  (threshold: <{THRESHOLDS['max_ambiguous_ratio']})\n"
        f"  Committed sims       : {metrics.get('committed_count')} / {metrics.get('n_sims')} named a specific decision\n"
        f"  Non-committed sims   : {metrics.get('non_committed_count')} / {metrics.get('n_sims')} failed to name a specific decision\n"
        f"  Outcome diversity    : {metrics.get('outcome_diversity')}  (threshold: >={THRESHOLDS['min_outcome_diversity']})\n"
        f"  Persona distribution : {metrics.get('personas_distribution')}\n\n"
        f"Your job:\n"
        f"1. Reason through what these metrics tell you about the quality of this run.\n"
        f"2. Identify the single most important failure mode if any exists.\n"
        f"3. Explain what you would change and why.\n"
        f"4. Issue one of these verdicts on the final line:\n"
        f"   VERDICT: ACCEPT\n"
        f"   VERDICT: RERUN_TIGHTEN\n"
        f"   VERDICT: RERUN_DIVERSIFY\n"
        f"   VERDICT: RERUN_DEEPEN\n\n"
        f"Verdict definitions:\n"
        f"  ACCEPT          — run quality meets all thresholds, commit this result\n"
        f"  RERUN_TIGHTEN   — too many non-committal recommendations, force specificity\n"
        f"  RERUN_DIVERSIFY — simulations converged on same answer, break consensus\n"
        f"  RERUN_DEEPEN    — recommendations not traceable to prior analysis, force linkage\n\n"
        f"Write your reasoning in full. Be direct and specific about what the numbers mean.\n\n"
        f"MANDATORY SELF-CHECK — before issuing your verdict, explicitly state YES or NO for each:\n"
        f"  spread > {THRESHOLDS['min_reward_spread']}?              YES or NO\n"
        f"  mean_c5 > {THRESHOLDS['min_c5_mean']}?            YES or NO\n"
        f"  ambiguous_ratio < {THRESHOLDS['max_ambiguous_ratio']}?      YES or NO\n"
        f"  outcome_diversity >= {THRESHOLDS['min_outcome_diversity']}?   YES or NO\n"
        f"  mean_reward > {THRESHOLDS['accept_reward_mean']}?         YES or NO\n\n"
        f"RULE: If ALL five checks are YES, your verdict MUST be ACCEPT. "
        f"You are not permitted to issue a RERUN verdict when all thresholds pass. "
        f"Qualitative observations and concerns do not override passing thresholds — "
        f"they belong in your reasoning text but cannot change a passing verdict to a rerun.\n\n"
        f"End with exactly one VERDICT line."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise autonomous evaluation agent. "
                        "Reason carefully about quantitative run metrics. "
                        "Be direct and specific. End with exactly one VERDICT line."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )

        reasoning = response.choices[0].message.content.strip()

        # Extract verdict from reasoning text
        verdict_match = re.search(
            r'VERDICT:\s*(ACCEPT|RERUN_TIGHTEN|RERUN_DIVERSIFY|RERUN_DEEPEN)',
            reasoning
        )
        extracted_verdict = verdict_match.group(1) if verdict_match else None

        return reasoning, extracted_verdict

    except Exception as e:
        fallback = f"[Agent reasoning call failed: {e}]\nVERDICT: RERUN_DEEPEN"
        return fallback, "RERUN_DEEPEN"


# ---------------------------------------------------------------------------
# Deterministic Fallback Verdict
# (used if LLM verdict is missing or parsing fails)
# ---------------------------------------------------------------------------

def deterministic_verdict(metrics: dict) -> str:
    if metrics.get("mean_c5", 1.0) < THRESHOLDS["min_c5_mean"]:
        return "RERUN_DEEPEN"
    if metrics.get("ambiguous_ratio", 0.0) > THRESHOLDS["max_ambiguous_ratio"]:
        return "RERUN_TIGHTEN"
    if metrics.get("spread", 1.0) < THRESHOLDS["min_reward_spread"]:
        return "RERUN_DIVERSIFY"
    if metrics.get("outcome_diversity", 99) < THRESHOLDS["min_outcome_diversity"]:
        return "RERUN_DIVERSIFY"
    if metrics.get("mean_reward", 1.0) < THRESHOLDS["accept_reward_mean"]:
        return "RERUN_DEEPEN"
    return "ACCEPT"


# ---------------------------------------------------------------------------
# Framework Mutator
# ---------------------------------------------------------------------------

def mutate_framework(framework: dict, verdict: str, pass_number: int) -> dict:
    import copy
    mutated = copy.deepcopy(framework)

    # Each verdict contributes its constraint text — all accumulate, none replace
    constraint_additions = {
        "RERUN_TIGHTEN": (
            "The DECISION_RECOMMENDATION must name exactly one specific option by name. "
            "A recommendation that does not commit to a single named choice is invalid. "
            "Hedging, equivocating, or recommending 'further analysis' is not acceptable."
        ),
        "RERUN_DIVERSIFY": (
            "The analysis must explicitly argue for the option that appears least obvious "
            "or most counterintuitive. Consensus answers require active challenge."
        ),
        "RERUN_DEEPEN": (
            "The DECISION_RECOMMENDATION must explicitly reference at least two specific risks "
            "from RISK_ASSESSMENT and at least one constraint from CONSTRAINT_MAPPING by name. "
            "Untraceable recommendations that do not cite prior findings are invalid."
        ),
    }

    # Directive labels per verdict type — these accumulate in the scenario text
    directive_labels = {
        "RERUN_TIGHTEN": "COMMIT",
        "RERUN_DIVERSIFY": "DIVERSIFY",
        "RERUN_DEEPEN": "TRACE",
    }

    directive_bodies = {
        "RERUN_TIGHTEN": (
            f"Name exactly one candidate to cut. Non-committal recommendations are rejected."
        ),
        "RERUN_DIVERSIFY": (
            f"Surface and argue the non-obvious case. Consensus is insufficient."
        ),
        "RERUN_DEEPEN": (
            f"Your recommendation must cite specific named risks and constraints from prior phases. "
            f"General reasoning without citation is rejected."
        ),
    }

    if verdict in constraint_additions:
        # Add constraint if not already present (accumulate, never replace)
        new_constraint = constraint_additions[verdict]
        if new_constraint not in mutated["constraints"]:
            mutated["constraints"].append(new_constraint)

        # Add directive to scenario text — tag by label so each type appears once
        # but all active types coexist
        label = directive_labels[verdict]
        body = directive_bodies[verdict]
        directive_line = f"[AGENT {label} PASS {pass_number}]: {body}"

        # Replace this label's prior entry if it exists, otherwise append
        label_pattern = rf'\[AGENT {label} PASS \d+\]:.*'
        if re.search(label_pattern, mutated["scenario"]):
            mutated["scenario"] = re.sub(
                label_pattern, directive_line,
                mutated["scenario"]
            )
        else:
            mutated["scenario"] += f"\n\n{directive_line}"

    return mutated


# ---------------------------------------------------------------------------
# Agent Log
# ---------------------------------------------------------------------------

class AgentLog:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"XYBERNETEX AUTONOMOUS AGENT LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")

    def log_pass(
        self,
        pass_number: int,
        metrics: dict,
        reasoning: str,
        llm_verdict: str,
        final_verdict: str,
        action: str,
        conflict_note: str = ""
    ):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"PASS {pass_number}  —  {datetime.now().strftime('%H:%M:%S')}\n")
            f.write(f"{'─'*70}\n\n")

            f.write(f"METRICS:\n")
            for k, v in metrics.items():
                f.write(f"  {k:<25}: {v}\n")

            f.write(f"\nAGENT REASONING:\n")
            f.write(f"{'─'*40}\n")
            f.write(reasoning)
            f.write(f"\n{'─'*40}\n\n")

            if llm_verdict != final_verdict:
                f.write(f"LLM verdict      : {llm_verdict} (overridden — see conflict note)\n")
                f.write(f"Conflict note    : {conflict_note}\n")
            f.write(f"FINAL VERDICT    : {final_verdict}\n")
            f.write(f"ACTION           : {action}\n")
            f.write(f"\n{'='*70}\n\n")

    def finalize(self, accepted_pass: int, best_reward: float, total_passes: int):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"AGENT COMPLETE\n")
            f.write(f"{'─'*70}\n")
            f.write(f"Accepted on pass : {accepted_pass}\n")
            f.write(f"Total passes     : {total_passes}\n")
            f.write(f"Best reward      : {best_reward:.4f}\n")
            f.write(f"Finished         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ---------------------------------------------------------------------------
# Agent Control Loop
# ---------------------------------------------------------------------------

def run_agent(
    framework: dict,
    client: OpenAI,
    max_passes: int = 3,
    n_simulations: int = 25,
    n_expand: int = 2,
    audit_dir: str = "agent_audit"
) -> dict:

    log_path = os.path.join(audit_dir, "AGENT_LOG.txt")
    agent_log = AgentLog(log_path)

    all_pass_results = []
    best_result      = None
    best_reward      = -1.0
    accepted_pass    = -1
    current_fw       = framework
    prior_verdicts   = []

    print(f"\n{'█'*70}")
    print(f"XYBERNETEX AUTONOMOUS AGENT")
    print(f"{'█'*70}")
    print(f"Scenario   : {framework['name']}")
    print(f"Max passes : {max_passes}  |  Sims/pass: {n_simulations}")
    print(f"Thresholds :")
    for k, v in THRESHOLDS.items():
        print(f"  {k:<30}: {v}")
    print(f"{'█'*70}\n")

    for pass_num in range(1, max_passes + 1):

        print(f"\n{'='*70}")
        print(f"AGENT PASS {pass_num}/{max_passes}  —  {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}\n")

        pass_audit_dir = os.path.join(audit_dir, f"pass_{pass_num:02d}")
        result = mcts_search(
            framework=current_fw,
            client=client,
            n_simulations=n_simulations,
            n_expand=n_expand,
            audit_dir=pass_audit_dir
        )

        # Track best result
        pass_rewards = [s["reward"] for s in result["simulation_trace"]]
        pass_max     = max(pass_rewards) if pass_rewards else 0.0
        if pass_max > best_reward:
            best_reward            = pass_max
            best_result            = result
            best_result["pass_number"] = pass_num

        all_pass_results.append({
            "pass":         pass_num,
            "max_reward":   pass_max,
            "n_sims":       result["n_simulations"],
            "best_sim_idx": result["best_sim_idx"],
        })

        # Extract metrics
        metrics = extract_run_metrics(result, audit_base_dir=pass_audit_dir)

        # ── AGENT REASONING STEP ──────────────────────────────────────────
        print(f"\n{'▓'*70}")
        print(f"AGENT REASONING — Pass {pass_num}")
        print(f"{'▓'*70}\n")

        reasoning, llm_verdict = agent_reason(
            pass_number    = pass_num,
            metrics        = metrics,
            scenario_name  = framework["name"],
            prior_verdicts = prior_verdicts,
            client         = client
        )

        # Print reasoning in full
        print(reasoning)
        print(f"\n{'▓'*70}\n")

        # Deterministic verdict is ground truth — thresholds are explicit contracts.
        # LLM verdict is used only when it AGREES with deterministic, or when the
        # deterministic verdict is ACCEPT but the LLM identifies a genuine escalation.
        # If deterministic says ACCEPT and LLM says RERUN_*, deterministic wins —
        # the LLM has invented a failure mode not supported by the numbers.
        det_verdict = deterministic_verdict(metrics)

        if not llm_verdict or llm_verdict not in ("ACCEPT", "RERUN_TIGHTEN", "RERUN_DIVERSIFY", "RERUN_DEEPEN"):
            # LLM returned unparseable output
            final_verdict = det_verdict
            conflict_note = f"[Agent: LLM returned no valid verdict. Using deterministic: {det_verdict}]"
            print(conflict_note)

        elif det_verdict == "ACCEPT" and llm_verdict != "ACCEPT":
            # LLM wants to rerun but all thresholds passed — LLM invented a failure mode
            final_verdict = det_verdict
            conflict_note = (
                f"[Agent: CONFLICT — deterministic says ACCEPT (all thresholds passed) "
                f"but LLM issued {llm_verdict}. "
                f"Deterministic overrides. LLM hallucinated a failure mode not in the rubric.]"
            )
            print(conflict_note)

        elif det_verdict != "ACCEPT" and llm_verdict == "ACCEPT":
            # LLM wants to accept but thresholds failed — LLM missed a real problem
            final_verdict = det_verdict
            conflict_note = (
                f"[Agent: CONFLICT — deterministic says {det_verdict} (threshold breach) "
                f"but LLM issued ACCEPT. "
                f"Deterministic overrides. LLM missed a real failure mode.]"
            )
            print(conflict_note)

        elif det_verdict != "ACCEPT" and llm_verdict != "ACCEPT" and det_verdict != llm_verdict:
            # Both say rerun but disagree on type — deterministic is priority-ordered, use it
            final_verdict = det_verdict
            conflict_note = (
                f"[Agent: RERUN TYPE CONFLICT — deterministic says {det_verdict} "
                f"but LLM says {llm_verdict}. "
                f"Deterministic overrides — its priority ordering is explicit and correct.]"
            )
            print(conflict_note)

        else:
            # Both agree — use LLM verdict
            final_verdict = llm_verdict
            conflict_note = f"[Agent: LLM and deterministic agree on {final_verdict}]"
            print(conflict_note)

        # Force accept on final pass
        if pass_num == max_passes and final_verdict != "ACCEPT":
            print(f"[Agent: Max passes reached. Forcing ACCEPT on best result from pass {best_result.get('pass_number', pass_num)}.]")
            final_verdict = "ACCEPT"

        prior_verdicts.append(final_verdict)

        # Build action description
        action_map = {
            "ACCEPT": (
                f"Run quality accepted. Best reward: {best_reward:.4f}. "
                f"Committing result from pass {best_result.get('pass_number', pass_num)}."
            ),
            "RERUN_TIGHTEN": (
                f"Non-committal ratio {metrics.get('ambiguous_ratio', '?')} too high. "
                f"Injecting commitment forcing into scenario. Rerunning pass {pass_num + 1}."
            ),
            "RERUN_DIVERSIFY": (
                f"Spread {metrics.get('spread', '?')} or diversity {metrics.get('outcome_diversity', '?')} insufficient. "
                f"Injecting adversarial constraint. Rerunning pass {pass_num + 1}."
            ),
            "RERUN_DEEPEN": (
                f"Mean C5 {metrics.get('mean_c5', '?')} below threshold. "
                f"Injecting traceability constraint. Rerunning pass {pass_num + 1}."
            ),
        }
        action = action_map.get(final_verdict, f"Verdict: {final_verdict}")

        # Log everything
        agent_log.log_pass(
            pass_number    = pass_num,
            metrics        = metrics,
            reasoning      = reasoning,
            llm_verdict    = llm_verdict or "NONE",
            final_verdict  = final_verdict,
            action         = action,
            conflict_note  = conflict_note
        )

        print(f"FINAL VERDICT  : {final_verdict}")
        print(f"ACTION         : {action}\n")

        if final_verdict == "ACCEPT":
            accepted_pass = pass_num
            break

        # Mutate for next pass
        current_fw = mutate_framework(current_fw, final_verdict, pass_num + 1)
        print(f"Framework mutated for pass {pass_num + 1}.")
        print(f"  Constraints now : {len(current_fw['constraints'])}")
        new_constraint = current_fw["constraints"][-1]
        print(f"  New constraint  : {new_constraint[:100]}...")

    agent_log.finalize(accepted_pass, best_reward, pass_num)

    # Write agent result JSON
    agent_output = {
        "scenario":            framework["name"],
        "max_passes":          max_passes,
        "passes_executed":     pass_num,
        "accepted_pass":       accepted_pass,
        "best_reward":         best_reward,
        "all_pass_summaries":  all_pass_results,
        "thresholds_used":     THRESHOLDS,
        "prior_verdicts":      prior_verdicts,
        "agent_log_path":      log_path,
        "best_result":         best_result,
    }

    output_path = os.path.join(audit_dir, "agent_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(agent_output, f, indent=2)

    print(f"\n{'█'*70}")
    print(f"AGENT COMPLETE")
    print(f"{'█'*70}")
    print(f"Accepted pass  : {accepted_pass}")
    print(f"Total passes   : {pass_num}")
    print(f"Best reward    : {best_reward:.4f}")
    print(f"Agent log      : {log_path}")
    print(f"Agent result   : {output_path}")
    print(f"{'█'*70}\n")

    return agent_output


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def main():
    parser = argparse.ArgumentParser(description="Xybernetex Autonomous Agent")
    parser.add_argument("--scenario",   default="scenario.txt")
    parser.add_argument("--max_passes", type=int, default=3)
    parser.add_argument("--sims",       type=int, default=25)
    parser.add_argument("--expand",     type=int, default=2)
    parser.add_argument("--audit_dir",  default="agent_audit")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Print thresholds and exit")
    args = parser.parse_args()

    if args.dry_run:
        print("Current thresholds:")
        for k, v in THRESHOLDS.items():
            print(f"  {k}: {v}")
        sys.exit(0)

    client   = get_client()
    framework = parse_scenario_file(args.scenario)
    print_loaded_scenario(framework)

    cost_per_pass = round(args.sims * 7 * 500 / 1000 * 0.002, 2)
    print(f"Est. cost/pass : ~${cost_per_pass}")
    print(f"Est. max cost  : ~${round(cost_per_pass * args.max_passes, 2)}\n")

    start = datetime.now()
    run_agent(
        framework    = framework,
        client       = client,
        max_passes   = args.max_passes,
        n_simulations = args.sims,
        n_expand     = args.expand,
        audit_dir    = args.audit_dir
    )
    print(f"Wall time: {round((datetime.now() - start).total_seconds(), 1)}s")


if __name__ == "__main__":
    main()
