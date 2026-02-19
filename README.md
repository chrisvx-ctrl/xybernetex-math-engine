# Xybernetex Decision Engine

Monte Carlo Tree Search over LLM reasoning paths for high-stakes decisions with no ground truth.

---

## What It Does

Runs structured multi-phase decision analysis across parallel simulations, each conditioned by an epistemological persona. An autonomous agent evaluates run quality against explicit thresholds and reruns with targeted mutations until quality criteria are met.

---

## Files

| File | Purpose |
|---|---|
| `mcts_engine.py` | Core MCTS engine, persona system, reward function (C1-C5) |
| `agent.py` | Autonomous multi-pass control loop with self-evaluation |
| `run.py` | CLI entry point |
| `scenario_loader.py` | Parses `scenario.txt` into framework object |
| `scenario.txt` | Active scenario definition |
| `baseline.py` | Single cold-call comparison (no MCTS) |

---

## Scenario Format

```
SCENARIO_NAME: Your Decision Title

SCENARIO_DESCRIPTION:
Describe the decision context, the options, and the tensions.

DIMENSIONS:
- Dimension One
- Dimension Two

PHASES:
- PROBLEM_FRAMING
- STAKEHOLDER_ANALYSIS
- CONSTRAINT_MAPPING
- RISK_ASSESSMENT
- OPTIONS_EVALUATION
- DECISION_RECOMMENDATION
- EXECUTION_PLAN

CONSTRAINTS:
- Hard requirement one
- Hard requirement two
```

---

## Running

```bash
# Full autonomous agent run (recommended)
python run.py --sims 25 --passes 3

# Single baseline call for comparison
python baseline.py
```

Output lands in `agent_audit/`:
- `AGENT_LOG.txt` — full pass-by-pass decision trace
- `agent_result.json` — structured results
- `pass_NN/` — per-simulation audit files

---

## Reward Function

| Component | Weight | What It Measures |
|---|---|---|
| C1 Constraint Satisfaction | 0.25 | Phase-scoped structural completeness |
| C2 Framework Alignment | 0.30 | Log-likelihood of reasoning given framework |
| C3 Internal Consistency | 0.20 | Rubric-scored cross-phase coherence |
| C4 User Feedback | 0.10 | Deferred — returns 0.5 neutral |
| C5 Defensibility | 0.15 | Recommendation traceability to prior phases |

---

## Personas

Four epistemological archetypes injected at system level. Each reasons from a different value hierarchy regardless of domain.

- **THE OPERATOR** — execution feasibility above all
- **THE QUANT** — everything measurable, probability ranges required
- **THE CONTRARIAN** — obvious answer is wrong; must name a specific alternative
- **THE STEWARD** — long-term sustainability and stakeholder preservation

---

## Agent Verdicts

| Verdict | Trigger | Action |
|---|---|---|
| `ACCEPT` | All thresholds pass | Commit best result across passes |
| `RERUN_TIGHTEN` | Ambiguous ratio ≥ 0.50 | Inject commitment forcing |
| `RERUN_DEEPEN` | Mean C5 < 0.55 | Inject traceability constraint |
| `RERUN_DIVERSIFY` | Outcome diversity < 2 | Inject diversification directive |

Deterministic verdict logic overrides LLM reasoning when they conflict.

---

## Requirements

```
openai>=1.0.0
python>=3.9
```

Set `OPENAI_API_KEY` in environment before running.

---

## Paper

See `Cognitive_State_Machines_v2.docx` for full theoretical foundations and empirical results.
