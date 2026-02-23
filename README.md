# Xybernetex Decision Engine

Monte Carlo Tree Search over LLM reasoning paths. The math is in charge. The LLM does the work.

---

## What It Does

Takes any problem — a hard decision, a business plan, a research question — runs it through structured multi-phase reasoning across parallel simulations, scores every reasoning path against a formal reward function, and surfaces the most defensible recommendation with a full audit trail.

The LLM does not self-evaluate. The engine does.

---

## Two Modes

**Selection** — you have options that already exist and need to choose between them.
> "We need to cut one of our three drug candidates. Which one?"

**Generation** — you have a goal and need to figure out what to do and how to do it.
> "Help me plan how to start a lawn care company in Salt Lake City."

The engine infers the mode from your input automatically. You can also set it explicitly.

---

## Running

### Plain English (no setup required)
```bash
python agent.py --freeform "Help me figure out how to start a lawn care company"
```
The engine infers the mode, assigns the right phase set, and runs.

### Scenario File
```bash
python agent.py --scenario scenario.txt --sims 25 --passes 3
```

### Options
```
--scenario     Path to scenario file (default: scenario.txt)
--freeform     Plain English problem description — bypasses scenario.txt
--sims         Simulations per pass (default: 25)
--max_passes   Max agent passes before forced accept (default: 3)
--expand       Nodes to expand per simulation (default: 2)
--audit_dir    Output directory for audit files (default: agent_audit)
--dry_run      Print thresholds and exit
```

---

## Scenario File Format

```
SCENARIO_NAME: Should we cut Candidate B from the pipeline?

MODE: selection          # optional — inferred if omitted

SCENARIO_DESCRIPTION:
We are a mid-size pharmaceutical company with three active drug candidates.
Budget constraints require cutting one before Q3. Candidate A has strong
efficacy but high manufacturing cost. Candidate B has moderate efficacy
and a faster regulatory path. Candidate C is early stage with high upside
and high risk.

DIMENSIONS:
- Clinical Risk
- Manufacturing Cost
- Regulatory Timeline
- Strategic Fit
- Resource Requirements

CONSTRAINTS:
- Exactly one candidate must be cut
- The decision must be defensible to the board
- Budget impact must be addressed explicitly
```

`PHASES` is optional. If omitted, the engine uses the default phase set for the inferred mode.

---

## Phase Sets

| Selection | Generation |
|---|---|
| PROBLEM_FRAMING | GOAL_FRAMING |
| STAKEHOLDER_ANALYSIS | MARKET_AND_CONTEXT |
| CONSTRAINT_MAPPING | RESOURCE_AND_CONSTRAINTS |
| RISK_ASSESSMENT | OPTIONS_DISCOVERY |
| OPTIONS_EVALUATION | RISK_ASSESSMENT |
| DECISION_RECOMMENDATION | RECOMMENDED_PATH |
| EXECUTION_PLAN | EXECUTION_ROADMAP |

Selection mode evaluates between options that already exist. Generation mode surfaces the options before evaluating them.

---

## Reward Function

Every reasoning path is scored against five components. The LLM does not decide quality — the math does.

| Component | Weight | What It Measures |
|---|---|---|
| C1 Constraint Satisfaction | 0.25 | Phase-scoped structural completeness |
| C2 Framework Alignment | 0.30 | Log-likelihood of reasoning given framework |
| C3 Internal Consistency | 0.20 | Rubric-scored cross-phase coherence |
| C4 User Feedback | 0.10 | Deferred — returns 0.5 neutral until live |
| C5 Defensibility | 0.15 | Recommendation traceability to prior phases |

---

## Personas

Four epistemological archetypes injected at system level. Each reasons from a different value hierarchy. Prevents consensus collapse.

| Persona | Epistemic Frame | Will Always Ask |
|---|---|---|
| THE OPERATOR | Execution feasibility above all | Can we actually do this? |
| THE QUANT | Everything measurable, probability ranges required | What are the numbers? |
| THE CONTRARIAN | The obvious answer is wrong | What is the non-obvious case? |
| THE STEWARD | Long-term sustainability and stakeholder preservation | Who gets hurt in year three? |

---

## Agent Verdicts

The autonomous agent evaluates each pass against explicit thresholds and decides whether to accept or rerun with a targeted mutation.

| Verdict | Trigger | Mutation Applied |
|---|---|---|
| `ACCEPT` | All thresholds pass | Commit best result |
| `RERUN_TIGHTEN` | Ambiguous ratio >= 0.50 | Inject commitment forcing |
| `RERUN_DEEPEN` | Mean C5 < 0.55 | Inject traceability constraint |
| `RERUN_DIVERSIFY` | Outcome diversity < 2 | Inject diversification directive |

Deterministic verdict logic overrides the LLM when they conflict. The LLM cannot hallucinate a failure mode past a passing threshold.

---

## Output

Every run writes to `agent_audit/`:

```
agent_audit/
  AGENT_LOG.txt          full pass-by-pass decision trace
  agent_result.json      structured results across all passes
  pass_01/               per-simulation audit files, pass 1
  pass_02/               per-simulation audit files, pass 2
```

---

## Files

| File | Purpose |
|---|---|
| `agent.py` | Autonomous multi-pass control loop |
| `mcts_engine.py` | Core MCTS, persona system, reward function |
| `scenario_loader.py` | Parses scenario.txt or free-form input into framework |
| `run.py` | CLI entry point |
| `baseline.py` | Single cold-call comparison (no MCTS) |

---

## Requirements

```
openai>=1.0.0
python>=3.9
```

Set `OPENAI_API_KEY` in environment before running.

---

## Paper

See `Cognitive_State_Machines_v2.docx` for the full theoretical foundation and empirical validation results.