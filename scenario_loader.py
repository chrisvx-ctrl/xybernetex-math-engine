"""
scenario_loader.py
Parses scenario.txt into a framework dict consumed by the MCTS engine.

Supports two modes:

  MODE: selection  — choose between named existing options (pharma, Cedar Falls)
  MODE: generation — figure out what to do and how (lawn care, research, planning)

If MODE is omitted the loader infers it from the scenario description.

Format of scenario.txt:
    SCENARIO_NAME: <name>
    MODE: selection | generation          # optional — inferred if omitted
    SCENARIO_DESCRIPTION: <multi-line>
    DIMENSIONS: <bullet list>
    PHASES: <bullet list>                 # optional — defaults from mode if omitted
    CONSTRAINTS: <bullet list>

Free-form intake:
    Pass a plain English string to parse_freeform() and it returns a
    fully-populated framework dict with mode inferred and phases assigned.
"""

import os
import re
import sys


# ---------------------------------------------------------------------------
# Phase Sets — one per mode
# ---------------------------------------------------------------------------

SELECTION_PHASES = [
    "PROBLEM_FRAMING",
    "STAKEHOLDER_ANALYSIS",
    "CONSTRAINT_MAPPING",
    "RISK_ASSESSMENT",
    "OPTIONS_EVALUATION",
    "DECISION_RECOMMENDATION",
    "EXECUTION_PLAN",
]

GENERATION_PHASES = [
    "GOAL_FRAMING",
    "MARKET_AND_CONTEXT",
    "RESOURCE_AND_CONSTRAINTS",
    "OPTIONS_DISCOVERY",
    "RISK_ASSESSMENT",
    "RECOMMENDED_PATH",
    "EXECUTION_ROADMAP",
]

PHASE_SETS = {
    "selection": SELECTION_PHASES,
    "generation": GENERATION_PHASES,
}


# ---------------------------------------------------------------------------
# Mode Inference
# ---------------------------------------------------------------------------

SELECTION_SIGNALS = [
    r"\bchoose between\b",
    r"\bdecide between\b",
    r"\bwhich (one|option|candidate|choice)\b",
    r"\bcut one of\b",
    r"\bselect (one|a|the)\b",
    r"\bpick (one|between|from)\b",
    r"\bshould (we|i|they) (cut|eliminate|select|choose|go with)\b",
    r"\bopt(ion)? [abc123]\b",
    r"\balternatives? (are|include)\b",
]

GENERATION_SIGNALS = [
    r"\bhow (do|should|can) (i|we|they) (start|build|create|launch|plan|grow|design|run)\b",
    r"\bhelp me (plan|build|start|create|think through|figure out)\b",
    r"\bstarting a\b",
    r"\bbuilding a\b",
    r"\blaunching a\b",
    r"\bresearch (on|into|about)\b",
    r"\bwhat (should|would|could) (i|we) do\b",
    r"\bstrategy for\b",
    r"\bplan (for|to|out)\b",
    r"\bsteps (to|for|needed)\b",
    r"\bfigure out\b",
]


def infer_mode(text: str) -> str:
    """
    Infers selection or generation from scenario description text.
    Counts signal matches for each mode and returns the winner.
    Defaults to generation if ambiguous.
    """
    text_lower = text.lower()

    sel_score = sum(1 for p in SELECTION_SIGNALS if re.search(p, text_lower))
    gen_score = sum(1 for p in GENERATION_SIGNALS if re.search(p, text_lower))

    if sel_score > gen_score:
        return "selection"
    return "generation"


# ---------------------------------------------------------------------------
# Free-form Intake
# ---------------------------------------------------------------------------

def parse_freeform(text: str, name: str = None) -> dict:
    """
    Takes a plain English problem description and returns a fully-populated
    framework dict. Mode is inferred. Phases default from mode.
    Dimensions and constraints are left minimal — the engine will populate
    them during the GOAL_FRAMING or PROBLEM_FRAMING phase.

    This is the intake path for the product UI.
    """
    mode = infer_mode(text)
    phases = PHASE_SETS[mode]

    # Derive a name if not provided
    if not name:
        # Take first sentence or first 60 chars
        first_sentence = re.split(r'[.!?]', text.strip())[0].strip()
        name = first_sentence[:60] if len(first_sentence) > 60 else first_sentence

    # Minimal default dimensions — engine expands these in phase 1
    default_dimensions = {
        "selection": [
            "Risk and Downside",
            "Resource Requirements",
            "Stakeholder Impact",
            "Strategic Fit",
            "Feasibility",
        ],
        "generation": [
            "Market Opportunity",
            "Resource Requirements",
            "Execution Feasibility",
            "Risk and Downside",
            "Long-term Viability",
        ],
    }

    # Minimal default constraints
    default_constraints = {
        "selection": [
            "The recommendation must name exactly one specific option.",
            "All major stakeholder concerns must be addressed.",
        ],
        "generation": [
            "The recommended path must be actionable with realistic resources.",
            "The execution roadmap must include concrete first steps.",
        ],
    }

    return {
        "name":        name,
        "mode":        mode,
        "scenario":    text.strip(),
        "dimensions":  default_dimensions[mode],
        "phases":      phases,
        "constraints": default_constraints[mode],
    }


# ---------------------------------------------------------------------------
# Scenario File Parser
# ---------------------------------------------------------------------------

def parse_scenario_file(filepath: str = "scenario.txt") -> dict:
    if not os.path.exists(filepath):
        print(f"ERROR: scenario file not found at '{filepath}'")
        print("Create a scenario.txt file in the same directory as run.py.")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split into labeled sections
    sections = {}
    current_key = None
    current_lines = []

    for line in raw.splitlines():
        stripped = line.strip()

        # Detect section headers: UPPER_KEY: value
        if ":" in stripped and stripped.split(":")[0].replace("_", "").isupper():
            if current_key:
                sections[current_key] = "\n".join(current_lines).strip()
            parts = stripped.split(":", 1)
            current_key = parts[0].strip()
            remainder = parts[1].strip()
            current_lines = [remainder] if remainder else []
        else:
            if current_key:
                current_lines.append(line)

    if current_key:
        sections[current_key] = "\n".join(current_lines).strip()

    # Required fields (PHASES now optional — defaults from mode)
    required = ["SCENARIO_NAME", "SCENARIO_DESCRIPTION", "DIMENSIONS", "CONSTRAINTS"]
    for req in required:
        if req not in sections:
            print(f"ERROR: Missing required section '{req}' in scenario.txt")
            sys.exit(1)

    def parse_bullet_list(text: str) -> list:
        items = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("- "):
                items.append(line[2:].strip())
            elif line and not line.startswith("#"):
                items.append(line.strip())
        return [i for i in items if i]

    # Resolve mode
    raw_mode = sections.get("MODE", "").strip().lower()
    if raw_mode in PHASE_SETS:
        mode = raw_mode
    else:
        # Infer from description
        mode = infer_mode(sections["SCENARIO_DESCRIPTION"])
        if raw_mode and raw_mode not in PHASE_SETS:
            print(f"WARNING: Unrecognised MODE '{raw_mode}'. Inferred: {mode}")

    # Resolve phases — explicit list in file wins, otherwise default from mode
    if "PHASES" in sections and sections["PHASES"].strip():
        phases = parse_bullet_list(sections["PHASES"])
    else:
        phases = PHASE_SETS[mode]

    framework = {
        "name":        sections["SCENARIO_NAME"].strip(),
        "mode":        mode,
        "scenario":    sections["SCENARIO_DESCRIPTION"].strip(),
        "dimensions":  parse_bullet_list(sections["DIMENSIONS"]),
        "phases":      phases,
        "constraints": parse_bullet_list(sections["CONSTRAINTS"]),
    }

    # Validate
    for key in ["dimensions", "constraints"]:
        if not framework[key]:
            print(f"ERROR: Section '{key.upper()}' is empty in scenario.txt")
            sys.exit(1)

    if len(framework["phases"]) < 3:
        print(f"WARNING: Only {len(framework['phases'])} phases defined. Recommend at least 5.")

    return framework


# ---------------------------------------------------------------------------
# Print Helper
# ---------------------------------------------------------------------------

def print_loaded_scenario(framework: dict):
    mode_label = framework.get("mode", "unknown").upper()
    print(f"\nLoaded scenario : {framework['name']}")
    print(f"Mode            : {mode_label}")
    print(f"Phases ({len(framework['phases'])}): {' → '.join(framework['phases'])}")
    print(f"Dimensions ({len(framework['dimensions'])}): {', '.join(framework['dimensions'])}")
    print(f"Constraints     : {len(framework['constraints'])} defined\n")
