"""
scenario_loader.py
Parses scenario.txt into a framework dict consumed by the MCTS engine.

Format of scenario.txt:
    SCENARIO_NAME: <name>
    SCENARIO_DESCRIPTION: <multi-line description>
    DIMENSIONS: <bullet list>
    PHASES: <bullet list>
    CONSTRAINTS: <bullet list>

Sections are separated by blank lines. Bullet items start with "- ".
"""

import os
import sys


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

        # Detect section headers: lines ending with ":"  that are ALL_CAPS_KEY: value
        if ":" in stripped and stripped.split(":")[0].replace("_", "").isupper():
            # Save previous section
            if current_key:
                sections[current_key] = "\n".join(current_lines).strip()
            # Start new section
            parts = stripped.split(":", 1)
            current_key = parts[0].strip()
            remainder = parts[1].strip()
            current_lines = [remainder] if remainder else []
        else:
            if current_key:
                current_lines.append(line)

    # Save last section
    if current_key:
        sections[current_key] = "\n".join(current_lines).strip()

    # Extract and validate required fields
    required = ["SCENARIO_NAME", "SCENARIO_DESCRIPTION", "DIMENSIONS", "PHASES", "CONSTRAINTS"]
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

    framework = {
        "name": sections["SCENARIO_NAME"].strip(),
        "scenario": sections["SCENARIO_DESCRIPTION"].strip(),
        "dimensions": parse_bullet_list(sections["DIMENSIONS"]),
        "phases": parse_bullet_list(sections["PHASES"]),
        "constraints": parse_bullet_list(sections["CONSTRAINTS"]),
    }

    # Validate non-empty
    for key in ["dimensions", "phases", "constraints"]:
        if not framework[key]:
            print(f"ERROR: Section '{key.upper()}' is empty in scenario.txt")
            sys.exit(1)

    if len(framework["phases"]) < 3:
        print(f"WARNING: Only {len(framework['phases'])} phases defined. Recommend at least 5.")

    return framework


def print_loaded_scenario(framework: dict):
    print(f"\nLoaded scenario: {framework['name']}")
    print(f"  Phases ({len(framework['phases'])}): {' → '.join(framework['phases'])}")
    print(f"  Dimensions ({len(framework['dimensions'])}): {', '.join(framework['dimensions'])}")
    print(f"  Constraints ({len(framework['constraints'])}): {len(framework['constraints'])} defined\n")
