"""
mcts_engine.py
Event-Sourced Cognitive State Machine — MCTS core engine.

Persona system in this version:
  Four epistemological archetypes replace directive strings.
  Personas are injected at the SYSTEM level, not appended to prompts.
  Each persona has: identity, epistemic style, value hierarchy, blind spots.
  Designed to generalize across any decision domain — enterprise, personal,
  creative, technical. Domain knowledge comes from the LLM. The persona
  shapes how that knowledge gets weighted and what conclusions it drives toward.

  THE OPERATOR   — execution-first, implementation risk above all
  THE QUANT      — everything measurable, probabilistic, data-demanding
  THE CONTRARIAN — default assumption is the obvious answer is wrong
  THE STEWARD    — long-term sustainability, stakeholder preservation

Reward engine (from previous version):
  C1: phase-scoped constraint satisfaction (weight 0.25)
  C2: framework alignment via full logprob sum (weight 0.30)
  C3: rubric-based consistency scoring (weight 0.20)
  C4: user feedback deferred (weight 0.10)
  C5: decision defensibility (weight 0.15)
  Depth bonus: conditional on substantive terminal phases

Requires: openai>=1.0.0
"""

import math
import random
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI


# ---------------------------------------------------------------------------
# Epistemological Personas
# Injected as system-level identity, not prompt appendages.
# Each persona reasons from a different epistemic ground regardless of domain.
# ---------------------------------------------------------------------------

PERSONAS = [
    {
        "name": "THE OPERATOR",
        "system_identity": (
            "You are THE OPERATOR. Your entire worldview is organized around one question: "
            "can this actually be executed by real people under real constraints?\n\n"
            "Your epistemic style:\n"
            "- You believe most decisions fail in implementation, not in conception. "
            "An elegant strategy that can't survive contact with operational reality is worthless.\n"
            "- You evaluate every option through: Who does this? With what resources? "
            "In what timeframe? What breaks first?\n"
            "- You are deeply suspicious of recommendations that ignore complexity, "
            "assume stakeholder cooperation, or require perfect information.\n"
            "- You will choose the suboptimal-on-paper option if it's the one that "
            "can actually be executed reliably.\n"
            "- You ask about dependencies, bottlenecks, failure modes, and recovery paths.\n\n"
            "Your blind spot: You sometimes underweight options that are hard to execute "
            "but transformatively valuable if they succeed.\n\n"
            "Apply this identity completely regardless of the decision domain. "
            "A dog breeding business, a pharmaceutical portfolio, a career change — "
            "your lens is always execution feasibility first."
        )
    },
    {
        "name": "THE QUANT",
        "system_identity": (
            "You are THE QUANT. You reduce every decision to its measurable variables "
            "and refuse to pretend certainty where none exists.\n\n"
            "Your epistemic style:\n"
            "- Every claim needs a number or a probability range attached to it. "
            "'Significant risk' means nothing. '40-60% probability of regulatory rejection "
            "based on Phase II failure rates' means something.\n"
            "- You explicitly flag when uncertainty ranges on key variables are too wide "
            "to differentiate between options. Insufficient data is a valid finding.\n"
            "- You think in expected value, variance, and downside scenarios simultaneously.\n"
            "- You distrust qualitative arguments that could justify any conclusion "
            "depending on how they're framed.\n"
            "- You will say 'the data does not support choosing A over B' even when "
            "everyone in the room wants a confident recommendation.\n\n"
            "Your blind spot: You sometimes underweight factors that are real but "
            "genuinely unmeasurable — morale, trust, reputation, timing.\n\n"
            "Apply this identity completely regardless of the decision domain. "
            "Dog breeding margins, drug efficacy signals, career tradeoffs — "
            "your lens is always: what are the actual numbers and what do they entail?"
        )
    },
    {
        "name": "THE CONTRARIAN",
        "system_identity": (
            "You are THE CONTRARIAN. Your default assumption is that the obvious answer "
            "is wrong — and you are willing to name a specific alternative and defend it.\n\n"
            "Your epistemic style:\n"
            "- Consensus recommendations are optimized for defensibility, not for value. "
            "You actively identify what the consensus is and then argue for the opposite "
            "with a specific, named position. You do not merely cast doubt — you commit.\n"
            "- You look for the option that looks worst on the surface but has asymmetric "
            "upside that conventional analysis ignores. You name that option explicitly.\n"
            "- You stress-test popular assumptions by asking: what would have to be true "
            "for the obvious answer to be wrong? Then you argue that those conditions exist.\n"
            "- You are comfortable recommending the option that is hardest to defend to a "
            "committee precisely because committees optimize for safety, not value.\n"
            "- CRITICAL: You always end your analysis with a specific named recommendation. "
            "Vague contrarianism is worthless. Your job is to identify the non-obvious option "
            "AND commit to it with a concrete, defensible argument for why it is correct.\n\n"
            "Your blind spot: You sometimes reject correct conventional wisdom simply because "
            "it is conventional, not because it is actually wrong.\n\n"
            "Apply this identity completely regardless of the decision domain. "
            "The obvious drug to cut, the obvious dog breed to raise, the obvious career move — "
            "your first instinct is always: what is everyone missing, and what is the specific "
            "alternative I am prepared to name and defend?"
        )
    },
    {
        "name": "THE STEWARD",
        "system_identity": (
            "You are THE STEWARD. You optimize for long-term sustainability and "
            "the preservation of relationships, systems, and trust over short-term returns.\n\n"
            "Your epistemic style:\n"
            "- Every decision creates a future — you ask what kind of future each option "
            "creates, not just what it returns in the next 12-18 months.\n"
            "- You ask who bears the costs of each option and whether those costs are "
            "distributed fairly across stakeholders.\n"
            "- You think about what the entity making this decision looks like in 10 years "
            "under each scenario. Is it still functioning? Is it trusted? Is it sustainable?\n"
            "- You weight reputational and relational capital heavily because they are "
            "slow to build and fast to destroy.\n"
            "- You flag decisions that optimize for short-term metrics at the expense "
            "of long-term capability, trust, or structural health.\n\n"
            "Your blind spot: You sometimes let long-term concerns paralyze decisions "
            "that need to be made quickly to survive short-term.\n\n"
            "Apply this identity completely regardless of the decision domain. "
            "Animal welfare in breeding, scientific integrity in pharma, "
            "community impact in infrastructure — your lens is always: "
            "who does this serve and for how long?"
        )
    },
]


# ---------------------------------------------------------------------------
# Tick Node
# ---------------------------------------------------------------------------

@dataclass
class TickNode:
    phase: str
    content: str
    phase_index: int = 0
    persona_name: str = ""
    parent: Optional["TickNode"] = None
    children: list = field(default_factory=list)

    visit_count: int = 0
    total_reward: float = 0.0

    reward_constraints: float = 0.0
    reward_alignment: float = 0.0
    reward_consistency: float = 0.0
    reward_defensibility: float = 0.0
    reward_hybrid: float = 0.0

    avg_logprob: float = 0.0
    token_count: int = 0

    def avg_reward(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        if self.visit_count == 0:
            return float("inf")
        parent_visits = self.parent.visit_count if self.parent else self.visit_count
        if parent_visits == 0:
            return float("inf")
        return self.avg_reward() + exploration_constant * math.sqrt(
            math.log(parent_visits) / self.visit_count
        )

    def path_to_root(self) -> list:
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))


# ---------------------------------------------------------------------------
# Audit writer
# ---------------------------------------------------------------------------

def write_simulation_audit(
    scenario_name: str,
    sim_idx: int,
    episode: list,
    reward: float,
    reward_components: dict,
    audit_dir: str = "audit"
) -> str:
    safe_name = scenario_name.replace(" ", "_").replace("/", "-")[:60]
    sim_dir = os.path.join(audit_dir, safe_name)
    os.makedirs(sim_dir, exist_ok=True)

    label = "BEST_PATH" if sim_idx == 999 else f"sim_{sim_idx:03d}"
    filepath = os.path.join(sim_dir, f"{label}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        title = "BEST PATH (highest-reward simulation)" if sim_idx == 999 else f"SIMULATION {sim_idx}"
        f.write(f"{title}\n")
        f.write(f"Scenario: {scenario_name}\n")
        f.write(f"{'='*70}\n\n")

        if reward_components:
            f.write(f"REWARD SUMMARY\n")
            f.write(f"  C1 Constraint Satisfaction : {reward_components.get('c1', 0):.4f}  (weight 0.25)\n")
            f.write(f"  C2 Framework Alignment     : {reward_components.get('c2', 0):.4f}  (weight 0.30)"
                    f"  [avg_logprob: {reward_components.get('raw_lp', 0):.4f}, tokens: {reward_components.get('tokens', 0)}]\n")
            f.write(f"  C3 Internal Consistency    : {reward_components.get('c3', 0):.4f}  (weight 0.20)  [rubric-scored]\n")
            f.write(f"  C4 User Feedback           : 0.5000  (weight 0.10)  [deferred]\n")
            f.write(f"  C5 Defensibility           : {reward_components.get('c5', 0):.4f}  (weight 0.15)\n")
            f.write(f"  Depth Bonus                : {reward_components.get('depth_bonus', 0):.4f}  [conditional]\n")
            f.write(f"  Rhybrid (final)            : {reward:.4f}\n")
            f.write(f"\n{'='*70}\n\n")

        personas_used = list(dict.fromkeys(n.persona_name for n in episode if n.persona_name))
        if personas_used:
            f.write(f"PERSONAS IN THIS PATH: {', '.join(personas_used)}\n")
            f.write(f"\n{'='*70}\n\n")

        f.write(f"FULL REASONING PATH — {len(episode)} phases\n\n")

        for i, node in enumerate(episode):
            f.write(f"{'─'*70}\n")
            f.write(f"PHASE {i+1}/{len(episode)}: {node.phase}\n")
            f.write(f"  persona: {node.persona_name or 'ROOT (no persona)'}  "
                    f"|  avg_logprob: {node.avg_logprob:.4f}  |  tokens: {node.token_count}\n")
            f.write(f"{'─'*70}\n\n")
            f.write(node.content)
            f.write(f"\n\n")

    return filepath


# ---------------------------------------------------------------------------
# Reward: Component 1 — Phase-scoped Constraint Satisfaction
# ---------------------------------------------------------------------------

def score_constraints(episode_nodes: list, framework: dict) -> float:
    base = 10.0
    violations = 0.0

    phase_content = {n.phase: n.content.lower() for n in episode_nodes}
    full_text = " ".join(n.content for n in episode_nodes).lower()

    constraint_phase_map = {
        0: "CONSTRAINT_MAPPING",
        1: "DECISION_RECOMMENDATION",
        2: "STAKEHOLDER_ANALYSIS",
        3: "DECISION_RECOMMENDATION",
        4: "EXECUTION_PLAN",
    }

    for i, constraint in enumerate(framework["constraints"]):
        key_words = [w for w in constraint.lower().split() if len(w) > 5]
        if not key_words:
            continue
        expected_phase = constraint_phase_map.get(i)
        if expected_phase and expected_phase in phase_content:
            search_text = phase_content[expected_phase]
        else:
            search_text = full_text
        if not any(kw in search_text for kw in key_words):
            violations += 1.2

    analysis_text = " ".join(
        phase_content.get(p, "")
        for p in ["STAKEHOLDER_ANALYSIS", "CONSTRAINT_MAPPING", "RISK_ASSESSMENT", "OPTIONS_EVALUATION"]
    )
    dims_covered = sum(
        1 for dim in framework["dimensions"]
        if any(word.lower() in analysis_text for word in dim.split() if len(word) > 4)
    )
    missing_dims = len(framework["dimensions"]) - dims_covered
    violations += 0.9 * missing_dims

    rec_content = phase_content.get("DECISION_RECOMMENDATION", "")
    if len(rec_content.split()) < 80:
        violations += 2.5

    return round(max(0.0, base - violations) / base, 4)


# ---------------------------------------------------------------------------
# Reward: Component 2 — Framework Alignment via full logprob sum
# ---------------------------------------------------------------------------

def score_alignment_full_logprobs(
    node: TickNode,
    framework: dict,
    client: OpenAI
) -> tuple[float, float, int]:
    scoring_prompt = (
        f"Framework: {framework['name']}\n"
        f"Dimensions: {', '.join(framework['dimensions'])}\n"
        f"Scenario context: {framework['scenario'][:400]}\n\n"
        f"Phase being evaluated: {node.phase}\n"
        f"Reasoning:\n{node.content[:700]}\n\n"
        f"Evaluate whether this reasoning covers the framework dimensions and "
        f"advances the decision analysis. Provide a concise analytical assessment."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strategic decision analyst. Evaluate the reasoning against the framework."
                },
                {"role": "user", "content": scoring_prompt}
            ],
            max_tokens=120,
            temperature=0.0,
            logprobs=True,
            top_logprobs=1
        )

        lp_data = response.choices[0].logprobs
        if lp_data and lp_data.content:
            token_logprobs = [t.logprob for t in lp_data.content]
            avg_lp = sum(token_logprobs) / len(token_logprobs)
            token_count = len(token_logprobs)
        else:
            avg_lp = -2.0
            token_count = 0

    except Exception as e:
        print(f"      [C2 error: {e}]")
        avg_lp = -2.0
        token_count = 0

    clamped = max(-5.0, min(0.0, avg_lp))
    normalized = (clamped + 5.0) / 5.0
    return round(normalized, 4), round(avg_lp, 4), token_count


# ---------------------------------------------------------------------------
# Reward: Component 3 — Rubric-based Consistency Scoring
# ---------------------------------------------------------------------------

def score_consistency_rubric(episode_nodes: list, framework: dict, client: OpenAI) -> float:
    phase_summaries = "\n\n".join(
        f"[{n.phase}]:\n{n.content[:300]}..."
        for n in episode_nodes
    )

    rubric_prompt = (
        f"You are auditing a multi-phase strategic analysis for internal consistency.\n\n"
        f"ANALYSIS PHASES:\n{phase_summaries}\n\n"
        f"Apply this scoring rubric. Start at 1.00 and subtract deductions:\n\n"
        f"  -0.20  A stakeholder position established in STAKEHOLDER_ANALYSIS is directly "
        f"contradicted or ignored in DECISION_RECOMMENDATION\n"
        f"  -0.15  The DECISION_RECOMMENDATION ignores a constraint explicitly named "
        f"in CONSTRAINT_MAPPING\n"
        f"  -0.15  Risks identified in RISK_ASSESSMENT are not addressed in EXECUTION_PLAN\n"
        f"  -0.10  Later phases restate earlier phases verbatim instead of advancing reasoning\n"
        f"  -0.10  The EXECUTION_PLAN does not reference the specific decision made in "
        f"DECISION_RECOMMENDATION\n"
        f"  -0.10  OPTIONS_EVALUATION evaluates options not present in the scenario\n\n"
        f"Apply each deduction independently. Multiple deductions can apply.\n"
        f"Respond with ONLY a single float between 0.0 and 1.0. No explanation."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a precise auditor. Return only a float between 0.0 and 1.0."},
                {"role": "user", "content": rubric_prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )

        raw = response.choices[0].message.content.strip()
        matches = re.findall(r"\d+\.?\d*", raw)
        if matches:
            score = float(matches[0])
            if score > 1.0:
                score = score / 10.0
            return round(min(1.0, max(0.0, score)), 4)
        return 0.70

    except Exception as e:
        print(f"      [C3 rubric error: {e}]")
        return 0.70


# ---------------------------------------------------------------------------
# Reward: Component 5 — Decision Defensibility
# ---------------------------------------------------------------------------

def score_defensibility(episode_nodes: list, framework: dict, client: OpenAI) -> float:
    phase_content = {n.phase: n.content for n in episode_nodes}

    constraint_text = phase_content.get("CONSTRAINT_MAPPING", "")[:400]
    risk_text = phase_content.get("RISK_ASSESSMENT", "")[:400]
    options_text = phase_content.get("OPTIONS_EVALUATION", "")[:400]
    rec_text = phase_content.get("DECISION_RECOMMENDATION", "")[:600]

    if not rec_text or len(rec_text.split()) < 30:
        return 0.20

    defensibility_prompt = (
        f"Framework: {framework['name']}\n\n"
        f"CONSTRAINT MAPPING (excerpt):\n{constraint_text}\n\n"
        f"RISK ASSESSMENT (excerpt):\n{risk_text}\n\n"
        f"OPTIONS EVALUATION (excerpt):\n{options_text}\n\n"
        f"DECISION RECOMMENDATION:\n{rec_text}\n\n"
        f"Is the DECISION RECOMMENDATION logically entailed by the prior analysis?\n"
        f"Score 1.0 if fully traceable. Score 0.5 if partially traceable. "
        f"Score 0.2 if it contradicts or ignores prior analysis.\n"
        f"Respond with ONLY a single float. No explanation."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a precise evaluator of logical entailment. Return only a float."},
                {"role": "user", "content": defensibility_prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )

        raw = response.choices[0].message.content.strip()
        matches = re.findall(r"\d+\.?\d*", raw)
        if matches:
            score = float(matches[0])
            if score > 1.0:
                score = score / 10.0
            return round(min(1.0, max(0.0, score)), 4)
        return 0.50

    except Exception as e:
        print(f"      [C5 error: {e}]")
        return 0.50


# ---------------------------------------------------------------------------
# Reward: Conditional Depth Bonus
# ---------------------------------------------------------------------------

def compute_depth_bonus(episode_nodes: list, total_phases: int) -> float:
    phase_content = {n.phase: n.content for n in episode_nodes}
    rec = phase_content.get("DECISION_RECOMMENDATION", "")
    plan = phase_content.get("EXECUTION_PLAN", "")

    if len(rec.split()) < 100:
        return 0.0

    depth_ratio = len(episode_nodes) / total_phases
    base_bonus = 0.15 * depth_ratio

    if len(plan.split()) < 50:
        base_bonus *= 0.5

    return round(base_bonus, 4)


# ---------------------------------------------------------------------------
# Hybrid Reward
# ---------------------------------------------------------------------------

def compute_hybrid_reward(
    episode_nodes: list,
    framework: dict,
    client: OpenAI,
    verbose: bool = True
) -> tuple[float, dict]:
    total_phases = len(framework["phases"])

    c1 = score_constraints(episode_nodes, framework)
    c2, raw_lp, tokens = score_alignment_full_logprobs(episode_nodes[-1], framework, client)
    c3 = score_consistency_rubric(episode_nodes, framework, client)
    c4 = 0.5
    c5 = score_defensibility(episode_nodes, framework, client)
    db = compute_depth_bonus(episode_nodes, total_phases)

    base = 0.25 * c1 + 0.30 * c2 + 0.20 * c3 + 0.10 * c4 + 0.15 * c5
    final = round(base * (1.0 + db), 4)

    components = {
        "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5,
        "raw_lp": raw_lp, "tokens": tokens,
        "depth_bonus": db, "base": round(base, 4)
    }

    leaf = episode_nodes[-1]
    leaf.reward_constraints = c1
    leaf.reward_alignment = c2
    leaf.reward_consistency = c3
    leaf.reward_defensibility = c5
    leaf.reward_hybrid = final

    if verbose:
        print(
            f"      REWARD  C1:{c1:.3f}  "
            f"C2:{c2:.3f}[lp:{raw_lp:.3f}]  "
            f"C3:{c3:.3f}  "
            f"C5:{c5:.3f}  "
            f"db:{db:.3f}  "
            f"→ Rhybrid:{final:.4f}"
        )

    return final, components


# ---------------------------------------------------------------------------
# LLM Generation — Persona injected at system level
# ---------------------------------------------------------------------------

def generate_tick_content(
    phase: str,
    framework: dict,
    prior_nodes: list,
    client: OpenAI,
    temperature: float = 0.9,
    max_tokens: int = 500,
    persona: dict = None
) -> tuple[str, float, int]:
    prior_context = ""
    if prior_nodes:
        prior_context = "\n\n".join(
            f"[{n.phase}]\n{n.content}" for n in prior_nodes[-3:]
        )

    # Persona is injected as the primary system identity
    # Framework analyst role is secondary — persona shapes how it's applied
    if persona:
        system_prompt = (
            f"{persona['system_identity']}\n\n"
            f"─────────────────────────────────────────\n"
            f"You are analyzing the following decision framework:\n"
            f"Framework: {framework['name']}\n"
            f"Dimensions to address: {', '.join(framework['dimensions'])}\n\n"
            f"Apply your identity fully. Reason as {persona['name']} would reason. "
            f"Your epistemic style and value hierarchy take precedence over generic "
            f"analytical conventions. Be specific. Make commitments. "
            f"Do not restate prior phases — advance the reasoning."
        )
    else:
        # Root node — neutral analyst, no persona
        system_prompt = (
            f"You are a senior strategic analyst conducting rigorous decision analysis.\n"
            f"Framework: {framework['name']}\n"
            f"Dimensions to address: {', '.join(framework['dimensions'])}\n\n"
            f"Be specific. Address each dimension. Make analytical commitments."
        )

    user_prompt = (
        f"SCENARIO:\n{framework['scenario']}\n\n"
        + (f"PRIOR ANALYSIS:\n{prior_context}\n\n" if prior_context else "")
        + f"Complete the {phase} phase. Cover all relevant framework dimensions. "
        f"Advance the reasoning — do not restate what has been said."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=1
        )

        content = response.choices[0].message.content or ""
        lp_data = response.choices[0].logprobs

        if lp_data and lp_data.content:
            lps = [t.logprob for t in lp_data.content]
            avg_lp = sum(lps) / len(lps)
            token_count = len(lps)
        else:
            avg_lp = -2.0
            token_count = 0

        return content.strip(), round(avg_lp, 4), token_count

    except Exception as e:
        print(f"      [generation error: {e}]")
        return f"[Generation failed: {e}]", -5.0, 0


# ---------------------------------------------------------------------------
# MCTS Operators
# ---------------------------------------------------------------------------

def select(node: TickNode) -> TickNode:
    while node.children:
        node = max(node.children, key=lambda c: c.ucb1())
    return node


def expand(
    node: TickNode,
    framework: dict,
    client: OpenAI,
    n_children: int = 2
) -> list:
    """
    Each child gets a distinct persona. Children at the same node
    always receive different personas to maximize branch divergence.
    Persona assignment rotates by phase_index to ensure all four
    personas get coverage across the tree over many simulations.
    """
    phases = framework["phases"]
    if node.phase not in phases:
        return []

    next_idx = phases.index(node.phase) + 1
    if next_idx >= len(phases):
        return []

    next_phase = phases[next_idx]
    prior_nodes = node.path_to_root()
    children = []

    for i in range(n_children):
        # Offset ensures adjacent children always get different personas
        persona_idx = (node.phase_index + i) % len(PERSONAS)
        persona = PERSONAS[persona_idx]

        content, avg_lp, tokens = generate_tick_content(
            next_phase, framework, prior_nodes, client,
            temperature=0.9, max_tokens=500,
            persona=persona
        )
        child = TickNode(
            phase=next_phase,
            content=content,
            phase_index=next_idx,
            persona_name=persona["name"],
            parent=node
        )
        child.avg_logprob = avg_lp
        child.token_count = tokens
        node.children.append(child)
        children.append(child)

    return children


def simulate(node: TickNode, framework: dict, client: OpenAI) -> list:
    """
    Simulation inherits the persona of the starting node through all
    remaining phases. This maintains epistemic consistency within a branch —
    a Contrarian branch reasons as the Contrarian from STAKEHOLDER_ANALYSIS
    all the way through EXECUTION_PLAN.
    """
    phases = framework["phases"]
    if node.phase not in phases:
        return node.path_to_root()

    current_idx = phases.index(node.phase)
    episode = node.path_to_root()
    current_node = node

    # Inherit persona from the node being simulated from
    inherited_persona = None
    if node.persona_name:
        inherited_persona = next(
            (p for p in PERSONAS if p["name"] == node.persona_name), None
        )

    for phase in phases[current_idx + 1:]:
        content, avg_lp, tokens = generate_tick_content(
            phase, framework, episode, client,
            temperature=0.7, max_tokens=450,
            persona=inherited_persona
        )
        next_idx = phases.index(phase)
        sim_node = TickNode(
            phase=phase, content=content,
            phase_index=next_idx,
            persona_name=node.persona_name,
            parent=current_node
        )
        sim_node.avg_logprob = avg_lp
        sim_node.token_count = tokens
        episode.append(sim_node)
        current_node = sim_node

    return episode


def backpropagate(node: TickNode, reward: float):
    current = node
    while current is not None:
        current.visit_count += 1
        current.total_reward += reward
        current = current.parent


def extract_best_path_from_trace(simulation_trace: list) -> tuple[list, int]:
    if not simulation_trace:
        return [], -1
    best = max(simulation_trace, key=lambda s: s["reward"])
    return best["episode"], best["sim_idx"]


# ---------------------------------------------------------------------------
# Full MCTS Search
# ---------------------------------------------------------------------------

def mcts_search(
    framework: dict,
    client: OpenAI,
    n_simulations: int = 25,
    n_expand: int = 2,
    audit_dir: str = "audit"
) -> dict:
    phases = framework["phases"]

    print(f"\n{'='*70}")
    print(f"SCENARIO: {framework['name']}")
    print(f"{'='*70}")
    print(f"Phases ({len(phases)}): {' → '.join(phases)}")
    print(f"Simulations: {n_simulations}  |  Expansions/node: {n_expand}")
    print(f"Personas: {', '.join(p['name'] for p in PERSONAS)}")
    print(f"Reward: C1:0.25  C2:0.30  C3:0.20  C4:0.10  C5:0.15  +conditional_depth\n")

    # Root — neutral, no persona
    root_content, root_lp, root_tokens = generate_tick_content(
        phases[0], framework, [], client,
        temperature=0.7, max_tokens=500, persona=None
    )
    root = TickNode(phase=phases[0], content=root_content, phase_index=0)
    root.avg_logprob = root_lp
    root.token_count = root_tokens
    root.visit_count = 1

    print(f"[ROOT] {phases[0]}  persona:NEUTRAL  logprob:{root_lp:.4f}  tokens:{root_tokens}")
    print(f"{root_content[:400]}...\n")

    write_simulation_audit(
        framework["name"], 0, [root], 0.0,
        {"c1":0,"c2":0,"c3":0,"c4":0,"c5":0,"raw_lp":root_lp,
         "tokens":root_tokens,"depth_bonus":0,"base":0},
        audit_dir=audit_dir
    )

    simulation_trace = []

    for sim_idx in range(1, n_simulations + 1):
        print(f"\n── Sim {sim_idx:02d}/{n_simulations} " + "─"*48)

        selected = select(root)
        print(
            f"  SELECT   {selected.phase}"
            f"  persona:{selected.persona_name or 'NEUTRAL'}"
            f"  visits:{selected.visit_count}"
            f"  avg_r:{selected.avg_reward():.4f}"
            f"  ucb1:{selected.ucb1():.4f}"
        )

        if selected.visit_count > 0 and not selected.children:
            children = expand(selected, framework, client, n_children=n_expand)
            if children:
                selected = random.choice(children)
                print(f"  EXPAND   → {selected.phase}  ({len(children)} children)")
                for c in children:
                    print(f"             persona: {c.persona_name}")
            else:
                print(f"  EXPAND   → terminal")

        episode = simulate(selected, framework, client)
        phase_seq = " → ".join(n.phase for n in episode)
        persona_seq = selected.persona_name or "NEUTRAL"
        print(f"  SIMULATE {len(episode)} phases  persona:{persona_seq}")
        print(f"           {phase_seq}")

        reward, components = compute_hybrid_reward(episode, framework, client, verbose=True)
        backpropagate(selected, reward)

        audit_path = write_simulation_audit(
            framework["name"], sim_idx, episode, reward, components, audit_dir=audit_dir
        )
        print(f"  AUDIT    → {audit_path}")

        simulation_trace.append({
            "sim_idx": sim_idx,
            "selected_phase": selected.phase,
            "persona": persona_seq,
            "phases_completed": len(episode),
            "phase_sequence": phase_seq,
            "reward": reward,
            "components": components,
            "audit_file": audit_path,
            "episode": episode
        })

        time.sleep(0.3)

    best_episode, best_sim_idx = extract_best_path_from_trace(simulation_trace)

    print(f"\n{'='*70}")
    print(f"BEST PATH — Simulation {best_sim_idx}  ({len(best_episode)} phases)")
    print(f"{'='*70}")
    for node in best_episode:
        print(f"\n[{node.phase}]  persona:{node.persona_name or 'NEUTRAL'}  lp:{node.avg_logprob:.4f}")
        print(f"  {node.content[:600]}{'...' if len(node.content) > 600 else ''}")

    best_reward = max(s["reward"] for s in simulation_trace) if simulation_trace else 0.0
    best_components = next(
        s["components"] for s in simulation_trace if s["sim_idx"] == best_sim_idx
    )
    best_audit = write_simulation_audit(
        framework["name"], 999, best_episode,
        best_reward, best_components, audit_dir=audit_dir
    )
    print(f"\n[AUDIT] Best path → {best_audit}")

    # Persona distribution summary
    persona_counts = {}
    for s in simulation_trace:
        p = s["persona"]
        persona_counts[p] = persona_counts.get(p, 0) + 1

    print(f"\nPersona distribution across simulations:")
    for p, count in sorted(persona_counts.items(), key=lambda x: -x[1]):
        avg_r = sum(s["reward"] for s in simulation_trace if s["persona"] == p) / count
        print(f"  {p:<20} {count} sims  avg_reward:{avg_r:.4f}")

    serializable_trace = [
        {k: v for k, v in s.items() if k != "episode"}
        for s in simulation_trace
    ]

    return {
        "scenario": framework["name"],
        "n_simulations": n_simulations,
        "phases": phases,
        "best_sim_idx": best_sim_idx,
        "persona_distribution": persona_counts,
        "best_path": [
            {
                "phase": n.phase,
                "phase_index": n.phase_index,
                "persona": n.persona_name or "NEUTRAL",
                "content": n.content,
                "visit_count": n.visit_count,
                "avg_reward": round(n.avg_reward(), 4),
                "avg_logprob": n.avg_logprob,
                "token_count": n.token_count,
                "reward_constraints": n.reward_constraints,
                "reward_alignment": n.reward_alignment,
                "reward_consistency": n.reward_consistency,
                "reward_defensibility": n.reward_defensibility,
                "reward_hybrid": n.reward_hybrid,
            }
            for n in best_episode
        ],
        "simulation_trace": serializable_trace,
        "root_visits": root.visit_count,
        "root_avg_reward": round(root.avg_reward(), 4),
    }
