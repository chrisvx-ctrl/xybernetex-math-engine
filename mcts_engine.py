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
    tick_latency_ms: float = 0.0

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
    safe_name = scenario_name
    for ch in r'\\/:*?"<>|':
        safe_name = safe_name.replace(ch, "")
    safe_name = safe_name.replace(" ", "_")[:60]
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
        t0 = time.perf_counter()
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
        tick_ms = round((time.perf_counter() - t0) * 1000, 1)

        content = response.choices[0].message.content or ""
        lp_data = response.choices[0].logprobs

        if lp_data and lp_data.content:
            lps = [t.logprob for t in lp_data.content]
            avg_lp = sum(lps) / len(lps)
            token_count = len(lps)
        else:
            avg_lp = -2.0
            token_count = 0

        return content.strip(), round(avg_lp, 4), token_count, tick_ms

    except Exception as e:
        print(f"      [generation error: {e}]")
        return f"[Generation failed: {e}]", -5.0, 0, 0.0


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

        content, avg_lp, tokens, tick_ms = generate_tick_content(
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
        child.tick_latency_ms = tick_ms
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

    tick_latencies = []
    for phase in phases[current_idx + 1:]:
        content, avg_lp, tokens, tick_ms = generate_tick_content(
            phase, framework, episode, client,
            temperature=0.7, max_tokens=450,
            persona=inherited_persona
        )
        tick_latencies.append(tick_ms)
        next_idx = phases.index(phase)
        sim_node = TickNode(
            phase=phase, content=content,
            phase_index=next_idx,
            persona_name=node.persona_name,
            parent=current_node
        )
        sim_node.avg_logprob = avg_lp
        sim_node.token_count = tokens
        sim_node.tick_latency_ms = tick_ms
        episode.append(sim_node)
        current_node = sim_node

    return episode, tick_latencies


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
# C6: Execution Plan Convergence
# ---------------------------------------------------------------------------

def score_execution_convergence(
    simulation_trace: list,
    framework: dict,
    client: OpenAI,
    sample_size: int = 8
) -> tuple[float, str]:
    plans = [
        s["episode"][-1].content
        for s in simulation_trace
        if s.get("episode") and s["episode"][-1].phase == "EXECUTION_PLAN"
    ]

    if len(plans) < 2:
        return 0.5, "Insufficient execution plans to assess convergence."

    sampled = plans[:sample_size] if len(plans) >= sample_size else plans
    plans_text = "\n\n---\n\n".join(
        f"[Plan {i+1}]:\n{p[:500]}" for i, p in enumerate(sampled)
    )

    prompt = (
        f"Framework: {framework['name']}\n\n"
        f"The following are execution plans from {len(sampled)} independent simulations "
        f"of the same decision scenario. Each simulation used a different reasoning persona.\n\n"
        f"{plans_text}\n\n"
        f"Do these plans converge on a common set of core actions, regardless of how they "
        f"label or frame the recommendation?\n\n"
        f"Score 1.0 if the core actions are essentially the same across plans.\n"
        f"Score 0.5 if there is partial overlap — some shared actions, some divergence.\n"
        f"Score 0.0 if the plans recommend fundamentally different actions.\n\n"
        f"Respond with a JSON object with two fields:\n"
        f"  score: float between 0.0 and 1.0\n"
        f"  summary: one sentence describing what the plans converge on (or why they diverge)\n"
        f"No other text."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a precise analyst. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120,
            temperature=0.0
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        data = __import__("json").loads(raw)
        score = float(data.get("score", 0.5))
        summary = str(data.get("summary", "No summary available."))
        return round(min(1.0, max(0.0, score)), 4), summary

    except Exception as e:
        print(f"      [C6 error: {e}]")
        return 0.5, "Convergence scoring unavailable."


# ---------------------------------------------------------------------------
# Step Decomposition — extract discrete steps from best path, expand each
# ---------------------------------------------------------------------------

def decompose_steps(
    best_episode: list,
    framework: dict,
    client: OpenAI
) -> list[dict]:
    """
    Single LLM call. Takes the full best path and extracts discrete actionable
    steps. Returns a list of dicts: {step_number, title, description, source_phase}.
    """
    phase_summaries = "\n\n".join(
        f"[{n.phase}  |  persona: {n.persona_name or 'NEUTRAL'}]\n{n.content[:600]}"
        for n in best_episode
    )

    prompt = (
        f"Framework: {framework['name']}\n"
        f"Scenario: {framework['scenario'][:400]}\n\n"
        f"The following is a complete multi-phase decision analysis spanning constraint mapping, "
        f"risk assessment, options evaluation, a recommendation, and an execution plan.\n\n"
        f"Your job is to extract discrete, actionable steps that cover the FULL reasoning chain — "
        f"not just implementation steps from the execution plan.\n\n"
        f"Steps must be drawn from ALL phases:\n"
        f"  - Constraint-mitigation steps from CONSTRAINT_MAPPING or RESOURCE_AND_CONSTRAINTS "
        f"(what must be addressed before acting)\n"
        f"  - Risk-mitigation steps from RISK_ASSESSMENT (what breaks first and how to prevent it)\n"
        f"  - Validation steps from OPTIONS_EVALUATION (what must be confirmed before committing)\n"
        f"  - Decision steps from DECISION_RECOMMENDATION or RECOMMENDED_PATH "
        f"(the core commitment and its rationale)\n"
        f"  - Implementation steps from EXECUTION_PLAN or EXECUTION_ROADMAP "
        f"(concrete actions to execute)\n\n"
        f"Rules:\n"
        f"  - Each step must be concrete and specific — not a restatement of analysis\n"
        f"  - Steps must be distinct actions, not variations of the same action\n"
        f"  - Steps must span at least 3 different source phases\n"
        f"  - Do not cluster all steps in the execution plan\n\n"
        f"ANALYSIS:\n{phase_summaries}\n\n"
        f"Return a JSON array. Each element must have:\n"
        f"  step_number: integer starting at 1\n"
        f"  title: short action title (5-10 words)\n"
        f"  description: one sentence describing exactly what this step involves\n"
        f"  source_phase: the phase name this step primarily derives from\n\n"
        f"Return between 7 and 12 steps. Return only valid JSON. No other text."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a precise analyst. Return only valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=900,
            temperature=0.0
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        steps = __import__("json").loads(raw)
        if isinstance(steps, list):
            return steps
        return []

    except Exception as e:
        print(f"      [decompose_steps error: {e}]")
        return []


def expand_step(
    step: dict,
    best_episode: list,
    framework: dict,
    client: OpenAI
) -> str:
    """
    One LLM call per step. Takes a step dict and the full best path context,
    returns a detailed expansion as plain text.
    """
    source_phase = step.get("source_phase", "")
    source_node = next(
        (n for n in best_episode if n.phase == source_phase),
        best_episode[-1]
    )

    prompt = (
        f"Framework: {framework['name']}\n"
        f"Scenario: {framework['scenario'][:300]}\n\n"
        f"You are expanding a single actionable step from a decision analysis.\n\n"
        f"STEP {step['step_number']}: {step['title']}\n"
        f"Description: {step['description']}\n"
        f"Derived from phase: {source_phase}\n\n"
        f"Source phase content:\n{source_node.content[:700]}\n\n"
        f"Expand this step into a detailed, actionable guide. Cover:\n"
        f"  - What exactly needs to happen\n"
        f"  - Who is responsible\n"
        f"  - What inputs or prerequisites are needed\n"
        f"  - What done looks like (definition of completion)\n"
        f"  - Key risks or dependencies to watch\n\n"
        f"Write in plain prose. Be specific. Do not restate the analysis — "
        f"translate it into action."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a precise execution planner. Write in clear, actionable prose."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"      [expand_step error: {e}]")
        return f"[Step expansion failed: {e}]"


def write_step_files(
    best_episode: list,
    framework: dict,
    client: OpenAI,
    best_sim_idx: int,
    best_reward: float,
    audit_dir: str = "audit"
) -> tuple[str, int]:
    """
    Orchestrates step decomposition and per-step expansion.
    Creates audit_dir/steps/<scenario>/ directory.
    Writes one .txt per step plus a steps_index.txt manifest.
    Returns (steps_dir_path, step_count).
    """
    safe_name = framework["name"]
    for ch in r'\/:*?"<>|':
        safe_name = safe_name.replace(ch, "")
    safe_name = safe_name.replace(" ", "_")[:60]

    ts = time.strftime("%Y%m%d_%H%M%S")
    steps_dir = os.path.join(audit_dir, f"steps_{safe_name}_{ts}")
    os.makedirs(steps_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"STEP DECOMPOSITION")
    print(f"{'='*70}")

    # ── Phase 1: extract steps ──────────────────────────────────────────────
    t0 = time.perf_counter()
    print(f"  Extracting steps from best path...")
    steps = decompose_steps(best_episode, framework, client)
    extract_ms = round((time.perf_counter() - t0) * 1000, 1)

    if not steps:
        print(f"  [ERROR] Step extraction returned nothing.")
        return steps_dir, 0

    print(f"  Extracted {len(steps)} steps  [{extract_ms:.0f}ms]")

    # ── Phase 2: expand each step ───────────────────────────────────────────
    expanded = []
    for step in steps:
        t_step = time.perf_counter()
        print(f"  Expanding step {step['step_number']:02d}: {step['title']}", end="", flush=True)
        expansion = expand_step(step, best_episode, framework, client)
        step_ms = round((time.perf_counter() - t_step) * 1000, 1)
        print(f"  [{step_ms:.0f}ms]")
        expanded.append((step, expansion))

    # ── Phase 3: write files ────────────────────────────────────────────────
    index_lines = [
        f"XYBERNETEX — ACTION STEPS INDEX",
        f"{'='*70}",
        f"Scenario   : {framework['name']}",
        f"Timestamp  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source     : Simulation {best_sim_idx:02d}  |  Reward: {best_reward:.4f}",
        f"Steps      : {len(expanded)}",
        f"{'='*70}",
        f"",
    ]

    for step, expansion in expanded:
        n = step["step_number"]
        title = step["title"]
        desc = step["description"]
        phase = step.get("source_phase", "UNKNOWN")

        # per-step file
        filename = f"step_{n:02d}_{title.lower().replace(' ', '_')[:40]}.txt"
        # sanitize filename
        for ch in r'\/:*?"<>|':
            filename = filename.replace(ch, "")
        filepath = os.path.join(steps_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"STEP {n}: {title}\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Scenario    : {framework['name']}\n")
            f.write(f"Source phase: {phase}\n")
            f.write(f"Source sim  : {best_sim_idx:02d}  |  Reward: {best_reward:.4f}\n")
            f.write(f"Generated   : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"{'─'*70}\n")
            f.write(f"DESCRIPTION\n")
            f.write(f"{'─'*70}\n")
            f.write(f"{desc}\n\n")
            f.write(f"{'─'*70}\n")
            f.write(f"EXPANDED GUIDANCE\n")
            f.write(f"{'─'*70}\n")
            f.write(f"{expansion}\n")

        index_lines += [
            f"STEP {n:02d}: {title}",
            f"  Phase  : {phase}",
            f"  File   : {filename}",
            f"  Summary: {desc}",
            f"",
        ]

    # write index
    index_path = os.path.join(steps_dir, "steps_index.txt")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))

    print(f"  Written to → {steps_dir}/")
    print(f"  Files: steps_index.txt + {len(expanded)} step files")

    return steps_dir, len(expanded)


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
    print(f"Reward: C1:0.25  C2:0.30  C3:0.20  C4:0.10  C5:0.15  +depth_bonus\n")

    # Root — neutral, no persona
    run_t0 = time.perf_counter()
    root_content, root_lp, root_tokens, root_tick_ms = generate_tick_content(
        phases[0], framework, [], client,
        temperature=0.7, max_tokens=500, persona=None
    )
    root = TickNode(phase=phases[0], content=root_content, phase_index=0)
    root.avg_logprob = root_lp
    root.token_count = root_tokens
    root.tick_latency_ms = root_tick_ms
    root.visit_count = 1

    print(f"[ROOT] {phases[0]}  persona:NEUTRAL  logprob:{root_lp:.4f}  tokens:{root_tokens}  tick:{root_tick_ms:.0f}ms")
    print(f"{root_content[:400]}...\n")

    write_simulation_audit(
        framework["name"], 0, [root], 0.0,
        {"c1":0,"c2":0,"c3":0,"c4":0,"c5":0,"raw_lp":root_lp,
         "tokens":root_tokens,"depth_bonus":0,"base":0},
        audit_dir=audit_dir
    )

    simulation_trace = []

    for sim_idx in range(1, n_simulations + 1):
        sim_t0 = time.perf_counter()
        print(f"\n── Sim {sim_idx:02d}/{n_simulations} " + "─"*48)

        # SELECT
        t_sel = time.perf_counter()
        selected = select(root)
        sel_ms = round((time.perf_counter() - t_sel) * 1000, 1)
        print(
            f"  SELECT   {selected.phase}"
            f"  persona:{selected.persona_name or 'NEUTRAL'}"
            f"  visits:{selected.visit_count}"
            f"  avg_r:{selected.avg_reward():.4f}"
            f"  ucb1:{selected.ucb1():.4f}"
            f"  [{sel_ms:.1f}ms]"
        )

        # EXPAND
        exp_ms = 0.0
        if selected.visit_count > 0 and not selected.children:
            t_exp = time.perf_counter()
            children = expand(selected, framework, client, n_children=n_expand)
            exp_ms = round((time.perf_counter() - t_exp) * 1000, 1)
            if children:
                selected = random.choice(children)
                print(f"  EXPAND   → {selected.phase}  ({len(children)} children)  [{exp_ms:.0f}ms]")
                for c in children:
                    print(f"             persona:{c.persona_name}  tick:{c.tick_latency_ms:.0f}ms")
            else:
                print(f"  EXPAND   → terminal  [{exp_ms:.0f}ms]")

        # SIMULATE
        episode, tick_latencies = simulate(selected, framework, client)
        llm_ms = round(sum(tick_latencies), 1)
        phase_seq = " → ".join(n.phase for n in episode)
        persona_seq = selected.persona_name or "NEUTRAL"
        avg_tick = round(llm_ms / len(tick_latencies), 1) if tick_latencies else 0
        print(f"  SIMULATE {len(episode)} phases  persona:{persona_seq}")
        print(
            f"  TIMING   llm:{llm_ms:.0f}ms  "
            f"avg_tick:{avg_tick:.0f}ms  "
            f"slowest:{max(tick_latencies):.0f}ms" if tick_latencies else
            f"  TIMING   llm:0ms"
        )

        # SCORE — full C1+C2+C3+C4+C5 always
        t_score = time.perf_counter()
        reward, components = compute_hybrid_reward(episode, framework, client, verbose=True)
        score_ms = round((time.perf_counter() - t_score) * 1000, 1)
        print(f"  SCORE    [{score_ms:.0f}ms]")

        # BACKPROPAGATE
        backpropagate(selected, reward)

        audit_path = write_simulation_audit(
            framework["name"], sim_idx, episode, reward, components, audit_dir=audit_dir
        )
        sim_wall_ms = round((time.perf_counter() - sim_t0) * 1000, 1)
        print(f"  TOTAL    {sim_wall_ms:.0f}ms  (sel:{sel_ms:.0f}  exp:{exp_ms:.0f}  score:{score_ms:.0f}ms)")

        simulation_trace.append({
            "sim_idx": sim_idx,
            "selected_phase": selected.phase,
            "persona": persona_seq,
            "phases_completed": len(episode),
            "phase_sequence": phase_seq,
            "reward": reward,
            "components": components,
            "llm_ms": llm_ms,
            "score_ms": score_ms,
            "sim_wall_ms": sim_wall_ms,
            "tick_latencies": tick_latencies,
            "avg_tick_ms": avg_tick,
            "audit_file": audit_path,
            "episode": episode
        })

    # -----------------------------------------------------------------------
    # C6: Execution Plan Convergence
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"C6: EXECUTION PLAN CONVERGENCE")
    print(f"{'='*70}")
    t_c6 = time.perf_counter()
    c6_score, c6_summary = score_execution_convergence(simulation_trace, framework, client)
    c6_ms = round((time.perf_counter() - t_c6) * 1000, 1)
    convergence_label = (
        "HIGH — action is robust across simulations" if c6_score >= 0.7
        else "MEDIUM — partial convergence, review alternatives" if c6_score >= 0.4
        else "LOW — simulations diverged, treat recommendation with caution"
    )
    print(f"  Score  : {c6_score:.4f}  [{c6_ms:.0f}ms]")
    print(f"  Label  : {convergence_label}")
    print(f"  Summary: {c6_summary}\n")

    run_wall_s = round(time.perf_counter() - run_t0, 2)

    # Best path
    best_episode, best_sim_idx = extract_best_path_from_trace(simulation_trace)
    best_reward = max(s["reward"] for s in simulation_trace) if simulation_trace else 0.0
    best_components = next(s["components"] for s in simulation_trace if s["sim_idx"] == best_sim_idx)
    best_components["c6_score"] = c6_score
    best_components["c6_summary"] = c6_summary

    # Persona distribution
    persona_counts = {}
    for s in simulation_trace:
        persona_counts[s["persona"]] = persona_counts.get(s["persona"], 0) + 1

    # -----------------------------------------------------------------------
    # Run Report
    # -----------------------------------------------------------------------
    all_llm_ms   = [s["llm_ms"] for s in simulation_trace]
    all_score_ms = [s["score_ms"] for s in simulation_trace]
    all_wall_ms  = [s["sim_wall_ms"] for s in simulation_trace]
    all_ticks    = [ms for s in simulation_trace for ms in s["tick_latencies"]]
    total_llm_s  = round(sum(all_llm_ms) / 1000, 2)
    total_score_s = round(sum(all_score_ms) / 1000, 2)
    total_tokens = sum(n.token_count for s in simulation_trace for n in s["episode"])
    rewards      = [s["reward"] for s in simulation_trace]
    slowest_sim  = max(simulation_trace, key=lambda s: s["sim_wall_ms"])
    fastest_sim  = min(simulation_trace, key=lambda s: s["sim_wall_ms"])

    report_lines = [
        f"XYBERNETEX DECISION ENGINE — RUN REPORT",
        f"{'='*70}",
        f"Scenario    : {framework['name']}",
        f"Timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Simulations : {n_simulations}  |  Phases: {len(phases)}  |  Personas: {len(PERSONAS)}",
        f"",
        f"{'='*70}",
        f"TIMING SUMMARY",
        f"{'='*70}",
        f"  Total wall time    : {run_wall_s:.1f}s  ({run_wall_s/60:.1f} min)",
        f"  Pure LLM time      : {total_llm_s:.1f}s  ({round(total_llm_s/run_wall_s*100,1) if run_wall_s else 0}% of wall)",
        f"  C3+C5 scoring time : {total_score_s:.1f}s  ({round(total_score_s/run_wall_s*100,1) if run_wall_s else 0}% of wall)",
        f"  C6 scoring time    : {c6_ms:.0f}ms",
        f"  Root tick time     : {root_tick_ms:.0f}ms",
        f"",
        f"  Avg sim wall time  : {round(sum(all_wall_ms)/len(all_wall_ms),0):.0f}ms",
        f"  Fastest sim        : {fastest_sim['sim_wall_ms']:.0f}ms  (Sim {fastest_sim['sim_idx']:02d} — {fastest_sim['persona']})",
        f"  Slowest sim        : {slowest_sim['sim_wall_ms']:.0f}ms  (Sim {slowest_sim['sim_idx']:02d} — {slowest_sim['persona']})",
        f"",
        f"  Avg tick latency   : {round(sum(all_ticks)/len(all_ticks),1) if all_ticks else 0:.0f}ms",
        f"  Fastest tick       : {min(all_ticks) if all_ticks else 0:.0f}ms",
        f"  Slowest tick       : {max(all_ticks) if all_ticks else 0:.0f}ms",
        f"  Total tokens       : {total_tokens:,}",
        f"",
        f"{'='*70}",
        f"REWARD SUMMARY",
        f"{'='*70}",
        f"  Best reward        : {best_reward:.4f}  (Sim {best_sim_idx:02d})",
        f"  Mean reward        : {round(sum(rewards)/len(rewards),4):.4f}",
        f"  Min reward         : {min(rewards):.4f}",
        f"  Spread             : {round(max(rewards)-min(rewards),4):.4f}",
        f"",
        f"  Best path components:",
        f"    C1 Constraint Sat : {best_components.get('c1',0):.4f}  (weight 0.25)",
        f"    C2 Framework Align: {best_components.get('c2',0):.4f}  (weight 0.30)  [logprob: {best_components.get('raw_lp',0):.4f}]",
        f"    C3 Consistency    : {best_components.get('c3',0):.4f}  (weight 0.20)",
        f"    C4 User Feedback  : 0.5000  (weight 0.10)  [deferred]",
        f"    C5 Defensibility  : {best_components.get('c5',0):.4f}  (weight 0.15)",
        f"    C6 Convergence    : {c6_score:.4f}  [post-hoc]",
        f"    Depth Bonus       : {best_components.get('depth_bonus',0):.4f}",
        f"    Rhybrid (final)   : {best_reward:.4f}",
        f"",
        f"{'='*70}",
        f"C6 CONVERGENCE",
        f"{'='*70}",
        f"  Score   : {c6_score:.4f}",
        f"  Label   : {convergence_label}",
        f"  Summary : {c6_summary}",
        f"",
        f"{'='*70}",
        f"PERSONA PERFORMANCE",
        f"{'='*70}",
    ]

    for p, count in sorted(persona_counts.items(), key=lambda x: -x[1]):
        p_sims = [s for s in simulation_trace if s["persona"] == p]
        p_avg_r = round(sum(s["reward"] for s in p_sims) / count, 4)
        p_avg_ms = round(sum(s["sim_wall_ms"] for s in p_sims) / count, 0)
        report_lines.append(
            f"  {p:<22} {count:2d} sims  avg_reward:{p_avg_r:.4f}  avg_time:{p_avg_ms:.0f}ms"
        )

    report_lines += [
        f"",
        f"{'='*70}",
        f"PER-SIM TIMING",
        f"{'='*70}",
        f"  {'Sim':<6} {'Persona':<22} {'Wall':>7} {'LLM':>7} {'Score':>7} {'AvgTick':>8} {'Reward':>8}",
        f"  {'─'*6} {'─'*22} {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*8}",
    ]

    for s in simulation_trace:
        report_lines.append(
            f"  {s['sim_idx']:02d}     {s['persona']:<22} "
            f"{s['sim_wall_ms']:>6.0f}ms "
            f"{s['llm_ms']:>6.0f}ms "
            f"{s['score_ms']:>6.0f}ms "
            f"{s['avg_tick_ms']:>7.0f}ms "
            f"{s['reward']:>8.4f}"
        )

    report_lines += [
        f"",
        f"{'='*70}",
        f"BEST PATH — Simulation {best_sim_idx}",
        f"{'='*70}",
    ]

    for node in best_episode:
        report_lines += [
            f"",
            f"[{node.phase}]  persona:{node.persona_name or 'NEUTRAL'}  "
            f"logprob:{node.avg_logprob:.4f}  tokens:{node.token_count}  tick:{node.tick_latency_ms:.0f}ms",
            f"{'─'*70}",
            node.content,
        ]

    safe_name = framework["name"]
    for ch in r'\/:*?"<>|':
        safe_name = safe_name.replace(ch, "")
    safe_name = safe_name.replace(" ", "_")[:60]
    os.makedirs(audit_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(audit_dir, f"RUN_REPORT_{safe_name}_{ts}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n{'='*70}")
    print(f"RUN COMPLETE")
    print(f"{'='*70}")
    print(f"  Wall time    : {run_wall_s:.1f}s  ({run_wall_s/60:.1f} min)")
    print(f"  LLM time     : {total_llm_s:.1f}s  |  Scoring: {total_score_s:.1f}s")
    print(f"  Best reward  : {best_reward:.4f}  (Sim {best_sim_idx:02d})")
    print(f"  C6           : {c6_score:.4f}  {convergence_label}")
    print(f"  Report       → {report_path}")

    best_audit = write_simulation_audit(
        framework["name"], 999, best_episode,
        best_reward, best_components, audit_dir=audit_dir
    )
    print(f"  Best path    → {best_audit}")

    # ── Step decomposition ──────────────────────────────────────────────────
    steps_dir, step_count = write_step_files(
        best_episode, framework, client,
        best_sim_idx=best_sim_idx,
        best_reward=best_reward,
        audit_dir=audit_dir
    )

    return {
        "scenario": framework["name"],
        "n_simulations": n_simulations,
        "best_sim_idx": best_sim_idx,
        "best_reward": best_reward,
        "persona_distribution": persona_counts,
        "c6_convergence": {"score": c6_score, "summary": c6_summary, "label": convergence_label},
        "timing": {
            "run_wall_s": run_wall_s,
            "total_llm_s": total_llm_s,
            "total_score_s": total_score_s,
            "avg_sim_wall_ms": round(sum(all_wall_ms)/len(all_wall_ms), 1),
            "avg_tick_ms": round(sum(all_ticks)/len(all_ticks), 1) if all_ticks else 0,
            "total_tokens": total_tokens,
        },
        "report_path": report_path,
        "steps_dir": steps_dir,
        "step_count": step_count,
        "best_path": best_episode,
    }
