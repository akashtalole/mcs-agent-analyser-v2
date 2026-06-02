"""Multi-Agent Conversation Flow Analyzer — dynamic runtime pattern analysis.

Analyzes conversation transcripts to detect agent hand-off patterns:
- Agent transition matrices (who hands off to whom)
- Circular routing detection (user bounced between agents)
- Agent capability overlap (multiple agents handle same query type)
- Agent black holes (receive traffic but never resolve)
- Per-agent resolution rates
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from models import AgentTurnSummary, ConversationTimeline, MultiTurnAgentAnalysis


@dataclass
class AgentTransition:
    """A transition between two agents."""

    from_agent: str
    to_agent: str
    count: int = 0
    avg_turns_before_switch: float = 0.0


@dataclass
class AgentResolution:
    """Resolution metrics for a single agent."""

    agent_name: str
    total_invocations: int = 0
    successful_resolutions: int = 0
    failed_resolutions: int = 0
    redirected_count: int = 0
    resolution_rate: float = 0.0
    avg_turns_to_resolve: float = 0.0


@dataclass
class CircularRoute:
    """A detected circular routing pattern."""

    agents: list[str] = field(default_factory=list)  # cycle path
    occurrences: int = 0
    example_conversation_id: str = ""


@dataclass
class AgentFlowAnalysis:
    """Complete multi-agent conversation flow analysis."""

    conversation_count: int = 0
    total_agent_switches: int = 0
    transitions: list[AgentTransition] = field(default_factory=list)
    agent_resolutions: list[AgentResolution] = field(default_factory=list)
    circular_routes: list[CircularRoute] = field(default_factory=list)
    black_hole_agents: list[str] = field(default_factory=list)
    overlap_pairs: list[dict] = field(default_factory=list)
    sankey_mermaid: str = ""
    warnings: list[str] = field(default_factory=list)


def _extract_agent_sequence(turns: list[AgentTurnSummary]) -> list[str]:
    """Extract the flat sequence of agents invoked across turns."""
    sequence: list[str] = []
    for turn in turns:
        for agent in turn.agents_invoked:
            if not sequence or sequence[-1] != agent:
                sequence.append(agent)
    return sequence


def _detect_cycles(sequence: list[str], min_length: int = 2, max_length: int = 4) -> list[list[str]]:
    """Detect repeated subsequences (cycles) in an agent sequence."""
    cycles: list[list[str]] = []
    seen_cycles: set[tuple[str, ...]] = set()

    for length in range(min_length, min(max_length + 1, len(sequence))):
        for i in range(len(sequence) - length * 2 + 1):
            subseq = tuple(sequence[i : i + length])
            if subseq in seen_cycles:
                continue
            # Check if this subsequence repeats immediately after
            next_subseq = tuple(sequence[i + length : i + length * 2])
            if subseq == next_subseq:
                seen_cycles.add(subseq)
                cycles.append(list(subseq))

    return cycles


def analyze_agent_flows(
    timelines: list[ConversationTimeline],
    multi_turn_analyses: list[MultiTurnAgentAnalysis] | None = None,
) -> AgentFlowAnalysis:
    """Analyze agent routing flows across multiple conversations.

    Args:
        timelines: List of conversation timelines.
        multi_turn_analyses: Optional pre-computed multi-turn analyses.

    Returns:
        AgentFlowAnalysis with transition matrices, cycles, and resolutions.
    """
    result = AgentFlowAnalysis(conversation_count=len(timelines))

    if not timelines:
        return result

    # Transition counting
    transition_counts: dict[tuple[str, str], int] = defaultdict(int)
    agent_invocation_counts: dict[str, int] = defaultdict(int)
    agent_success_counts: dict[str, int] = defaultdict(int)
    agent_fail_counts: dict[str, int] = defaultdict(int)
    agent_redirect_counts: dict[str, int] = defaultdict(int)
    agent_turns_to_resolve: dict[str, list[int]] = defaultdict(list)

    # Per-agent query tracking for overlap detection
    agent_queries: dict[str, list[str]] = defaultdict(list)

    all_cycles: list[tuple[list[str], str]] = []  # (cycle, conversation_id)

    analyses = multi_turn_analyses or []

    for i, analysis in enumerate(analyses):
        if not analysis.turns:
            continue

        conv_id = timelines[i].conversation_id if i < len(timelines) else f"conv-{i}"
        agent_sequence = _extract_agent_sequence(analysis.turns)

        # Count transitions
        for j in range(len(agent_sequence) - 1):
            from_agent = agent_sequence[j]
            to_agent = agent_sequence[j + 1]
            if from_agent != to_agent:
                transition_counts[(from_agent, to_agent)] += 1
                result.total_agent_switches += 1

        # Per-turn analysis
        current_agent = ""
        turns_with_agent = 0

        for turn in analysis.turns:
            for agent in turn.agents_invoked:
                agent_invocation_counts[agent] += 1

                if agent != current_agent:
                    if current_agent and turns_with_agent > 0:
                        agent_turns_to_resolve[current_agent].append(turns_with_agent)
                    current_agent = agent
                    turns_with_agent = 0

                turns_with_agent += 1

                if turn.outcome == "success":
                    agent_success_counts[agent] += 1
                elif turn.outcome == "failed":
                    agent_fail_counts[agent] += 1
                elif turn.outcome == "redirected":
                    agent_redirect_counts[agent] += 1

                # Track queries for overlap detection
                if turn.user_message:
                    agent_queries[agent].append(turn.user_message.lower())

        # Final agent
        if current_agent and turns_with_agent > 0:
            agent_turns_to_resolve[current_agent].append(turns_with_agent)

        # Detect cycles
        cycles = _detect_cycles(agent_sequence)
        for cycle in cycles:
            all_cycles.append((cycle, conv_id))

    # Build transition list
    for (from_a, to_a), count in sorted(transition_counts.items(), key=lambda x: x[1], reverse=True):
        result.transitions.append(AgentTransition(
            from_agent=from_a,
            to_agent=to_a,
            count=count,
        ))

    # Build resolution metrics
    for agent_name in sorted(agent_invocation_counts.keys()):
        total = agent_invocation_counts[agent_name]
        successes = agent_success_counts.get(agent_name, 0)
        fails = agent_fail_counts.get(agent_name, 0)
        redirects = agent_redirect_counts.get(agent_name, 0)
        turns_list = agent_turns_to_resolve.get(agent_name, [])

        result.agent_resolutions.append(AgentResolution(
            agent_name=agent_name,
            total_invocations=total,
            successful_resolutions=successes,
            failed_resolutions=fails,
            redirected_count=redirects,
            resolution_rate=successes / total if total > 0 else 0.0,
            avg_turns_to_resolve=sum(turns_list) / len(turns_list) if turns_list else 0.0,
        ))

    # Detect black holes (agents that never resolve)
    for res in result.agent_resolutions:
        if res.total_invocations >= 3 and res.resolution_rate == 0.0:
            result.black_hole_agents.append(res.agent_name)
            result.warnings.append(
                f"Agent '{res.agent_name}' was invoked {res.total_invocations} times "
                f"but never produced a successful resolution."
            )

    # Deduplicate cycles
    cycle_counts: dict[tuple[str, ...], tuple[int, str]] = {}
    for cycle, conv_id in all_cycles:
        key = tuple(cycle)
        if key in cycle_counts:
            count, _ = cycle_counts[key]
            cycle_counts[key] = (count + 1, conv_id)
        else:
            cycle_counts[key] = (1, conv_id)

    for cycle_tuple, (count, example_id) in sorted(cycle_counts.items(), key=lambda x: x[1][0], reverse=True):
        result.circular_routes.append(CircularRoute(
            agents=list(cycle_tuple),
            occurrences=count,
            example_conversation_id=example_id,
        ))
        if count >= 3:
            route = " -> ".join(cycle_tuple)
            result.warnings.append(
                f"Circular routing detected: {route} (occurred {count} times). "
                f"Users may be bouncing between these agents without resolution."
            )

    # Agent overlap detection (simple token-based)
    agent_token_sets: dict[str, set[str]] = {}
    for agent, queries in agent_queries.items():
        tokens = set()
        for q in queries:
            tokens.update(q.split())
        agent_token_sets[agent] = tokens

    agents = list(agent_token_sets.keys())
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = agents[i], agents[j]
            tokens_a, tokens_b = agent_token_sets[a], agent_token_sets[b]
            if not tokens_a or not tokens_b:
                continue
            shared = tokens_a & tokens_b
            min_len = min(len(tokens_a), len(tokens_b))
            overlap = len(shared) / min_len if min_len > 0 else 0
            if overlap > 0.4:
                result.overlap_pairs.append({
                    "agent_a": a,
                    "agent_b": b,
                    "overlap_pct": round(overlap * 100, 1),
                })

    # Build Sankey diagram (Mermaid)
    if result.transitions:
        result.sankey_mermaid = _build_sankey(result.transitions)

    return result


def _build_sankey(transitions: list[AgentTransition]) -> str:
    """Build a Mermaid sankey diagram from agent transitions."""
    lines: list[str] = []
    lines.append("sankey-beta")
    lines.append("")
    for t in transitions[:15]:  # Limit for readability
        safe_from = t.from_agent.replace('"', "'")[:30]
        safe_to = t.to_agent.replace('"', "'")[:30]
        lines.append(f'"{safe_from}","{safe_to}",{t.count}')
    return "\n".join(lines)


def render_agent_flow_report(analysis: AgentFlowAnalysis) -> str:
    """Render agent flow analysis as markdown."""
    lines: list[str] = []

    lines.append("# Multi-Agent Conversation Flow Report")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Conversations Analyzed | {analysis.conversation_count} |")
    lines.append(f"| Total Agent Switches | {analysis.total_agent_switches} |")
    lines.append(f"| Circular Routes Detected | {len(analysis.circular_routes)} |")
    lines.append(f"| Black Hole Agents | {len(analysis.black_hole_agents)} |")
    lines.append(f"| Overlapping Agent Pairs | {len(analysis.overlap_pairs)} |")
    lines.append("")

    # Warnings
    if analysis.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in analysis.warnings:
            lines.append(f"- \u26a0\ufe0f {w}")
        lines.append("")

    # Agent resolution table
    if analysis.agent_resolutions:
        lines.append("## Agent Resolution Metrics")
        lines.append("")
        lines.append("| Agent | Invocations | Resolution Rate | Avg Turns | Failures | Redirects |")
        lines.append("|-------|-------------|-----------------|-----------|----------|-----------|")
        for ar in sorted(analysis.agent_resolutions, key=lambda x: x.total_invocations, reverse=True):
            lines.append(
                f"| {ar.agent_name} | {ar.total_invocations} | {ar.resolution_rate:.1%} | "
                f"{ar.avg_turns_to_resolve:.1f} | {ar.failed_resolutions} | {ar.redirected_count} |"
            )
        lines.append("")

    # Transition matrix
    if analysis.transitions:
        lines.append("## Agent Transitions")
        lines.append("")
        lines.append("| From Agent | To Agent | Count |")
        lines.append("|------------|----------|-------|")
        for t in analysis.transitions[:20]:
            lines.append(f"| {t.from_agent} | {t.to_agent} | {t.count} |")
        lines.append("")

    # Circular routes
    if analysis.circular_routes:
        lines.append("## Circular Routes")
        lines.append("")
        for cr in analysis.circular_routes:
            route = " \u2192 ".join(cr.agents)
            lines.append(f"- **{route}** \u2014 {cr.occurrences} occurrence(s) (e.g. `{cr.example_conversation_id}`)")
        lines.append("")

    # Sankey diagram
    if analysis.sankey_mermaid:
        lines.append("## Agent Flow Diagram")
        lines.append("")
        lines.append("```mermaid")
        lines.append(analysis.sankey_mermaid)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
