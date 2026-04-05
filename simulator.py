"""Conversation Simulator — dry-run routing prediction for Copilot Studio agents.

Given a user utterance and a parsed BotProfile, simulates the routing decision:
which topic would be triggered, which child agent would handle it (for orchestrators),
and what execution path the conversation would likely follow.

Produces a predicted execution path as structured data and as a Mermaid sequence diagram.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from models import BotProfile, ComponentSummary, TopicConnection
from parser import match_query_to_triggers


@dataclass
class RoutingStep:
    """A single step in the predicted routing path."""

    step_number: int
    actor: str  # "User", "Orchestrator", "Topic", "Agent", "System"
    action: str  # "sends message", "routes to", "triggers", "invokes", "responds"
    target: str  # name of target component
    confidence: float = 0.0  # 0.0-1.0
    details: str = ""


@dataclass
class SimulationResult:
    """Result of a conversation routing simulation."""

    query: str
    matched_topic: str = ""
    match_score: float = 0.0
    is_orchestrator: bool = False
    selected_agent: str = ""
    agent_score: float = 0.0
    routing_path: list[RoutingStep] = field(default_factory=list)
    alternative_topics: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mermaid_diagram: str = ""


def _score_agent_match(query: str, agent: ComponentSummary) -> float:
    """Score how well a query matches a child agent's description and instructions."""
    query_tokens = set(query.lower().split())
    if not query_tokens:
        return 0.0

    # Combine agent description and instructions for matching
    agent_text = " ".join(filter(None, [
        agent.model_description or "",
        agent.description or "",
        agent.agent_instructions or "",
    ])).lower()

    if not agent_text:
        return 0.0

    agent_tokens = set(agent_text.split())
    if not agent_tokens:
        return 0.0

    shared = query_tokens & agent_tokens
    # Use query coverage as score
    score = len(shared) / len(query_tokens) if query_tokens else 0.0
    return min(score, 1.0)


def _find_downstream_topics(
    source_schema: str,
    connections: list[TopicConnection],
    components_map: dict[str, ComponentSummary],
    visited: set[str] | None = None,
    depth: int = 0,
    max_depth: int = 5,
) -> list[dict]:
    """Walk the topic connection graph from a source topic."""
    if visited is None:
        visited = set()
    if depth >= max_depth or source_schema in visited:
        return []

    visited.add(source_schema)
    downstream: list[dict] = []

    for conn in connections:
        if conn.source_schema != source_schema:
            continue
        target = conn.target_schema
        target_comp = components_map.get(target)
        target_name = conn.target_display or target
        downstream.append({
            "schema": target,
            "display_name": target_name,
            "condition": conn.condition,
            "depth": depth + 1,
            "kind": target_comp.kind if target_comp else "Unknown",
        })
        # Recurse
        downstream.extend(
            _find_downstream_topics(target, connections, components_map, visited, depth + 1, max_depth)
        )

    return downstream


def _build_mermaid(result: SimulationResult) -> str:
    """Build a Mermaid sequence diagram from the simulation result."""
    lines: list[str] = []
    lines.append("sequenceDiagram")
    lines.append("    participant U as User")

    if result.is_orchestrator:
        lines.append("    participant O as Orchestrator")
        if result.selected_agent:
            # Sanitize for Mermaid
            safe_agent = re.sub(r"[^\w\s]", "", result.selected_agent)[:30]
            lines.append(f"    participant A as {safe_agent}")
    if result.matched_topic:
        safe_topic = re.sub(r"[^\w\s]", "", result.matched_topic)[:30]
        lines.append(f"    participant T as {safe_topic}")
    lines.append("    participant B as Bot")

    lines.append("")

    for step in result.routing_path:
        safe_target = re.sub(r"[^\w\s]", "", step.target)[:30]
        conf = f" ({step.confidence:.0%})" if step.confidence > 0 else ""

        if step.actor == "User" and step.action == "sends message":
            lines.append(f"    U->>+O: {result.query[:50]}" if result.is_orchestrator
                         else f"    U->>+B: {result.query[:50]}")
        elif step.actor == "Orchestrator" and step.action == "routes to":
            lines.append(f"    O->>+A: Route to {safe_target}{conf}")
        elif step.actor == "Orchestrator" and step.action == "triggers":
            lines.append(f"    O->>+T: Trigger {safe_target}{conf}")
        elif step.actor == "System" and step.action == "triggers":
            lines.append(f"    B->>+T: Trigger {safe_target}{conf}")
        elif step.actor == "Topic" and step.action == "invokes":
            lines.append(f"    T->>B: Invoke {safe_target}")
        elif step.actor == "Bot" and step.action == "responds":
            lines.append("    B->>-U: Response")

    # Add note for alternatives
    if result.alternative_topics:
        alt_names = ", ".join(a["display_name"][:20] for a in result.alternative_topics[:3])
        lines.append(f"    Note over U,B: Alternatives: {alt_names}")

    # Add warnings
    for warn in result.warnings[:2]:
        safe_warn = warn[:60].replace("\n", " ")
        lines.append(f"    Note over U,B: \u26a0 {safe_warn}")

    return "\n".join(lines)


def simulate_routing(
    query: str,
    profile: BotProfile,
    *,
    threshold: float = 0.3,
    max_results: int = 5,
) -> SimulationResult:
    """Simulate conversation routing for a user utterance.

    Args:
        query: The user's message/utterance to simulate.
        profile: Parsed bot profile with components and connections.
        threshold: Minimum score threshold for trigger matching.
        max_results: Maximum number of alternative topics to return.

    Returns:
        SimulationResult with predicted routing path and Mermaid diagram.
    """
    result = SimulationResult(query=query, is_orchestrator=profile.is_orchestrator)
    step_num = 0

    # Step 1: User sends message
    step_num += 1
    result.routing_path.append(RoutingStep(
        step_number=step_num,
        actor="User",
        action="sends message",
        target="Bot",
        details=query[:100],
    ))

    # Step 2: Match against trigger queries
    matches = match_query_to_triggers(query, profile.components, threshold=threshold, max_results=max_results)

    if not matches:
        # No trigger match — check for orchestrator routing
        if profile.is_orchestrator:
            result.warnings.append("No direct trigger match found. Orchestrator will use AI routing.")
        else:
            result.warnings.append("No trigger match found. Fallback/unknown intent topic will handle this.")
            # Check for fallback topic
            fallback_topics = [c for c in profile.components
                               if c.trigger_kind in ("OnUnknownIntent", "OnConversationStart")
                               or "fallback" in (c.display_name or "").lower()]
            if fallback_topics:
                result.matched_topic = fallback_topics[0].display_name
                step_num += 1
                result.routing_path.append(RoutingStep(
                    step_number=step_num,
                    actor="System",
                    action="triggers",
                    target=fallback_topics[0].display_name,
                    details="Fallback topic (no trigger match)",
                ))
    else:
        # Primary match
        primary = matches[0]
        result.matched_topic = primary["display_name"]
        result.match_score = primary["score"]
        result.alternative_topics = matches[1:]

        step_num += 1
        actor = "Orchestrator" if profile.is_orchestrator else "System"
        result.routing_path.append(RoutingStep(
            step_number=step_num,
            actor=actor,
            action="triggers",
            target=primary["display_name"],
            confidence=primary["score"],
            details=f"Best phrase match: '{primary['best_phrase']}'",
        ))

    # Step 3: For orchestrators, also score child agents
    if profile.is_orchestrator:
        child_agents = [c for c in profile.components if c.tool_type in ("TaskDialog", "AgentDialog")]
        agent_scores = []
        for agent in child_agents:
            score = _score_agent_match(query, agent)
            if score > 0:
                agent_scores.append({
                    "display_name": agent.display_name,
                    "score": round(score, 4),
                    "type": agent.tool_type,
                })

        agent_scores.sort(key=lambda x: x["score"], reverse=True)

        if agent_scores:
            best_agent = agent_scores[0]
            result.selected_agent = best_agent["display_name"]
            result.agent_score = best_agent["score"]

            step_num += 1
            result.routing_path.append(RoutingStep(
                step_number=step_num,
                actor="Orchestrator",
                action="routes to",
                target=best_agent["display_name"],
                confidence=best_agent["score"],
                details=f"Agent type: {best_agent['type']}",
            ))

            # Warn if scores are close (routing ambiguity)
            if len(agent_scores) >= 2:
                score_diff = agent_scores[0]["score"] - agent_scores[1]["score"]
                if score_diff < 0.15:
                    result.warnings.append(
                        f"Routing ambiguity: '{agent_scores[0]['display_name']}' "
                        f"({agent_scores[0]['score']:.0%}) vs '{agent_scores[1]['display_name']}' "
                        f"({agent_scores[1]['score']:.0%}). Consider improving agent descriptions."
                    )
        elif child_agents:
            result.warnings.append(
                f"No child agent matched the query. {len(child_agents)} agent(s) available "
                f"but none have descriptions matching this utterance."
            )

    # Step 4: Walk downstream connections from matched topic
    if result.matched_topic:
        comp_map = {c.schema_name: c for c in profile.components}
        # Find schema for matched topic
        matched_schema = ""
        for c in profile.components:
            if c.display_name == result.matched_topic:
                matched_schema = c.schema_name
                break

        if matched_schema:
            downstream = _find_downstream_topics(
                matched_schema, profile.topic_connections, comp_map
            )
            for ds in downstream[:3]:  # Limit depth for readability
                step_num += 1
                result.routing_path.append(RoutingStep(
                    step_number=step_num,
                    actor="Topic",
                    action="invokes",
                    target=ds["display_name"],
                    details=f"Condition: {ds['condition']}" if ds["condition"] else "Unconditional",
                ))

    # Final step: Bot responds
    step_num += 1
    result.routing_path.append(RoutingStep(
        step_number=step_num,
        actor="Bot",
        action="responds",
        target="User",
        details="Generated response",
    ))

    # Build Mermaid diagram
    result.mermaid_diagram = _build_mermaid(result)

    return result


def render_simulation_report(result: SimulationResult) -> str:
    """Render a simulation result as a markdown report."""
    lines: list[str] = []

    lines.append("# Routing Simulation Report")
    lines.append("")
    lines.append(f"**User Query:** {result.query}")
    lines.append("")

    # Summary
    lines.append("## Routing Decision")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Matched Topic | {result.matched_topic or 'None'} |")
    lines.append(f"| Match Score | {result.match_score:.0%} |")
    lines.append(f"| Is Orchestrator | {result.is_orchestrator} |")
    if result.selected_agent:
        lines.append(f"| Selected Agent | {result.selected_agent} |")
        lines.append(f"| Agent Score | {result.agent_score:.0%} |")
    lines.append("")

    # Warnings
    if result.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in result.warnings:
            lines.append(f"- \u26a0\ufe0f {w}")
        lines.append("")

    # Routing path
    lines.append("## Predicted Routing Path")
    lines.append("")
    lines.append("| Step | Actor | Action | Target | Confidence |")
    lines.append("|------|-------|--------|--------|------------|")
    for step in result.routing_path:
        conf = f"{step.confidence:.0%}" if step.confidence > 0 else "-"
        lines.append(f"| {step.step_number} | {step.actor} | {step.action} | {step.target} | {conf} |")
    lines.append("")

    # Alternative topics
    if result.alternative_topics:
        lines.append("## Alternative Topic Matches")
        lines.append("")
        lines.append("| Topic | Score | Best Phrase |")
        lines.append("|-------|-------|-------------|")
        for alt in result.alternative_topics:
            lines.append(f"| {alt['display_name']} | {alt['score']:.0%} | {alt.get('best_phrase', '')} |")
        lines.append("")

    # Mermaid diagram
    lines.append("## Sequence Diagram")
    lines.append("")
    lines.append("```mermaid")
    lines.append(result.mermaid_diagram)
    lines.append("```")
    lines.append("")

    return "\n".join(lines)
