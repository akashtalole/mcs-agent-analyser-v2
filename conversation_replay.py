"""Conversation Replay with Annotation — interactive step-through of transcripts.

Provides a stateful replay engine that allows stepping through conversation
events, inspecting context at each point, and adding user annotations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from models import (
    ConversationTimeline,
    EventType,
    ExecutionPhase,
    KnowledgeSearchInfo,
    TimelineEvent,
)


@dataclass
class Annotation:
    """A user annotation on a conversation event."""

    event_index: int
    label: str  # "bug", "bookmark", "question", "note"
    text: str
    author: str = ""


@dataclass
class ReplayStep:
    """A single step in the conversation replay."""

    step_index: int
    event: TimelineEvent
    active_topic: str = ""
    variables_in_scope: dict[str, str] = field(default_factory=dict)
    knowledge_search: KnowledgeSearchInfo | None = None
    current_phase: ExecutionPhase | None = None
    annotations: list[Annotation] = field(default_factory=list)
    elapsed_ms: float = 0.0


@dataclass
class ReplaySession:
    """A complete replay session for a conversation."""

    conversation_id: str
    bot_name: str
    total_steps: int = 0
    current_step: int = 0
    steps: list[ReplayStep] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    bookmarks: list[int] = field(default_factory=list)


def build_replay_session(timeline: ConversationTimeline) -> ReplaySession:
    """Build a replay session from a conversation timeline.

    Converts timeline events into annotatable replay steps with context
    at each point (active topic, variables, knowledge searches).
    """
    session = ReplaySession(
        conversation_id=timeline.conversation_id,
        bot_name=timeline.bot_name,
    )

    active_topic = ""
    variables: dict[str, str] = {}
    phase_map: dict[str, ExecutionPhase] = {p.label: p for p in timeline.phases}
    ks_map: dict[int, KnowledgeSearchInfo] = {ks.position: ks for ks in timeline.knowledge_searches}

    for i, event in enumerate(timeline.events):
        # Track active topic
        if event.topic_name:
            active_topic = event.topic_name

        # Track variable assignments
        if event.event_type == EventType.VARIABLE_ASSIGNMENT and event.summary:
            parts = event.summary.split("=", 1)
            if len(parts) == 2:
                variables[parts[0].strip()] = parts[1].strip()[:100]

        # Find matching knowledge search
        ks = ks_map.get(event.position)

        # Find current phase
        current_phase = phase_map.get(active_topic)

        step = ReplayStep(
            step_index=i,
            event=event,
            active_topic=active_topic,
            variables_in_scope=dict(variables),
            knowledge_search=ks,
            current_phase=current_phase,
            elapsed_ms=0.0,
        )
        session.steps.append(step)

    session.total_steps = len(session.steps)
    return session


def add_annotation(
    session: ReplaySession,
    event_index: int,
    label: str,
    text: str,
    author: str = "",
) -> Annotation:
    """Add an annotation to a specific event in the replay session."""
    annotation = Annotation(
        event_index=event_index,
        label=label,
        text=text,
        author=author,
    )
    session.annotations.append(annotation)

    # Also add to the specific step
    if 0 <= event_index < len(session.steps):
        session.steps[event_index].annotations.append(annotation)

    # Auto-bookmark if it's a bug report
    if label == "bug" and event_index not in session.bookmarks:
        session.bookmarks.append(event_index)

    return annotation


def export_annotations(session: ReplaySession) -> str:
    """Export annotations as JSON for sharing/archiving."""
    data = {
        "conversation_id": session.conversation_id,
        "bot_name": session.bot_name,
        "total_steps": session.total_steps,
        "annotations": [
            {
                "event_index": a.event_index,
                "label": a.label,
                "text": a.text,
                "author": a.author,
            }
            for a in session.annotations
        ],
        "bookmarks": session.bookmarks,
    }
    return json.dumps(data, indent=2)


def render_replay_report(session: ReplaySession) -> str:
    """Render a replay session with annotations as markdown."""
    lines: list[str] = []

    lines.append("# Conversation Replay Report")
    lines.append("")
    lines.append(f"**Conversation:** {session.conversation_id}")
    lines.append(f"**Bot:** {session.bot_name}")
    lines.append(f"**Total Steps:** {session.total_steps}")
    lines.append("")

    # Bookmarks
    if session.bookmarks:
        lines.append("## Bookmarks")
        lines.append("")
        for bk in session.bookmarks:
            if bk < len(session.steps):
                step = session.steps[bk]
                lines.append(f"- Step {bk}: [{step.event.event_type.value}] {step.event.summary[:60]}")
        lines.append("")

    # Annotations summary
    if session.annotations:
        lines.append("## Annotations")
        lines.append("")
        _LABEL_ICONS = {"bug": "\U0001f41b", "bookmark": "\U0001f516", "question": "\u2753", "note": "\U0001f4dd"}
        for a in session.annotations:
            icon = _LABEL_ICONS.get(a.label, "\U0001f4dd")
            step_info = ""
            if a.event_index < len(session.steps):
                step = session.steps[a.event_index]
                step_info = f" [{step.event.event_type.value}]"
            author_info = f" ({a.author})" if a.author else ""
            lines.append(f"- {icon} **Step {a.event_index}**{step_info}{author_info}: {a.text}")
        lines.append("")

    # Step-by-step replay
    lines.append("## Step-by-Step Replay")
    lines.append("")

    for step in session.steps:
        event = step.event
        _TYPE_ICONS = {
            EventType.USER_MESSAGE: "\U0001f464",
            EventType.BOT_MESSAGE: "\U0001f916",
            EventType.KNOWLEDGE_SEARCH: "\U0001f50d",
            EventType.ERROR: "\u274c",
            EventType.STEP_TRIGGERED: "\u25b6\ufe0f",
            EventType.STEP_FINISHED: "\u2705",
            EventType.VARIABLE_ASSIGNMENT: "\U0001f4be",
            EventType.DIALOG_REDIRECT: "\u21aa\ufe0f",
            EventType.ORCHESTRATOR_THINKING: "\U0001f9e0",
        }
        icon = _TYPE_ICONS.get(event.event_type, "\u2022")
        has_annotations = bool(step.annotations)
        marker = " \U0001f4cc" if has_annotations else ""

        lines.append(f"### Step {step.step_index} {icon}{marker}")
        lines.append("")
        lines.append(f"**Type:** {event.event_type.value} | **Topic:** {step.active_topic or 'N/A'}")
        if event.summary:
            lines.append(f"**Summary:** {event.summary[:200]}")
        lines.append("")

        # Show knowledge search if present
        if step.knowledge_search:
            ks = step.knowledge_search
            lines.append(f"> \U0001f50d **Knowledge Search:** \"{ks.search_query or 'N/A'}\"")
            lines.append(f"> Results: {len(ks.search_results)} | Sources: {', '.join(ks.knowledge_sources) or 'N/A'}")
            lines.append("")

        # Show variables if they changed
        if step.variables_in_scope:
            lines.append(f"<details><summary>Variables ({len(step.variables_in_scope)})</summary>")
            lines.append("")
            for var, val in step.variables_in_scope.items():
                lines.append(f"- `{var}` = `{val}`")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        # Show annotations
        if step.annotations:
            for a in step.annotations:
                _LABEL_ICONS = {"bug": "\U0001f41b", "bookmark": "\U0001f516", "question": "\u2753", "note": "\U0001f4dd"}
                a_icon = _LABEL_ICONS.get(a.label, "\U0001f4dd")
                lines.append(f"> {a_icon} **{a.label.capitalize()}:** {a.text}")
            lines.append("")

    return "\n".join(lines)
