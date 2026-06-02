"""Power Automate Flow Impact Analysis — parse Cloud Flows from solution ZIPs.

Extracts and analyzes Power Automate Cloud Flow definitions found in
solution ZIPs under the Workflows/ directory.  Maps flows to the bot
topics that invoke them and checks for error handling, timeouts, retry
policies, and connector chains.
"""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from utils import safe_extractall


@dataclass
class FlowAction:
    """A single action within a Power Automate flow."""

    name: str
    action_type: str  # e.g. "OpenApiConnection", "Http", "Compose", "Condition"
    connector: str = ""
    has_error_handling: bool = False
    has_timeout: bool = False
    has_retry: bool = False
    inputs_summary: str = ""


@dataclass
class FlowDefinition:
    """Parsed Power Automate flow definition."""

    flow_name: str
    display_name: str = ""
    trigger_type: str = ""  # "manual", "automated", "scheduled"
    actions: list[FlowAction] = field(default_factory=list)
    total_action_count: int = 0
    connectors_used: list[str] = field(default_factory=list)
    has_error_scope: bool = False
    has_timeout_config: bool = False
    has_retry_policy: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class FlowBotMapping:
    """Mapping between a bot topic and a Power Automate flow."""

    topic_name: str
    flow_name: str
    connection_reference: str = ""


@dataclass
class FlowAnalysisResult:
    """Complete flow impact analysis."""

    flows: list[FlowDefinition] = field(default_factory=list)
    flow_bot_mappings: list[FlowBotMapping] = field(default_factory=list)
    unmapped_flows: list[str] = field(default_factory=list)
    total_flows: int = 0
    flows_without_error_handling: int = 0
    flows_without_timeout: int = 0
    warnings: list[str] = field(default_factory=list)
    mermaid_diagram: str = ""


def _parse_flow_actions(actions_dict: dict) -> list[FlowAction]:
    """Parse actions from a flow definition."""
    parsed: list[FlowAction] = []
    if not isinstance(actions_dict, dict):
        return parsed

    for name, action_def in actions_dict.items():
        if not isinstance(action_def, dict):
            continue

        action_type = action_def.get("type", "Unknown")
        connector = ""

        # Extract connector info
        inputs = action_def.get("inputs", {}) or {}
        if isinstance(inputs, dict):
            host = inputs.get("host", {}) or {}
            if isinstance(host, dict):
                connector = host.get("apiId", "") or host.get("connectionName", "")

        # Check error handling
        run_after = action_def.get("runAfter", {}) or {}
        has_error_handling = False
        if isinstance(run_after, dict):
            for _dep, statuses in run_after.items():
                if isinstance(statuses, list) and any(s in ("Failed", "TimedOut", "Skipped") for s in statuses):
                    has_error_handling = True
                    break

        # Check timeout
        has_timeout = "timeout" in str(action_def.get("limit", {}))

        # Check retry policy
        retry = action_def.get("retryPolicy", {})
        has_retry = bool(retry) and retry.get("type", "none") != "none"

        parsed.append(FlowAction(
            name=name,
            action_type=action_type,
            connector=connector,
            has_error_handling=has_error_handling,
            has_timeout=has_timeout,
            has_retry=has_retry,
        ))

        # Recurse into nested actions (conditions, switches, scopes)
        for nested_key in ("actions", "cases", "default"):
            nested = action_def.get(nested_key)
            if isinstance(nested, dict):
                if nested_key == "cases":
                    for _case_name, case_def in nested.items():
                        if isinstance(case_def, dict):
                            parsed.extend(_parse_flow_actions(case_def.get("actions", {})))
                else:
                    parsed.extend(_parse_flow_actions(nested))

    return parsed


def _parse_flow_json(flow_data: dict, flow_name: str) -> FlowDefinition:
    """Parse a single flow JSON definition."""
    props = flow_data.get("properties", {}) or {}
    definition = props.get("definition", {}) or {}
    display_name = props.get("displayName", flow_name)

    # Trigger type
    triggers = definition.get("triggers", {}) or {}
    trigger_type = "unknown"
    for _trigger_name, trigger_def in triggers.items():
        t_type = trigger_def.get("type", "").lower() if isinstance(trigger_def, dict) else ""
        if "manual" in t_type or "request" in t_type:
            trigger_type = "manual"
        elif "recurrence" in t_type or "schedule" in t_type:
            trigger_type = "scheduled"
        else:
            trigger_type = "automated"
        break

    # Parse actions
    actions_dict = definition.get("actions", {}) or {}
    actions = _parse_flow_actions(actions_dict)

    # Unique connectors
    connectors = sorted(set(a.connector for a in actions if a.connector))

    # Check for error scope
    has_error_scope = any(
        a.action_type.lower() in ("scope",) and a.has_error_handling
        for a in actions
    ) or any(a.has_error_handling for a in actions)

    # Check for any retry policy
    has_retry = any(a.has_retry for a in actions)

    # Check for timeout config
    has_timeout = any(a.has_timeout for a in actions)

    # Generate warnings
    warnings: list[str] = []
    if not has_error_scope:
        warnings.append("No error handling (runAfter with Failed status) detected in any action")
    if not has_retry:
        warnings.append("No retry policies configured on any action")
    if not has_timeout:
        warnings.append("No timeout configurations found — flow may hang indefinitely")
    if len(actions) > 50:
        warnings.append(f"Flow has {len(actions)} actions — consider refactoring for maintainability")

    return FlowDefinition(
        flow_name=flow_name,
        display_name=display_name,
        trigger_type=trigger_type,
        actions=actions,
        total_action_count=len(actions),
        connectors_used=connectors,
        has_error_scope=has_error_scope,
        has_timeout_config=has_timeout,
        has_retry_policy=has_retry,
        warnings=warnings,
    )


def analyze_solution_flows(zip_bytes: bytes) -> FlowAnalysisResult:
    """Analyze Power Automate flows in a solution ZIP.

    Args:
        zip_bytes: Raw bytes of the solution ZIP file.

    Returns:
        FlowAnalysisResult with parsed flows and health analysis.
    """
    result = FlowAnalysisResult()

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        result.warnings.append("Invalid ZIP file")
        return result

    # Find workflow files
    workflow_files = [n for n in names if n.startswith("Workflows/") and n.endswith(".json")]

    if not workflow_files:
        result.warnings.append("No Power Automate flows found in solution (no Workflows/ directory)")
        return result

    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            safe_extractall(zf, work_dir)

        for wf_path_str in workflow_files:
            wf_path = work_dir / wf_path_str
            if not wf_path.exists():
                continue

            try:
                flow_data = json.loads(wf_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                result.warnings.append(f"Failed to parse flow: {wf_path_str}")
                continue

            flow_name = Path(wf_path_str).stem
            flow_def = _parse_flow_json(flow_data, flow_name)
            result.flows.append(flow_def)

        # Find bot-flow mappings by scanning botcomponents for InvokeFlowAction
        botcomponents_dir = work_dir / "botcomponents"
        if botcomponents_dir.exists():
            for comp_dir in sorted(botcomponents_dir.iterdir()):
                if not comp_dir.is_dir():
                    continue
                data_path = comp_dir / "data"
                if not data_path.exists():
                    continue
                try:
                    text = data_path.read_text(encoding="utf-8", errors="replace")
                    if "InvokeFlowAction" in text:
                        topic_name = comp_dir.name
                        # Try to find which flow is referenced
                        for flow in result.flows:
                            if flow.flow_name.lower() in text.lower():
                                result.flow_bot_mappings.append(FlowBotMapping(
                                    topic_name=topic_name,
                                    flow_name=flow.flow_name,
                                ))
                                break
                        else:
                            result.flow_bot_mappings.append(FlowBotMapping(
                                topic_name=topic_name,
                                flow_name="(unknown flow)",
                            ))
                except Exception:
                    continue

    result.total_flows = len(result.flows)
    result.flows_without_error_handling = sum(1 for f in result.flows if not f.has_error_scope)
    result.flows_without_timeout = sum(1 for f in result.flows if not f.has_timeout_config)

    # Find unmapped flows
    mapped_flows = {m.flow_name for m in result.flow_bot_mappings}
    result.unmapped_flows = [f.flow_name for f in result.flows if f.flow_name not in mapped_flows]

    # Build Mermaid diagram
    if result.flows:
        result.mermaid_diagram = _build_flow_diagram(result)

    return result


def _build_flow_diagram(result: FlowAnalysisResult) -> str:
    """Build a Mermaid flowchart of bot-flow-connector relationships."""
    lines: list[str] = []
    lines.append("flowchart LR")

    # Bot topics
    for mapping in result.flow_bot_mappings[:10]:
        safe_topic = mapping.topic_name.replace('"', "'")[:30]
        safe_flow = mapping.flow_name.replace('"', "'")[:30]
        lines.append(f'    T_{hash(mapping.topic_name) % 10000}["{safe_topic}"] --> F_{hash(mapping.flow_name) % 10000}["{safe_flow}"]')

    # Flow to connector relationships
    for flow in result.flows[:10]:
        safe_flow = flow.flow_name.replace('"', "'")[:30]
        flow_id = f"F_{hash(flow.flow_name) % 10000}"
        for conn in flow.connectors_used[:5]:
            safe_conn = conn.replace('"', "'")[:30]
            conn_id = f"C_{hash(conn) % 10000}"
            lines.append(f'    {flow_id} --> {conn_id}["{safe_conn}"]')

    return "\n".join(lines)


def render_flow_report(result: FlowAnalysisResult) -> str:
    """Render flow analysis as markdown."""
    lines: list[str] = []

    lines.append("# Power Automate Flow Impact Analysis")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Flows | {result.total_flows} |")
    lines.append(f"| Without Error Handling | {result.flows_without_error_handling} |")
    lines.append(f"| Without Timeout Config | {result.flows_without_timeout} |")
    lines.append(f"| Bot-Flow Mappings | {len(result.flow_bot_mappings)} |")
    lines.append(f"| Unmapped Flows | {len(result.unmapped_flows)} |")
    lines.append("")

    # Warnings
    all_warnings = list(result.warnings)
    for flow in result.flows:
        for w in flow.warnings:
            all_warnings.append(f"**{flow.display_name}**: {w}")

    if all_warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in all_warnings[:20]:
            lines.append(f"- \u26a0\ufe0f {w}")
        lines.append("")

    # Flow details
    if result.flows:
        lines.append("## Flow Details")
        lines.append("")
        lines.append("| Flow | Trigger | Actions | Connectors | Error Handling | Timeout | Retry |")
        lines.append("|------|---------|---------|------------|----------------|---------|-------|")
        for flow in result.flows:
            err = "\u2705" if flow.has_error_scope else "\u274c"
            timeout = "\u2705" if flow.has_timeout_config else "\u274c"
            retry = "\u2705" if flow.has_retry_policy else "\u274c"
            conns = len(flow.connectors_used)
            lines.append(
                f"| {flow.display_name} | {flow.trigger_type} | {flow.total_action_count} | "
                f"{conns} | {err} | {timeout} | {retry} |"
            )
        lines.append("")

    # Bot-Flow mappings
    if result.flow_bot_mappings:
        lines.append("## Bot Topic \u2192 Flow Mappings")
        lines.append("")
        lines.append("| Topic | Flow |")
        lines.append("|-------|------|")
        for m in result.flow_bot_mappings:
            lines.append(f"| {m.topic_name} | {m.flow_name} |")
        lines.append("")

    # Diagram
    if result.mermaid_diagram:
        lines.append("## Flow Diagram")
        lines.append("")
        lines.append("```mermaid")
        lines.append(result.mermaid_diagram)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
