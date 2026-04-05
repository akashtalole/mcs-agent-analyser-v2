"""Environment Drift Detector — compare bot configuration across environments.

Compares the same bot across multiple environments (Dev, Test, Prod) to detect
configuration drift: instruction changes, model version mismatches, knowledge
source URL differences, connector endpoint mismatches, and more.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from models import BotProfile


@dataclass
class DriftItem:
    """A single configuration drift between environments."""

    property_name: str
    category: str  # "instructions", "model", "knowledge", "connections", "settings", "components"
    severity: str  # "critical", "high", "medium", "low"
    values: dict[str, str] = field(default_factory=dict)  # env_name -> value
    detail: str = ""


@dataclass
class EnvironmentDriftReport:
    """Complete drift analysis across N environments."""

    environment_names: list[str] = field(default_factory=list)
    drift_items: list[DriftItem] = field(default_factory=list)
    total_drifts: int = 0
    critical_drifts: int = 0
    instruction_drift: bool = False
    model_drift: bool = False
    unified_diff: str = ""  # instruction diff between first and last env


def compare_environments(
    profiles: dict[str, BotProfile],
) -> EnvironmentDriftReport:
    """Compare bot profiles across multiple environments.

    Args:
        profiles: Dict mapping environment name -> BotProfile.

    Returns:
        EnvironmentDriftReport with all detected drifts.
    """
    env_names = list(profiles.keys())
    report = EnvironmentDriftReport(environment_names=env_names)

    if len(profiles) < 2:
        return report

    # --- Model drift ---
    model_hints: dict[str, str] = {}
    for name, p in profiles.items():
        hint = (p.gpt_info.model_hint or "Unknown") if p.gpt_info else "Unknown"
        model_hints[name] = hint

    if len(set(model_hints.values())) > 1:
        report.model_drift = True
        report.drift_items.append(DriftItem(
            property_name="Model Hint",
            category="model",
            severity="critical",
            values=model_hints,
            detail="Different AI models are configured across environments. This can cause "
                   "significant behaviour differences between environments.",
        ))

    # --- Instruction drift ---
    instructions: dict[str, str] = {}
    for name, p in profiles.items():
        instr = (p.gpt_info.instructions or "") if p.gpt_info else ""
        instructions[name] = instr

    unique_instructions = set(instructions.values())
    if len(unique_instructions) > 1:
        report.instruction_drift = True
        # Compute lengths for comparison
        instr_lengths = {name: str(len(instr)) for name, instr in instructions.items()}
        report.drift_items.append(DriftItem(
            property_name="System Instructions",
            category="instructions",
            severity="critical",
            values=instr_lengths,
            detail="System instructions differ across environments. This is the most "
                   "common cause of environment-specific bot behaviour differences.",
        ))

        # Generate unified diff between first and last env
        first_env = env_names[0]
        last_env = env_names[-1]
        diff_lines = difflib.unified_diff(
            instructions[first_env].splitlines(keepends=True),
            instructions[last_env].splitlines(keepends=True),
            fromfile=first_env,
            tofile=last_env,
        )
        report.unified_diff = "".join(diff_lines)

    # --- Component count drift ---
    component_counts: dict[str, str] = {}
    for name, p in profiles.items():
        component_counts[name] = str(len(p.components))

    if len(set(component_counts.values())) > 1:
        report.drift_items.append(DriftItem(
            property_name="Component Count",
            category="components",
            severity="high",
            values=component_counts,
            detail="Different number of components across environments. Topics, skills, or "
                   "knowledge sources may be missing in some environments.",
        ))

    # --- Component-level drift (check which components exist in each env) ---
    all_schemas: set[str] = set()
    env_schemas: dict[str, set[str]] = {}
    for name, p in profiles.items():
        schemas = {c.schema_name for c in p.components}
        env_schemas[name] = schemas
        all_schemas.update(schemas)

    for schema in sorted(all_schemas):
        present_in: dict[str, str] = {}
        for name in env_names:
            present_in[name] = "Present" if schema in env_schemas[name] else "Missing"

        if len(set(present_in.values())) > 1:
            report.drift_items.append(DriftItem(
                property_name=f"Component: {schema}",
                category="components",
                severity="high",
                values=present_in,
                detail=f"Component '{schema}' is not present in all environments.",
            ))

    # --- Authentication drift ---
    auth_modes: dict[str, str] = {}
    for name, p in profiles.items():
        auth_modes[name] = p.authentication_mode

    if len(set(auth_modes.values())) > 1:
        report.drift_items.append(DriftItem(
            property_name="Authentication Mode",
            category="settings",
            severity="critical",
            values=auth_modes,
            detail="Authentication mode differs across environments. This is a security risk — "
                   "one environment may have weaker authentication than others.",
        ))

    # --- Knowledge source drift ---
    for name, p in profiles.items():
        for comp in p.components:
            if comp.source_site:
                # Check if same component has different source_site in other envs
                for other_name, other_p in profiles.items():
                    if other_name == name:
                        continue
                    other_comp = next(
                        (c for c in other_p.components if c.schema_name == comp.schema_name),
                        None,
                    )
                    if other_comp and other_comp.source_site and other_comp.source_site != comp.source_site:
                        values = {name: comp.source_site or "", other_name: other_comp.source_site or ""}
                        report.drift_items.append(DriftItem(
                            property_name=f"Knowledge Source URL: {comp.display_name}",
                            category="knowledge",
                            severity="high",
                            values=values,
                            detail=f"Knowledge source '{comp.display_name}' points to different URLs "
                                   f"in different environments. This may cause the bot to reference "
                                   f"different or stale data.",
                        ))

    # --- Settings drift ---
    _compare_bool_setting(profiles, "ai_settings.content_moderation",
                          lambda p: p.ai_settings.content_moderation, "settings", "medium", report)
    _compare_bool_setting(profiles, "ai_settings.file_analysis",
                          lambda p: str(p.ai_settings.file_analysis), "settings", "medium", report)
    _compare_bool_setting(profiles, "is_orchestrator",
                          lambda p: str(p.is_orchestrator), "settings", "high", report)
    _compare_bool_setting(profiles, "access_control_policy",
                          lambda p: p.access_control_policy, "settings", "high", report)
    _compare_bool_setting(profiles, "recognizer_kind",
                          lambda p: p.recognizer_kind, "settings", "medium", report)

    # Summary
    report.total_drifts = len(report.drift_items)
    report.critical_drifts = sum(1 for d in report.drift_items if d.severity == "critical")

    return report


def _compare_bool_setting(
    profiles: dict[str, BotProfile],
    name: str,
    getter: callable,
    category: str,
    severity: str,
    report: EnvironmentDriftReport,
) -> None:
    """Helper to compare a single setting across environments."""
    values: dict[str, str] = {}
    for env_name, p in profiles.items():
        values[env_name] = str(getter(p))

    if len(set(values.values())) > 1:
        report.drift_items.append(DriftItem(
            property_name=name,
            category=category,
            severity=severity,
            values=values,
            detail=f"Setting '{name}' differs across environments.",
        ))


def render_drift_report(report: EnvironmentDriftReport) -> str:
    """Render an environment drift report as markdown."""
    lines: list[str] = []

    lines.append("# Environment Drift Report")
    lines.append("")
    lines.append(f"**Environments compared:** {', '.join(report.environment_names)}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Drifts | {report.total_drifts} |")
    lines.append(f"| Critical Drifts | {report.critical_drifts} |")
    lines.append(f"| Instruction Drift | {'Yes' if report.instruction_drift else 'No'} |")
    lines.append(f"| Model Drift | {'Yes' if report.model_drift else 'No'} |")
    lines.append("")

    if not report.drift_items:
        lines.append("> No configuration drift detected. All environments are in sync.")
        return "\n".join(lines)

    # Group by category
    by_category: dict[str, list[DriftItem]] = {}
    for item in report.drift_items:
        by_category.setdefault(item.category, []).append(item)

    _CATEGORY_LABELS = {
        "instructions": "Instruction Drift",
        "model": "Model Configuration",
        "components": "Component Differences",
        "knowledge": "Knowledge Source URLs",
        "connections": "Connection References",
        "settings": "Settings Differences",
    }

    for cat in ("instructions", "model", "components", "knowledge", "connections", "settings"):
        items = by_category.get(cat, [])
        if not items:
            continue

        label = _CATEGORY_LABELS.get(cat, cat.capitalize())
        lines.append(f"## {label}")
        lines.append("")

        for item in sorted(items, key=lambda x: ["critical", "high", "medium", "low"].index(x.severity)):
            _SEV_ICONS = {"critical": "\U0001f534", "high": "\U0001f7e0", "medium": "\U0001f7e1", "low": "\U0001f535"}
            icon = _SEV_ICONS.get(item.severity, "")
            lines.append(f"### {icon} {item.property_name}")
            lines.append("")
            lines.append(f"**Severity:** {item.severity.capitalize()}")
            lines.append("")
            lines.append(f"{item.detail}")
            lines.append("")

            # Values table
            lines.append("| Environment | Value |")
            lines.append("|-------------|-------|")
            for env_name, value in item.values.items():
                display_val = value[:100] if value else "(empty)"
                lines.append(f"| {env_name} | {display_val} |")
            lines.append("")

    # Instruction diff
    if report.unified_diff:
        lines.append("## Instruction Diff")
        lines.append("")
        lines.append("```diff")
        lines.append(report.unified_diff)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
