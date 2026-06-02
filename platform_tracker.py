"""Platform Changelog Tracker — flag deprecated models, preview features, and known issues.

Cross-references bot configuration against a registry of Copilot Studio platform
capabilities, deprecations, and known bugs.  Returns warnings when the bot uses
deprecated patterns, preview-only features, or configurations affected by known issues.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from models import BotProfile


@dataclass
class PlatformIssue:
    """A known platform issue, deprecation, or capability change."""

    issue_id: str
    category: str  # "deprecation", "preview", "known_issue", "capability"
    title: str
    description: str
    severity: str  # "critical", "high", "medium", "low", "info"
    affected_models: list[str] = field(default_factory=list)  # model hint patterns
    affected_features: list[str] = field(default_factory=list)  # feature keys
    date_reported: date | None = None
    date_resolved: date | None = None
    workaround: str = ""
    reference_url: str = ""


# ── Platform Changelog Registry ──────────────────────────────────────────────

_PLATFORM_ISSUES: list[PlatformIssue] = [
    # Deprecations
    PlatformIssue(
        issue_id="DEP-001",
        category="deprecation",
        title="GPT-4o models deprecated in favour of GPT-4.1 family",
        description=(
            "GPT-4o and GPT-4o Mini are deprecated in Copilot Studio. Microsoft recommends "
            "migrating to the GPT-4.1 family (GPT-4.1, GPT-4.1 Mini, GPT-4.1 Nano) which "
            "offer better instruction following and lower latency."
        ),
        severity="high",
        affected_models=["GPT4o", "gpt-4o", "GPT4oMini", "gpt-4o-mini"],
        date_reported=date(2025, 4, 14),
        workaround="Update the model hint to GPT41, GPT41Mini, or GPT41Nano.",
        reference_url="https://learn.microsoft.com/copilot-studio/advanced-ai-models",
    ),
    PlatformIssue(
        issue_id="DEP-002",
        category="deprecation",
        title="GPT-3.5 Turbo is end-of-life",
        description=(
            "GPT-3.5 Turbo has reached end-of-life and is no longer supported in Copilot Studio. "
            "Agents using this model may experience degraded performance or service interruptions."
        ),
        severity="critical",
        affected_models=["GPT35Turbo", "gpt-3.5-turbo", "gpt-35-turbo"],
        date_reported=date(2025, 1, 15),
        workaround="Migrate to GPT-4.1 Nano for cost-effective scenarios or GPT-4.1 for full capability.",
    ),
    PlatformIssue(
        issue_id="DEP-003",
        category="deprecation",
        title="GPT-4 (non-o/turbo) is deprecated",
        description=(
            "The original GPT-4 model is deprecated in Copilot Studio. "
            "It has been superseded by GPT-4.1 which offers significantly better "
            "instruction following and tool use capabilities."
        ),
        severity="high",
        affected_models=["GPT4", "gpt-4"],
        date_reported=date(2025, 3, 1),
        workaround="Migrate to GPT-4.1 for equivalent or better capability.",
    ),

    # Preview features
    PlatformIssue(
        issue_id="PRV-001",
        category="preview",
        title="GPT-5 Chat multi-agent orchestration is in preview",
        description=(
            "GPT-5 Chat's multi-agent orchestration capability is in public preview. "
            "Known instabilities include system topic intrusions during mid-conversation routing, "
            "context drops when switching between connected agents, and inconsistent plan evolution. "
            "Not recommended for production multi-agent deployments."
        ),
        severity="high",
        affected_models=["GPT5Chat", "gpt-5-chat"],
        affected_features=["orchestrator", "multi_agent"],
        workaround="Use GPT-5 (non-Chat) for orchestrator bots, or limit to single-agent configurations.",
    ),
    PlatformIssue(
        issue_id="PRV-002",
        category="preview",
        title="Code interpreter capability is in preview",
        description=(
            "The code interpreter capability is currently in public preview. Behaviour may change "
            "without notice, and there are known limitations around execution timeouts and "
            "output size restrictions. Not recommended for mission-critical workflows."
        ),
        severity="medium",
        affected_features=["code_interpreter"],
        workaround="Implement fallback handling for code interpreter failures.",
    ),
    PlatformIssue(
        issue_id="PRV-003",
        category="preview",
        title="Web browsing capability is in preview",
        description=(
            "Web browsing as a knowledge source is in public preview. Content retrieval "
            "quality varies, and there are known issues with JavaScript-heavy websites. "
            "Results may not be consistent across invocations."
        ),
        severity="medium",
        affected_features=["web_browsing"],
        workaround="Use SharePoint or file-based knowledge sources for production-critical content.",
    ),

    # Known issues
    PlatformIssue(
        issue_id="KI-001",
        category="known_issue",
        title="Conversation Boosting fires unexpectedly in orchestrator bots",
        description=(
            "In multi-agent orchestrator configurations, the system topic 'Conversation Boosting' "
            "may fire mid-conversation when the orchestrator fails to route to a connected agent. "
            "This causes the agent to generate generic responses instead of delegating to the "
            "appropriate child agent."
        ),
        severity="high",
        affected_features=["orchestrator"],
        workaround=(
            "Monitor for Conversation Boosting events in conversation traces. Consider adding "
            "explicit routing instructions in the orchestrator's system prompt."
        ),
    ),
    PlatformIssue(
        issue_id="KI-002",
        category="known_issue",
        title="o-series reasoning models ignore 'think step by step' instructions",
        description=(
            "The o1, o3, and o4-mini reasoning models perform internal chain-of-thought "
            "automatically. Explicit 'think step by step' instructions are redundant and "
            "consume token budget without benefit. They may also cause the model to output "
            "its reasoning process to the user."
        ),
        severity="low",
        affected_models=["o1", "o3", "o4-mini", "o4mini", "o1-preview", "o3-mini"],
        workaround="Remove 'think step by step' and similar directives from system instructions.",
    ),
    PlatformIssue(
        issue_id="KI-003",
        category="known_issue",
        title="Context drops in GPT-5 Chat multi-turn conversations",
        description=(
            "GPT-5 Chat may lose conversation context when routing between different connected "
            "agents in the same session. Variables set in earlier turns may not be available "
            "to agents invoked in later turns."
        ),
        severity="high",
        affected_models=["GPT5Chat", "gpt-5-chat"],
        affected_features=["orchestrator", "multi_agent"],
        workaround=(
            "Store critical context in session variables rather than relying on conversation "
            "history. Consider implementing context-passing explicitly in topic logic."
        ),
    ),

    # Capability changes
    PlatformIssue(
        issue_id="CAP-001",
        category="capability",
        title="GPT-4.1 family supports enhanced tool use",
        description=(
            "The GPT-4.1 family (GPT-4.1, Mini, Nano) supports enhanced tool use with better "
            "function calling accuracy compared to GPT-4o. If your agent uses Power Automate "
            "flows or HTTP actions, the GPT-4.1 family will route to them more reliably."
        ),
        severity="info",
        affected_models=["GPT41", "gpt-4.1", "GPT41Mini", "gpt-4.1-mini", "GPT41Nano", "gpt-4.1-nano"],
    ),
    PlatformIssue(
        issue_id="CAP-002",
        category="capability",
        title="GPT-5 supports extended context windows up to 200K tokens",
        description=(
            "GPT-5 models support up to 200K token context windows, significantly expanding "
            "the amount of knowledge content and conversation history that can be processed. "
            "Longer system instructions are viable with this model."
        ),
        severity="info",
        affected_models=["GPT5", "gpt-5", "GPT5Chat", "gpt-5-chat"],
    ),
]


# ── Tracker functions ─────────────────────────────────────────────────────────


def _check_model_issues(profile: BotProfile) -> list[dict]:
    """Check if the bot's model is affected by any known issues."""
    findings: list[dict] = []
    model_hint = (profile.gpt_info.model_hint or "") if profile.gpt_info else ""

    if not model_hint:
        return findings

    for issue in _PLATFORM_ISSUES:
        if not issue.affected_models:
            continue

        # Case-insensitive match against any affected model pattern
        matched = any(
            model_hint.lower() == am.lower() or re.match(re.escape(am) + r"$", model_hint, re.I)
            for am in issue.affected_models
        )

        if not matched:
            continue

        # Skip resolved issues
        if issue.date_resolved:
            continue

        finding = {
            "issue_id": issue.issue_id,
            "category": issue.category,
            "severity": issue.severity,
            "title": issue.title,
            "detail": issue.description,
            "workaround": issue.workaround,
            "reference_url": issue.reference_url,
            "component": f"Model: {model_hint}",
        }
        findings.append(finding)

    return findings


def _check_feature_issues(profile: BotProfile) -> list[dict]:
    """Check if the bot's enabled features are affected by known issues."""
    findings: list[dict] = []
    gpt = profile.gpt_info

    # Build set of active features
    active_features: set[str] = set()
    if profile.is_orchestrator:
        active_features.add("orchestrator")
        active_features.add("multi_agent")
    if gpt and gpt.code_interpreter:
        active_features.add("code_interpreter")
    if gpt and gpt.web_browsing:
        active_features.add("web_browsing")

    if not active_features:
        return findings

    model_hint = (gpt.model_hint or "").lower() if gpt else ""

    for issue in _PLATFORM_ISSUES:
        if not issue.affected_features:
            continue

        # Check if any affected feature is active
        feature_match = any(f in active_features for f in issue.affected_features)
        if not feature_match:
            continue

        # If issue also has model constraints, check those too
        if issue.affected_models:
            model_match = any(model_hint == am.lower() for am in issue.affected_models)
            if not model_match:
                continue

        # Skip resolved issues
        if issue.date_resolved:
            continue

        finding = {
            "issue_id": issue.issue_id,
            "category": issue.category,
            "severity": issue.severity,
            "title": issue.title,
            "detail": issue.description,
            "workaround": issue.workaround,
            "reference_url": issue.reference_url,
            "component": ", ".join(sorted(active_features & set(issue.affected_features))),
        }
        findings.append(finding)

    return findings


# ── Public API ────────────────────────────────────────────────────────────────


def check_platform_compatibility(profile: BotProfile) -> dict:
    """Check bot configuration against known platform issues, deprecations, and capabilities.

    Returns:
        {
            "findings": list[dict],
            "summary": {
                "total": int,
                "deprecations": int,
                "preview_warnings": int,
                "known_issues": int,
                "capabilities": int,
            },
        }
    """
    findings: list[dict] = []

    findings.extend(_check_model_issues(profile))
    findings.extend(_check_feature_issues(profile))

    # Deduplicate by issue_id
    seen: set[str] = set()
    unique: list[dict] = []
    for f in findings:
        if f["issue_id"] not in seen:
            seen.add(f["issue_id"])
            unique.append(f)

    summary = {
        "total": len(unique),
        "deprecations": sum(1 for f in unique if f["category"] == "deprecation"),
        "preview_warnings": sum(1 for f in unique if f["category"] == "preview"),
        "known_issues": sum(1 for f in unique if f["category"] == "known_issue"),
        "capabilities": sum(1 for f in unique if f["category"] == "capability"),
    }

    return {"findings": unique, "summary": summary}


def render_platform_report(result: dict) -> str:
    """Render platform compatibility findings as markdown."""
    lines: list[str] = []
    findings = result["findings"]
    summary = result["summary"]

    lines.append("# Platform Compatibility Report")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    lines.append(f"| Deprecations | {summary['deprecations']} |")
    lines.append(f"| Preview Warnings | {summary['preview_warnings']} |")
    lines.append(f"| Known Issues | {summary['known_issues']} |")
    lines.append(f"| Capability Notes | {summary['capabilities']} |")
    lines.append(f"| **Total** | **{summary['total']}** |")
    lines.append("")

    if not findings:
        lines.append("> No platform compatibility issues detected. Your configuration is up to date.")
        return "\n".join(lines)

    _CATEGORY_ICONS = {
        "deprecation": "\u26a0\ufe0f",
        "preview": "\U0001f9ea",
        "known_issue": "\U0001f41b",
        "capability": "\u2139\ufe0f",
    }

    _CATEGORY_LABELS = {
        "deprecation": "Deprecations",
        "preview": "Preview Features",
        "known_issue": "Known Issues",
        "capability": "Capability Notes",
    }

    # Group by category
    by_category: dict[str, list[dict]] = {}
    for f in findings:
        by_category.setdefault(f["category"], []).append(f)

    for cat in ("deprecation", "preview", "known_issue", "capability"):
        cat_findings = by_category.get(cat, [])
        if not cat_findings:
            continue

        icon = _CATEGORY_ICONS.get(cat, "")
        label = _CATEGORY_LABELS.get(cat, cat)
        lines.append(f"## {icon} {label}")
        lines.append("")

        for f in cat_findings:
            lines.append(f"### [{f['issue_id']}] {f['title']}")
            lines.append("")
            lines.append(f"**Severity:** {f['severity'].capitalize()} | **Component:** {f['component']}")
            lines.append("")
            lines.append(f"{f['detail']}")
            if f.get("workaround"):
                lines.append("")
                lines.append(f"**Workaround:** {f['workaround']}")
            if f.get("reference_url"):
                lines.append("")
                lines.append(f"**Reference:** {f['reference_url']}")
            lines.append("")

    return "\n".join(lines)
