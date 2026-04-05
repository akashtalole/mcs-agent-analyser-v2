"""SARIF Export — export analysis findings in SARIF 2.1.0 format.

SARIF (Static Analysis Results Interchange Format) is the standard consumed by
GitHub Code Scanning, Azure DevOps, and other CI/CD platforms.  This enables
automated bot quality gates in CI pipelines.

Usage:
    from sarif_export import export_sarif
    sarif_json = export_sarif(solution_results, validator_results, security_findings)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

_SARIF_VERSION = "2.1.0"
_SARIF_SCHEMA = "https://docs.oasis-open.org/sarif/sarif/v2.1.0/cos02/schemas/sarif-schema-2.1.0.json"
_TOOL_NAME = "Agent Analyser"
_TOOL_VERSION = "2.0.0"

# Severity mapping from internal conventions to SARIF levels
_SEVERITY_TO_SARIF_LEVEL: dict[str, str] = {
    "critical": "error",
    "fail": "error",
    "high": "error",
    "warning": "warning",
    "medium": "warning",
    "low": "note",
    "info": "note",
    "pass": "none",
}

_SEVERITY_TO_SARIF_KIND: dict[str, str] = {
    "pass": "pass",
    "info": "informational",
}


def _map_severity(severity: str) -> tuple[str, str]:
    """Map internal severity to (SARIF kind, SARIF level).

    Returns (kind, level) where kind is 'fail'|'pass'|'informational'|'notApplicable'
    and level is 'error'|'warning'|'note'|'none'.
    """
    kind = _SEVERITY_TO_SARIF_KIND.get(severity, "fail")
    level = _SEVERITY_TO_SARIF_LEVEL.get(severity, "warning")
    return kind, level


def _build_rule(rule_id: str, title: str, detail: str, category: str = "") -> dict:
    """Build a SARIF reportingDescriptor (rule definition)."""
    rule: dict = {
        "id": rule_id,
        "shortDescription": {"text": title[:200]},
        "fullDescription": {"text": detail[:1000]},
        "helpUri": f"https://github.com/Roelzz/Agent_analyser#rule-{rule_id.lower()}",
    }
    if category:
        rule["properties"] = {"tags": [category]}
    return rule


def _build_result(
    rule_id: str,
    title: str,
    detail: str,
    severity: str,
    component: str = "",
) -> dict:
    """Build a SARIF result entry."""
    kind, level = _map_severity(severity)
    result: dict = {
        "ruleId": rule_id,
        "kind": kind,
        "level": level,
        "message": {"text": detail},
    }
    # Add logical location (component name) when available
    if component:
        result["locations"] = [
            {
                "logicalLocations": [
                    {
                        "name": component,
                        "kind": "module",
                    }
                ]
            }
        ]
    return result


def export_sarif(
    solution_results: list[dict] | None = None,
    validator_results: list[dict] | None = None,
    security_findings: list[dict] | None = None,
    *,
    include_passes: bool = False,
) -> str:
    """Export analysis findings as a SARIF 2.1.0 JSON string.

    Args:
        solution_results: Results from solution_checker (rule_id, category, title, severity, detail).
        validator_results: Results from validator (rule_id, title, severity, detail).
        security_findings: Findings from security_scanner (vuln_id, category, severity, title, detail, component).
        include_passes: If True, include passing checks in the output.

    Returns:
        SARIF JSON string.
    """
    rules: dict[str, dict] = {}  # rule_id -> rule definition
    results: list[dict] = []

    # Process solution checker results
    for item in (solution_results or []):
        rule_id = item.get("rule_id", "UNKNOWN")
        severity = item.get("severity", "info")

        if severity == "pass" and not include_passes:
            continue

        if rule_id not in rules:
            rules[rule_id] = _build_rule(
                rule_id,
                item.get("title", ""),
                item.get("detail", ""),
                item.get("category", "Solution"),
            )

        results.append(_build_result(
            rule_id,
            item.get("title", ""),
            item.get("detail", ""),
            severity,
        ))

    # Process validator results
    for item in (validator_results or []):
        rule_id = f"VAL-{item.get('rule_id', 'UNKNOWN')}"
        severity = item.get("severity", "info")

        if severity == "pass" and not include_passes:
            continue

        if rule_id not in rules:
            rules[rule_id] = _build_rule(
                rule_id,
                item.get("title", ""),
                item.get("detail", ""),
                "Validation",
            )

        results.append(_build_result(
            rule_id,
            item.get("title", ""),
            item.get("detail", ""),
            severity,
        ))

    # Process security scanner findings
    for item in (security_findings or []):
        rule_id = item.get("vuln_id", "UNKNOWN")
        severity = item.get("severity", "info")

        if severity == "pass" and not include_passes:
            continue

        if rule_id not in rules:
            rules[rule_id] = _build_rule(
                rule_id,
                item.get("title", ""),
                item.get("detail", ""),
                item.get("category", "Security"),
            )

        results.append(_build_result(
            rule_id,
            item.get("title", ""),
            item.get("detail", ""),
            severity,
            item.get("component", ""),
        ))

    # Assemble SARIF document
    sarif = {
        "$schema": _SARIF_SCHEMA,
        "version": _SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": _TOOL_NAME,
                        "version": _TOOL_VERSION,
                        "informationUri": "https://github.com/Roelzz/Agent_analyser",
                        "rules": list(rules.values()),
                    }
                },
                "results": results,
                "invocations": [
                    {
                        "executionSuccessful": True,
                        "startTimeUtc": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            }
        ],
    }

    return json.dumps(sarif, indent=2)
