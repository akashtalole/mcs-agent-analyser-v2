"""Agent Performance Profiling — percentile-based latency and quality analysis from transcripts.

Analyses transcript data to build per-topic and per-action performance profiles:
p50/p95/p99 latency distributions, knowledge search hit rates, token usage
estimates, and bottleneck identification.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from models import ConversationTimeline, EventType


@dataclass
class TopicLatencyProfile:
    """Latency statistics for a single topic."""

    topic_name: str
    invocation_count: int = 0
    durations_ms: list[float] = field(default_factory=list)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    avg_ms: float = 0.0
    error_count: int = 0


@dataclass
class KnowledgeSearchProfile:
    """Performance profile for knowledge search operations."""

    total_searches: int = 0
    hit_count: int = 0  # searches that returned results
    miss_count: int = 0  # searches with empty results
    hit_rate: float = 0.0
    avg_result_count: float = 0.0
    avg_execution_time_ms: float = 0.0
    execution_times_ms: list[float] = field(default_factory=list)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    error_count: int = 0


@dataclass
class BottleneckInfo:
    """Identifies a performance bottleneck."""

    component: str
    bottleneck_type: str  # "latency", "error_rate", "search_miss"
    severity: str  # "critical", "high", "medium", "low"
    detail: str
    metric_value: float = 0.0


@dataclass
class PerformanceProfile:
    """Complete performance profile from transcript analysis."""

    conversation_count: int = 0
    total_elapsed_ms: float = 0.0
    avg_elapsed_ms: float = 0.0
    p50_elapsed_ms: float = 0.0
    p95_elapsed_ms: float = 0.0
    p99_elapsed_ms: float = 0.0
    topic_profiles: list[TopicLatencyProfile] = field(default_factory=list)
    knowledge_profile: KnowledgeSearchProfile = field(default_factory=KnowledgeSearchProfile)
    bottlenecks: list[BottleneckInfo] = field(default_factory=list)
    slowest_conversation_id: str = ""
    slowest_elapsed_ms: float = 0.0


def _percentile(data: list[float], pct: float) -> float:
    """Compute percentile from sorted data. Returns 0.0 for empty lists."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def _parse_execution_time(time_str: str | None) -> float | None:
    """Parse execution time string (e.g. '123ms', '1.5s') to milliseconds."""
    if not time_str:
        return None
    time_str = time_str.strip().lower()
    try:
        if time_str.endswith("ms"):
            return float(time_str[:-2])
        if time_str.endswith("s"):
            return float(time_str[:-1]) * 1000
        return float(time_str)
    except (ValueError, TypeError):
        return None


def build_performance_profile(
    timelines: list[ConversationTimeline],
) -> PerformanceProfile:
    """Build a performance profile from multiple conversation timelines.

    Args:
        timelines: List of parsed conversation timelines.

    Returns:
        PerformanceProfile with latency distributions and bottleneck analysis.
    """
    if not timelines:
        return PerformanceProfile()

    profile = PerformanceProfile(conversation_count=len(timelines))

    # Overall elapsed times
    elapsed_times = [t.total_elapsed_ms for t in timelines if t.total_elapsed_ms > 0]
    if elapsed_times:
        profile.total_elapsed_ms = sum(elapsed_times)
        profile.avg_elapsed_ms = statistics.mean(elapsed_times)
        profile.p50_elapsed_ms = _percentile(elapsed_times, 50)
        profile.p95_elapsed_ms = _percentile(elapsed_times, 95)
        profile.p99_elapsed_ms = _percentile(elapsed_times, 99)

    # Find slowest conversation
    slowest = max(timelines, key=lambda t: t.total_elapsed_ms)
    profile.slowest_conversation_id = slowest.conversation_id or "unknown"
    profile.slowest_elapsed_ms = slowest.total_elapsed_ms

    # Per-topic latency profiles
    topic_durations: dict[str, list[float]] = defaultdict(list)
    topic_errors: dict[str, int] = defaultdict(int)

    for timeline in timelines:
        for phase in timeline.phases:
            if phase.duration_ms > 0:
                topic_durations[phase.label].append(phase.duration_ms)
            if phase.state and phase.state.lower() in ("failed", "error"):
                topic_errors[phase.label] += 1

        for event in timeline.events:
            if event.event_type == EventType.ERROR and event.topic_name:
                topic_errors[event.topic_name] += 1

    for topic_name, durations in sorted(topic_durations.items(), key=lambda x: len(x[1]), reverse=True):
        tp = TopicLatencyProfile(
            topic_name=topic_name,
            invocation_count=len(durations),
            durations_ms=durations,
            p50_ms=_percentile(durations, 50),
            p95_ms=_percentile(durations, 95),
            p99_ms=_percentile(durations, 99),
            min_ms=min(durations),
            max_ms=max(durations),
            avg_ms=statistics.mean(durations),
            error_count=topic_errors.get(topic_name, 0),
        )
        profile.topic_profiles.append(tp)

    # Knowledge search profile
    ks_profile = KnowledgeSearchProfile()
    all_exec_times: list[float] = []
    total_results = 0

    for timeline in timelines:
        for ks in timeline.knowledge_searches:
            ks_profile.total_searches += 1
            has_results = len(ks.search_results) > 0
            if has_results:
                ks_profile.hit_count += 1
                total_results += len(ks.search_results)
            else:
                ks_profile.miss_count += 1

            if ks.search_errors:
                ks_profile.error_count += len(ks.search_errors)

            exec_time = _parse_execution_time(ks.execution_time)
            if exec_time is not None:
                all_exec_times.append(exec_time)

    if ks_profile.total_searches > 0:
        ks_profile.hit_rate = ks_profile.hit_count / ks_profile.total_searches
        ks_profile.avg_result_count = total_results / ks_profile.total_searches

    if all_exec_times:
        ks_profile.execution_times_ms = all_exec_times
        ks_profile.avg_execution_time_ms = statistics.mean(all_exec_times)
        ks_profile.p50_ms = _percentile(all_exec_times, 50)
        ks_profile.p95_ms = _percentile(all_exec_times, 95)

    profile.knowledge_profile = ks_profile

    # Bottleneck detection
    bottlenecks: list[BottleneckInfo] = []

    # Topic latency bottlenecks
    for tp in profile.topic_profiles:
        if tp.p95_ms > 5000:  # > 5 seconds at p95
            bottlenecks.append(BottleneckInfo(
                component=tp.topic_name,
                bottleneck_type="latency",
                severity="critical" if tp.p95_ms > 10000 else "high",
                detail=f"p95 latency is {tp.p95_ms:.0f}ms ({tp.invocation_count} invocations)",
                metric_value=tp.p95_ms,
            ))
        if tp.error_count > 0 and tp.invocation_count > 0:
            error_rate = tp.error_count / tp.invocation_count
            if error_rate > 0.1:
                bottlenecks.append(BottleneckInfo(
                    component=tp.topic_name,
                    bottleneck_type="error_rate",
                    severity="critical" if error_rate > 0.3 else "high",
                    detail=f"Error rate is {error_rate:.1%} ({tp.error_count}/{tp.invocation_count})",
                    metric_value=error_rate,
                ))

    # Knowledge search bottlenecks
    if ks_profile.total_searches > 5 and ks_profile.hit_rate < 0.5:
        bottlenecks.append(BottleneckInfo(
            component="Knowledge Search",
            bottleneck_type="search_miss",
            severity="high" if ks_profile.hit_rate < 0.3 else "medium",
            detail=f"Hit rate is {ks_profile.hit_rate:.1%} ({ks_profile.hit_count}/{ks_profile.total_searches})",
            metric_value=ks_profile.hit_rate,
        ))

    if ks_profile.p95_ms > 3000:
        bottlenecks.append(BottleneckInfo(
            component="Knowledge Search",
            bottleneck_type="latency",
            severity="high",
            detail=f"p95 search latency is {ks_profile.p95_ms:.0f}ms",
            metric_value=ks_profile.p95_ms,
        ))

    bottlenecks.sort(key=lambda b: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(b.severity, 4))
    profile.bottlenecks = bottlenecks

    return profile


def render_performance_report(profile: PerformanceProfile) -> str:
    """Render a performance profile as a markdown report."""
    lines: list[str] = []

    lines.append("# Performance Profile Report")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Conversations Analyzed | {profile.conversation_count} |")
    lines.append(f"| Avg Elapsed Time | {profile.avg_elapsed_ms:.0f} ms |")
    lines.append(f"| p50 Elapsed | {profile.p50_elapsed_ms:.0f} ms |")
    lines.append(f"| p95 Elapsed | {profile.p95_elapsed_ms:.0f} ms |")
    lines.append(f"| p99 Elapsed | {profile.p99_elapsed_ms:.0f} ms |")
    lines.append(f"| Slowest Conversation | {profile.slowest_conversation_id} ({profile.slowest_elapsed_ms:.0f} ms) |")
    lines.append("")

    # Bottlenecks
    if profile.bottlenecks:
        lines.append("## Bottlenecks Detected")
        lines.append("")
        _SEV_ICONS = {"critical": "\U0001f534", "high": "\U0001f7e0", "medium": "\U0001f7e1", "low": "\U0001f535"}
        for b in profile.bottlenecks:
            icon = _SEV_ICONS.get(b.severity, "")
            lines.append(f"- {icon} **{b.component}** ({b.bottleneck_type}): {b.detail}")
        lines.append("")

    # Topic latency table
    if profile.topic_profiles:
        lines.append("## Topic Latency Distribution")
        lines.append("")
        lines.append("| Topic | Count | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) | Errors |")
        lines.append("|-------|-------|----------|----------|----------|----------|--------|")
        for tp in profile.topic_profiles[:20]:
            lines.append(
                f"| {tp.topic_name} | {tp.invocation_count} | "
                f"{tp.p50_ms:.0f} | {tp.p95_ms:.0f} | {tp.p99_ms:.0f} | "
                f"{tp.max_ms:.0f} | {tp.error_count} |"
            )
        lines.append("")

    # Knowledge search profile
    ks = profile.knowledge_profile
    if ks.total_searches > 0:
        lines.append("## Knowledge Search Performance")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Searches | {ks.total_searches} |")
        lines.append(f"| Hit Rate | {ks.hit_rate:.1%} |")
        lines.append(f"| Avg Results per Search | {ks.avg_result_count:.1f} |")
        lines.append(f"| Avg Execution Time | {ks.avg_execution_time_ms:.0f} ms |")
        lines.append(f"| p50 Execution Time | {ks.p50_ms:.0f} ms |")
        lines.append(f"| p95 Execution Time | {ks.p95_ms:.0f} ms |")
        lines.append(f"| Search Errors | {ks.error_count} |")
        lines.append("")

    # Slowest conversation Gantt chart
    if profile.conversation_count > 0:
        lines.append("## Slowest Conversation Breakdown")
        lines.append("")
        lines.append("```mermaid")
        lines.append("gantt")
        lines.append("    title Slowest Conversation Phases")
        lines.append("    dateFormat X")
        lines.append("    axisFormat %s ms")

        # Find the slowest conversation's phases from topic profiles
        if profile.topic_profiles:
            offset = 0
            for tp in profile.topic_profiles[:10]:
                if tp.max_ms > 0:
                    safe_name = tp.topic_name[:30].replace(":", " ")
                    end = offset + int(tp.max_ms)
                    lines.append(f"    {safe_name} : {offset}, {end}")
                    offset = end
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
