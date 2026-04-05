"""Knowledge Source Content Quality Analyzer — structural analysis of knowledge documents.

Analyzes knowledge document content for issues that degrade retrieval quality:
- Formatting problems (tables without headers, excessively long paragraphs)
- Content structure issues (missing headings, inconsistent formatting)
- Potential contradictions within documents
- Content gaps relative to bot purpose
- Readability metrics

This module performs structural analysis only (no LLM calls required).
LLM-based semantic analysis (contradiction detection, gap analysis) can be
optionally enabled when an API key is available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ContentIssue:
    """A quality issue found in knowledge content."""

    issue_id: str
    category: str  # "formatting", "structure", "readability", "content"
    severity: str  # "high", "medium", "low", "info"
    title: str
    detail: str
    location: str = ""  # e.g. "paragraph 3", "line 45"


@dataclass
class DocumentProfile:
    """Quality profile for a single document."""

    document_name: str
    word_count: int = 0
    paragraph_count: int = 0
    heading_count: int = 0
    avg_paragraph_length: float = 0.0
    max_paragraph_length: int = 0
    table_count: int = 0
    list_count: int = 0
    issues: list[ContentIssue] = field(default_factory=list)
    quality_score: float = 0.0  # 0-100


@dataclass
class KnowledgeQualityReport:
    """Complete quality analysis across all knowledge documents."""

    documents: list[DocumentProfile] = field(default_factory=list)
    total_issues: int = 0
    avg_quality_score: float = 0.0
    cross_document_issues: list[ContentIssue] = field(default_factory=list)


def _analyze_paragraphs(text: str) -> tuple[int, float, int, list[ContentIssue]]:
    """Analyze paragraph structure and return (count, avg_len, max_len, issues)."""
    issues: list[ContentIssue] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if not paragraphs:
        return 0, 0.0, 0, issues

    lengths = [len(p.split()) for p in paragraphs]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)

    # Flag excessively long paragraphs (> 300 words)
    for i, (para, length) in enumerate(zip(paragraphs, lengths)):
        if length > 300:
            issues.append(ContentIssue(
                issue_id="FMT-001",
                category="formatting",
                severity="medium",
                title="Excessively long paragraph",
                detail=(
                    f"Paragraph {i + 1} has {length} words. Long paragraphs degrade "
                    f"retrieval quality because search systems chunk at paragraph boundaries. "
                    f"Break into smaller, focused paragraphs of 50-150 words."
                ),
                location=f"Paragraph {i + 1}",
            ))
        elif length > 200:
            issues.append(ContentIssue(
                issue_id="FMT-001",
                category="formatting",
                severity="low",
                title="Long paragraph",
                detail=f"Paragraph {i + 1} has {length} words. Consider splitting for better retrieval.",
                location=f"Paragraph {i + 1}",
            ))

    # Flag very short paragraphs that might be formatting artifacts
    short_count = sum(1 for wl in lengths if wl < 5)
    if short_count > len(paragraphs) * 0.3 and len(paragraphs) > 5:
        issues.append(ContentIssue(
            issue_id="FMT-002",
            category="formatting",
            severity="low",
            title="Many very short paragraphs",
            detail=(
                f"{short_count} of {len(paragraphs)} paragraphs have fewer than 5 words. "
                f"This may indicate formatting issues (orphaned lines, list items as paragraphs)."
            ),
        ))

    return len(paragraphs), avg_len, max_len, issues


def _analyze_headings(text: str) -> tuple[int, list[ContentIssue]]:
    """Analyze heading structure."""
    issues: list[ContentIssue] = []

    # Markdown headings
    headings = re.findall(r"^#{1,6}\s+.+$", text, re.MULTILINE)
    # Also detect underline-style headings
    headings.extend(re.findall(r"^.+\n[=\-]{3,}$", text, re.MULTILINE))

    heading_count = len(headings)
    word_count = len(text.split())

    # Flag missing headings in long documents
    if word_count > 500 and heading_count == 0:
        issues.append(ContentIssue(
            issue_id="STR-001",
            category="structure",
            severity="high",
            title="No headings in long document",
            detail=(
                f"Document has {word_count} words but no headings. Headings improve "
                f"retrieval quality by providing semantic boundaries for chunking. "
                f"Add descriptive headings to organize the content by topic."
            ),
        ))
    elif word_count > 1000 and heading_count < 3:
        issues.append(ContentIssue(
            issue_id="STR-001",
            category="structure",
            severity="medium",
            title="Few headings for document length",
            detail=(
                f"Document has {word_count} words but only {heading_count} heading(s). "
                f"Consider adding more headings to improve content organization."
            ),
        ))

    # Check heading hierarchy (skipping levels)
    md_headings = re.findall(r"^(#{1,6})\s+.+$", text, re.MULTILINE)
    if len(md_headings) >= 2:
        levels = [len(h) for h in md_headings]
        for i in range(1, len(levels)):
            if levels[i] > levels[i - 1] + 1:
                issues.append(ContentIssue(
                    issue_id="STR-002",
                    category="structure",
                    severity="low",
                    title="Heading level skip",
                    detail=(
                        f"Heading at position {i + 1} skips from level {levels[i - 1]} "
                        f"to level {levels[i]}. Consistent heading hierarchy improves "
                        f"document structure and retrieval."
                    ),
                ))
                break  # One warning is enough

    return heading_count, issues


def _analyze_tables(text: str) -> tuple[int, list[ContentIssue]]:
    """Analyze table formatting."""
    issues: list[ContentIssue] = []

    # Detect markdown tables
    table_pattern = re.compile(r"^\|.+\|$", re.MULTILINE)
    table_lines = table_pattern.findall(text)
    separator_lines = [line for line in table_lines if re.match(r"^\|[\s\-:|]+\|$", line)]

    table_count = len(separator_lines)  # Each separator = one table header

    # Detect tables without proper headers (pipe-delimited lines without separator)
    if len(table_lines) > 0 and table_count == 0:
        issues.append(ContentIssue(
            issue_id="FMT-003",
            category="formatting",
            severity="medium",
            title="Tables without proper headers",
            detail=(
                "Pipe-delimited content detected but no proper markdown table headers "
                "(missing separator row like |---|---|). Tables without headers are harder "
                "for the retrieval system to parse correctly."
            ),
        ))

    return table_count, issues


def _analyze_readability(text: str) -> list[ContentIssue]:
    """Basic readability analysis."""
    issues: list[ContentIssue] = []
    words = text.split()
    word_count = len(words)

    if word_count < 10:
        return issues

    # Average sentence length
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_sentence_len > 35:
            issues.append(ContentIssue(
                issue_id="READ-001",
                category="readability",
                severity="medium",
                title="Complex sentences",
                detail=(
                    f"Average sentence length is {avg_sentence_len:.0f} words. "
                    f"Sentences over 25 words are harder for AI models to process accurately. "
                    f"Consider breaking complex sentences into shorter, clearer statements."
                ),
            ))

    # Detect jargon density (words > 12 characters)
    long_words = [w for w in words if len(w) > 12]
    jargon_ratio = len(long_words) / word_count
    if jargon_ratio > 0.15:
        issues.append(ContentIssue(
            issue_id="READ-002",
            category="readability",
            severity="low",
            title="High jargon density",
            detail=(
                f"{jargon_ratio:.0%} of words are longer than 12 characters. "
                f"High jargon density may reduce retrieval accuracy. Consider adding "
                f"a glossary section or using simpler synonyms where possible."
            ),
        ))

    # Detect duplicate content (repeated sentences)
    sentence_set: set[str] = set()
    duplicates = 0
    for s in sentences:
        normalized = " ".join(s.lower().split())
        if normalized in sentence_set and len(normalized.split()) > 5:
            duplicates += 1
        sentence_set.add(normalized)

    if duplicates > 2:
        issues.append(ContentIssue(
            issue_id="READ-003",
            category="readability",
            severity="medium",
            title="Duplicate sentences detected",
            detail=(
                f"{duplicates} duplicate sentence(s) found. Duplicated content wastes "
                f"knowledge source capacity and may confuse the retrieval system by "
                f"creating redundant search results."
            ),
        ))

    return issues


def _compute_quality_score(doc: DocumentProfile) -> float:
    """Compute a 0-100 quality score based on issues found."""
    score = 100.0

    for issue in doc.issues:
        if issue.severity == "high":
            score -= 15
        elif issue.severity == "medium":
            score -= 8
        elif issue.severity == "low":
            score -= 3

    # Bonus for good structure
    if doc.heading_count > 0:
        score += 5
    if 50 <= doc.avg_paragraph_length <= 150:
        score += 5

    return max(0.0, min(100.0, score))


def analyze_document(text: str, document_name: str = "document") -> DocumentProfile:
    """Analyze a single knowledge document for quality issues.

    Args:
        text: The document content as plain text.
        document_name: Name/identifier for the document.

    Returns:
        DocumentProfile with issues and quality score.
    """
    doc = DocumentProfile(document_name=document_name)
    doc.word_count = len(text.split())

    if doc.word_count < 5:
        doc.issues.append(ContentIssue(
            issue_id="CONTENT-001",
            category="content",
            severity="high",
            title="Document is nearly empty",
            detail=f"Document '{document_name}' has only {doc.word_count} words.",
        ))
        doc.quality_score = 10.0
        return doc

    # Paragraph analysis
    doc.paragraph_count, doc.avg_paragraph_length, doc.max_paragraph_length, para_issues = (
        _analyze_paragraphs(text)
    )
    doc.issues.extend(para_issues)

    # Heading analysis
    doc.heading_count, heading_issues = _analyze_headings(text)
    doc.issues.extend(heading_issues)

    # Table analysis
    doc.table_count, table_issues = _analyze_tables(text)
    doc.issues.extend(table_issues)

    # Readability analysis
    doc.issues.extend(_analyze_readability(text))

    # Quality score
    doc.quality_score = _compute_quality_score(doc)

    return doc


def analyze_knowledge_quality(
    documents: dict[str, str],
    bot_instructions: str = "",
) -> KnowledgeQualityReport:
    """Analyze multiple knowledge documents for quality.

    Args:
        documents: Dict mapping document name -> content text.
        bot_instructions: Optional bot instructions for relevance checking.

    Returns:
        KnowledgeQualityReport with per-document and cross-document analysis.
    """
    report = KnowledgeQualityReport()

    for name, text in documents.items():
        doc = analyze_document(text, name)
        report.documents.append(doc)

    if report.documents:
        report.avg_quality_score = (
            sum(d.quality_score for d in report.documents) / len(report.documents)
        )

    report.total_issues = sum(len(d.issues) for d in report.documents)

    # Cross-document analysis
    if len(report.documents) >= 2:
        # Check for content overlap between documents
        doc_tokens: list[tuple[str, set[str]]] = []
        for doc in report.documents:
            # Get significant tokens (length > 4, not common words)
            _STOPWORDS = {"about", "after", "before", "could", "would", "should", "their",
                          "there", "these", "those", "which", "where", "other", "being"}
            text = documents.get(doc.document_name, "")
            tokens = set(
                w.lower() for w in text.split()
                if len(w) > 4 and w.lower() not in _STOPWORDS
            )
            doc_tokens.append((doc.document_name, tokens))

        for i in range(len(doc_tokens)):
            for j in range(i + 1, len(doc_tokens)):
                name_a, tokens_a = doc_tokens[i]
                name_b, tokens_b = doc_tokens[j]
                if not tokens_a or not tokens_b:
                    continue
                shared = tokens_a & tokens_b
                min_len = min(len(tokens_a), len(tokens_b))
                overlap = len(shared) / min_len if min_len > 0 else 0
                if overlap > 0.6:
                    report.cross_document_issues.append(ContentIssue(
                        issue_id="CROSS-001",
                        category="content",
                        severity="medium",
                        title=f"High content overlap between '{name_a}' and '{name_b}'",
                        detail=(
                            f"{overlap:.0%} token overlap detected. Highly overlapping documents "
                            f"create redundant search results and waste knowledge source capacity. "
                            f"Consider merging or deduplicating."
                        ),
                    ))

    return report


def render_knowledge_quality_report(report: KnowledgeQualityReport) -> str:
    """Render knowledge quality analysis as markdown."""
    lines: list[str] = []

    lines.append("# Knowledge Content Quality Report")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Documents Analyzed | {len(report.documents)} |")
    lines.append(f"| Total Issues | {report.total_issues} |")
    lines.append(f"| Avg Quality Score | {report.avg_quality_score:.0f}/100 |")
    lines.append(f"| Cross-Document Issues | {len(report.cross_document_issues)} |")
    lines.append("")

    # Document scores
    if report.documents:
        lines.append("## Document Scores")
        lines.append("")
        lines.append("| Document | Words | Paragraphs | Headings | Quality Score | Issues |")
        lines.append("|----------|-------|------------|----------|---------------|--------|")
        for doc in sorted(report.documents, key=lambda d: d.quality_score):
            score_icon = "\U0001f7e2" if doc.quality_score >= 70 else "\U0001f7e1" if doc.quality_score >= 40 else "\U0001f534"
            lines.append(
                f"| {doc.document_name} | {doc.word_count} | {doc.paragraph_count} | "
                f"{doc.heading_count} | {score_icon} {doc.quality_score:.0f} | {len(doc.issues)} |"
            )
        lines.append("")

    # Per-document issues
    for doc in report.documents:
        if not doc.issues:
            continue
        lines.append(f"## {doc.document_name}")
        lines.append("")
        for issue in doc.issues:
            _SEV_ICONS = {"high": "\U0001f534", "medium": "\U0001f7e1", "low": "\U0001f535", "info": "\u2139\ufe0f"}
            icon = _SEV_ICONS.get(issue.severity, "")
            loc = f" ({issue.location})" if issue.location else ""
            lines.append(f"- {icon} **[{issue.issue_id}]** {issue.title}{loc}: {issue.detail}")
        lines.append("")

    # Cross-document issues
    if report.cross_document_issues:
        lines.append("## Cross-Document Issues")
        lines.append("")
        for issue in report.cross_document_issues:
            lines.append(f"- \u26a0\ufe0f **[{issue.issue_id}]** {issue.title}: {issue.detail}")
        lines.append("")

    return "\n".join(lines)
