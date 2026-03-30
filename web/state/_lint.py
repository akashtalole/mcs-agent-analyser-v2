import os

import reflex as rx
from loguru import logger

from linter import run_lint as _run_lint  # noqa: E402
from models import BotProfile  # noqa: E402

from web.mermaid import split_markdown_mermaid


class LintMixin(rx.State, mixin=True):
    """Lint vars and handlers."""

    # Lint vars
    bot_profile_json: str = ""
    lint_report_markdown: str = ""
    is_linting: bool = False
    lint_error: str = ""

    @rx.var
    def can_lint(self) -> bool:
        return bool(self.bot_profile_json) and bool(self.report_markdown)  # type: ignore[attr-defined]

    @rx.var
    def has_lint_report(self) -> bool:
        return bool(self.lint_report_markdown)

    @rx.var
    def lint_report_segments(self) -> list[dict[str, str]]:
        if not self.lint_report_markdown:
            return []
        segments = split_markdown_mermaid(self.lint_report_markdown)
        return [{"type": t, "content": c} for t, c in segments]

    # --- Lint handlers ---

    @rx.event
    async def run_lint(self):
        openai_key = os.getenv("OPENAI_API_KEY", "")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

        use_azure = bool(azure_key and azure_endpoint)
        if not openai_key and not anthropic_key and not use_azure:
            self.lint_error = (
                "No API key configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "or AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT in .env."
            )
            return

        if not self.bot_profile_json:
            self.lint_error = "No bot profile available for linting."
            return

        self.is_linting = True
        self.lint_error = ""
        self.lint_report_markdown = ""
        yield

        try:
            profile = BotProfile.model_validate_json(self.bot_profile_json)
            report, model_used = await _run_lint(
                profile,
                openai_api_key=openai_key,
                anthropic_api_key=anthropic_key,
                azure_openai_api_key=azure_key,
                azure_openai_endpoint=azure_endpoint,
                azure_openai_api_version=azure_api_version,
                azure_openai_deployment=azure_deployment,
            )
            self.lint_report_markdown = report
            logger.info(f"Lint complete using {model_used}")
        except Exception as e:
            logger.error(f"Lint failed: {e}")
            self.lint_error = f"Lint failed: {e}"
        finally:
            self.is_linting = False

    @rx.event
    def download_lint_report(self):
        title = self.report_title if self.report_title else "report"  # type: ignore[attr-defined]
        filename = f"{title}_lint.md"
        yield rx.download(data=self.lint_report_markdown, filename=filename)
        yield rx.toast(f"Lint report saved as {filename}", duration=3000)

    def clear_lint(self):
        self.lint_report_markdown = ""
        self.lint_error = ""
