import abc
import typing as t
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import jinja2
import rich.markup
import rich.progress
from rich.console import Console
from rich.syntax import Syntax

from contraqctor.qc.base import Result, ResultsStatistics, Status, _TaggedResult
from contraqctor.qc.serializers import ContextExportableObjSerializer

try:
    __version__ = version("contraqctor")
except PackageNotFoundError:
    __version__ = "0.0.0"

STATUS_COLOR = {
    Status.PASSED: "green",
    Status.FAILED: "red",
    Status.ERROR: "magenta",
    Status.SKIPPED: "yellow",
    Status.WARNING: "orange1",
}


class Reporter(abc.ABC):
    """Base class for test result reporters.

    Reporters handle the presentation of test results in different formats
    such as console output or HTML files.
    """

    @abc.abstractmethod
    def report_results(
        self,
        results: dict[str | None, list[Result]] | list[_TaggedResult],
        *,
        render_context: bool = True,
        render_description: bool = True,
        render_traceback: bool = True,
        render_message: bool = True,
        serialize_context_exportable_obj: bool = False,
        **kwargs,
    ) -> None:
        """Report test results.

        Args:
            results: List of tagged test results.
            statistics: Overall statistics for the test run.
            render_context: Whether to include test context.
            render_description: Whether to include test descriptions.
            render_traceback: Whether to include tracebacks for errors.
            render_message: Whether to include test result messages.
            serialize_context_exportable_obj: Whether to serialize ContextExportableObj instances.
        """
        pass


class ConsoleReporter(Reporter):
    """Reporter that outputs test results to a rich console.

    Args:
        console: Optional rich Console instance. If not provided, creates a new one.
        include_status: Set of statuses to include in detailed output.
        default_group_name: Name to use for ungrouped tests.
    """

    def __init__(
        self,
        console: t.Optional[Console] = None,
        include_status: t.Set[Status] = frozenset({Status.FAILED, Status.ERROR, Status.WARNING}),
        default_group_name: str = "Ungrouped",
    ):
        self.console = console or Console()
        self.include_status = include_status
        self.default_group_name = default_group_name

    def report_results(
        self,
        results: dict[str | None, list[Result]] | list[_TaggedResult],
        *,
        render_context: bool = True,
        render_description: bool = True,
        render_traceback: bool = True,
        render_message: bool = True,
        serialize_context_exportable_obj: bool = False,
        asset_output_dir: str | Path = Path("./report/assets"),
        **kwargs,
    ) -> None:
        """Print detailed test results to the console.

        Args:
            results: List of tagged test results.
            statistics: Overall statistics for the test run.
            render_context: Whether to include test context.
            render_description: Whether to include test descriptions.
            render_traceback: Whether to include tracebacks for errors.
            render_message: Whether to include test result messages.
            serialize_context_exportable_obj: Whether to serialize ContextExportableObj instances.
            asset_output_dir: Directory for saving serialized assets. Defaults to "./report/assets".
        """
        if not results:
            return

        results = _normalize_results(results)
        # Setup serializer and serialize ALL results if needed
        # (not just the ones being displayed)
        serializer = None
        output_dir = None
        serialized_contexts = {}

        if serialize_context_exportable_obj:
            serializer = ContextExportableObjSerializer()
            if asset_output_dir is None:
                output_dir = Path("./report/assets")
            else:
                output_dir = Path(asset_output_dir)

            # Serialize all results, not just displayed ones
            for idx, tagged_result in enumerate(results):
                if tagged_result.result.context is not None:
                    serialized_contexts[idx] = serializer.serialize_as_file(
                        tagged_result.result.context, output_dir, f"test_{idx}"
                    )

        all_included_results = [
            tagged_result for tagged_result in results if tagged_result.result.status in self.include_status
        ]

        if not all_included_results:
            return

        self.console.print()
        self.console.print(f"[bold]contraqctor v{__version__}[/bold]")
        self.console.print(f"[dim]Test run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}[/dim]")
        self.console.print()
        self._print_status_header(self.include_status)
        self.console.print()

        idx = 0
        for group, test_results in _TaggedResult.group_by_group(all_included_results):
            group_name = group or self.default_group_name
            for result in test_results:
                # Find the original index in the full results list
                original_idx = results.index(result)
                context = serialized_contexts.get(original_idx, result.result.context)

                self._print_test_result(
                    result.result,
                    group_name,
                    idx,
                    render_message,
                    render_description,
                    render_traceback,
                    render_context,
                    context,
                )
                self.console.print()
                idx += 1

    def _print_status_header(self, include: t.Set[Status]) -> None:
        """Print header showing which statuses are included."""
        if include:
            self.console.print("Including ", end="")
            for i, status in enumerate(include):
                color = STATUS_COLOR[status]
                self.console.print(f"[{color}]{status}[/{color}]", end="")
                if i < len(include) - 1:
                    self.console.print(", ", end="")
            self.console.print()

    def _print_test_result(
        self,
        test_result: Result,
        group_name: str,
        idx: int,
        render_message: bool = True,
        render_description: bool = True,
        render_traceback: bool = True,
        render_context: bool = True,
        context: t.Optional[t.Any] = None,
    ) -> None:
        """Print details of a single test result.

        Args:
            test_result: The test result to print.
            group_name: Name of the test group.
            idx: Index of the test result.
            render_message: Whether to render the message.
            render_description: Whether to render the description.
            render_traceback: Whether to render the traceback.
            render_context: Whether to render the context.
            context: Optional serialized context to use instead of test_result.context.
        """
        color = STATUS_COLOR[test_result.status]

        group_name_escaped = rich.markup.escape(f"[{group_name}]")
        self.console.print(
            f"[bold {color}]{idx}. {group_name_escaped} {test_result.suite_name}.{test_result.test_name}[/bold {color}]"
        )

        self.console.print(f"[{color}]Result:[/{color}] {test_result.result}")

        if render_message and test_result.message:
            self.console.print(f"[{color}]Message:[/{color}] {test_result.message}")

        if render_description and test_result.description:
            self.console.print(f"[{color}]Description:[/{color}] {test_result.description}")

        if render_traceback and test_result.traceback:
            self.console.print(f"[{color}]Traceback:[/{color}]")
            syntax = Syntax(test_result.traceback, "pytb", theme="ansi", line_numbers=False)
            self.console.print(syntax)

        if render_context:
            # Use provided context (possibly serialized) or fallback to test_result.context
            ctx = context if context is not None else test_result.context
            if ctx:
                self.console.print(f"[{color}]Context:[/{color}] {ctx}")

        self.console.print("=" * 80)


class HtmlReporter(Reporter):
    """Reporter that generates HTML output for test results.

    Args:
        output_path: Path where the HTML report should be written.
        template_dir: Optional directory containing custom Jinja2 templates.
        default_group_name: Name to use for ungrouped tests.
        serializer: Optional custom ContextExportableObjSerializer instance.
    """

    def __init__(
        self,
        output_path: t.Union[str, Path] = "report.html",
        template_dir: t.Optional[t.Union[str, Path]] = None,
        default_group_name: str = "Ungrouped",
        serializer: t.Optional[ContextExportableObjSerializer] = None,
    ):
        self.output_path = Path(output_path)
        self.default_group_name = default_group_name
        self.serializer = serializer or ContextExportableObjSerializer()

        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            self.template_dir = Path(__file__).parent / "templates"

        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)), autoescape=jinja2.select_autoescape(["html", "xml"])
        )

    def report_results(
        self,
        results: dict[str | None, list[Result]] | list[_TaggedResult],
        *,
        render_context: bool = True,
        render_description: bool = True,
        render_traceback: bool = True,
        render_message: bool = True,
        serialize_context_exportable_obj: bool = True,
        **kwargs,
    ) -> None:
        """Generate HTML report of test results.

        Args:
            results: List of tagged test results.
            statistics: Overall statistics for the test run.
            render_context: Whether to include test context.
            render_description: Whether to include test descriptions.
            render_traceback: Whether to include tracebacks for errors.
            render_message: Whether to include test result messages.
            serialize_context_exportable_obj: Whether to serialize ContextExportableObj instances.
            asset_output_dir: Directory for saving serialized assets (not used for HTML, uses base64).
        """
        template = self.env.get_template("report.html")

        results = _normalize_results(results)

        grouped_results = []
        for group, test_results in _TaggedResult.group_by_group(results):
            group_name = group or self.default_group_name
            group_stats = ResultsStatistics.from_results([tr.result for tr in test_results])

            # Group by suite within the group
            suites: dict[str, list[dict]] = {}
            for tr in test_results:
                suite_name = tr.suite.name
                if suite_name not in suites:
                    suites[suite_name] = []

                # Serialize context if requested
                context = tr.result.context
                if serialize_context_exportable_obj and context is not None:
                    context = self.serializer.serialize_as_bytes(context)

                suites[suite_name].append(
                    {
                        "result": tr.result,
                        "suite_name": suite_name,
                        "serialized_context": context,
                    }
                )

            grouped_results.append(
                {
                    "name": group_name,
                    "statistics": group_stats,
                    "suites": suites,
                    "results": [
                        {
                            "result": tr.result,
                            "suite_name": tr.suite.name,
                            "serialized_context": self.serializer.serialize_as_bytes(tr.result.context)
                            if serialize_context_exportable_obj and tr.result.context is not None
                            else tr.result.context,
                        }
                        for tr in test_results
                    ],
                }
            )

        html_content = template.render(
            groups=grouped_results,
            statistics=ResultsStatistics.from_results([tr.result for tr in results]),
            status_color=STATUS_COLOR,
            render_context=render_context,
            render_description=render_description,
            render_traceback=render_traceback,
            render_message=render_message,
            serialize_context_exportable_obj=serialize_context_exportable_obj,
            version=__version__,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        self.output_path.write_text(html_content, encoding="utf-8")


def _normalize_results(
    results: dict[str | None, list[Result]] | list[_TaggedResult],
) -> list[_TaggedResult]:
    """Normalize results to a list of _TaggedResult instances.

    Args:
        results: Either a dict mapping group names to lists of Results,
                 or a list of _TaggedResult instances.

    Returns:
        A list of _TaggedResult instances.
    """
    if isinstance(results, dict):
        normalized_results = []
        for group, test_results in results.items():
            for result in test_results:
                assert result.suite_reference is not None, "Result must have suite_reference set"
                normalized_results.append(
                    _TaggedResult(suite=result.suite_reference, result=result, group=group, test=result.test_reference)
                )
        return normalized_results
    else:
        return results
