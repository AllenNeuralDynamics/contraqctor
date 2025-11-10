import abc
import contextvars
import dataclasses
import functools
import inspect
import itertools
import traceback
import typing as t
from contextlib import contextmanager
from enum import Enum, auto

import rich.markup
import rich.progress
from rich.console import Console

if t.TYPE_CHECKING:
    from contraqctor.qc.reporters import Reporter

_elevate_skippable = contextvars.ContextVar("elevate_skippable", default=False)
_elevate_warning = contextvars.ContextVar("elevate_warning", default=False)
_allow_null_as_pass_ctx = contextvars.ContextVar("allow_null_as_pass", default=False)


@contextmanager
def allow_null_as_pass(value: bool = True):
    """Context manager to control whether null results are allowed as pass.

    When enabled, tests that return None will be treated as passing tests
    rather than producing errors.

    Args:
        value: True to allow null results as passing, False otherwise.

    Examples:
        ```python
        from contraqctor.qc.base import allow_null_as_pass, Runner

        # Create a test suite with methods that return None
        class SimpleTestSuite(Suite):
            def test_basic_check(self):
                # This method returns None, which would normally be an error
                print("Performing a check")
                # No explicit return

        # Run with allow_null_as_pass to treat None returns as passing
        suite = SimpleTestSuite()
        runner = Runner().add_suite(suite)

        with allow_null_as_pass():
            # None returns will be treated as passing tests
            results = runner.run_all_with_progress()

        # Outside the context manager, None returns would cause errors
        ```
    """
    token = _allow_null_as_pass_ctx.set(value)
    try:
        yield
    finally:
        _allow_null_as_pass_ctx.reset(token)


@contextmanager
def elevated_skips(value: bool = True):
    """Context manager to control whether skipped tests are treated as failures.

    When enabled, skipped tests will be treated as failing tests rather than
    being merely marked as skipped.

    Args:
        value: True to elevate skipped tests to failures, False otherwise.

    Examples:
        ```python
        from contraqctor.qc.base import elevated_skips, Runner

        # Create a test suite with some skipped tests
        class FeatureTestSuite(Suite):
            def test_implemented_feature(self):
                return self.pass_test(True, "Feature works")

            def test_unimplemented_feature(self):
                return self.skip_test("Feature not yet implemented")

        # Run with elevated_skips to fail when tests are skipped
        suite = FeatureTestSuite()
        runner = Runner().add_suite(suite)

        with elevated_skips():
            # Skipped tests will be treated as failures
            results = runner.run_all_with_progress()

        # Without the context manager, skips are just marked as skipped
        ```
    """
    token = _elevate_skippable.set(value)
    try:
        yield
    finally:
        _elevate_skippable.reset(token)


@contextmanager
def elevated_warnings(value: bool = True):
    """Context manager to control whether warnings are treated as failures.

    When enabled, warning results will be treated as failing tests rather than
    just being marked as warnings.

    Args:
        value: True to elevate warnings to failures, False otherwise.

    Examples:
        ```python
        from contraqctor.qc.base import elevated_warnings, Runner

        # Create a test suite with warning conditions
        class PerformanceTestSuite(Suite):
            def test_response_time(self):
                response_time = measure_response()

                if response_time < 100:
                    return self.pass_test(response_time, "Response time acceptable")
                elif response_time < 200:
                    # This would normally be a warning
                    return self.warn_test(response_time, "Response time degraded")
                else:
                    return self.fail_test(response_time, "Response time unacceptable")

        # Run with elevated_warnings to fail on warnings
        suite = PerformanceTestSuite()
        runner = Runner().add_suite(suite)

        with elevated_warnings():
            # Warning results will be treated as failures
            # Useful in CI/CD pipelines where warnings should trigger failures
            results = runner.run_all_with_progress()
        ```
    """
    token = _elevate_warning.set(value)
    try:
        yield
    finally:
        _elevate_warning.reset(token)


class Status(Enum):
    """Enum representing possible test result statuses.

    Defines the different states a test can be in after execution.
    """

    PASSED = auto()
    FAILED = auto()
    ERROR = auto()
    SKIPPED = auto()
    WARNING = auto()

    def __str__(self) -> str:
        """Convert status to lowercase string representation.

        Returns:
            str: Lowercase string representation of the status.
        """
        return self.name.lower()


STATUS_COLOR = {
    Status.PASSED: "green",
    Status.FAILED: "red",
    Status.ERROR: "bright_red",
    Status.SKIPPED: "yellow",
    Status.WARNING: "dark_orange",
}


@t.runtime_checkable
class ITest(t.Protocol):
    """Protocol defining the interface for test functions.

    A test function should be callable, return a Result object or generator,
    and have a __name__ attribute.
    """

    def __call__(self) -> "Result" | t.Generator["Result", None, None]: ...

    @property
    def __name__(self) -> str: ...


TResult = t.TypeVar("TResult", bound=t.Any)


@dataclasses.dataclass(frozen=True)
class Result(t.Generic[TResult]):
    """Container for test execution results.

    Stores the outcome of a test execution including status, returned value,
    contextual information, and any exception details.

    Attributes:
        status: The status of the test execution.
        result: The value returned by the test.
        test_name: Name of the test that generated this result.
        suite_name: Name of the test suite containing the test.
        message: Optional message describing the test outcome.
        context: Optional contextual data for the test result.
        description: Optional description of the test.
        exception: Optional exception that occurred during test execution.
        traceback: Optional traceback string if an exception occurred.
        test_reference: Optional reference to the test function.
        suite_reference: Optional reference to the suite that ran this test.
    """

    status: Status
    result: TResult
    test_name: str
    suite_name: str
    message: t.Optional[str] = None
    context: t.Optional[t.Any] = dataclasses.field(default=None, repr=False)
    description: t.Optional[str] = dataclasses.field(default=None, repr=False)
    exception: t.Optional[Exception] = dataclasses.field(default=None, repr=False)
    traceback: t.Optional[str] = dataclasses.field(default=None, repr=False)
    test_reference: t.Optional[ITest] = dataclasses.field(default=None, repr=False)
    suite_reference: t.Optional["Suite"] = dataclasses.field(default=None, repr=False)


def implicit_pass(func: t.Callable[..., t.Any]) -> t.Callable[..., Result]:
    """Decorator to automatically convert non-Result return values to passing results.

    If a test method returns something other than a Result object, this decorator
    will wrap it in a passing Result with the original return value.

    Args:
        func: The function to decorate.

    Returns:
        callable: Decorated function that ensures Result objects are returned.

    Raises:
        TypeError: If the decorated function is not a method of a Suite object.

    Examples:
        ```python
        from contraqctor.qc.base import Suite, implicit_pass

        class SimplifiedTestSuite(Suite):
            # Regular test method that explicitly returns a Result object
            def test_regular_approach(self):
                value = 42
                return self.pass_test(value, "Explicitly created pass result")

            # Using the decorator to simplify - just return the value
            @implicit_pass
            def test_implicit_approach(self):
                # This will automatically be wrapped in a passing Result
                return 42

            # The decorator handles different return types
            @implicit_pass
            def test_with_dict(self):
                # This dictionary will be wrapped in a passing Result
                return {"status": "ok", "value": 100}
        ```
    """

    @functools.wraps(func)
    def wrapper(self: Suite, *args: t.Any, **kwargs: t.Any) -> Result:
        """Wrapper function to ensure the decorated function returns a Result."""
        result = func(self, *args, **kwargs)

        if isinstance(result, Result):
            return result

        # Just in case someone tries to do funny stuff
        if isinstance(self, Suite):
            return self.pass_test(result=result, message=f"Auto-converted return value: {result}")
        else:
            # Not in a Suite - can't convert properly
            raise TypeError(
                f"The auto_test decorator was used on '{func.__name__}' in a non-Suite "
                f"class ({self.__class__.__name__}). This is not supported."
            )

    return wrapper


class Suite(abc.ABC):
    """Base class for test suites.

    Provides the core functionality for defining, running, and reporting on tests.
    All test suites should inherit from this class and implement test methods
    that start with 'test'.

    Examples:
        ```python
        from contraqctor.qc.base import Suite

        class MyTestSuite(Suite):
            \"\"\"Test suite for validating my component.\"\"\"

            def __init__(self, component):
                self.component = component

            def test_has_required_property(self):
                if hasattr(self.component, "required_property"):
                    return self.pass_test(True, "Component has required property")
                else:
                    return self.fail_test(False, "Component is missing required property")

            def test_performs_calculation(self):
                try:
                    result = self.component.calculate(10)
                    if result == 20:
                        return self.pass_test(result, "Calculation correct")
                    else:
                        return self.fail_test(result, f"Expected 20 but got {result}")
                except Exception as e:
                    return self.fail_test(None, f"Calculation failed: {str(e)}")
        ```
    """

    def get_tests(self) -> t.Generator[ITest, None, None]:
        """Find all methods starting with 'test'.

        Yields:
            ITest: Test methods found in the suite.
        """
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("test"):
                yield method

    @property
    def description(self) -> t.Optional[str]:
        """Get the description of the test suite from its docstring.

        Returns:
            Optional[str]: The docstring of the class, or None if not available.
        """
        return getattr(self, "__doc__", None)

    @property
    def name(self) -> str:
        """Get the name of the test suite.

        Returns:
            str: The name of the test suite class.
        """
        return self.__class__.__name__

    def _get_caller_info(self):
        """Get information about the calling function.

        Retrieves the name and docstring of the test method that called
        one of the result-generating methods.

        Returns:
            tuple: Containing the calling function name and its docstring.

        Raises:
            RuntimeError: If unable to retrieve the calling frame information.
        """
        if (f := inspect.currentframe()) is None:
            raise RuntimeError("Unable to retrieve the calling frame.")
        if (frame := f.f_back) is None:
            raise RuntimeError("Unable to retrieve the calling frame.")
        if (frame := frame.f_back) is None:  # Need to go one frame further as we're in a helper
            raise RuntimeError("Unable to retrieve the calling frame.")

        calling_func_name = frame.f_code.co_name
        description = getattr(frame.f_globals.get(calling_func_name), "__doc__", None)

        return calling_func_name, description

    @t.overload
    def pass_test(self) -> Result:
        """Create a passing test result with no result value.

        Returns:
            Result: A Result object with PASSED status and None result value.
        """
        ...

    @t.overload
    def pass_test(self, result: t.Any) -> Result:
        """Create a passing test result with a result value.

        Args:
            result: The value to include in the test result.

        Returns:
            Result: A Result object with PASSED status and the provided result value.
        """
        ...

    @t.overload
    def pass_test(self, result: t.Any, message: str) -> Result:
        """Create a passing test result with a result value and message.

        Args:
            result: The value to include in the test result.
            message: Message describing why the test passed.

        Returns:
            Result: A Result object with PASSED status, result value, and message.
        """
        ...

    @t.overload
    def pass_test(self, result: t.Any, *, context: t.Any) -> Result:
        """Create a passing test result with a result value and context data.

        Args:
            result: The value to include in the test result.
            context: Contextual data for the test result.

        Returns:
            Result: A Result object with PASSED status, result value, and context.
        """
        ...

    @t.overload
    def pass_test(self, result: t.Any, message: str, *, context: t.Any) -> Result:
        """Create a passing test result with a result value, message, and context data.

        Args:
            result: The value to include in the test result.
            message: Message describing why the test passed.
            context: Contextual data for the test result.

        Returns:
            Result: A Result object with PASSED status, result value, message, and context.
        """
        ...

    def pass_test(
        self, result: t.Any = None, message: t.Optional[str] = None, *, context: t.Optional[t.Any] = None
    ) -> Result:
        """Create a passing test result.

        Args:
            result: The value to include in the test result.
            message: Optional message describing why the test passed.
            context: Optional contextual data for the test result.

        Returns:
            Result: A Result object with PASSED status.
        """
        calling_func_name, description = self._get_caller_info()

        return Result(
            status=Status.PASSED,
            result=result,
            test_name=calling_func_name,
            suite_name=self.name,
            message=message,
            context=context,
            description=description,
        )

    @t.overload
    def warn_test(self) -> Result:
        """Create a warning test result with no result value.

        Returns:
            Result: A Result object with WARNING status (or FAILED if warnings are elevated)
                   and None result value.
        """
        ...

    @t.overload
    def warn_test(self, result: t.Any) -> Result:
        """Create a warning test result with a result value.

        Args:
            result: The value to include in the test result.

        Returns:
            Result: A Result object with WARNING status (or FAILED if warnings are elevated)
                   and the provided result value.
        """
        ...

    @t.overload
    def warn_test(self, result: t.Any, message: str) -> Result:
        """Create a warning test result with a result value and message.

        Args:
            result: The value to include in the test result.
            message: Message describing the warning.

        Returns:
            Result: A Result object with WARNING status (or FAILED if warnings are elevated),
                   result value, and message.
        """
        ...

    @t.overload
    def warn_test(self, result: t.Any, *, context: t.Any) -> Result:
        """Create a warning test result with a result value and context data.

        Args:
            result: The value to include in the test result.
            context: Contextual data for the test result.

        Returns:
            Result: A Result object with WARNING status (or FAILED if warnings are elevated),
                   result value, and context.
        """
        ...

    @t.overload
    def warn_test(self, result: t.Any, message: str, *, context: t.Any) -> Result:
        """Create a warning test result with a result value, message, and context data.

        Args:
            result: The value to include in the test result.
            message: Message describing the warning.
            context: Contextual data for the test result.

        Returns:
            Result: A Result object with WARNING status (or FAILED if warnings are elevated),
                   result value, message, and context.
        """
        ...

    def warn_test(
        self, result: t.Any = None, message: t.Optional[str] = None, *, context: t.Optional[t.Any] = None
    ) -> Result:
        """Create a warning test result.

        Creates a result with WARNING status, or FAILED if warnings are elevated.

        Args:
            result: The value to include in the test result.
            message: Optional message describing the warning.
            context: Optional contextual data for the test result.

        Returns:
            Result: A Result object with WARNING or FAILED status.
        """
        calling_func_name, description = self._get_caller_info()

        return Result(
            status=Status.WARNING if not _elevate_warning.get() else Status.FAILED,
            result=result,
            test_name=calling_func_name,
            suite_name=self.name,
            message=message,
            context=context,
            description=description,
        )

    @t.overload
    def fail_test(self) -> Result:
        """Create a failing test result with no result value.

        Returns:
            Result: A Result object with FAILED status and None result value.
        """
        ...

    @t.overload
    def fail_test(self, result: t.Any) -> Result:
        """Create a failing test result with a result value.

        Args:
            result: The value to include in the test result.

        Returns:
            Result: A Result object with FAILED status and the provided result value.
        """
        ...

    @t.overload
    def fail_test(self, result: t.Any, message: str) -> Result:
        """Create a failing test result with a result value and message.

        Args:
            result: The value to include in the test result.
            message: Message describing why the test failed.

        Returns:
            Result: A Result object with FAILED status, result value, and message.
        """
        ...

    @t.overload
    def fail_test(self, result: t.Any, message: str, *, context: t.Any) -> Result:
        """Create a failing test result with a result value, message, and context data.

        Args:
            result: The value to include in the test result.
            message: Message describing why the test failed.
            context: Contextual data for the test result.

        Returns:
            Result: A Result object with FAILED status, result value, message, and context.
        """
        ...

    def fail_test(
        self, result: t.Optional[t.Any] = None, message: t.Optional[str] = None, *, context: t.Optional[t.Any] = None
    ) -> Result:
        """Create a failing test result.

        Args:
            result: The value to include in the test result.
            message: Optional message describing why the test failed.
            context: Optional contextual data for the test result.

        Returns:
            Result: A Result object with FAILED status.
        """
        calling_func_name, description = self._get_caller_info()

        return Result(
            status=Status.FAILED,
            result=result,
            test_name=calling_func_name,
            suite_name=self.name,
            message=message,
            context=context,
            description=description,
        )

    @t.overload
    def skip_test(self) -> Result:
        """Create a skipped test result with no message.

        Returns:
            Result: A Result object with SKIPPED status (or FAILED if skips are elevated)
                   and None result value.
        """
        ...

    @t.overload
    def skip_test(self, message: str) -> Result:
        """Create a skipped test result with a message.

        Args:
            message: Message explaining why the test was skipped.

        Returns:
            Result: A Result object with SKIPPED status (or FAILED if skips are elevated)
                   and the provided message.
        """
        ...

    @t.overload
    def skip_test(self, message: str, *, context: t.Any) -> Result:
        """Create a skipped test result with a message and context data.

        Args:
            message: Message explaining why the test was skipped.
            context: Contextual data for the test result.

        Returns:
            Result: A Result object with SKIPPED status (or FAILED if skips are elevated),
                   message, and context.
        """
        ...

    def skip_test(self, message: t.Optional[str] = None, *, context: t.Optional[t.Any] = None) -> Result:
        """Create a skipped test result.

        Creates a result with SKIPPED status, or FAILED if skips are elevated.

        Args:
            message: Optional message explaining why the test was skipped.
            context: Optional contextual data for the test result.

        Returns:
            Result: A Result object with SKIPPED or FAILED status.
        """
        calling_func_name, description = self._get_caller_info()
        return Result(
            status=Status.SKIPPED if not _elevate_skippable.get() else Status.FAILED,
            result=None,
            test_name=calling_func_name,
            suite_name=self.name,
            message=message,
            context=context,
            description=description,
        )

    def setup(self) -> None:
        """Run before each test method.

        This method can be overridden by subclasses to implement
        setup logic that runs before each test.
        """
        pass

    def teardown(self) -> None:
        """Run after each test method.

        This method can be overridden by subclasses to implement
        teardown logic that runs after each test.
        """
        pass

    def _process_test_result(
        self, result: t.Optional[Result], test_method: ITest, test_name: str, description: t.Optional[str]
    ) -> Result:
        """Process and validate a test result.

        Ensures that the result is properly formatted and contains all necessary information.

        Args:
            result: The result returned by the test method.
            test_method: The test method that produced the result.
            test_name: The name of the test method.
            description: The description of the test method.

        Returns:
            Result: A properly formatted Result object.
        """
        if result is None and _allow_null_as_pass_ctx.get():
            result = self.pass_test(None, "Test passed with <null> result implicitly.")

        if isinstance(result, Result):
            result = dataclasses.replace(
                result,
                test_reference=test_method,
                suite_reference=self,
                test_name=test_name,
                suite_name=self.name,
                description=description,
            )

            return result

        error_msg = f"Test method '{test_name}' must return a TestResult instance or generator, but got {type(result).__name__}."
        return Result(
            status=Status.ERROR,
            result=result,
            test_name=test_name,
            suite_name=self.name,
            description=description,
            message=error_msg,
            exception=TypeError(error_msg),
            test_reference=test_method,
            suite_reference=self,
        )

    def run_test(self, test_method: ITest) -> t.Generator[Result, None, None]:
        """Run a single test method and yield its results.

        Handles setup, test execution, result processing, and teardown.

        Args:
            test_method: The test method to run.

        Yields:
            Result: Result objects produced by the test method.
        """
        test_name = test_method.__name__
        suite_name = self.name
        test_description = getattr(test_method, "__doc__", None)

        try:
            self.setup()
            result = test_method()
            if inspect.isgenerator(result):
                for sub_result in result:
                    yield self._process_test_result(sub_result, test_method, test_name, test_description)
            else:
                yield self._process_test_result(result, test_method, test_name, test_description)
        except Exception as e:
            tb = traceback.format_exc()
            yield Result(
                status=Status.ERROR,
                result=None,
                test_name=test_name,
                suite_name=suite_name,
                description=test_description,
                message=f"Error during test execution: {str(e)}",
                exception=e,
                traceback=tb,
                test_reference=test_method,
                suite_reference=self,
            )
        finally:
            self.teardown()

    def run_all(self) -> t.Generator[Result, None, None]:
        """Run all test methods in the suite.

        Finds all test methods and runs them in sequence.

        Yields:
            Result: Result objects produced by all test methods.
        """

        for test in self.get_tests():
            yield from self.run_test(test)


@dataclasses.dataclass
class ResultsStatistics:
    """Statistics about test results.

    Aggregates counts of test results by status and provides methods for
    calculating statistics like pass rate.

    Attributes:
        passed: Number of passed tests.
        failed: Number of failed tests.
        error: Number of tests that produced errors.
        skipped: Number of skipped tests.
        warnings: Number of tests with warnings.
    """

    passed: int
    failed: int
    error: int
    skipped: int
    warnings: int

    @property
    def total(self) -> int:
        """Get the total number of tests.

        Returns:
            int: Sum of all test result counts.
        """
        return self.passed + self.failed + self.error + self.skipped + self.warnings

    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate.

        Returns:
            float: Ratio of passed tests to total tests, or 0 if no tests.
        """
        total = self.total
        return (self.passed / total) if total > 0 else 0.0

    def get_status_summary(self) -> str:
        """Generate a compact string summary of result counts.

        Returns:
            str: Summary string with counts for each status type.
        """
        return f"P:{self[Status.PASSED]} F:{self[Status.FAILED]} E:{self[Status.ERROR]} S:{self[Status.SKIPPED]} W:{self[Status.WARNING]}"

    def __getitem__(self, item: Status) -> int:
        """Get the count for a specific status.

        Args:
            item: The status to get the count for.

        Returns:
            int: Number of tests with the specified status.

        Raises:
            KeyError: If an invalid status is provided.
        """
        if item == Status.PASSED:
            return self.passed
        elif item == Status.FAILED:
            return self.failed
        elif item == Status.ERROR:
            return self.error
        elif item == Status.SKIPPED:
            return self.skipped
        elif item == Status.WARNING:
            return self.warnings
        else:
            raise KeyError(f"Invalid key: {item}. Valid keys are: {list(Status)}")

    @classmethod
    def from_results(cls, results: t.List[Result]) -> "ResultsStatistics":
        """Create statistics from a list of test results.

        Args:
            results: List of test results to analyze.

        Returns:
            ResultsStatistics: Statistics object summarizing the results.
        """
        stats = {status: sum(1 for r in results if r.status == status) for status in Status}
        return cls(
            passed=stats[Status.PASSED],
            failed=stats[Status.FAILED],
            error=stats[Status.ERROR],
            skipped=stats[Status.SKIPPED],
            warnings=stats[Status.WARNING],
        )

    def __add__(self, other: "ResultsStatistics") -> "ResultsStatistics":
        """Add two ResultsStatistics objects together.

        Args:
            other: Another ResultsStatistics object to add.

        Returns:
            ResultsStatistics: New object with combined statistics.
        """
        return ResultsStatistics(
            passed=self.passed + other.passed,
            failed=self.failed + other.failed,
            error=self.error + other.error,
            skipped=self.skipped + other.skipped,
            warnings=self.warnings + other.warnings,
        )


@dataclasses.dataclass(frozen=True)
class _Tagged(abc.ABC):
    """Abstract base class for tagged items.

    Used internally to associate suites, groups, and tests for organization
    and filtering of test executions and results.

    Attributes:
        suite: The test suite this item belongs to.
        group: Optional group name this item belongs to.
    """

    suite: Suite
    group: t.Optional[str]

    @classmethod
    def group_by_suite(cls, values: t.Iterable[t.Self]) -> t.Generator[t.Tuple[Suite, t.List[t.Self]], None, None]:
        """Group items by their associated test suite.

        Args:
            values: Iterable of tagged items to group.

        Yields:
            Tuple containing a suite and all items associated with that suite.
        """
        for suite, group in itertools.groupby(values, key=lambda x: x.suite):
            yield suite, list(group)

    @classmethod
    def group_by_group(
        cls, values: t.Iterable[t.Self]
    ) -> t.Generator[t.Tuple[t.Optional[str], t.List[t.Self]], None, None]:
        """Group items by their associated group name.

        Args:
            values: Iterable of tagged items to group.

        Yields:
            Tuple containing a group name and all items associated with that group.
        """
        for group, group_items in itertools.groupby(values, key=lambda x: x.group):
            yield group, list(group_items)

    @classmethod
    def get_by_group(cls, values: t.Iterable[t.Self], group: t.Optional[str]) -> t.List[t.Self]:
        """Get all items in a specific group.

        Args:
            values: Iterable of tagged items.
            group: Group name to filter by.

        Returns:
            List of tagged items in the specified group.
        """
        return [item for item in values if item.group == group]

    @classmethod
    def get_by_suite(cls, values: t.Iterable[t.Self], suite: Suite) -> t.List[t.Self]:
        """Get all items in a specific suite.

        Args:
            values: Iterable of tagged items.
            suite: Suite to filter by.

        Returns:
            List of tagged items in the specified suite.
        """
        return [item for item in values if item.suite == suite]


@dataclasses.dataclass(frozen=True)
class _TaggedResult(_Tagged):
    """Container for a test result with suite and group information.

    Associates a test result with its test, suite, and group for organization
    and reporting.

    Attributes:
        result: The test execution result.
        test: Optional reference to the test that produced this result.
    """

    result: Result
    test: t.Optional[ITest]


@dataclasses.dataclass(frozen=True)
class _TaggedTest(_Tagged):
    """Container for a test with suite and group information.

    Associates a test with its suite and group for organization and execution.

    Attributes:
        test: The test function to execute.
    """

    test: ITest


class Runner:
    """Test runner for executing suites and reporting results.

    Handles executing test suites, collecting results, and generating reports.

    Attributes:
        suites: Dictionary mapping group names to lists of test suites.
        _results: Optional dictionary of collected test results by group.

    Examples:
        ```python
        from contraqctor.qc.base import Runner

        # Create test suites
        suite1 = MyTestSuite(component1)
        suite2 = AnotherTestSuite(component2)
        suite3 = YetAnotherTestSuite(component2)

        # Create runner and add suites with group names
        runner = Runner()
        runner.add_suite(suite1, "Component Tests")
        runner.add_suite(suite2, "Integration Tests")
        runner.add_suite(suite3, "Integration Tests")

        # Run all tests with progress display
        results = runner.run_all_with_progress()

        # Access results by group
        component_results = results["Component Tests"]
        ```
    """

    _DEFAULT_TEST_GROUP = "Ungrouped"

    def __init__(self, console: t.Optional[Console] = None):
        """Initialize the test runner.

        Args:
            console: Optional rich Console instance for progress display.
        """
        self.suites: t.Dict[t.Optional[str], t.List[Suite]] = {}
        self._results: t.Optional[t.List[_TaggedResult]] = None
        self._console = console or Console()

    @t.overload
    def add_suite(self, suite: Suite) -> t.Self:
        """Add a test suite to the runner without specifying a group.

        Args:
            suite: Test suite to add.

        Returns:
            Runner: Self for method chaining.
        """

    @t.overload
    def add_suite(self, suite: Suite, group: str) -> t.Self:
        """Add a test suite to the runner with a specific group.

        Args:
            suite: Test suite to add.
            group: Group name for organizing suites.

        Returns:
            Runner: Self for method chaining.
        """

    def add_suite(self, suite: Suite, group: t.Optional[str] = None) -> t.Self:
        """Add a test suite to the runner.

        Args:
            suite: Test suite to add.
            group: Optional group name for organizing suites. Defaults to None.

        Returns:
            Runner: Self for method chaining.

        Examples:
            ```python
            runner = Runner()

            # Add a suite without a group
            runner.add_suite(BasicSuite())

            # Add suites with named groups for organization
            runner.add_suite(DataSuite(), "Data Validation")
            runner.add_suite(VisualizationSuite(), "Data Validation")
            runner.add_suite(ApiSuite(), "API Tests")
            ```
        """
        self._update_suites(suite, group)
        return self

    def _update_suites(self, suite: Suite, group: t.Optional[str] = None) -> t.Self:
        """Add a suite to the specified group.

        Args:
            suite: Test suite to add.
            group: Optional group name. If None, uses the default group.

        Returns:
            Runner: Self for method chaining.
        """
        if group in self.suites:
            self.suites[group].append(suite)
        else:
            self.suites[group] = [suite]
        return self

    def _collect_tests(self) -> t.List[_TaggedTest]:
        """Collect all tests across all suites and groups.

        Iterates through all registered suites and groups, collecting all test methods
        and tagging them with their source suite and group.

        Returns:
            List of tagged tests ready for execution.
        """
        tests: t.List[_TaggedTest] = []
        for group, suites in self.suites.items():
            for suite in suites:
                for test in suite.get_tests():
                    tests.append(_TaggedTest(suite=suite, group=group, test=test))
        return tests

    def _render_status_bar(self, stats: ResultsStatistics, bar_width: int = 20) -> str:
        """Render a colored status bar representing test result proportions.

        Args:
            stats: Statistics to render.
            bar_width: Width of the status bar in characters.

        Returns:
            str: Rich-formatted string containing the status bar.
        """
        total = stats.total
        if total == 0:
            return ""

        status_bar = ""
        allocated_width = 0
        statuses_with_counts = [(status, stats[status]) for status in Status if stats[status]]

        for idx, (status, count) in enumerate(statuses_with_counts):
            color = STATUS_COLOR[status]

            if idx == len(statuses_with_counts) - 1:
                # Last segment gets remaining width to avoid rounding errors
                segment_width = bar_width - allocated_width
            else:
                # Round to nearest integer for better distribution
                segment_width = round(bar_width * (count / total))

            allocated_width += segment_width
            status_bar += f"[{color}]{'█' * segment_width}[/{color}]"

        return status_bar

    def _setup_progress_display(self, suite_name_width: int, test_name_width: int = 20) -> t.List:
        """Configure the progress display format.

        Creates a list of components for the rich progress display with proper spacing
        and column widths.

        Args:
            suite_name_width: Width to allocate for suite names.
            test_name_width: Width to allocate for test names.

        Returns:
            List of progress display format components.
        """
        return [
            f"[progress.description]{{task.description:<{suite_name_width + test_name_width + 5}}}",
            rich.progress.BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            rich.progress.TimeElapsedColumn(),
        ]

    def _run_suite_tests(
        self,
        progress: rich.progress.Progress,
        suite: Suite,
        tests: t.List[ITest],
        suite_name_width: int,
        test_name_width: int,
        total_task: rich.progress.TaskID,
        group_task: rich.progress.TaskID,
    ) -> t.List[Result]:
        """Run all tests for a specific suite with progress reporting.

        Args:
            progress: Progress display instance.
            suite: Test suite to run.
            tests: List of test methods to run.
            suite_name_width: Width allocated for suite names.
            test_name_width: Width allocated for test names.
            total_task: Task ID for the overall progress bar.
            group_task: Task ID for the group progress bar.

        Returns:
            List of test results.
        """
        suite_name = suite.name
        suite_task = progress.add_task(f"[cyan]{suite_name}".ljust(suite_name_width + 5), total=len(tests))
        suite_results = []

        for test in tests:
            test_name = test.__name__
            test_desc = f"[cyan]{suite_name:<{suite_name_width}} • {test_name:<{test_name_width}}"
            progress.update(suite_task, description=test_desc)

            test_results = list(suite.run_test(test))
            suite_results.extend(test_results)

            progress.advance(total_task)
            progress.advance(group_task)
            progress.advance(suite_task)

        if tests:
            self._update_suite_progress(progress, suite_task, suite_name, suite_results, suite_name_width)

        return suite_results

    def _update_suite_progress(
        self,
        progress: rich.progress.Progress,
        suite_task: rich.progress.TaskID,
        suite_name: str,
        suite_results: t.List[Result],
        suite_name_width: int,
        bar_width: int = 20,
    ) -> None:
        """Update progress display with suite results summary.

        Calculates statistics for a suite's test results and updates the progress
        display with a visual status bar and summary statistics.

        Args:
            progress: Progress display instance.
            suite_task: Task ID for the suite progress bar.
            suite_name: Name of the suite.
            suite_results: List of test results.
            suite_name_width: Width allocated for suite names.
            bar_width: Width of status bars in characters.
        """
        stats = ResultsStatistics.from_results(suite_results)
        status_bar = self._render_status_bar(stats, bar_width)
        summary_line = f"[cyan]{suite_name:<{suite_name_width}} | {status_bar} | {stats.get_status_summary()}"
        progress.update(suite_task, description=summary_line)

    def run_all(self) -> t.Dict[t.Optional[str], t.List[Result]]:
        """Run all tests in all suites without progress display.

        Executes all tests and collects results without visual progress reporting.

        Returns:
            Dict[Optional[str], List[Result]]: Results grouped by test group name.
        """
        collected_tests = self._collect_tests()
        collected_results: t.List[_TaggedResult] = []

        for group, tests_in_group in _TaggedTest.group_by_group(collected_tests):
            for suite, tests_in_suite in _TaggedTest.group_by_suite(tests_in_group):
                results: t.List[Result] = []
                for test in tests_in_suite:
                    results.extend(suite.run_test(test.test))
                for result in results:
                    collected_results.append(
                        _TaggedResult(suite=suite, group=group, result=result, test=result.test_reference)
                    )

        self._results = collected_results

        out: t.Dict[t.Optional[str], t.List[Result]] = {}
        for group, grouped_results in _TaggedResult.group_by_group(collected_results):
            out[group] = [tagged_result.result for tagged_result in grouped_results]
        return out

    def run_all_with_progress(
        self,
        *,
        reporter: t.Optional["Reporter"] = None,
        **reporter_kwargs: t.Any,
    ) -> t.Dict[t.Optional[str], t.List[Result]]:
        """Run all tests in all suites with a rich progress display.

        Executes all tests with a visual progress bar and detailed reporting
        of test outcomes.

        Args:
            reporter: Optional reporter to use for output. If None, uses ConsoleReporter.

        Returns:
            Dict[Optional[str], List[Result]]: Results grouped by test group name.

        Examples:
            ```python
            from contraqctor.qc.base import Runner
            from contraqctor.qc.reporters import ConsoleReporter, HtmlReporter

            runner = Runner()
            runner.add_suite(DataValidationSuite(), "Validation")
            runner.add_suite(PerformanceSuite(), "Performance")

            # Run with default console reporter
            results = runner.run_all_with_progress()

            # Run with HTML reporter
            html_reporter = HtmlReporter("test_report.html")
            results = runner.run_all_with_progress(reporter=html_reporter)

            # Run with simplified output (no context or traceback)
            results = runner.run_all_with_progress(
                render_context=False,
                render_traceback=False
            )

            # Check if any tests failed
            all_passed = all(
                result.status == Status.PASSED
                for group_results in results.values()
                for result in group_results
            )
            ```
        """
        from contraqctor.qc.reporters import ConsoleReporter

        if reporter is None:
            reporter = ConsoleReporter(console=self._console)

        collected_tests = self._collect_tests()
        total_test_count = len(collected_tests)

        suite_name_lengths = [len(suite.name) for suite, _ in _TaggedTest.group_by_suite(collected_tests)]
        group_lengths = [
            len(group) + 2 for group, _ in _TaggedTest.group_by_group(collected_tests) if group is not None
        ]
        full_name_width = max(suite_name_lengths + group_lengths) if suite_name_lengths else 10
        test_name_width = 20
        bar_width = 20

        progress_format = [
            f"[progress.description]{{task.description:<{full_name_width + test_name_width + 5}}}",
            rich.progress.BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            rich.progress.TimeElapsedColumn(),
        ]

        with rich.progress.Progress(*progress_format, console=self._console) as progress:
            total_task = progress.add_task(
                "[bold green]TOTAL PROGRESS".ljust(full_name_width + test_name_width + 5), total=total_test_count
            )

            collected_results: t.List[_TaggedResult] = []
            for group, tests_in_group in _TaggedTest.group_by_group(collected_tests):
                _title = (
                    rich.markup.escape(f"[{group}]") if group else rich.markup.escape(f"[{self._DEFAULT_TEST_GROUP}]")
                )
                group_task = progress.add_task(
                    f"[honeydew2]{_title}".ljust(full_name_width + test_name_width + 5),
                    total=len(tests_in_group),
                )
                for suite, tests_in_suite in _TaggedTest.group_by_suite(tests_in_group):
                    results = self._run_suite_tests(
                        progress,
                        suite,
                        [t.test for t in tests_in_suite],
                        full_name_width,
                        test_name_width,
                        total_task,
                        group_task,
                    )
                    for result in results:
                        collected_results.append(
                            _TaggedResult(suite=suite, group=group, result=result, test=result.test_reference)
                        )

                if len(_TaggedResult.get_by_group(collected_results, group)) > 0:
                    group_results = [
                        tagged_result.result for tagged_result in _TaggedResult.get_by_group(collected_results, group)
                    ]
                    group_stats = ResultsStatistics.from_results(group_results)
                    group_status_bar = self._render_status_bar(group_stats, bar_width)
                    padding_width = max(0, full_name_width - len(self._rich_unscape(_title)))
                    group_line = f"[honeydew2]{_title}{' ' * padding_width} | {group_status_bar} | {group_stats.get_status_summary()}"
                    progress.update(group_task, description=group_line)

            if total_test_count > 0:
                total_stats = ResultsStatistics.from_results(
                    [tagged_result.result for tagged_result in collected_results]
                )
                total_status_bar = self._render_status_bar(total_stats, bar_width)

                _title = "TOTAL PROGRESS"
                padding_width = max(0, full_name_width - len(_title))
                total_line = f"[bold green]{_title}{' ' * padding_width} | {total_status_bar} | {total_stats.get_status_summary()}"
                progress.update(total_task, description=total_line)

        self._results = collected_results
        if self._results:
            reporter.report_results(
                self._results,
                **reporter_kwargs,
            )

        out: t.Dict[t.Optional[str], t.List[Result]] = {}
        for group, grouped_results in _TaggedResult.group_by_group(collected_results):
            out[group] = [tagged_result.result for tagged_result in grouped_results]
        return out

    @staticmethod
    def _rich_unscape(value: str) -> str:
        """Unescape rich markup in a string.

        Args:
            value: String containing rich markup to unescape.

        Returns:
            str: Unescaped string.
        """
        return value.replace(r"\[", "[").replace(r"\]", "]")
