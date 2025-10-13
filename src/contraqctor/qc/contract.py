import typing as t

from .._typing import ErrorOnLoad
from ..contract.base import DataStream
from .base import Suite


class ContractTestSuite(Suite):
    """Test suite for validating data stream loading.

    Converts the output of DataStream.load_all() into a test suite that can report
    on loading errors and distinguish between true errors and expected/excluded issues.

    Attributes:
        loading_errors: List of tuples containing the data stream and the exception that occurred.
        exclude: Optional list of data streams to exclude from error reporting.

    Examples:
        ```python
        from contraqctor.contract import Dataset
        from contraqctor.qc.contract import ContractTestSuite
        from contraqctor.qc.base import Runner

        # Create and attempt to load a dataset
        dataset = create_complex_dataset()
        loading_errors = dataset.load_all(strict=False)

        # Create test suite to analyze loading errors
        suite = ContractTestSuite(
            loading_errors,
            exclude=[dataset["optional_stream"]]  # Exclude known problematic stream
        )

        # Run tests
        runner = Runner().add_suite(suite)
        results = runner.run_all_with_progress()
        ```
    """

    def __init__(self, loading_errors: list[ErrorOnLoad], exclude: t.Optional[list[DataStream]] = None):
        """Initialize the contract test suite.

        Args:
            loading_errors: List of tuples containing data streams and their loading errors.
            exclude: Optional list of data streams to exclude from error reporting.
                These will be reported as warnings instead of failures.
        """
        self.loading_errors = loading_errors
        self.exclude = exclude if exclude is not None else []

    def test_has_errors_on_load(self):
        """Check if any non-excluded data streams had loading errors."""
        errors = [err for err in self.loading_errors if err.data_stream not in self.exclude]
        if errors:
            str_errors = "\n".join([f"{err.data_stream.resolved_name}" for err in errors])
            return self.fail_test(
                None,
                f"The following DataStreams raised errors on load: \n {str_errors}",
                context={"errors": errors},
            )
        else:
            return self.pass_test(None, "All DataStreams loaded successfully")

    def test_has_excluded_as_warnings(self):
        """Check if any excluded data streams had loading errors and report as warnings."""
        warnings = [err for err in self.loading_errors if err.data_stream in self.exclude]
        if warnings:
            return self.warn_test(
                None,
                f"Found {len(warnings)} DataStreams that raised ignored errors on load.",
                context={"warnings": warnings},
            )
        else:
            return self.pass_test(None, "No excluded DataStreams raised errors on load")
