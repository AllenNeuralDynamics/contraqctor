"""Example demonstrating the HTML reporter functionality.

This example shows how to generate an HTML test report instead of
console output.
"""

from pathlib import Path

import contraqctor.qc as qc


class DataValidationSuite(qc.Suite):
    """Example test suite for data validation."""

    def __init__(self, data):
        self.data = data

    def test_data_not_empty(self):
        """Check that data is not empty."""
        if len(self.data) > 0:
            return self.pass_test(len(self.data), f"Data contains {len(self.data)} items")
        return self.fail_test(0, "Data is empty")

    def test_data_type(self):
        """Verify data type is correct."""
        if isinstance(self.data, list):
            return self.pass_test(type(self.data).__name__, "Data type is correct")
        return self.fail_test(type(self.data).__name__, "Expected list type")

    def test_data_values_positive(self):
        """Check that all values are positive."""
        negative_values = [x for x in self.data if x < 0]
        if not negative_values:
            return self.pass_test(True, "All values are positive")
        return self.fail_test(
            negative_values,
            f"Found {len(negative_values)} negative values",
            context={"negative_values": negative_values, "total_values": len(self.data)},
        )


class DataRangeSuite(qc.Suite):
    """Another test suite for data range validation."""

    def __init__(self, data):
        self.data = data

    def test_values_in_range(self):
        """Check if all values are within expected range."""
        out_of_range = [x for x in self.data if x < 0 or x > 100]
        if not out_of_range:
            return self.pass_test(True, "All values are within expected range [0, 100]")
        return self.fail_test(
            out_of_range,
            f"Found {len(out_of_range)} values out of range",
            context={"out_of_range": out_of_range, "expected_range": "[0, 100]"},
        )

    def test_mean_value(self):
        """Check if mean value is reasonable."""
        if not self.data:
            return self.skip_test("No data to calculate mean")

        mean = sum(self.data) / len(self.data)
        if 0 <= mean <= 50:
            return self.pass_test(mean, f"Mean value {mean:.2f} is within acceptable range")
        return self.warn_test(mean, f"Mean value {mean:.2f} is outside typical range", context={"mean": mean})


class PerformanceTimingSuite(qc.Suite):
    """Test suite for timing performance checks."""

    def __init__(self, execution_time):
        self.execution_time = execution_time

    def test_execution_time(self):
        """Check if execution time is acceptable."""
        if self.execution_time < 1.0:
            return self.pass_test(self.execution_time, "Execution time is excellent")
        elif self.execution_time < 2.0:
            return self.warn_test(
                self.execution_time,
                "Execution time is acceptable but could be improved",
                context={"threshold_ms": 1000, "actual_ms": self.execution_time * 1000},
            )
        else:
            return self.fail_test(
                self.execution_time,
                "Execution time is too slow",
                context={"max_allowed_ms": 2000, "actual_ms": self.execution_time * 1000},
            )


class PerformanceMemorySuite(qc.Suite):
    """Test suite for memory performance checks."""

    def __init__(self, memory_mb):
        self.memory_mb = memory_mb

    def test_memory_usage(self):
        """Check if memory usage is acceptable."""
        if self.memory_mb < 100:
            return self.pass_test(self.memory_mb, f"Memory usage {self.memory_mb} MB is excellent")
        elif self.memory_mb < 200:
            return self.warn_test(
                self.memory_mb,
                f"Memory usage {self.memory_mb} MB is higher than expected",
                context={"threshold_mb": 100, "actual_mb": self.memory_mb},
            )
        else:
            return self.fail_test(
                self.memory_mb,
                f"Memory usage {self.memory_mb} MB exceeds limit",
                context={"max_allowed_mb": 200, "actual_mb": self.memory_mb},
            )


def main():
    # Create test data
    sample_data = [1, 2, 3, 4, 5, -1, 7, 150]
    execution_time = 1.5
    memory_usage = 120

    # Create test suites
    data_validation_suite = DataValidationSuite(sample_data)
    data_range_suite = DataRangeSuite(sample_data)
    perf_timing_suite = PerformanceTimingSuite(execution_time)
    perf_memory_suite = PerformanceMemorySuite(memory_usage)

    # Create runner and add suites
    # Note: Multiple suites can be added to the same group
    runner = qc.Runner()
    runner.add_suite(data_validation_suite, "Data Quality")
    runner.add_suite(data_range_suite, "Data Quality")  # Same group, different suite
    runner.add_suite(perf_timing_suite, "Performance")
    runner.add_suite(perf_memory_suite, "Performance")  # Same group, different suite

    # Generate HTML report
    output_path = Path("test_report.html")
    html_reporter = qc.HtmlReporter(output_path)

    print("Running tests and generating HTML report...")
    results = runner.run_all_with_progress(reporter=html_reporter)

    print("\nHTML report generated:", output_path.absolute())
    print("Open it in your browser to view the results!")

    # You can still access results programmatically
    data_results = results["Data Quality"]
    perf_results = results["Performance"]

    print(f"\nData Quality: {len(data_results)} tests")
    print(f"Performance: {len(perf_results)} tests")

    # Also demonstrate console reporter (default)
    print("\n" + "=" * 80)
    print("Running with console reporter (default):")
    print("=" * 80 + "\n")

    runner2 = qc.Runner()
    runner2.add_suite(data_validation_suite, "Data Quality")
    runner2.add_suite(data_range_suite, "Data Quality")
    runner2.add_suite(perf_timing_suite, "Performance")
    runner2.add_suite(perf_memory_suite, "Performance")
    runner2.run_all_with_progress()


if __name__ == "__main__":
    main()
