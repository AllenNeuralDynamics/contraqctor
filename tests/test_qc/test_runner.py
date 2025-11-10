from unittest.mock import patch

import pytest

from contraqctor.qc.base import Result, ResultsStatistics, Runner, Status, Suite
from contraqctor.qc.reporters import ConsoleReporter


class MockSuite(Suite):
    """A mock test suite for testing the TestRunner."""

    def test_pass1(self):
        return self.pass_test("pass1")

    def test_pass2(self):
        return self.pass_test("pass2")

    def test_fail(self):
        return self.fail_test("fail")

    def test_warning(self):
        return self.warn_test("warning")

    def test_error(self):
        raise ValueError("Test error")

    def test_skip(self):
        return self.skip_test("Skip test")


class TestResultsStatistics:
    """Tests for the TestStatistics class."""

    def test_from_results(self):
        """Test creating statistics from results."""
        results = [
            Result(Status.PASSED, "pass1", "test_pass1", "MockSuite"),
            Result(Status.PASSED, "pass2", "test_pass2", "MockSuite"),
            Result(Status.FAILED, "fail", "test_fail", "MockSuite"),
            Result(Status.ERROR, None, "test_error", "MockSuite"),
            Result(Status.WARNING, "warning", "test_warning", "MockSuite"),
            Result(Status.SKIPPED, None, "test_skip", "MockSuite"),
        ]

        stats = ResultsStatistics.from_results(results)

        assert stats.passed == 2
        assert stats.failed == 1
        assert stats.error == 1
        assert stats.skipped == 1
        assert stats.warnings == 1
        assert stats.total == 6
        assert stats.pass_rate == 0.3333333333333333

    def test_get_item(self):
        """Test accessing statistics by test status."""
        stats = ResultsStatistics(passed=5, failed=3, error=2, skipped=1, warnings=0)

        assert stats[Status.PASSED] == 5
        assert stats[Status.FAILED] == 3
        assert stats[Status.ERROR] == 2
        assert stats[Status.SKIPPED] == 1
        assert stats[Status.WARNING] == 0

        with pytest.raises(KeyError):
            # Invalid key should raise KeyError
            stats["INVALID"]

    def test_status_summary(self):
        """Test getting a status summary string."""
        stats = ResultsStatistics(passed=5, failed=3, error=2, skipped=1, warnings=0)

        summary = stats.get_status_summary()
        assert summary == "P:5 F:3 E:2 S:1 W:0"


class TestRunner:
    """Tests for the TestRunner class."""

    def test_add_suite(self):
        """Test adding a test suite to the runner."""
        runner = Runner()
        suite = MockSuite()

        result = runner.add_suite(suite)
        assert result is runner
        assert None in runner.suites
        assert len(runner.suites[None]) == 1
        assert runner.suites[None][0] is suite

        suite2 = MockSuite()
        result = runner.add_suite(suite2, group="TestGroup")
        assert result is runner
        assert "TestGroup" in runner.suites
        assert len(runner.suites["TestGroup"]) == 1
        assert runner.suites["TestGroup"][0] is suite2

    @patch("rich.progress.Progress")
    def test_run_all_with_progress(self, mock_progress):
        """Test running all tests with progress reporting."""

        runner = Runner()
        suite1 = MockSuite()
        suite2 = MockSuite()

        runner.add_suite(suite1)  # Default group (None)
        runner.add_suite(suite2, group="TestGroup")

        with patch.object(ConsoleReporter, "report_results"):
            grouped_results = runner.run_all_with_progress()

            assert None in grouped_results
            assert "TestGroup" in grouped_results

            assert len(grouped_results[None]) == 6  # All tests in MockSuite
            assert len(grouped_results["TestGroup"]) == 6  # All tests in MockSuite

            all_results = grouped_results[None] + grouped_results["TestGroup"]
            stats = ResultsStatistics.from_results(all_results)

            assert stats[Status.PASSED] == 4
            assert stats[Status.FAILED] == 2
            assert stats[Status.ERROR] == 2
            assert stats[Status.SKIPPED] == 2
            assert stats[Status.WARNING] == 2

    def test_run_all(self):
        """Test running all tests without progress reporting."""

        runner = Runner()
        suite1 = MockSuite()
        suite2 = MockSuite()

        runner.add_suite(suite1)  # Default group (None)
        runner.add_suite(suite2, group="TestGroup")

        grouped_results = runner.run_all()

        assert None in grouped_results
        assert "TestGroup" in grouped_results

        assert len(grouped_results[None]) == 6  # All tests in MockSuite
        assert len(grouped_results["TestGroup"]) == 6  # All tests in MockSuite

        all_results = grouped_results[None] + grouped_results["TestGroup"]
        stats = ResultsStatistics.from_results(all_results)

        assert stats[Status.PASSED] == 4
        assert stats[Status.FAILED] == 2
        assert stats[Status.ERROR] == 2
        assert stats[Status.SKIPPED] == 2
        assert stats[Status.WARNING] == 2
