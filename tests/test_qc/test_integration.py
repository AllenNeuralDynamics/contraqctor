from unittest.mock import MagicMock, patch

import pytest

from contraqctor.qc.base import (
    ResultsStatistics,
    Runner,
    Status,
    Suite,
    allow_null_as_pass,
)

from contraqctor.qc.reporters import ConsoleReporter

class MockHarpDevice:
    def __init__(self, name, data=None):
        self.name = name
        self._data = data or {}

    def __getitem__(self, key):
        return self._data.get(key, MagicMock())


class ExampleBoardTestSuite(Suite):
    def __init__(self, device):
        self.device = device

    def test_whoami(self):
        """Check if the WhoAmI value exists."""
        return self.pass_test(1234, "WhoAmI value exists")

    def test_check_name(self):
        """Check if the device name is valid."""
        if self.device.name == "ValidDevice":
            return self.pass_test(self.device.name)
        else:
            return self.fail_test(self.device.name, "Invalid device name")

    def test_with_generator(self):
        """Test that yields multiple results."""
        yield self.pass_test("First check")

        if self.device.name == "ValidDevice":
            yield self.pass_test(self.device.name, "Device name is valid")
        else:
            yield self.fail_test(self.device.name, "Invalid device name")

    def test_that_returns_none(self):
        """Test that returns None."""
        # This would normally return None, which is invalid by default
        return None


@pytest.fixture
def mock_device():
    return MockHarpDevice("ValidDevice")


@pytest.fixture
def invalid_device():
    return MockHarpDevice("InvalidDevice")


class TestIntegration:
    """Integration tests for the QC module."""

    def test_full_test_flow(self, mock_device):
        """Test a complete flow from creating suites to running tests."""

        suite = ExampleBoardTestSuite(mock_device)

        runner = Runner()
        runner.add_suite(suite, group="TestDevices")

        with patch.object(ConsoleReporter, "report_results"):
            grouped_results = runner.run_all_with_progress()

            assert "TestDevices" in grouped_results
            results = grouped_results["TestDevices"]

            stats = ResultsStatistics.from_results(results)
            assert stats[Status.PASSED] == 4
            assert stats[Status.ERROR] == 1

    def test_with_invalid_device(self, invalid_device):
        """Test with an invalid device that should cause failures."""

        suite = ExampleBoardTestSuite(invalid_device)
        runner = Runner()
        runner.add_suite(suite)

        with patch.object(ConsoleReporter, "report_results"):
            grouped_results = runner.run_all_with_progress()

            results = grouped_results[None]

            stats = ResultsStatistics.from_results(results)

            # With an invalid device, we should get failures
            assert stats[Status.PASSED] == 2
            assert stats[Status.FAILED] == 2
            assert stats[Status.ERROR] == 1

    def test_with_context_managers(self, mock_device):
        """Test using context managers to modify test behavior."""

        suite = ExampleBoardTestSuite(mock_device)
        runner = Runner()
        runner.add_suite(suite)

        # Allow None to be treated as pass
        with allow_null_as_pass():
            with patch.object(ConsoleReporter, "report_results"):
                grouped_results = runner.run_all_with_progress()
                results = grouped_results[None]

                stats = ResultsStatistics.from_results(results)

                assert stats[Status.PASSED] == 5
                assert stats[Status.ERROR] == 0

    def test_multiple_groups(self, mock_device, invalid_device):
        """Test running suites in multiple groups."""

        valid_suite = ExampleBoardTestSuite(mock_device)
        invalid_suite = ExampleBoardTestSuite(invalid_device)

        runner = Runner()
        runner.add_suite(valid_suite, "ValidDevices")
        runner.add_suite(invalid_suite, "InvalidDevices")

        with patch.object(ConsoleReporter, "report_results"):
            grouped_results = runner.run_all_with_progress()

            assert "ValidDevices" in grouped_results
            assert "InvalidDevices" in grouped_results

            valid_stats = ResultsStatistics.from_results(grouped_results["ValidDevices"])
            assert valid_stats[Status.PASSED] == 4
            assert valid_stats[Status.ERROR] == 1

            invalid_stats = ResultsStatistics.from_results(grouped_results["InvalidDevices"])
            assert invalid_stats[Status.PASSED] == 2
            assert invalid_stats[Status.FAILED] == 2
            assert invalid_stats[Status.ERROR] == 1
