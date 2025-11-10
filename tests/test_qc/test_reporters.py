import tempfile
from pathlib import Path

from contraqctor.qc.base import Runner, Suite
from contraqctor.qc.reporters import HtmlReporter


class SimpleTestSuite(Suite):
    """Simple test suite for demonstrating HTML reporter."""

    def test_passing_example(self):
        """This test always passes."""
        return self.pass_test(42, "Everything is working correctly")

    def test_failing_example(self):
        """This test demonstrates a failure."""
        return self.fail_test(None, "Something went wrong")

    def test_warning_example(self):
        """This test demonstrates a warning."""
        return self.warn_test(99, "This is a potential issue")

    def test_with_context(self):
        """Test that includes context information."""
        context = {"file": "data.csv", "rows": 1000, "columns": 5}
        return self.pass_test(True, "Data file processed successfully", context=context)


class AnotherTestSuite(Suite):
    """Another test suite in the same group."""

    def test_another_pass(self):
        """Another passing test."""
        return self.pass_test(100, "Another test passed")

    def test_with_dict_context(self):
        """Test with dictionary context."""
        context = {"key1": "value1", "key2": "value2", "count": 42}
        return self.pass_test(True, "Test with context", context=context)


def test_html_reporter():
    """Test that HtmlReporter generates valid HTML output."""
    suite1 = SimpleTestSuite()
    suite2 = AnotherTestSuite()
    runner = Runner()
    runner.add_suite(suite1, "Demo Tests")
    runner.add_suite(suite2, "Demo Tests")  # Same group, different suite

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_report.html"
        html_reporter = HtmlReporter(output_path)

        results = runner.run_all_with_progress(reporter=html_reporter)

        # Check that HTML file was created
        assert output_path.exists()

        # Check that HTML contains expected content
        html_content = output_path.read_text(encoding="utf-8")
        assert "Test Results Report" in html_content
        assert "SimpleTestSuite" in html_content
        assert "AnotherTestSuite" in html_content
        assert "test_passing_example" in html_content
        assert "test_failing_example" in html_content
        assert "test_warning_example" in html_content
        assert "test_another_pass" in html_content
        assert "PASSED" in html_content
        assert "FAILED" in html_content
        assert "WARNING" in html_content

        # Check for overview tree structure
        assert "Test Results" in html_content
        assert "tree-group" in html_content
        assert "tree-suite" in html_content
        assert "tree-test-item" in html_content

        # Check for collapsible structure
        assert "collapsed" in html_content
        assert "toggleGroup" in html_content
        assert "toggleSuite" in html_content
        assert "toggleTest" in html_content

        # Check context is rendered
        assert "data.csv" in html_content
        assert "key1" in html_content
        assert "value1" in html_content

        # Check results are still returned properly
        assert "Demo Tests" in results
        assert len(results["Demo Tests"]) == 6  # 4 tests from suite1 + 2 from suite2


if __name__ == "__main__":
    test_html_reporter()
    print("HTML reporter test passed!")
