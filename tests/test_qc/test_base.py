import pytest

from contraqctor.qc.base import (
    Status,
    Suite,
    allow_null_as_pass,
    elevated_skips,
    elevated_warnings,
    implicit_pass,
)


class SimpleSuite(Suite):
    """A simple test suite for testing the TestSuite class."""

    def test_always_pass(self):
        """A test that always passes."""
        return self.pass_test("pass result", "This test passed")

    def test_always_fail(self):
        """A test that always fails."""
        return self.fail_test("fail result", "This test failed")

    def test_always_skip(self):
        """A test that always skips."""
        return self.skip_test("This test was skipped")

    def test_always_warn(self):
        """A test that always warns."""
        return self.warn_test("warning", "This test was a warning")

    def test_return_none(self):
        """A test that returns None."""
        return None

    def test_yielding_results(self):
        """A test that yields multiple results."""
        yield self.pass_test("first", "First yielded test")
        yield self.pass_test("second", "Second yielded test")
        yield self.fail_test("third", "Third yielded test")

    @implicit_pass
    def test_implicit_pass(self):
        """A test using the implicit_pass decorator."""
        return "This should be auto-converted to a pass"

    def test_implicit_fail(self):
        """A test that fails because it returns None and is not decorated."""
        return "I should fail"

    def not_a_test(self):
        """This is not a test method."""
        return "Not a test"


class TestSuite:
    """Tests for the TestSuite class."""

    def test_get_tests(self):
        """Test that get_tests returns all test methods."""
        suite = SimpleSuite()
        tests = list(suite.get_tests())

        # The number of tests in the SimpleTestSuite
        assert len(tests) == 8

        for test in tests:
            assert callable(test)

        # Should only include methods starting with 'test_'
        test_names = [test.__name__ for test in tests]
        assert "not_a_test" not in test_names
        assert "test_always_pass" in test_names

    def test_run_test_pass(self):
        """Test running a test that passes."""
        suite = SimpleSuite()
        test_method = suite.test_always_pass
        results = list(suite.run_test(test_method))

        assert len(results) == 1
        assert results[0].status == Status.PASSED
        assert results[0].result == "pass result"
        assert results[0].message == "This test passed"

    def test_run_test_fail(self):
        """Test running a test that fails."""
        suite = SimpleSuite()
        test_method = suite.test_always_fail
        results = list(suite.run_test(test_method))

        assert len(results) == 1
        assert results[0].status == Status.FAILED
        assert results[0].result == "fail result"
        assert results[0].message == "This test failed"

    def test_run_test_skip(self):
        """Test running a test that skips."""
        suite = SimpleSuite()
        test_method = suite.test_always_skip

        with elevated_skips(False):
            results = list(suite.run_test(test_method))

        assert len(results) == 1
        assert results[0].status == Status.SKIPPED
        assert results[0].message == "This test was skipped"

    def test_run_test_skip_elevated(self):
        """Test running a test that skips in a non-skippable context."""
        suite = SimpleSuite()
        test_method = suite.test_always_skip

        with elevated_skips(True):
            results = list(suite.run_test(test_method))

        assert len(results) == 1
        assert results[0].status == Status.FAILED
        assert results[0].message == "This test was skipped"

    def test_run_test_warn(self):
        """Test running a test that warns."""
        suite = SimpleSuite()
        test_method = suite.test_always_warn

        with elevated_warnings(False):
            results = list(suite.run_test(test_method))

        assert len(results) == 1
        assert results[0].status == Status.WARNING
        assert results[0].message == "This test was a warning"

    def test_run_test_warn_elevated(self):
        """Test running a test that skips in a non-skippable context."""
        suite = SimpleSuite()
        test_method = suite.test_always_warn

        with elevated_warnings(True):
            results = list(suite.run_test(test_method))

        assert len(results) == 1
        assert results[0].status == Status.FAILED
        assert results[0].message == "This test was a warning"

    def test_run_test_none(self):
        """Test running a test that returns None."""
        suite = SimpleSuite()
        test_method = suite.test_return_none

        with allow_null_as_pass(value=True):
            results = list(suite.run_test(test_method))
            assert len(results) == 1
            assert results[0].status == Status.PASSED

    def test_run_test_none_not_allowed(self):
        suite = SimpleSuite()
        test_method = suite.test_return_none
        with allow_null_as_pass(value=False):
            results = list(suite.run_test(test_method))
            assert len(results) == 1
            assert results[0].status == Status.ERROR

    def test_run_test_yielding(self):
        """Test running a test that yields multiple results."""
        suite = SimpleSuite()
        test_method = suite.test_yielding_results
        results = list(suite.run_test(test_method))

        assert len(results) == 3
        assert results[0].status == Status.PASSED
        assert results[1].status == Status.PASSED
        assert results[2].status == Status.FAILED

        assert results[0].result == "first"
        assert results[1].result == "second"
        assert results[2].result == "third"

    def test_run_test_with_implicit_pass(self):
        """Test running a test with the implicit_pass decorator."""
        suite = SimpleSuite()
        test_method = suite.test_implicit_pass
        results = list(suite.run_test(test_method))

        assert len(results) == 1
        assert results[0].status == Status.PASSED
        assert "auto-converted" in results[0].message.lower()

    def test_run_test_implicit_fail(self):
        """Test running a test that fails because it returns None."""
        suite = SimpleSuite()
        test_method = suite.test_implicit_fail
        results = list(suite.run_test(test_method))

        assert len(results) == 1
        assert results[0].status == Status.ERROR
        assert isinstance(results[0].exception, TypeError)

    def test_run_all(self):
        """Test running all tests in a suite."""
        suite = SimpleSuite()
        results = list(suite.run_all())

        assert len(results) == 10

        statuses = [r.status for r in results]
        assert statuses.count(Status.PASSED) == 4
        assert statuses.count(Status.FAILED) == 2
        assert statuses.count(Status.SKIPPED) == 1
        assert statuses.count(Status.ERROR) == 2
        assert statuses.count(Status.WARNING) == 1

    def test_run_all_with_context(self):
        """Test running all tests with context managers."""
        suite = SimpleSuite()

        with allow_null_as_pass():
            with elevated_skips(True):
                with elevated_warnings(True):
                    results = list(suite.run_all())

                    statuses = [r.status for r in results]
                    assert statuses.count(Status.PASSED) == 5
                    assert statuses.count(Status.FAILED) == 4
                    assert statuses.count(Status.SKIPPED) == 0
                    assert statuses.count(Status.WARNING) == 0
                    assert statuses.count(Status.ERROR) == 1

    def test_setup_teardown(self):
        class SetupTeardownSuite(Suite):
            def __init__(self):
                self.setup_called = 0
                self.teardown_called = 0
                self.test_called = 0

            def setup(self):
                self.setup_called += 1

            def teardown(self):
                self.teardown_called += 1

            def test_something(self):
                self.test_called += 1
                return self.pass_test()

            def test_something_with_yield(self):
                self.test_called += 1
                for i in range(3):
                    yield self.pass_test()

        suite = SetupTeardownSuite()
        list(suite.run_all())

        assert suite.setup_called == 2
        assert suite.teardown_called == 2
        assert suite.test_called == 2

    def test_teardown_error(self):
        """Test that teardown errors are properly reported."""

        class ErrorTeardownSuite(Suite):
            def test_something(self):
                return self.pass_test()

            def teardown(self):
                raise ValueError("Teardown error")

        suite = ErrorTeardownSuite()
        with pytest.raises(ValueError) as e_info:
            list(suite.run_all())
        assert "Teardown error" in str(e_info.value)

    def test_suite_reference(self):
        """Test that _suite_reference is properly set on results."""
        suite = SimpleSuite()

        pass_result = list(suite.run_test(suite.test_always_pass))[0]
        assert pass_result.suite_reference is suite

        error_result = list(suite.run_test(suite.test_implicit_fail))[0]
        assert error_result.suite_reference is suite

        class ExceptionSuite(Suite):
            def test_exception(self):
                raise ValueError("Test exception")

        exception_suite = ExceptionSuite()
        exception_result = list(exception_suite.run_test(exception_suite.test_exception))[0]
        assert exception_result.suite_reference is exception_suite

    def test_setup_suite_not_called_at_construction_or_collection(self):
        """Data loading deferred to setup_suite must not run at construction or collection.

        Regression test: I/O belongs in ``setup_suite`` (not ``__init__``) so that a
        load error is captured by the runner. Neither constructing the suite nor
        discovering its tests should trigger ``setup_suite``.
        """

        class LazySuite(Suite):
            def __init__(self):
                self.setup_suite_calls = 0

            def setup_suite(self):
                self.setup_suite_calls += 1
                raise FileNotFoundError("register .bin missing")

            def test_uses_data(self):
                return self.pass_test(True)

            def test_independent(self):
                return self.pass_test(True)

        # Construction must be side-effect free.
        suite = LazySuite()
        assert suite.setup_suite_calls == 0

        # Collecting tests must not call setup_suite either.
        test_names = sorted(m.__name__ for m in suite.get_tests())
        assert test_names == ["test_independent", "test_uses_data"]
        assert suite.setup_suite_calls == 0

    def test_setup_suite_runs_once_per_suite(self):
        """setup_suite/teardown_suite run once per suite, not once per test."""

        class LifecycleSuite(Suite):
            def __init__(self):
                self.setup_suite_calls = 0
                self.teardown_suite_calls = 0
                self.setup_calls = 0

            def setup_suite(self):
                self.setup_suite_calls += 1

            def teardown_suite(self):
                self.teardown_suite_calls += 1

            def setup(self):
                self.setup_calls += 1

            def test_a(self):
                return self.pass_test(True)

            def test_b(self):
                return self.pass_test(True)

        suite = LifecycleSuite()
        results = list(suite.run_all())

        assert len(results) == 2
        assert all(r.status == Status.PASSED for r in results)
        # Suite-level hooks run exactly once; per-test setup runs per test.
        assert suite.setup_suite_calls == 1
        assert suite.teardown_suite_calls == 1
        assert suite.setup_calls == 2

    def test_failing_setup_suite_surfaces_as_test_results(self):
        """A failing setup_suite errors every test in the suite, not a construction crash."""

        class LazySuite(Suite):
            def __init__(self):
                self.teardown_suite_calls = 0

            def setup_suite(self):
                raise FileNotFoundError("register .bin missing")

            def teardown_suite(self):
                self.teardown_suite_calls += 1

            def test_uses_data(self):
                return self.pass_test(True)

            def test_independent(self):
                return self.pass_test(True)

        # Construction must not raise even though setup_suite would fail.
        suite = LazySuite()
        results = {r.test_name: r for r in suite.run_all()}

        # Every test errors because suite setup failed before any test ran.
        assert results["test_independent"].status == Status.ERROR
        assert results["test_uses_data"].status == Status.ERROR
        assert isinstance(results["test_uses_data"].exception, FileNotFoundError)
        # teardown_suite must NOT run when setup_suite failed (mirrors unittest).
        assert suite.teardown_suite_calls == 0

    def test_failing_setup_suite_via_runner(self):
        """The Runner also captures a failing setup_suite as errored tests."""

        class LazySuite(Suite):
            def setup_suite(self):
                raise FileNotFoundError("register .bin missing")

            def test_a(self):
                return self.pass_test(True)

            def test_b(self):
                return self.pass_test(True)

        from contraqctor.qc.base import Runner

        runner = Runner().add_suite(LazySuite())
        results = runner.run_all()
        flat = [r for group in results.values() for r in group]
        assert len(flat) == 2
        assert all(r.status == Status.ERROR for r in flat)
