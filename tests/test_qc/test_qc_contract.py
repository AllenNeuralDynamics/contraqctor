import pytest
from conftest import SimpleDataStream, SimpleParams

from contraqctor._typing import ErrorOnLoad
from contraqctor.contract.base import DataStreamCollection
from contraqctor.qc.base import Status
from contraqctor.qc.contract import ContractTestSuite


def raise_value_error(*args, **kwargs):
    raise ValueError("Simulated load error")


def raise_io_error(*args, **kwargs):
    raise IOError("Simulated load error")


@pytest.fixture
def loading_errors(text_file) -> list[ErrorOnLoad]:
    stream1 = SimpleDataStream(name="stream1", reader_params=SimpleParams(path=text_file))
    stream1._reader = raise_value_error

    stream2 = SimpleDataStream(name="stream2", reader_params=SimpleParams(path=text_file))
    stream2._reader = raise_io_error

    collection = DataStreamCollection(name="collection", data_streams=[stream1, stream2])

    collection.load_all()
    return collection.collect_errors()


@pytest.fixture
def excluded_streams(loading_errors):
    return [loading_errors[0].data_stream]


class TestContractTestSuite:
    """Tests for the ContractTestSuite class."""

    def test_init(self, loading_errors, excluded_streams):
        """Test initializing the ContractTestSuite."""
        suite = ContractTestSuite(loading_errors)
        assert suite.loading_errors == loading_errors
        assert suite.exclude == []

        suite = ContractTestSuite(loading_errors, exclude=excluded_streams)
        assert suite.loading_errors == loading_errors
        assert suite.exclude == excluded_streams

    def test_has_errors_on_load_with_errors(self, loading_errors):
        """Test test_has_errors_on_load method with errors."""
        suite = ContractTestSuite(loading_errors)
        result = suite.test_has_errors_on_load()

        assert result.status == Status.FAILED
        assert result.message is not None
        assert "raised errors on load" in result.message
        assert result.context is not None
        assert "errors" in result.context
        assert len(result.context["errors"]) == len(loading_errors)

    def test_has_errors_on_load_no_errors(self):
        """Test test_has_errors_on_load method with no errors."""
        suite = ContractTestSuite([])
        result = suite.test_has_errors_on_load()

        assert result.status == Status.PASSED
        assert result.message is not None
        assert "All DataStreams loaded successfully" in result.message

    def test_has_errors_on_load_with_excludes(self, loading_errors, excluded_streams):
        """Test test_has_errors_on_load method with excluded streams."""
        suite = ContractTestSuite(loading_errors, exclude=excluded_streams)
        result = suite.test_has_errors_on_load()

        assert result.status == Status.FAILED
        assert result.context is not None
        assert "errors" in result.context
        assert len(result.context["errors"]) == len(loading_errors) - len(excluded_streams)
        excluded_names = [ds.resolved_name for ds in excluded_streams]
        for err in result.context["errors"]:
            assert err.data_stream.resolved_name not in excluded_names

    def test_has_excluded_as_warnings_with_excludes(self, loading_errors, excluded_streams):
        """Test test_has_excluded_as_warnings method with excluded streams."""
        suite = ContractTestSuite(loading_errors, exclude=excluded_streams)
        result = suite.test_has_excluded_as_warnings()

        assert result.status == Status.WARNING
        assert result.context is not None
        assert "warnings" in result.context
        assert len(result.context["warnings"]) == len(excluded_streams)
        for err in result.context["warnings"]:
            assert err.data_stream in excluded_streams

    def test_has_excluded_as_warnings_no_excludes(self, loading_errors):
        """Test test_has_excluded_as_warnings method with no excluded streams."""
        suite = ContractTestSuite(loading_errors)
        result = suite.test_has_excluded_as_warnings()

        assert result.status == Status.PASSED
        assert result.message is not None
        assert "No excluded DataStreams raised errors" in result.message

    def test_has_excluded_as_warnings_empty_errors(self, excluded_streams):
        """Test test_has_excluded_as_warnings with empty errors but excluded streams."""
        suite = ContractTestSuite([], exclude=excluded_streams)
        result = suite.test_has_excluded_as_warnings()

        assert result.status == Status.PASSED
        assert result.message is not None
        assert "No excluded DataStreams raised errors" in result.message
