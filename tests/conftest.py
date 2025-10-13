import json
import tempfile
import typing as t
from pathlib import Path

import pandas as pd
import pytest
from pydantic import BaseModel

from contraqctor.contract.base import (
    DataStream,
    FilePathBaseParam,
)


class SimpleParams(FilePathBaseParam):
    """Simple params implementation for testing."""

    pass


class SimpleDataStream(DataStream[str, SimpleParams]):
    """Simple DataStream implementation for testing."""

    @staticmethod
    def _reader(params: SimpleParams) -> str:
        with open(params.path, "r") as f:
            return f.read()

    make_params = SimpleParams


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def text_file(temp_dir):
    """Fixture providing a text file."""
    file_path = temp_dir / "test.txt"
    with open(file_path, "w") as f:
        f.write("Test content")
    return file_path


def make_mock_data(n: int = 1) -> t.List[dict[str, t.Any]]:
    return [{"name": f"Test {i}", "value": i, "timestamp": i * 1000} for i in range(n)]


@pytest.fixture
def json_file(temp_dir):
    """Fixture providing a JSON file."""
    file_path = temp_dir / "test.json"
    data = make_mock_data(1)[0]
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def jsonl_file(temp_dir):
    """Fixture providing a multi-line JSON file."""
    file_path = temp_dir / "test.jsonl"
    data = make_mock_data(5)
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return file_path


@pytest.fixture
def csv_file(temp_dir):
    """Fixture providing a CSV file."""
    file_path = temp_dir / "test.csv"
    data = pd.DataFrame(make_mock_data(5))
    data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def multiple_files(temp_dir):
    """Fixture providing multiple files with different extensions."""
    file_paths = []

    # Create text files
    for i in range(3):
        path = temp_dir / f"file{i}.txt"
        with open(path, "w") as f:
            f.write(f"Content of file {i}")
        file_paths.append(path)

    # Create JSON files
    for i in range(2):
        path = temp_dir / f"file{i}.json"
        with open(path, "w") as f:
            json.dump(make_mock_data(1)[0], f)
        file_paths.append(path)

    return file_paths


class MockModel(BaseModel):
    """A simple Pydantic model for testing."""

    name: str
    value: int
    timestamp: int
