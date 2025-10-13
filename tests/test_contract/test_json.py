import json

import pandas as pd
from conftest import MockModel

from contraqctor.contract.json import (
    Json,
    JsonParams,
    ManyPydanticModel,
    ManyPydanticModelParams,
    MultiLineJson,
    PydanticModel,
    PydanticModelParams,
)


class TestJson:
    """Tests for the Json class."""

    def test_read_json(self, json_file):
        """Test reading a JSON file."""
        json_stream = Json(name="test", reader_params=JsonParams(path=json_file))

        data = json_stream.read()
        assert isinstance(data, dict)
        assert data["name"] == "Test 0"
        assert data["value"] == 0
        assert data["timestamp"] == 0

    def test_load_json(self, json_file):
        """Test loading a JSON file."""
        json_stream = Json(name="test", reader_params=JsonParams(path=json_file))

        json_stream.load()
        assert json_stream.has_data
        assert isinstance(json_stream.data, dict)
        assert json_stream.data["name"] == "Test 0"
        assert json_stream.data["value"] == 0
        assert json_stream.data["timestamp"] == 0


class TestMultiLineJson:
    """Tests for the MultiLineJson class."""

    def test_read_jsonl(self, jsonl_file):
        """Test reading a JSONL file."""
        jsonl_stream = MultiLineJson(name="test", reader_params=JsonParams(path=jsonl_file))

        data = jsonl_stream.read()
        assert isinstance(data, list)
        assert len(data) == 5
        assert [d["name"] for d in data] == [f"Test {i}" for i in range(5)]
        assert [d["value"] for d in data] == list(range(5))
        assert [d["timestamp"] for d in data] == [i * 1000 for i in range(5)]

    def test_load_jsonl(self, jsonl_file):
        """Test loading a JSONL file."""
        jsonl_stream = MultiLineJson(name="test", reader_params=JsonParams(path=jsonl_file))

        jsonl_stream.load()
        assert jsonl_stream.has_data
        assert len(jsonl_stream.data) == 5
        assert [d["name"] for d in jsonl_stream.data] == [f"Test {i}" for i in range(5)]
        assert [d["value"] for d in jsonl_stream.data] == list(range(5))
        assert [d["timestamp"] for d in jsonl_stream.data] == [i * 1000 for i in range(5)]


class TestPydanticModel:
    """Tests for the PydanticModel class."""

    def test_read_pydantic(self, temp_dir):
        """Test reading a JSON file into a Pydantic model."""
        file_path = temp_dir / "model.json"
        data_raw = MockModel(name="Test", value=42, timestamp=0)

        with open(file_path, "w") as f:
            json.dump(data_raw.model_dump(), f)

        model_stream = PydanticModel(name="test", reader_params=PydanticModelParams(path=file_path, model=MockModel))

        data = model_stream.read()
        assert isinstance(data, MockModel)
        assert data == data_raw

    def test_load_pydantic(self, temp_dir):
        """Test loading a JSON file into a Pydantic model."""
        # Create a file with model data
        file_path = temp_dir / "model.json"
        data_raw = MockModel(name="Test", value=42, timestamp=0)

        with open(file_path, "w") as f:
            json.dump(data_raw.model_dump(), f)

        model_stream = PydanticModel(name="test", reader_params=PydanticModelParams(path=file_path, model=MockModel))

        model_stream.load()
        assert model_stream.has_data
        assert isinstance(model_stream.data, MockModel)
        assert model_stream.data == data_raw


class TestManyPydanticModel:
    """Tests for the ManyPydanticModel class."""

    def test_read_many_pydantic(self, temp_dir):
        """Test reading a JSONL file into a DataFrame of Pydantic models."""
        file_path = temp_dir / "models.jsonl"
        data = [MockModel(name=f"Test {i}", value=i, timestamp=i * 1000) for i in range(2)]

        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item.model_dump()) + "\n")

        models_stream = ManyPydanticModel(
            name="test", reader_params=ManyPydanticModelParams(path=file_path, model=MockModel)
        )

        df = models_stream.read()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert data == [MockModel(**row) for row in df.to_dict(orient="records")]

    def test_with_column_renaming(self, temp_dir):
        """Test reading models with column renaming."""
        file_path = temp_dir / "models.jsonl"
        data = [MockModel(name=f"Test {i}", value=i, timestamp=i * 1000) for i in range(2)]

        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item.model_dump()) + "\n")

        models_stream = ManyPydanticModel(
            name="test",
            reader_params=ManyPydanticModelParams(
                path=file_path, model=MockModel, column_names={"name": "subject_name", "value": "score"}
            ),
        )

        df = models_stream.read()
        assert "subject_name" in df.columns
        assert "score" in df.columns
        assert list(df["subject_name"]) == ["Test 0", "Test 1"]
        assert list(df["score"]) == [0, 1]
        assert list(df["timestamp"]) == [0, 1000]
        assert "name" not in df.columns
        assert "value" not in df.columns
        assert "timestamp" in df.columns

    def test_with_index(self, temp_dir):
        """Test reading models with a specific index."""
        file_path = temp_dir / "models.jsonl"
        data = [MockModel(name=f"Test {i}", value=i, timestamp=i * 1000) for i in range(2)]

        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item.model_dump()) + "\n")

        models_stream = ManyPydanticModel(
            name="test", reader_params=ManyPydanticModelParams(path=file_path, model=MockModel, index="timestamp")
        )

        df = models_stream.read()
        assert df.index.name == "timestamp"
        assert df.index.tolist() == [0, 1000]
