from pathlib import Path

import pytest

from contraqctor.contract.mux import MapFromPaths, MapFromPathsParams
from contraqctor.contract.text import Text, TextParams


class TestMapFromPaths:
    """Tests for the MapFromPaths class."""

    def test_basic_operation(self, multiple_files):  # grab multiple_files from conftest
        """Test basic operation of MapFromPaths."""
        parent_dir = Path(multiple_files[0]).parent

        mux = MapFromPaths(
            name="text_files",
            reader_params=MapFromPathsParams(
                paths=[parent_dir],
                include_glob_pattern=["*.txt"],
                inner_data_stream=Text,
                inner_param_factory=lambda path: TextParams(path=path),
            ),
        )

        mux.load()
        assert mux.has_data
        assert len(mux.data) == 3

        for stream in mux.data:
            assert isinstance(stream, Text)
            assert stream.name.startswith("file")

        file0 = mux.at("file0")
        assert file0.name == "file0"

        file0.load()
        assert file0.data == "Content of file 0"

    def test_multiple_paths(self, temp_dir):  # grab temp_dir from conftest
        """Test MapFromPaths with multiple paths."""
        dir1 = temp_dir / "dir1"
        dir2 = temp_dir / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        for i in range(2):
            with open(dir1 / f"file{i}.txt", "w") as f:
                f.write(f"Dir1 file {i}")

            with open(dir2 / f"file{i}.txt", "w") as f:
                f.write(f"Dir2 file {i}")

        mux = MapFromPaths(
            name="text_files",
            reader_params=MapFromPathsParams(
                paths=[dir1, dir2],
                include_glob_pattern=["*.txt"],
                inner_data_stream=Text,
                inner_param_factory=lambda path: TextParams(path=path),
            ),
        )

        mux.load()
        assert mux.has_error
        with pytest.raises(ValueError):
            _ = mux.data

    def test_include_exclude_patterns(self, temp_dir):
        """Test include and exclude patterns."""
        for i in range(3):
            with open(temp_dir / f"file{i}.txt", "w") as f:
                f.write(f"Text file {i}")

            with open(temp_dir / f"file{i}.csv", "w") as f:
                f.write(f"name,value\nRow{i},1")

            with open(temp_dir / f"skip{i}.txt", "w") as f:
                f.write(f"Skip file {i}")

        # Create a MapFromPaths that includes txt but excludes skip*.txt
        mux = MapFromPaths(
            name="filtered_files",
            reader_params=MapFromPathsParams(
                paths=[temp_dir],
                include_glob_pattern=["*.txt"],
                exclude_glob_pattern=["skip*.txt"],
                inner_data_stream=Text,
                inner_param_factory=lambda path: TextParams(path=path),
            ),
        )

        mux.load()
        assert len(mux.data) == 3  # Should find 3 text files (file0-2.txt)

        for stream in mux.data:
            assert not stream.name.startswith("skip")

    def test_with_descriptions(self, multiple_files):
        """Test MapFromPaths with custom descriptions."""
        parent_dir = Path(multiple_files[0]).parent

        descriptions = {"file0": "First file", "file1": "Second file"}

        mux = MapFromPaths(
            name="text_files",
            reader_params=MapFromPathsParams(
                paths=[parent_dir],
                include_glob_pattern=["*.txt"],
                inner_data_stream=Text,
                inner_param_factory=lambda path: TextParams(path=path),
                inner_descriptions=descriptions,
            ),
        )

        mux.load()

        assert mux.at("file0").description == "First file"
        assert mux.at("file1").description == "Second file"
        assert mux.at("file2").description is None  # No description provided
