import pytest
from conftest import SimpleDataStream, SimpleParams

from contraqctor import _typing
from contraqctor.contract.base import DataStream, DataStreamCollection, implicit_loading

class TestDataStream:
    """Tests for the DataStream class."""

    def test_creation_without_implicit_loading(self, text_file):
        """Test creating a DataStream."""
        stream = SimpleDataStream(name="test", description="Test stream", reader_params=SimpleParams(path=text_file))

        assert stream.name == "test"
        assert stream.description == "Test stream"
        assert not stream.is_collection
        assert stream.parent is None
        assert not stream.has_data

        with pytest.raises(ValueError):
            with implicit_loading(False):
                # Accessing data before loading should raise ValueError
                _ = stream.data

    def test_creation_with_implicit_loading(self, text_file):
        """Test creating a DataStream."""
        stream = SimpleDataStream(name="test", description="Test stream", reader_params=SimpleParams(path=text_file))

        assert stream.name == "test"
        assert stream.description == "Test stream"
        assert not stream.is_collection
        assert stream.parent is None
        assert not stream.has_data

        with implicit_loading(True):
            # Accessing data should trigger implicit loading
            _ = stream.data

    def test_load(self, text_file):
        """Test loading data from a DataStream."""
        stream = SimpleDataStream(name="test", reader_params=SimpleParams(path=text_file))

        stream.load()
        assert stream.has_data
        assert stream.data == "Test content"

    def test_read(self, text_file):
        """Test reading data without loading it."""
        stream = SimpleDataStream(name="test", reader_params=SimpleParams(path=text_file))

        data = stream.read()
        assert data == "Test content"
        assert not stream.has_data  # read() doesn't store the data

    def test_bind_reader_params(self, text_file):
        """Test post-instantiating binding of reader parameters."""
        stream = SimpleDataStream(name="test")

        assert _typing.is_unset(stream.reader_params)

        stream.bind_reader_params(SimpleParams(path=text_file))
        assert not _typing.is_unset(stream.reader_params)

        with pytest.raises(ValueError):
            # Binding params again should raise ValueError
            stream.bind_reader_params(SimpleParams(path=text_file))

    def test_at_not_implemented(self):
        """Test that at() method raises NotImplementedError."""
        stream = SimpleDataStream(name="test")

        with pytest.raises(NotImplementedError):
            stream.at("key")

    def test_resolved_name(self):
        """Test resolved_name property."""
        stream = SimpleDataStream(name="test")

        assert stream.resolved_name == "test"

        # Name with prohibited characters should raise an error
        with pytest.raises(ValueError):
            SimpleDataStream(name="test::invalid")

    def test_invalid_name(self, text_file):
        """Test creating a DataStream with an invalid name."""
        with pytest.raises(ValueError, match="Name cannot contain '::' character."):
            SimpleDataStream(
                name="test::invalid", description="Test stream", reader_params=SimpleParams(path=text_file)
            )

    def test_clear_data_without_implicit_loading(self, text_file):
        """Test clearing loaded data."""
        stream = SimpleDataStream(name="test", reader_params=SimpleParams(path=text_file))

        stream.load()
        assert stream.has_data

        stream.clear()
        assert not stream.has_data

        with pytest.raises(ValueError):
            with implicit_loading(False):
                # Accessing data after clearing should raise ValueError
                _ = stream.data
        stream.clear()
        assert not stream.has_data

        with implicit_loading(True):
            # Accessing data should trigger implicit loading
            _ = stream.data

    def test_null_data_stream(self):
        """Test DataStream with None data type."""

        class _NullDataStream(DataStream[None, None]):
            @staticmethod
            def _reader(params: None = None) -> None:
                return None

        null_stream = _NullDataStream(name="null_stream", description="Null data stream", reader_params=None)
        null_stream.load()
        assert null_stream.data is None


class TestDataStreamCollection:
    """Tests for the DataStreamCollection anonymous class."""

    def test_creation(self, text_file):
        """Test creating a DataStreamCollection."""
        stream1 = SimpleDataStream(name="stream1", reader_params=SimpleParams(path=text_file))
        stream2 = SimpleDataStream(name="stream2", reader_params=SimpleParams(path=text_file))

        collection = DataStreamCollection(
            name="collection", description="Test collection", data_streams=[stream1, stream2]
        )

        assert collection.name == "collection"
        assert collection.description == "Test collection"
        assert collection.is_collection
        assert collection.has_data  # data_streams are set directly
        assert len(collection.data) == 2

    def test_at_method(self, text_file):
        """Test accessing streams with at() method."""
        stream1 = SimpleDataStream(name="stream1", reader_params=SimpleParams(path=text_file))
        stream2 = SimpleDataStream(name="stream2", reader_params=SimpleParams(path=text_file))

        collection = DataStreamCollection(name="collection", data_streams=[stream1, stream2])

        assert collection.at("stream1") == stream1
        assert collection.at("stream2") == stream2

        with pytest.raises(KeyError):
            collection.at("nonexistent")

    def test_indexing(self, text_file):
        """Test accessing streams with indexing."""
        stream1 = SimpleDataStream(name="stream1", reader_params=SimpleParams(path=text_file))
        stream2 = SimpleDataStream(name="stream2", reader_params=SimpleParams(path=text_file))

        collection = DataStreamCollection(name="collection", data_streams=[stream1, stream2])

        assert collection["stream1"] == stream1
        assert collection["stream2"] == stream2

        with pytest.raises(KeyError):
            collection["nonexistent"]

    def test_add_stream(self, text_file):
        """Test adding a stream to a collection."""
        stream1 = SimpleDataStream(name="stream1", reader_params=SimpleParams(path=text_file))

        collection = DataStreamCollection(name="collection", data_streams=[stream1])

        stream2 = SimpleDataStream(name="stream2", reader_params=SimpleParams(path=text_file))

        collection.add_stream(stream2)
        assert len(collection.data) == 2
        assert collection.at("stream2") == stream2

        # Adding a stream with an existing name should raise KeyError
        stream3 = SimpleDataStream(name="stream1", reader_params=SimpleParams(path=text_file))

        with pytest.raises(KeyError):
            collection.add_stream(stream3)

    def test_remove_stream(self, text_file):
        """Test removing a stream from a collection."""
        stream1 = SimpleDataStream(name="stream1", reader_params=SimpleParams(path=text_file))
        stream2 = SimpleDataStream(name="stream2", reader_params=SimpleParams(path=text_file))

        collection = DataStreamCollection(name="collection", data_streams=[stream1, stream2])

        collection.remove_stream("stream1")
        assert len(collection.data) == 1

        with pytest.raises(KeyError):
            collection.at("stream1")

        with pytest.raises(KeyError):
            collection.remove_stream("nonexistent")

    def test_parent_references(self, text_file):
        """Test that parent references are properly set."""
        stream1 = SimpleDataStream(name="stream1", reader_params=SimpleParams(path=text_file))
        stream2 = SimpleDataStream(name="stream2", reader_params=SimpleParams(path=text_file))

        collection = DataStreamCollection(name="collection", data_streams=[stream1, stream2])

        assert stream1.parent == collection
        assert stream2.parent == collection

    def test_iter_streams(self, text_file):
        """Test iterating through data streams."""
        stream1 = SimpleDataStream(name="stream1", reader_params=SimpleParams(path=text_file))
        stream2 = SimpleDataStream(name="stream2", reader_params=SimpleParams(path=text_file))

        inner_collection = DataStreamCollection(name="inner", data_streams=[stream2])

        outer_collection = DataStreamCollection(name="outer", data_streams=[stream1, inner_collection])

        streams = [x for x in outer_collection.iter_all()]
        assert len(streams) == 3  # stream1, stream2, and inner_collection
        assert stream1 in streams
        assert stream2 in streams
        assert inner_collection in streams

        streams = [x for x in outer_collection]
        assert len(streams) == 2  # stream1, inner_collection
        assert stream1 in streams
        assert stream2 not in streams
        assert inner_collection in streams

    def test_duplicate_names(self, text_file):
        """Test that duplicate names raise an error."""
        stream1 = SimpleDataStream(name="duplicate", reader_params=SimpleParams(path=text_file))
        stream2 = SimpleDataStream(name="duplicate", reader_params=SimpleParams(path=text_file))

        with pytest.raises(ValueError):
            DataStreamCollection(name="collection", data_streams=[stream1, stream2])

    def test_resolved_name(self, text_file):
        """Test resolved_name property in nested collections."""
        stream1 = SimpleDataStream(name="stream1", reader_params=SimpleParams(path=text_file))
        stream2 = SimpleDataStream(name="stream2", reader_params=SimpleParams(path=text_file))

        inner_collection = DataStreamCollection(name="inner", data_streams=[stream2])
        DataStreamCollection(name="outer", data_streams=[stream1, inner_collection])

        assert stream1.resolved_name == "outer::stream1"
        assert inner_collection.resolved_name == "outer::inner"
        assert stream2.resolved_name == "outer::inner::stream2"

        level3 = SimpleDataStream(name="level3", reader_params=SimpleParams(path=text_file))
        level2 = DataStreamCollection(name="level2", data_streams=[level3])
        level1 = DataStreamCollection(name="level1", data_streams=[level2])
        DataStreamCollection(name="root", data_streams=[level1])

        assert level3.resolved_name == "root::level1::level2::level3"

    def test_collection_with_implicit_loading(self, text_file):
        """Test implicit loading behavior in a collection."""
        level3 = SimpleDataStream(name="level3", reader_params=SimpleParams(path=text_file))
        level2 = DataStreamCollection(name="level2", data_streams=[level3])
        level1 = DataStreamCollection(name="level1", data_streams=[level2])
        root = DataStreamCollection(name="root", data_streams=[level1])

        with implicit_loading(True):
            # Using all accessor styles just in case
            assert root.at("level1")["level2"].at("level3").data == "Test content"

    def test_collection_without_implicit_loading(self, text_file):
        """Test behavior without implicit loading in a collection."""
        level3 = SimpleDataStream(name="level3", reader_params=SimpleParams(path=text_file))
        level2 = DataStreamCollection(name="level2", data_streams=[level3])
        level1 = DataStreamCollection(name="level1", data_streams=[level2])
        root = DataStreamCollection(name="root", data_streams=[level1])

        with implicit_loading(False):
            with pytest.raises(ValueError):
                # Using all accessor styles just in case
                _ = root.at("level1")["level2"].at.level3.data

    def test_nested_implicit_loading_with_error_stream(self, text_file, temp_dir):
        """Test implicit loading with a nested stream that has an error."""
        working_stream = SimpleDataStream(name="working", reader_params=SimpleParams(path=text_file))

        nonexistent_path = temp_dir / "nonexistent.txt"
        failing_stream = SimpleDataStream(name="failing", reader_params=SimpleParams(path=nonexistent_path))

        inner_collection = DataStreamCollection(name="inner", data_streams=[working_stream, failing_stream])
        outer_collection = DataStreamCollection(name="outer", data_streams=[inner_collection])

        with implicit_loading(True):
            assert outer_collection.at("inner").at("working").data == "Test content"

            with pytest.raises(FileNotFoundError):
                _ = outer_collection.at("inner").at("failing").data

    def test_nested_implicit_loading_disabled_with_error_stream(self, text_file, temp_dir):
        """Test that implicit loading disabled prevents loading even working streams."""
        working_stream = SimpleDataStream(name="working", reader_params=SimpleParams(path=text_file))

        nonexistent_path = temp_dir / "nonexistent.txt"
        failing_stream = SimpleDataStream(name="failing", reader_params=SimpleParams(path=nonexistent_path))

        inner_collection = DataStreamCollection(name="inner", data_streams=[working_stream, failing_stream])
        outer_collection = DataStreamCollection(name="outer", data_streams=[inner_collection])

        with implicit_loading(False):
            with pytest.raises(ValueError, match="Data has not been loaded yet"):
                _ = outer_collection.at("inner").at("working").data

            with pytest.raises(ValueError, match="Data has not been loaded yet"):
                _ = outer_collection.at("inner").at("failing").data

    def test_mixed_loading_states_with_implicit_loading(self, text_file, temp_dir):
        """Test implicit loading with mixed pre-loaded and unloaded streams."""
        preloaded_stream = SimpleDataStream(name="preloaded", reader_params=SimpleParams(path=text_file))
        unloaded_working_stream = SimpleDataStream(name="unloaded_working", reader_params=SimpleParams(path=text_file))

        nonexistent_path = temp_dir / "nonexistent.txt"
        unloaded_failing_stream = SimpleDataStream(
            name="unloaded_failing", reader_params=SimpleParams(path=nonexistent_path)
        )

        preloaded_stream.load()
        assert preloaded_stream.has_data

        inner_collection = DataStreamCollection(
            name="inner", data_streams=[preloaded_stream, unloaded_working_stream, unloaded_failing_stream]
        )
        outer_collection = DataStreamCollection(name="outer", data_streams=[inner_collection])

        with implicit_loading(True):
            assert outer_collection.at("inner").at("preloaded").data == "Test content"
            assert outer_collection.at("inner").at("unloaded_working").data == "Test content"

            with pytest.raises(FileNotFoundError):
                _ = outer_collection.at("inner").at("unloaded_failing").data

            assert outer_collection.at("inner").at("unloaded_failing").has_error

    def test_error_propagation_in_deep_nesting(self, text_file, temp_dir):
        """Test error handling in deeply nested collections."""
        level3_working = SimpleDataStream(name="level3_working", reader_params=SimpleParams(path=text_file))

        nonexistent_path = temp_dir / "nonexistent.txt"
        level3_failing = SimpleDataStream(name="level3_failing", reader_params=SimpleParams(path=nonexistent_path))

        level2 = DataStreamCollection(name="level2", data_streams=[level3_working, level3_failing])
        level1 = DataStreamCollection(name="level1", data_streams=[level2])
        root = DataStreamCollection(name="root", data_streams=[level1])

        with implicit_loading(True):
            assert root.at("level1").at("level2").at("level3_working").data == "Test content"

            with pytest.raises(FileNotFoundError):
                _ = root.at("level1").at("level2").at("level3_failing").data

        errors = root.collect_errors()
        assert len(errors) == 1
        assert errors[0].data_stream == level3_failing
        assert isinstance(errors[0].exception, FileNotFoundError)

    def test_retry_after_error_with_implicit_loading(self, temp_dir):
        """Test that streams with errors don't auto-retry but can be manually retried."""
        nonexistent_path = temp_dir / "nonexistent.txt"
        failing_stream = SimpleDataStream(name="failing", reader_params=SimpleParams(path=nonexistent_path))

        collection = DataStreamCollection(name="collection", data_streams=[failing_stream])

        with implicit_loading(True):
            with pytest.raises(FileNotFoundError):
                _ = collection.at("failing").data

            assert collection.at("failing").has_error

            with pytest.raises(FileNotFoundError):
                _ = collection.at("failing").data

        nonexistent_path.write_text("Fixed content")
        collection.at("failing").clear()

        with implicit_loading(True):
            assert collection.at("failing").data == "Fixed content"


class TestLoadAllChildren:
    """Tests for loading all children datastreams recursively."""

    def test_load_all_success(self, text_file):
        """Test load_all with successful loads."""
        stream1 = SimpleDataStream(name="stream1", reader_params=SimpleParams(path=text_file))
        stream2 = SimpleDataStream(name="stream2", reader_params=SimpleParams(path=text_file))

        collection = DataStreamCollection(name="collection", data_streams=[stream1, stream2])

        result = collection.load_all()
        assert result.collect_errors() == []
        assert stream1.has_data
        assert stream2.has_data

    def test_load_all_with_exception(self, text_file, temp_dir):
        """Test load_all with an exception."""
        stream1 = SimpleDataStream(name="stream1", reader_params=SimpleParams(path=text_file))

        nonexistent_path = temp_dir / "nonexistent.txt"
        stream2 = SimpleDataStream(name="stream2", reader_params=SimpleParams(path=nonexistent_path))

        collection = DataStreamCollection(name="collection", data_streams=[stream1, stream2])

        result = collection.load_all()
        errors = result.collect_errors()
        assert len(errors) == 1
        assert errors[0].data_stream == stream2
        assert isinstance(errors[0].exception, FileNotFoundError)

        assert stream1.has_data
        assert not stream2.has_data

        with pytest.raises(FileNotFoundError):
            raise errors[0].exception

    def test_load_all_strict(self, text_file, temp_dir):
        """Test load_all with strict=True."""
        stream1 = SimpleDataStream(name="stream1", reader_params=SimpleParams(path=text_file))

        nonexistent_path = temp_dir / "nonexistent.txt"
        stream2 = SimpleDataStream(name="stream2", reader_params=SimpleParams(path=nonexistent_path))

        collection = DataStreamCollection(name="collection", data_streams=[stream1, stream2])

        with pytest.raises(FileNotFoundError):
            collection.load_all(strict=True)
