import pytest
from conftest import SimpleDataStream, SimpleParams

from contraqctor import _typing
from contraqctor.contract.base import DataStreamCollection


class TestDataStream:
    """Tests for the DataStream class."""

    def test_creation(self, text_file):
        """Test creating a DataStream."""
        stream = SimpleDataStream(name="test", description="Test stream", reader_params=SimpleParams(path=text_file))

        assert stream.name == "test"
        assert stream.description == "Test stream"
        assert not stream.is_collection
        assert stream.parent is None
        assert not stream.has_data

        with pytest.raises(ValueError):
            # Accessing data before loading should raise ValueError
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

    def test_clear_data(self, text_file):
        """Test clearing loaded data."""
        stream = SimpleDataStream(name="test", reader_params=SimpleParams(path=text_file))

        stream.load()
        assert stream.has_data

        stream.clear()
        assert not stream.has_data

        with pytest.raises(ValueError):
            _ = stream.data  # Accessing data after clearing should raise ValueError


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
        outer_collection = DataStreamCollection(name="outer", data_streams=[stream1, inner_collection])  # noqa: F841

        assert stream1.resolved_name == "outer::stream1"
        assert inner_collection.resolved_name == "outer::inner"
        assert stream2.resolved_name == "outer::inner::stream2"

        level3 = SimpleDataStream(name="level3", reader_params=SimpleParams(path=text_file))
        level2 = DataStreamCollection(name="level2", data_streams=[level3])
        level1 = DataStreamCollection(name="level1", data_streams=[level2])
        root = DataStreamCollection(name="root", data_streams=[level1])  # noqa: F841

        assert level3.resolved_name == "root::level1::level2::level3"


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
