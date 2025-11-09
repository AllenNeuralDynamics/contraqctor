import abc
import contextvars
import dataclasses
import os
from contextlib import contextmanager
from typing import (
    Any,
    ClassVar,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Protocol,
    Self,
    TypeVar,
    cast,
    runtime_checkable,
)

from semver import Version
from typing_extensions import override

from contraqctor import _typing

_implicit_loading = contextvars.ContextVar("implicit_loading", default=True)


@contextmanager
def implicit_loading(value: bool = True):
    """Context manager to control whether streams automatically load data on access.

    When enabled, data streams will automatically load their data when accessed. When disabled,
    accessing a data stream without prior loading will raise an error. Call `load()` explicitly
    instead.

    Args:
        value: True to enable auto-loading, False to disable. Default is True.

    Examples:
        ```python
        # Assume you have nested collections already created
        # collection.at("sensors").at("temperature") -> temperature sensor data
        # collection.at("sensors").at("humidity") -> humidity sensor data
        # collection.at("logs").at("error_log") -> error log file

        # With implicit loading enabled (default behavior)
        with implicit_loading(True):
            # Data loads automatically on access
            temp_data = collection.at("sensors").at("temperature").data
            humidity_data = collection.at("sensors").at("humidity").data

        # With implicit loading disabled - requires explicit loading
        with implicit_loading(False):
            # This would raise ValueError: "Data has not been loaded yet"
            try:
                temp_data = collection.at("sensors").at("temperature").data
            except ValueError:
                # Must load explicitly first
                collection.load_all()
                temp_data = collection.at("sensors").at("temperature").data
        ```
    """
    token = _implicit_loading.set(value)
    try:
        yield
    finally:
        _implicit_loading.reset(token)


@runtime_checkable
class _AtProtocol(Protocol):
    """Protocol for the "at" property"""

    _data_stream: "DataStreamCollectionBase[Any, Any]"

    def __call__(self, name: str) -> "DataStream": ...
    def __dir__(self) -> list[str]: ...
    def __getattribute__(self, name: str) -> Any: ...


class DataStream(abc.ABC, Generic[_typing.TData, _typing.TReaderParams]):
    """Abstract base class for all data streams.

    Provides a generic interface for data reading operations with configurable parameters
    and hierarchical organization.

    Args:
        name: Name identifier for the data stream.
        description: Optional description of the data stream.
        reader_params: Optional parameters for the data reader.
        **kwargs: Additional keyword arguments.

    Attributes:
        _is_collection: Class variable indicating if this is a collection of data streams.

    Raises:
        ValueError: If name contains '::' characters which are reserved for path resolution.
    """

    _is_collection: ClassVar[bool] = False

    def __init__(
        self: Self,
        name: str,
        *,
        description: Optional[str] = None,
        reader_params: _typing.TReaderParams = _typing.UnsetParams,
        **kwargs,
    ) -> None:
        if "::" in name:
            raise ValueError("Name cannot contain '::' character.")
        self._name = name

        self._description = description
        self._reader_params = reader_params
        self._data = _typing.UnsetData
        self._parent: Optional["DataStream"] = None

    @property
    def name(self) -> str:
        """Get the name of the data stream.

        Returns:
            str: Name identifier of the data stream.
        """
        return self._name

    @property
    def resolved_name(self) -> str:
        """Get the full hierarchical name of the data stream.

        Generates a path-like name showing the stream's position in the hierarchy,
        using '::' as a separator between parent and child names.

        Returns:
            str: The fully resolved name including all parent names.
        """
        builder = self.name
        d = self
        while d.parent is not None:
            builder = f"{d.parent.name}::{builder}"
            d = d.parent
        return builder

    @property
    def description(self) -> Optional[str]:
        """Get the description of the data stream.

        Returns:
            Optional[str]: Description of the data stream, or None if not provided.
        """
        return self._description

    @property
    def parent(self) -> Optional["DataStream"]:
        """Get the parent data stream.

        Returns:
            Optional[DataStream]: Parent data stream, or None if this is a root stream.
        """
        return self._parent

    def set_parent(self, parent: "DataStream") -> None:
        """Set the parent data stream.

        Args:
            parent: The parent data stream to set.
        """
        self._parent = parent

    @property
    def is_collection(self) -> bool:
        """Check if this data stream is a collection of other streams.

        Returns:
            bool: True if this is a collection stream, False otherwise.
        """
        return self._is_collection

    _reader: _typing.IReader[_typing.TData, _typing.TReaderParams] = _typing.UnsetReader

    make_params = NotImplementedError("make_params is not implemented for DataStream.")

    @property
    def reader_params(self) -> _typing.TReaderParams:
        """Get the parameters for the data reader.

        Returns:
            TReaderParams: Parameters for the data reader.
        """
        return self._reader_params

    def read(self, reader_params: Optional[_typing.TReaderParams] = None) -> _typing.TData:
        """Read data using the configured reader.

        Args:
            reader_params: Optional parameters to override the default reader parameters.

        Returns:
            TData: Data read from the source.

        Raises:
            ValueError: If reader parameters are not set.
        """
        reader_params = reader_params if reader_params is not None else self._reader_params
        if _typing.is_unset(reader_params):
            raise ValueError("Reader parameters are not set. Cannot read data.")
        return self._reader(reader_params)

    def bind_reader_params(self, params: _typing.TReaderParams) -> Self:
        """Bind reader parameters to the data stream.

        Args:
            params: Parameters to bind to the data stream's reader.

        Returns:
            Self: The data stream instance for method chaining.

        Raises:
            ValueError: If reader parameters have already been set.
        """
        if not _typing.is_unset(self._reader_params):
            raise ValueError("Reader parameters are already set. Cannot bind again.")
        self._reader_params = params
        return self

    @property
    def at(self) -> _AtProtocol:
        """Get a child data stream by name.

        Args:
            name: Name of the child data stream to retrieve.

        Returns:
            DataStream: The child data stream with the given name.

        Raises:
            NotImplementedError: If the data stream does not support child access.

        Examples:
            ```python
            # Access stream in a collection
            collection = data_collection.load()
            temp_stream = collection.at("temperature")

            # Or using dictionary-style syntax
            humidity_stream = collection["humidity"]
            ```
        """
        raise NotImplementedError("This method is not implemented for DataStream.")

    def __getitem__(self, name: str) -> "DataStream":
        """Get a child data stream by name using dictionary-like syntax.

        Args:
            name: Name of the child data stream to retrieve.

        Returns:
            DataStream: The child data stream with the given name.
        """
        return self.at(name)

    @property
    def has_data(self) -> bool:
        """Check if the data stream has loaded data.

        Returns:
            bool: True if data has been loaded, False otherwise.
        """
        return not (_typing.is_unset(self._data) or self.has_error)

    @property
    def has_error(self) -> bool:
        """Check if the data stream encountered an error during loading.

        Returns:
            bool: True if an error occurred, False otherwise.
        """
        return isinstance(self._data, _typing.ErrorOnLoad)

    @property
    def data(self) -> _typing.TData:
        """Get the loaded data.

        Returns:
            TData: The loaded data.

        Raises:
            ValueError: If data has not been loaded yet.
        """
        return self._solve_data_load()

    def _solve_data_load(self) -> _typing.TData:
        """Resolve data loading based on the current state and implicit loading setting."""
        if self.has_data:
            return cast(_typing.TData, self._data)

        # If there is an error we do not auto load
        # and instead raise the existing error
        # We use .load() to explicitly retry loading
        if (not self.has_error) and _implicit_loading.get():
            self.load()

        if self.has_error:
            cast(_typing.ErrorOnLoad, self._data).raise_from_error()

        if not (self.has_data):
            raise ValueError("Data has not been loaded yet.")

        return cast(_typing.TData, self._data)

    def clear(self) -> Self:
        """Clear the loaded data from the data stream.

        Resets the data to an unset state, allowing for reloading.

        Returns:
            Self: The data stream instance for method chaining.
        """
        self._data = _typing.UnsetData
        return self

    def load(self) -> Self:
        """Load data into the data stream.

        Reads data from the source and stores it in the data stream.

        Returns:
            Self: The data stream instance for method chaining.

        Examples:
            ```python
            from contraqctor.contract import csv

            # Create and load a CSV stream
            params = csv.CsvParams(path="data/measurements.csv")
            csv_stream = csv.Csv("measurements", reader_params=params)
            csv_stream.load()

            # Access the data
            df = csv_stream.data
            print(f"Loaded {len(df)} rows")
            ```
        """
        try:
            self._data = self.read()
        except Exception as e:  # pylint: disable=broad-except
            self._data = _typing.ErrorOnLoad(self, exception=e)
        return self

    def __str__(self):
        """Generate a string representation of the data stream.

        Returns:
            str: String representation showing the stream's type, name, description,
                 reader parameters, and data type if loaded.
        """
        return (
            f"DataStream("
            f"stream_type={self.__class__.__name__}, "
            f"name={self._name}, "
            f"description={self._description}, "
            f"reader_params={self._reader_params}, "
            f"data_type={self.data.__class__.__name__ if self.has_data else 'Data not loaded'}"
        )

    def __iter__(self) -> Generator["DataStream", None, None]:
        """Iterator interface for data streams.

        For non-collection data streams, this yields nothing.

        Yields:
            DataStream: Child data streams (none for base DataStream).
        """
        return
        yield  # This line is unreachable but needed for the generator type

    def collect_errors(self) -> List[_typing.ErrorOnLoad]:
        """Collect all errors from this stream and its children.

        Performs a depth-first traversal to gather all ErrorOnLoad instances.

        Returns:
            List[ErrorOnLoad]: List of all errors raised on load encountered in the hierarchy.
        """
        errors = []
        if self.has_error:
            errors.append(cast(_typing.ErrorOnLoad, self._data))
        for stream in self:
            if stream is None:
                continue
            errors.extend(stream.collect_errors())
        return errors

    def load_all(self, strict: bool = False) -> Self:
        """Recursively load this data stream and all child streams.

        Performs depth-first traversal to load all streams in the hierarchy.

        Args:
            strict: If True, raises exceptions immediately; otherwise collects and returns them.

        Returns:
            list: List of tuples containing streams and exceptions that occurred during loading.

        Raises:
            Exception: If strict is True and an exception occurs during loading.

        Examples:
            ```python
            # Load all streams and handle errors
            errors = collection.load_all(strict=False)

            if errors:
                for stream, error in errors:
                    print(f"Error loading {stream.name}: {error}")
            ```
        """
        self.load()
        for stream in self:
            if stream is None:
                continue
            stream.load_all(strict=strict)
            if stream.has_error and strict:
                cast(_typing.ErrorOnLoad, stream.data).raise_from_error()
        return self


TDataStream = TypeVar("TDataStream", bound=DataStream[Any, Any])


class _At(Generic[TDataStream]):
    """A class for accessing data streams by name using dot notation."""

    def __init__(self, data_stream: "DataStreamCollectionBase[TDataStream, Any]"):
        """Initialize the At accessor."""
        self._data_stream = data_stream

    def __call__(self, name: str) -> TDataStream:
        """Access a data stream by name."""

        self._data_stream._solve_data_load()

        try:
            return self._data_stream._data_stream_mapping[name]
        except KeyError as exc:
            raise KeyError(f"Stream with name: '{name}' not found in data streams.") from exc

    def __dir__(self):
        """List available attributes for the At accessor. This ensures autocompletion at runtime."""
        base = list(object.__dir__(self))
        if hasattr(self, "_data_stream") and hasattr(self._data_stream, "_data_stream_mapping"):
            h = list(self._data_stream._data_stream_mapping.keys())
            return h + base
        else:
            return base

    def __getattribute__(self, name: str) -> Any:
        """Get an attribute by dot notation."""
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            _data_stream = object.__getattribute__(self, "_data_stream")
            if name in _data_stream._data_stream_mapping:
                # Redirect to __call__ to get the stream by name
                return self.__call__(name)
            raise


class DataStreamCollectionBase(
    DataStream[List[TDataStream], _typing.TReaderParams],
    Generic[TDataStream, _typing.TReaderParams],
):
    """Base class for collections of data streams.

    Provides functionality for managing and accessing multiple child data streams.

    Args:
        name: Name identifier for the collection.
        description: Optional description of the collection.
        reader_params: Optional parameters for the reader.
        **kwargs: Additional keyword arguments.
    """

    _is_collection: ClassVar[bool] = True

    def __init__(
        self: Self,
        name: str,
        *,
        description: Optional[str] = None,
        reader_params: Optional[_typing.TReaderParams] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, description=description, reader_params=reader_params, **kwargs)
        self._data_stream_mapping: Dict[str, TDataStream] = {}
        self._update_data_stream_mapping()
        self._at = _At(self)

    def _update_data_stream_mapping(self) -> None:
        """Update the internal mapping of name: child data streams.

        Validates that all child streams have unique names and updates the lookup table.

        Raises:
            ValueError: If duplicate names are found among child streams.
        """
        if not self.has_data:
            return
        stream_keys = [stream.name for stream in self._data]
        duplicates = [name for name in stream_keys if stream_keys.count(name) > 1]
        if duplicates:
            raise ValueError(f"Duplicate names found in the data stream collection: {set(duplicates)}")
        self._data_stream_mapping = {stream.name: stream for stream in self._data}
        self._update_parent_references()
        return

    def _update_parent_references(self) -> None:
        """Update parent references for all child data streams.

        Sets this collection as the parent for all child streams.
        """
        for stream in self._data_stream_mapping.values():
            stream.set_parent(self)

    @property
    def at(self) -> _At[TDataStream]:
        """Get the accessor for child data streams.

        Returns:
            _At: Accessor object for retrieving child streams by name.
        """
        return self._at

    @override
    def load(self) -> Self:
        """Load data for this collection.

        Overrides the base method to add validation that loaded data is a list of DataStreams.

        Returns:
            Self: The collection instance for method chaining.

        Raises:
            ValueError: If loaded data is not a list of DataStreams.
        """
        super().load()
        if not isinstance(self._data, list):
            self._data = _typing.UnsetData
            raise ValueError("Data must be a list of DataStreams.")
        self._update_data_stream_mapping()
        return self

    def __str__(self: Self) -> str:
        """Generate a formatted table representation of the collection.

        Returns:
            str: Formatted table showing child streams, their types, and load status.
        """
        table = []
        table.append(["Stream Name", "Stream Type", "Is Loaded"])
        table.append(["-" * 20, "-" * 20, "-" * 20])

        if not self.has_data:
            return f"{self.__class__.__name__} has not been loaded yet."

        for key, value in self._data_stream_mapping.items():
            table.append(
                [
                    key,
                    value.data.__class__.__name__ if value.has_data else "Unknown",
                    "Yes" if value.has_data else "No",
                ]
            )

        max_lengths = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]

        formatted_table = []
        for row in table:
            formatted_row = [str(cell).ljust(max_lengths[i]) for i, cell in enumerate(row)]
            formatted_table.append(formatted_row)

        table_str = ""
        for row in formatted_table:
            table_str += " | ".join(row) + "\n"

        return table_str

    def __iter__(self) -> Generator[DataStream, None, None]:
        """Iterator for child data streams.

        Yields:
            DataStream: Child data streams.

        """
        # We intentionally yield from self.data to trigger
        # automatic loading if needed
        yield from self.data

    def iter_all(self) -> Generator[DataStream, None, None]:
        """Iterator for all child data streams, including nested collections.

        Implements a depth-first traversal of the stream hierarchy.

        Yields:
            DataStream: All recursively yielded child data streams.
        """
        for value in self:
            if isinstance(value, DataStream):
                yield value
            if isinstance(value, DataStreamCollectionBase):
                yield from value.iter_all()


class DataStreamCollection(DataStreamCollectionBase[DataStream, _typing.UnsetParamsType]):
    """Collection of data streams with direct initialization.

    A specialized collection where child streams are passed directly instead of being
    created by a reader function.

    Args:
        name: Name identifier for the collection.
        data_streams: List of child data streams to include.
        description: Optional description of the collection.

    Examples:
        ```python
        from contraqctor.contract import csv, text, DataStreamCollection

        # Create streams
        text_stream = text.Text("readme", reader_params=text.TextParams(path="README.md"))
        csv_stream = csv.Csv("data", reader_params=csv.CsvParams(path="data.csv"))

        # Create the collection
        collection = DataStreamCollection("project_files", [text_stream, csv_stream])

        # Load and use
        collection.load_all()
        readme_content = collection["readme"].data
        ```
    """

    @override
    def __init__(
        self,
        name: str,
        data_streams: List[DataStream],
        *,
        description: Optional[str] = None,
    ) -> None:
        """Initializes a special DataStreamGroup where the data streams are passed directly, without a reader."""
        super().__init__(
            name=name,
            description=description,
            reader_params=_typing.UnsetParams,
        )
        self.bind_data_streams(data_streams)

    @staticmethod
    def parameters(*args, **kwargs) -> _typing.UnsetParamsType:
        """Parameters function to return UnsetParams.

        Returns:
            UnsetParamsType: Special unset parameters value.
        """
        return _typing.UnsetParams

    def _reader(self, *args, **kwargs) -> List[DataStream]:
        """Reader function that returns the pre-set data streams.

        Returns:
            List[DataStream]: The pre-set data streams.
        """
        return self._data

    @override
    def read(self, *args, **kwargs) -> List[DataStream]:
        """Read data from the collection.

        Returns:
            List[DataStream]: The pre-set data streams.

        Raises:
            ValueError: If data streams have not been set yet.
        """
        if not self.has_data:
            raise ValueError("Data streams have not been read yet.")
        return self._data

    def bind_data_streams(self, data_streams: List[DataStream]) -> Self:
        """Bind a list of data streams to the collection.

        Args:
            data_streams: List of data streams to include in the collection.

        Returns:
            Self: The collection instance for method chaining.

        Raises:
            ValueError: If data streams have already been set.
        """
        if self.has_data:
            raise ValueError("Data streams are already set. Cannot bind again.")
        self._data = data_streams
        self._update_data_stream_mapping()
        return self

    def add_stream(self, stream: DataStream) -> Self:
        """Add a new data stream to the collection.

        Args:
            stream: Data stream to add to the collection.

        Returns:
            Self: The collection instance for method chaining.

        Raises:
            KeyError: If a stream with the same name already exists.

        Examples:
            ```python
            from contraqctor.contract import json, DataStreamCollection

            # Create an empty collection
            collection = DataStreamCollection("api_data", [])

            # Add streams
            collection.add_stream(
                json.Json("config", reader_params=json.JsonParams(path="config.json"))
            )

            # Load the data
            collection.load_all()
            ```
        """
        if not self.has_data:
            self._data = [stream]
            self._update_data_stream_mapping()
            return self

        if stream.name in self._data_stream_mapping:
            raise KeyError(f"Stream with name: '{stream.name}' already exists in data streams.")

        self._data.append(stream)
        self._update_data_stream_mapping()
        return self

    def remove_stream(self, name: str) -> None:
        """Remove a data stream from the collection.

        Args:
            name: Name of the data stream to remove.

        Raises:
            ValueError: If data streams have not been set yet.
            KeyError: If no stream with the given name exists.
        """
        if not self.has_data:
            raise ValueError("Data streams have not been read yet. Cannot access data streams.")

        if name not in self._data_stream_mapping:
            raise KeyError(f"Data stream with name '{name}' not found in data streams.")
        self._data.remove(self._data_stream_mapping[name])
        self._update_data_stream_mapping()
        return

    @classmethod
    def from_data_stream(cls, data_stream: DataStream) -> Self:
        """Create a DataStreamCollection from a DataStream object.

        Factory method to convert a single data stream or collection into a DataStreamCollection.

        Args:
            data_stream: Source data stream to convert.

        Returns:
            DataStreamCollection: New collection containing the source stream's data.

        Raises:
            TypeError: If the source is not a DataStream.
            ValueError: If the source has not been loaded yet.
        """
        if not isinstance(data_stream, DataStream):
            raise TypeError("data_stream must be an instance of DataStream.")
        if not data_stream.has_data:
            raise ValueError("DataStream has not been loaded yet. Cannot create DataStreamCollection.")
        data = data_stream._data if data_stream.is_collection else [data_stream._data]
        return cls(name=data_stream.name, data_streams=data, description=data_stream.description)


class Dataset(DataStreamCollection):
    """A version-tracked collection of data streams.

    Extends DataStreamCollection by adding semantic versioning support.

    Args:
        name: Name identifier for the dataset.
        data_streams: List of data streams to include in the dataset.
        version: Semantic version string or Version object. Defaults to "0.0.0".
        description: Optional description of the dataset.

    Examples:
        ```python
        from contraqctor.contract import text, csv, Dataset

        # Create streams
        text_stream = text.Text("notes", reader_params=text.TextParams(path="notes.txt"))
        csv_stream = csv.Csv("data", reader_params=csv.CsvParams(path="data.csv"))

        # Create a versioned dataset
        dataset = Dataset(
            "experiment_results",
            [text_stream, csv_stream],
            version="1.2.3"
        )

        # Load the dataset
        dataset.load_all(strict=True)

        # Access streams
        txt = dataset["notes"].data
        csv_data = dataset["data"].data

        print(f"Dataset version: {dataset.version}")
        ```
    """

    @override
    def __init__(
        self,
        name: str,
        data_streams: List[DataStream],
        *,
        version: str | Version = "0.0.0",
        description: Optional[str] = None,
    ) -> None:
        """Initializes a Dataset with a version and a list of data streams."""
        super().__init__(
            name=name,
            data_streams=data_streams,
            description=description,
        )
        self._version = self._parse_semver(version)

    @staticmethod
    def _parse_semver(version: str | Version) -> Version:
        """Parse a version string into a Version object.

        Args:
            version: Version string or object to parse.

        Returns:
            Version: Semantic version object.
        """
        if isinstance(version, str):
            return Version.parse(version)
        return version

    @property
    def version(self) -> Version:
        """Get the semantic version of the dataset.

        Returns:
            Version: Semantic version object.
        """
        return self._version


@dataclasses.dataclass
class FilePathBaseParam(abc.ABC):
    """Abstract base class for file-based reader parameters.

    Base parameter class for readers that access files by path.

    Attributes:
        path: Path to the file or directory to read from.
    """

    path: os.PathLike
