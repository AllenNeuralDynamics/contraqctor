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
    final
)
from pydantic import BaseModel, computed_field, Field, field_validator, model_validator
from functools import cache

from semver import Version
from typing_extensions import override

from contraqctor import _typing


class _CoreModel(BaseModel):
    """Base model for core data structures in the contraqctor contract system."""
    pass


class DataStream(abc.ABC, _CoreModel, Generic[_typing.TData]):
    """Abstract base class for all data streams.
    """

    description: Optional[str] = Field(default=None, description="Optional description of the data stream.")
    parent: Optional["DataStream"] = Field(default=None, description="Parent data stream, if any.")
    path: str = Field(description="Path to the data source.")

    def set_parent(self, parent: "DataStream") -> None:
        """Set the parent data stream.

        Args:
            parent: The parent data stream to set.
        """
        self.parent = parent

    @abc.abstractmethod
    def _reader(self) -> _typing.TData:
        """Reader function to be implemented by subclasses.

        This function should contain the logic to read data from the source.

        Args:
            cls: The class of the data stream being read.
            instance: An instance of the class to read data from.

        Returns:
            TData: Data read from the source.
        """
        raise NotImplementedError("Subclasses must implement the _reader method.")

    @cache
    @final
    def read(self: Self) -> _typing.TData:
        """Read data using the configured reader.

        Returns:
            TData: Data read from the source.
        """
        return self._reader()
    
    def try_read(self: Self) -> _typing.Maybe[_typing.TData]:
        try:
            return _typing.Maybe(self.read(), None)
        except Exception as err:
            return _typing.Maybe(None, err)


TDataStream = TypeVar("TDataStream", bound=DataStream)


class DataStreamCollection(
    DataStream[List[TDataStream]],
    Generic[TDataStream],
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
    data_streams: List[TDataStream] = Field(default_factory=list[TDataStream], description="The data streams known to the collection")

    @model_validator(mode="after")
    def _ensure_parent(self) -> Self:
        for ds in self.data_streams:
            ds.set_parent(self)
        return self

    def iter(self) -> Generator[DataStream, None, None]:
        """Iterator for child data streams.

        Yields:
            DataStream: Child data streams.

        """
        # We intentionally yield from self.data to trigger
        # automatic loading if needed
        yield from self.data_streams

    def walk(self) -> Generator[DataStream, None, None]:
        """Iterator for all child data streams, including nested collections.

        Implements a depth-first traversal of the stream hierarchy.

        Yields:
            DataStream: All recursively yielded child data streams.
        """
        for value in self.walk():
            if isinstance(value, DataStream):
                yield value
            if isinstance(value, "DataStreamCollection"):
                yield from value.walk()

class Dataset(DataStream):
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
    version: str = Field(description="Stores the version of the dataset")
