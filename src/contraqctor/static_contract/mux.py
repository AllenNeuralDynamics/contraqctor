import dataclasses
import os
from pathlib import Path
from typing import Any, Annotated, Callable, Generic, List, Optional, Type, TypeVar
import pydantic
from pydantic.fields import FieldInfo
from .. import _typing
from .base import DataStream, DataStreamCollection

_TDataStream = TypeVar("_TDataStream", bound=DataStream[Any])

def map_from_paths(
    model_name: str,
    paths: List[os.PathLike] | os.PathLike,
    cls: Type[_TDataStream],
    field_factory: Callable[[os.PathLike], FieldInfo] | None = None,
    include_glob_pattern: List[str] | None = None,
    exclude_glob_pattern: List[str] | None = None) -> Type["MappedFromPaths[_TDataStream]"]:

    if not isinstance(paths, list):
        paths = [paths]
    if len(paths) == 0:
        raise ValueError("At least one path must be provided.")
    paths = [Path(p) for p in paths]
    
    include_glob_pattern = include_glob_pattern or []
    exclude_glob_pattern = exclude_glob_pattern or []
    _hits: List[Path] = []

    for p in paths:
        for pattern in include_glob_pattern:
            _hits.extend(list(Path(p).glob(pattern)))
        for pattern in exclude_glob_pattern:
            _hits = [f for f in _hits if not f.match(pattern)]
    _hits = list(set(_hits))
    if len(list(set([f.stem for f in _hits]))) != len(_hits):
        raise ValueError(f"Duplicate stems found in glob pattern: {include_glob_pattern}.")

    _keyed_hits = {f.stem: f for f in _hits}
    _fields: dict[str, FieldInfo] = {}
    if field_factory is None:
        for hit in _keyed_hits:
            _inner_path = _keyed_hits[hit]

            def _builder() -> _TDataStream:
                return cls(path=str(_inner_path))

            _fields[hit] = pydantic.Field(default_factory=_builder)
    else:
        for hit in _keyed_hits:
            _fields[hit] = field_factory(_keyed_hits[hit])
    
    new_props: dict[str, Any] = {}
    for new_prop in _fields:
        new_props[new_prop] = Annotated[MappedFromPaths, _fields[new_prop]]

    return pydantic.create_model(model_name, __base__=MappedFromPaths[_TDataStream], **new_props)

class MappedFromPaths(DataStreamCollection[_TDataStream]):
    """File path mapper data stream provider.

    A data stream implementation for creating multiple child data streams
    by searching for files matching glob patterns and creating a stream for each.

    Args:
        DataStreamCollectionBase: Base class for data stream collection providers.

    Examples:
        ```python
        from contraqctor.contract import mux, text

        # Define a factory function for TextParams
        def create_text_params(file_path):
            return text.TextParams(path=file_path)

        # Create and load a text file collection
        params = mux.MapFromPathsParams(
            paths=["documents/"],
            include_glob_pattern=["*.txt"],
            inner_data_stream=text.Text,
            inner_param_factory=create_text_params
        )

        docs = mux.MapFromPaths("documents", reader_params=params).load()
        readme = docs["readme"].data
        ```
    """

    @staticmethod
    def _reader(params: MapFromPathsParams[_TDataStream]) -> List[_TDataStream]:
        """Create data streams for files matching the specified patterns.

        Args:
            params: Parameters for file path mapping configuration.

        Returns:
            List[_TDataStream]: List of data stream objects, one per matched file.

        Raises:
            ValueError: If duplicate file stems (names without extensions) are found.

        Examples:
            ```python
            from contraqctor.contract import mux, csv

            def make_csv_params(file_path):
                return csv.CsvParams(path=file_path)

            params = mux.MapFromPathsParams(
                paths=["data/sensors/"],
                include_glob_pattern=["*.csv"],
                inner_data_stream=csv.Csv,
                inner_param_factory=make_csv_params
            )

            # Get streams directly
            streams = mux.MapFromPaths._reader(params)
            ```
        """
        _hits: List[Path] = []

        for p in params.paths:
            for pattern in params.include_glob_pattern:
                _hits.extend(list(Path(p).glob(pattern)))
            for pattern in params.exclude_glob_pattern:
                _hits = [f for f in _hits if not f.match(pattern)]
            _hits = list(set(_hits))

        if len(list(set([f.stem for f in _hits]))) != len(_hits):
            raise ValueError(f"Duplicate stems found in glob pattern: {params.include_glob_pattern}.")

        _out: List[_TDataStream] = []
        _descriptions = params.inner_descriptions
        for f in _hits:
            _out.append(
                params.inner_data_stream(
                    name=f.stem,
                    description=_descriptions.get(f.stem, None),
                    reader_params=params.inner_param_factory(f),
                )
            )
        return _out
