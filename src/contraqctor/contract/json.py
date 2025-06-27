import dataclasses
import json
import os
from typing import Generic, Optional, Type, TypeVar

import aind_behavior_services
import aind_behavior_services.data_types
import pandas as pd
import pydantic

from .base import DataStream, FilePathBaseParam


@dataclasses.dataclass
class JsonParams:
    """Parameters for JSON file processing.

    Defines parameters for reading JSON files with specified encoding.

    Attributes:
        path: Path to the JSON file.
        encoding: Character encoding for the JSON file. Defaults to UTF-8.
    """

    path: os.PathLike
    encoding: str = "UTF-8"


class Json(DataStream[dict[str, str], JsonParams]):
    """JSON file data stream provider.

    A data stream implementation for reading single JSON objects from files.

    Args:
        DataStream: Base class for data stream providers.

    Examples:
        ```python
        from contraqctor.contract.json import Json, JsonParams

        # Create and load a JSON stream
        config_stream = Json(
            "config",
            reader_params=JsonParams(path="config/settings.json")
        )
        config_stream.load()

        # Access the data
        config = config_stream.data
        api_key = config.get("api_key")
        ```
    """

    @staticmethod
    def _reader(params: JsonParams) -> dict[str, str]:
        """Read JSON file into a dictionary.

        Args:
            params: Parameters for JSON file reading configuration.

        Returns:
            dict: Dictionary containing the parsed JSON data.

        Examples:
            ```python
            from contraqctor.contract.json import Json, JsonParams

            params = JsonParams(path="user_profile.json")
            data = Json._reader(params)
            username = data.get("username")
            ```
        """
        with open(params.path, "r", encoding=params.encoding) as file:
            data = json.load(file)
        return data

    make_params = JsonParams


class MultiLineJson(DataStream[list[dict[str, str]], JsonParams]):
    """Multi-line JSON file data stream provider.

    A data stream implementation for reading JSON files where each line
    contains a separate JSON object.

    Args:
        DataStream: Base class for data stream providers.

    Examples:
        ```python
        from contraqctor.contract.json import MultiLineJson, JsonParams

        # Create and load a multi-line JSON stream
        logs_stream = MultiLineJson(
            "server_logs",
            reader_params=JsonParams(path="logs/server_logs.jsonl")
        )
        logs_stream.load()

        # Process log entries
        for entry in logs_stream.data:
            if entry.get("level") == "ERROR":
                print(f"Error: {entry.get('message')}")
        ```
    """

    @staticmethod
    def _reader(params: JsonParams) -> list[dict[str, str]]:
        """Read multi-line JSON file into a list of dictionaries.

        Args:
            params: Parameters for JSON file reading configuration.

        Returns:
            list: List of dictionaries, each containing a parsed JSON object from one line.

        Examples:
            Using the reader directly to process events:

            ```python
            from contraqctor.contract.json import MultiLineJson, JsonParams

            # Set up parameters
            params = JsonParams(path="events/user_clicks.jsonl")

            # Read the JSON directly
            events = MultiLineJson._reader(params)

            # Calculate statistics
            clicks_by_user = {}
            for event in events:
                user_id = event.get("user_id")
                clicks_by_user[user_id] = clicks_by_user.get(user_id, 0) + 1
            ```
        """
        with open(params.path, "r", encoding=params.encoding) as file:
            data = [json.loads(line) for line in file]
        return data

    make_params = JsonParams


_TModel = TypeVar("_TModel", bound=pydantic.BaseModel)


@dataclasses.dataclass
class PydanticModelParams(FilePathBaseParam, Generic[_TModel]):
    """Parameters for Pydantic model-based JSON file processing.

    Extends the base file path parameters with Pydantic model specification
    for parsing JSON into typed objects.

    Attributes:
        model: Pydantic model class to use for parsing JSON data.
        encoding: Character encoding for the JSON file. Defaults to UTF-8.

    Examples:
        ```python
        from pydantic import BaseModel
        from contraqctor.contract.json import PydanticModelParams

        class User(BaseModel):
            user_id: str
            name: str
            active: bool = True

        params = PydanticModelParams(path="users/profile.json", model=User)
        ```
    """

    model: Type[_TModel]
    encoding: str = "UTF-8"


class PydanticModel(DataStream[_TModel, PydanticModelParams[_TModel]]):
    """Pydantic model-based JSON data stream provider.

    A data stream implementation for reading JSON files as Pydantic model instances.

    Args:
        DataStream: Base class for data stream providers.

    Examples:
        ```python
        from pydantic import BaseModel
        from contraqctor.contract.json import PydanticModel, PydanticModelParams

        class ServerConfig(BaseModel):
            host: str
            port: int
            debug: bool = False

        params = PydanticModelParams(path="config/server.json", model=ServerConfig)

        config_stream = PydanticModel("server_config", reader_params=params).load()
        server_config = config_stream.data
        print(f"Server: {server_config.host}:{server_config.port}")
        ```
    """

    @staticmethod
    def _reader(params: PydanticModelParams[_TModel]) -> _TModel:
        """Read JSON file and parse it as a Pydantic model.

        Args:
            params: Parameters for Pydantic model-based reading configuration.

        Returns:
            _TModel: Instance of the specified Pydantic model populated from JSON data.

        Examples:
            Using the reader directly with a model:

            ```python
            from pydantic import BaseModel
            from datetime import datetime
            from contraqctor.contract.json import PydanticModel, PydanticModelParams

            # Define a model for an experiment
            class Experiment(BaseModel):
                id: str
                name: str
                start_date: datetime
                completed: bool
                parameters: dict

            # Set up parameters
            params = PydanticModelParams(
                path="experiments/exp_001.json",
                model=Experiment
            )

            # Read and validate the JSON as an Experiment
            experiment = PydanticModel._reader(params)

            # Work with the validated model
            if experiment.completed:
                print(f"Experiment {experiment.name} completed on {experiment.start_date}")
            ```
        """
        with open(params.path, "r", encoding=params.encoding) as file:
            return params.model.model_validate_json(file.read())

    make_params = PydanticModelParams


@dataclasses.dataclass
class ManyPydanticModelParams(FilePathBaseParam, Generic[_TModel]):
    """Parameters for loading multiple Pydantic models from a file.

    Extends the base file path parameters with Pydantic model specification
    and options for converting to a DataFrame.

    Attributes:
        model: Pydantic model class to use for parsing JSON data.
        encoding: Character encoding for the JSON file. Defaults to UTF-8.
        index: Optional column name to set as the DataFrame index.
        column_names: Optional dictionary mapping original column names to new names.

    Examples:
        Defining parameters to load multiple models:

        ```python
        from pydantic import BaseModel
        from contraqctor.contract.json import ManyPydanticModelParams

        # Define a Pydantic model for log entries
        class LogEntry(BaseModel):
            timestamp: str
            level: str
            message: str

        # Create parameters for loading log entries
        params = ManyPydanticModelParams(
            path="logs/server_logs.json",
            model=LogEntry,
            index="timestamp",
            column_names={"level": "log_level", "message": "log_message"}
        )
        ```
    """

    model: Type[_TModel]
    encoding: str = "UTF-8"
    index: Optional[str] = None
    column_names: Optional[dict[str, str]] = None


class ManyPydanticModel(DataStream[pd.DataFrame, ManyPydanticModelParams[_TModel]]):
    """Multi-model JSON data stream provider.

    A data stream implementation for reading multiple JSON objects from a file,
    parsing them as Pydantic models, and returning them as a DataFrame.

    Args:
        DataStream: Base class for data stream providers.

    Examples:
        Loading server logs into a DataFrame:

        ```python
        from contraqctor.contract.json import ManyPydanticModel, ManyPydanticModelParams

        # Create and load the data stream
        logs_stream = ManyPydanticModel(
            "server_logs_df",
            reader_params=params
        )
        logs_stream.load()

        # Access the logs as a DataFrame
        logs_df = logs_stream.data

        # Analyze the logs
        error_logs = logs_df[logs_df["log_level"] == "ERROR"]
        ```
    """

    @staticmethod
    def _reader(params: ManyPydanticModelParams[_TModel]) -> pd.DataFrame:
        """Read multiple JSON objects and convert them to a DataFrame.

        Args:
            params: Parameters for multi-model reading configuration.

        Returns:
            pd.DataFrame: DataFrame containing data from multiple model instances.

        Examples:
            Using the reader directly to create a DataFrame:

            ```python
            from contraqctor.contract.json import ManyPydanticModel, ManyPydanticModelParams

            # Set up parameters
            params = ManyPydanticModelParams(
                path="data/transactions.json",
                model=Transaction,
                index="transaction_id",
                column_names={"amount": "transaction_amount"}
            )

            # Read the JSON lines and create the DataFrame
            transactions_df = ManyPydanticModel._reader(params)

            # Perform analysis
            total_amount = transactions_df["transaction_amount"].sum()
            ```
        """
        with open(params.path, "r", encoding=params.encoding) as file:
            model_ls = pd.DataFrame([params.model.model_validate_json(line).model_dump() for line in file])
        if params.column_names is not None:
            model_ls.rename(columns=params.column_names, inplace=True)
        if params.index is not None:
            model_ls.set_index(params.index, inplace=True)
        return model_ls

    make_params = ManyPydanticModelParams


@dataclasses.dataclass
class SoftwareEventsParams(ManyPydanticModelParams):
    """Parameters for software events file processing.

    A specialized version of ManyPydanticModelParams that defaults to using
    the SoftwareEvent model from aind_behavior_services.

    Attributes:
        model: Set to SoftwareEvent model and not modifiable after initialization.
        encoding: Character encoding for the JSON file. Defaults to UTF-8.
        index: Optional column name to set as the DataFrame index.
        column_names: Optional dictionary mapping original column names to new names.

    Examples:
        Defining parameters for loading software events:

        ```python
        from contraqctor.contract.json import SoftwareEventsParams

        # Create parameters for software events
        params = SoftwareEventsParams(
            path="events/software_events.json",
            index="event_id",
            column_names={"timestamp": "event_time"}
        )
        ```
    """

    model: Type[aind_behavior_services.data_types.SoftwareEvent] = dataclasses.field(
        default=aind_behavior_services.data_types.SoftwareEvent, init=False
    )
    encoding: str = "UTF-8"
    index: Optional[str] = "timestamp"
    column_names: Optional[dict[str, str]] = None


class SoftwareEvents(ManyPydanticModel[aind_behavior_services.data_types.SoftwareEvent]):
    """Software events data stream provider.

    A specialized data stream for reading software event logs from JSON files
    using the SoftwareEvent model from aind_behavior_services.

    Args:
        ManyPydanticModel: Base class for multi-model data stream providers.

    Examples:
        Analyzing software events data:

        ```python
        from contraqctor.contract.json import SoftwareEvents, SoftwareEventsParams

        # Create parameters for software events
        params = SoftwareEventsParams(
            path="events/software_events.json",
            index="event_id"
        )

        # Create and load the software events stream
        events_stream = SoftwareEvents(
            "software_events",
            reader_params=params
        )
        events_stream.load()

        # Access the events data
        events_df = events_stream.data

        # Perform analysis, e.g., count events by type
        event_counts = events_df["event_type"].value_counts()
        ```
    """

    make_params = SoftwareEventsParams
