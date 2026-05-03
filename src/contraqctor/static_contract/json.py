import dataclasses
import json
import os
from typing import Generic, Optional, Type, TypeVar, Any

import aind_behavior_services
import aind_behavior_services.data_types
import pandas as pd
import pydantic

from .base import DataStream

class Json(DataStream[dict[str, Any]]):
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
    encoding: str = "UTF-8"

    def _reader(self) -> dict[str, Any]:
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
        with open(self.path, "r", encoding=self.encoding) as file:
            data = json.load(file)
        return data


class MultiLineJson(DataStream[list[dict[str, Any]]]):
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
    encoding: str = "UTF-8"


    def _reader(self) -> list[dict[str, Any]]:
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
        with open(self.path, "r", encoding=self.encoding) as file:
            data = [json.loads(line) for line in file]
        return data


_TModel = TypeVar("_TModel", bound=pydantic.BaseModel)

class PydanticModel(DataStream[_TModel]):
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
    encoding: str = "UTF-8"
    _model: Type[_TModel]

    def _reader(self) -> _TModel:
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
        with open(self.path, "r", encoding=self.encoding) as file:
            # todo handle the TypeAdapter case here too
            return self._model.model_validate_json(file.read())


class ManyPydanticModel(DataStream[pd.DataFrame], Generic[_TModel]):
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
    _model: Type[_TModel]
    encoding: str = "UTF-8"
    index: Optional[str] = None
    column_names: Optional[dict[str, str]] = None

    def _reader(self) -> pd.DataFrame:
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
        with open(self.path, "r", encoding=self.encoding) as file:
            model_ls = pd.DataFrame([self._model.model_validate_json(line).model_dump() for line in file])
        if self.column_names is not None:
            model_ls.rename(columns=self.column_names, inplace=True)
        if self.index is not None:
            model_ls.set_index(self.index, inplace=True)
        return model_ls


_T = TypeVar("_T", bound=Any, default=Any)


class SoftwareEvents(ManyPydanticModel[aind_behavior_services.data_types.SoftwareEvent[_T]], Generic[_T]):
    pass