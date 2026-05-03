from dataclasses import dataclass
from typing import Optional, Self

import pandas as pd

from .base import DataStream


class Csv(DataStream[pd.DataFrame]):
    """CSV file data stream provider.

    A data stream implementation for reading CSV files into pandas DataFrames
    with configurable parameters for delimiter, header handling, and indexing.

    Args:
        DataStream: Base class for data stream providers.

    Examples:
        ```python
        from contraqctor.contract.csv import Csv, CsvParams

        # Create and load a CSV stream
        params = CsvParams(path="data/european_data.csv", delimiter=";")
        csv_stream = Csv("measurements", reader_params=params)
        csv_stream.load()

        # Access the DataFrame
        df = csv_stream.data
        filtered = df[df["temperature"] > 25]
        ```
    """

    delimiter: Optional[str] = None
    strict_header: bool = True
    index: Optional[str] = None

    def _reader(self) -> pd.DataFrame:
        """Read CSV file into a pandas DataFrame.

        Args:
            params: Parameters for CSV reading configuration.

        Returns:
            pd.DataFrame: DataFrame containing the parsed CSV data.
        """
        data = pd.read_csv(self.path, delimiter=self.delimiter, header=0 if self.strict_header else None)
        if self.index is not None:
            data.set_index(self.index, inplace=True)
        return data