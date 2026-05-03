from dataclasses import dataclass

from .base import DataStream

class Text(DataStream[str]):
    """Text file data stream provider.

    A data stream implementation for reading text files as a single string
    with configurable character encoding.

    Args:
        DataStream: Base class for data stream providers.

    Examples:
        ```python
        from contraqctor.contract.text import Text, TextParams

        # Create and load a text stream
        params = TextParams(path="README.md")
        readme_stream = Text("readme", reader_params=params).load()

        # Access the content
        content = readme_stream.data
        ```
    """
    encoding: str = "UTF-8"

    def _reader(self) -> str:
        """Read text file into a string.

        Args:
            params: Parameters for text file reading configuration.

        Returns:
            str: String containing the contents of the text file.
        """
        with open(self.path, "r", encoding=self.encoding) as file:
            return file.read()
