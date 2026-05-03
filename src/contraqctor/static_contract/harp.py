import datetime
import io
import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Optional, Self, TextIO, Tuple, TypeAlias, Union

import harp
import harp.reader
import pandas as pd
import requests
import yaml
from pydantic import AnyHttpUrl, BaseModel, Field, dataclasses
from typing_extensions import TypeAliasType, override

from .. import _typing
from .base import DataStream, DataStreamCollectionBase, FilePathBaseParam

HarpRegisterParams: TypeAlias = harp.reader._ReaderParams

_DEFAULT_HARP_READER_PARAMS = HarpRegisterParams(base_path=None, epoch=None, keep_type=True)


class HarpRegister(DataStream[pd.DataFrame, HarpRegisterParams]):
    """Harp device register data stream provider.

    A data stream implementation for reading Harp device register data
    using the Harp Python library.

    Args:
        DataStream: Base class for data stream providers.
    """

    make_params = HarpRegisterParams

    @override
    def read(self, reader_params: Optional[HarpRegisterParams] = None) -> pd.DataFrame:
        """Read register data from Harp binary files.

        Args:
            reader_params: Parameters for register reading configuration.

        Returns:
            pd.DataFrame: DataFrame containing the register data.

        Raises:
            ValueError: If reader parameters are not set.
        """
        reader_params = reader_params if reader_params is not None else self._reader_params
        if _typing.is_unset(reader_params):
            raise ValueError("Reader parameters are not set. Cannot read data.")
        return self._reader(reader_params.base_path, epoch=reader_params.epoch, keep_type=reader_params.keep_type)

    @classmethod
    def from_register_reader(
        cls,
        name: str,
        reg_reader: harp.reader.RegisterReader,
        params: HarpRegisterParams = _DEFAULT_HARP_READER_PARAMS,
    ) -> Self:
        """Create a HarpRegister data stream from a RegisterReader.

        Factory method to create a HarpRegister instance from an existing
        Harp RegisterReader object.

        Args:
            name: Name for the register data stream.
            reg_reader: Harp RegisterReader object.
            params: Parameters for register reading configuration.

        Returns:
            HarpRegister: Newly created HarpRegister data stream.
        """
        c = cls(
            name=name,
            description=reg_reader.register.description,
        )
        c.bind_reader_params(params)
        c._reader = reg_reader.read
        c.make_params = cls.make_params
        return c


class _DeviceYmlSource(BaseModel):
    """Base class for device YAML file sources.

    Abstract base model for different methods of obtaining device YAML files.

    Attributes:
        method: The method used to obtain the device YAML file.
    """

    method: str


class DeviceYmlByWhoAmI(_DeviceYmlSource):
    """Device YAML source that finds the file using WhoAmI value.

    Specifies that the device YAML should be obtained by looking up a device
    by its WhoAmI identifier.

    Attributes:
        method: Fixed as "whoami".
        who_am_i: The WhoAmI value of the device (0-9999).
    """

    method: Literal["whoami"] = "whoami"
    who_am_i: Annotated[int, Field(ge=0, le=9999, description="WhoAmI value")]


class DeviceYmlByFile(_DeviceYmlSource):
    """Device YAML source that specifies a file path.

    Specifies that the device YAML should be loaded from a local file path.

    Attributes:
        method: Fixed as "file".
        path: Optional path to the device YAML file. If None, assumes "device.yml" in the data directory.
    """

    method: Literal["file"] = "file"
    path: Optional[os.PathLike | str] = Field(default=None, description="Path to the device yml file")


class DeviceYmlByUrl(_DeviceYmlSource):
    """Device YAML source that fetches from a URL.

    Specifies that the device YAML should be downloaded from a URL.

    Attributes:
        method: Fixed as "http".
        url: HTTP URL to download the device YAML file from.
    """

    method: Literal["http"] = "http"
    url: AnyHttpUrl = Field(description="URL to the device yml file")


class DeviceYmlByRegister0(_DeviceYmlSource):
    """Device YAML source that infers from register 0 file.

    Specifies that the device YAML should be determined by finding and reading
    the WhoAmI value from register 0 files.

    Attributes:
        method: Fixed as "register0".
        register0_glob_pattern: List of glob patterns to locate register 0 files.
    """

    method: Literal["register0"] = "register0"
    register0_glob_pattern: List[str] = Field(
        default=["*_0.bin", "*whoami*.bin"],
        min_length=1,
        description="Glob pattern to match the WhoAmI (0) register file",
    )


if TYPE_CHECKING:
    DeviceYmlSource = Union[DeviceYmlByWhoAmI, DeviceYmlByFile, DeviceYmlByUrl, DeviceYmlByRegister0]
else:
    DeviceYmlSource: TypeAliasType = Annotated[
        Union[DeviceYmlByWhoAmI, DeviceYmlByFile, DeviceYmlByUrl, DeviceYmlByRegister0], Field(discriminator="method")
    ]


@dataclasses.dataclass
class HarpDeviceParams(FilePathBaseParam):
    """Parameters for Harp device data reading.

    Defines parameters for locating and reading Harp device data.

    Attributes:
        device_yml_hint: Source for the device YAML configuration file.
        include_common_registers: Whether to include common registers. Defaults to True.
        keep_type: Whether to preserve type information. Defaults to True.
        epoch: Reference datetime for timestamp calculations. If provided, timestamps are converted to datetime.
    """

    device_yml_hint: DeviceYmlSource = Field(
        default=DeviceYmlByFile(), description="Device yml hint", validate_default=True
    )
    include_common_registers: bool = Field(default=True, description="Include common registers")
    keep_type: bool = Field(default=True, description="Keep message type information")
    epoch: Optional[datetime.datetime] = Field(
        default=None,
        description="Reference datetime at which time zero begins. If specified, the result data frame will have a datetime index.",
    )


def _harp_device_reader(
    params: HarpDeviceParams,
) -> Tuple[List[HarpRegister], harp.reader.DeviceReader]:
    """Internal function to read Harp device data.

    Creates Harp register streams based on the provided parameters.

    Args:
        params: Parameters for Harp device reading configuration.

    Returns:
        tuple: A tuple containing a list of HarpRegister objects and the DeviceReader.

    Raises:
        FileNotFoundError: If required files cannot be found.
        ValueError: If there are issues with the device YAML configuration.
    """
    _yml_stream: str | os.PathLike | TextIO
    match params.device_yml_hint:
        case DeviceYmlByWhoAmI(who_am_i=who_am_i):
            # If WhoAmI is provided we xref it to the device list to find the correct device.yml
            _yml_stream = io.TextIOWrapper(fetch_yml_from_who_am_i(who_am_i))

        case DeviceYmlByRegister0(register0_glob_pattern=glob_pattern):
            # If we are allowed to infer the WhoAmI, we try to find it
            _reg_0_hint: List[os.PathLike] = []
            for pattern in glob_pattern:
                _reg_0_hint.extend(Path(params.path).glob(pattern))
            if len(_reg_0_hint) == 0:
                raise FileNotFoundError(
                    "File corresponding to WhoAmI register not found given the provided glob patterns."
                )
            device_hint = int(
                harp.read(_reg_0_hint[0]).values[0][0]
            )  # We read the first line of the file to get the WhoAmI value
            _yml_stream = io.TextIOWrapper(fetch_yml_from_who_am_i(device_hint))

        case DeviceYmlByFile(path=path):
            # If a device.yml is provided we trivially pass it to the reader
            if path is None:
                path = Path(params.path) / "device.yml"
            else:
                path = Path(path)
            _yml_stream = io.TextIOWrapper(open(path, "rb"))

        case DeviceYmlByUrl(url=url):
            # If a device.yml URL is provided we fetch it and pass it to the reader
            response = requests.get(url, allow_redirects=True, timeout=5)
            response.raise_for_status()
            if response.status_code == 200:
                _yml_stream = io.TextIOWrapper(io.BytesIO(response.content))
            else:
                raise ValueError(f"Failed to fetch device yml from {url}")

        case _:
            raise ValueError("Invalid device yml hint")

    reader = _make_device_reader(_yml_stream, params)
    data_streams: List[HarpRegister] = []

    for name, reg_reader in reader.registers.items():
        # todo we can add custom file name interpolation here
        data_streams.append(HarpRegister.from_register_reader(name, reg_reader, _DEFAULT_HARP_READER_PARAMS))
    return (data_streams, reader)


def _make_device_reader(yml_stream: str | os.PathLike | TextIO, params: HarpDeviceParams) -> harp.reader.DeviceReader:
    """Create a Harp DeviceReader from a YAML stream.

    Args:
        yml_stream: Device YAML file as a stream, path, or string.
        params: Parameters for Harp device reading configuration.

    Returns:
        harp.reader.DeviceReader: Harp DeviceReader configured for the device.
    """
    device = harp.read_schema(yml_stream, include_common_registers=params.include_common_registers)
    path = Path(params.path)
    base_path = path / device.device if path.is_dir() else path.parent / device.device
    reg_readers = {
        name: harp.reader._create_register_handler(
            device,
            name,
            HarpRegisterParams(base_path=base_path, epoch=params.epoch, keep_type=params.keep_type),
        )
        for name in device.registers.keys()
    }
    return harp.reader.DeviceReader(device, reg_readers)


def fetch_yml_from_who_am_i(who_am_i: int, release: str = "main") -> io.BytesIO:
    """Fetch a device YAML file based on its WhoAmI identifier.

    Looks up the device in the WhoAmI registry and downloads its YAML file.

    Args:
        who_am_i: WhoAmI identifier of the device.
        release: Git branch or tag to use for fetching the YAML file.

    Returns:
        io.BytesIO: Memory buffer containing the device YAML content.

    Raises:
        KeyError: If the WhoAmI identifier is not found in the registry.
        ValueError: If required repository information is missing or YAML file cannot be found.
    """
    try:
        device = fetch_who_am_i_list()[who_am_i]
    except KeyError as e:
        raise KeyError(f"WhoAmI {who_am_i} not found in whoami.yml") from e

    repository_url = device.get("repositoryUrl", None)

    if repository_url is None:
        raise ValueError("Device's repositoryUrl not found in whoami.yml")

    _repo_hint_paths = [
        "{repository_url}/{release}/device.yml",
        "{repository_url}/{release}/software/bonsai/device.yml",
    ]

    yml = None
    for hint in _repo_hint_paths:
        url = hint.format(repository_url=repository_url, release=release)
        if "github.com" in url:
            url = url.replace("github.com", "raw.githubusercontent.com")
        response = requests.get(url, allow_redirects=True, timeout=5)
        if response.status_code == 200:
            yml = io.BytesIO(response.content)
            return yml

    raise ValueError("device.yml not found in any repository")


@cache
def fetch_who_am_i_list(
    url: str = "https://raw.githubusercontent.com/harp-tech/whoami/main/whoami.yml",
) -> Dict[int, Any]:
    """Fetch and parse the Harp WhoAmI registry.

    Downloads and parses the WhoAmI registry YAML file from GitHub.
    Results are cached for efficiency.

    Args:
        url: URL to the WhoAmI registry YAML file.

    Returns:
        Dict[int, Any]: Dictionary mapping WhoAmI identifiers to device information.
    """
    response = requests.get(url, allow_redirects=True, timeout=5)
    content = response.content.decode("utf-8")
    content = yaml.safe_load(content)
    devices = content["devices"]
    return devices


class HarpDevice(DataStreamCollectionBase[HarpRegister, HarpDeviceParams]):
    """Harp device data stream collection provider.

    A data stream collection for accessing all registers of a Harp device.

    Args:
        DataStreamCollectionBase: Base class for data stream collection providers.

    Examples:
        ```python
        from contraqctor.contract.harp import HarpDevice, HarpDeviceParams, DeviceYmlByWhoAmI

        # Create and load a device stream
        params = HarpDeviceParams(
            path="behavior.harp",
            device_yml_hint=DeviceYmlByWhoAmI(who_am_i=1216)
        )

        behavior = HarpDevice("behavior", reader_params=params).load()

        # Access registers
        digital_input = behavior["DigitalInputState"].data
        adc = behavior["AnalogData"].data
        ```
    """

    make_params = HarpDeviceParams
    _device_reader: Optional[harp.reader.DeviceReader]

    @property
    def device_reader(self) -> harp.reader.DeviceReader:
        """Get the underlying Harp device reader.

        Returns:
            harp.reader.DeviceReader: Harp device reader for accessing raw device functionality.

        Raises:
            ValueError: If the device reader has not been initialized.
        """
        if not hasattr(self, "_device_reader"):
            raise ValueError("Device reader is not set. Cannot read data.")
        if self._device_reader is None:
            raise ValueError("Device reader is not set. Cannot read data.")
        return self._device_reader

    def _reader(self, params: HarpDeviceParams) -> List[HarpRegister]:
        """Create register data streams from Harp device data.

        Args:
            params: Parameters for Harp device reading configuration.

        Returns:
            List[HarpRegister]: List of data streams, one per device register.
        """
        regs, reader = _harp_device_reader(params)
        self._device_reader = reader
        return regs
