import datetime
import io
import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Optional, Self, Tuple, Union

import harp.data
import pandas as pd
import requests
import yaml
from harp.data import DatasetReader, open_dataset
from harp.device import core
from harp.device.schema import create_device_module, parse_device_schema
from harp.protocol import RegisterBase
from pydantic import AnyHttpUrl, BaseModel, Field, dataclasses
from typing_extensions import TypeAliasType

from .base import DataStream, DataStreamCollectionBase, FilePathBaseParam

if TYPE_CHECKING:
    from harp.device.schema._model import DeviceModel


@dataclasses.dataclass(frozen=True)
class HarpRegisterParams:
    """Parameters for decoding a Harp register into a DataFrame.

    Attributes:
        keep_type: Whether to include the ``message_type`` column. Defaults to True.
        decode_enums: Whether to decode group mask values as ``pd.Categorical``. Defaults to True.
        demux_bit_masks: Whether to expand bit mask registers into one boolean column per flag.
            Defaults to False, which keeps a single integer ``value`` column.
    """

    keep_type: bool = Field(default=True, description="Keep message type information")
    decode_enums: bool = Field(default=True, description="Decode group mask values as categoricals")
    demux_bit_masks: bool = Field(default=False, description="Expand bit masks into one column per flag")


_DEFAULT_HARP_READER_PARAMS = HarpRegisterParams()


class HarpRegister(DataStream[pd.DataFrame, HarpRegisterParams]):
    """Harp device register data stream provider.

    A data stream implementation for reading Harp device register data
    using the ``harp.data`` library.

    Args:
        DataStream: Base class for data stream providers.
    """

    make_params = HarpRegisterParams
    _register_class: type[RegisterBase[Any]]

    @property
    def register_class(self) -> type[RegisterBase[Any]]:
        """Get the ``harp.protocol`` register class, which carries e.g. its ``address`` and ``payload_type``.

        Returns:
            type[RegisterBase]: The register class from the device module.
        """
        return self._register_class

    @classmethod
    def from_dataset_reader(
        cls,
        name: str,
        dataset_reader: DatasetReader,
        register_class: type[RegisterBase[Any]],
        *,
        description: Optional[str] = None,
        params: HarpRegisterParams = _DEFAULT_HARP_READER_PARAMS,
    ) -> Self:
        """Create a HarpRegister data stream backed by a ``harp.data.DatasetReader``.

        Args:
            name: Name for the register data stream.
            dataset_reader: Reader over the device dataset folder.
            register_class: The register class from the device module.
            description: Optional description of the register.
            params: Parameters for register decoding.

        Returns:
            HarpRegister: Newly created HarpRegister data stream.
        """
        c = cls(name=name, description=description)
        c.bind_reader_params(params)
        c._register_class = register_class

        def _reader(params: HarpRegisterParams) -> pd.DataFrame:
            """Read the register data from the dataset.

            Args:
                params: Parameters for register decoding.

            Returns:
                pd.DataFrame: DataFrame containing the register data.

            Raises:
                FileNotFoundError: If the register has no data file in the dataset.
            """
            # harp.data returns an empty frame for a register with no file; we raise
            # instead so that a missing register is reported as a load error.
            if register_class.address not in dataset_reader.paths:
                raise FileNotFoundError(f"No data file found for register {name}.")
            return dataset_reader.read(
                register_class,
                keep_type=params.keep_type,
                decode_enums=params.decode_enums,
                demux_bit_masks=params.demux_bit_masks,
            )

        c._reader = _reader
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
        DeviceYmlByWhoAmI | DeviceYmlByFile | DeviceYmlByUrl | DeviceYmlByRegister0, Field(discriminator="method")
    ]


@dataclasses.dataclass
class HarpDeviceParams(FilePathBaseParam):
    """Parameters for Harp device data reading.

    Defines parameters for locating and reading Harp device data.

    Attributes:
        device_yml_hint: Source for the device YAML configuration file.
        include_common_registers: Whether to include common registers. Defaults to True.
        keep_type: Whether to include the ``message_type`` column. Defaults to True.
        epoch: Reference datetime for timestamp calculations. If provided, timestamps are converted to datetime.
        decode_enums: Whether to decode group mask values as ``pd.Categorical``. Defaults to True.
        demux_bit_masks: Whether to expand bit mask registers into one boolean column per flag. Defaults to False.
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
    decode_enums: bool = Field(default=True, description="Decode group mask values as categoricals")
    demux_bit_masks: bool = Field(default=False, description="Expand bit masks into one column per flag")


def _resolve_device_yml(params: HarpDeviceParams) -> bytes:
    """Internal function to obtain the device YAML content from the configured source.

    Args:
        params: Parameters for Harp device reading configuration.

    Returns:
        bytes: The device YAML content.

    Raises:
        FileNotFoundError: If required files cannot be found.
        ValueError: If there are issues with the device YAML configuration.
    """
    match params.device_yml_hint:
        case DeviceYmlByWhoAmI(who_am_i=who_am_i):
            # If WhoAmI is provided we xref it to the device list to find the correct device.yml
            return _fetch_yml_bytes_from_who_am_i(who_am_i, "main")

        case DeviceYmlByRegister0(register0_glob_pattern=glob_pattern):
            # If we are allowed to infer the WhoAmI, we try to find it
            _reg_0_hint: List[Path] = []
            for pattern in glob_pattern:
                _reg_0_hint.extend(Path(params.path).glob(pattern))
            if len(_reg_0_hint) == 0:
                raise FileNotFoundError(
                    "File corresponding to WhoAmI register not found given the provided glob patterns."
                )
            device_hint = int(
                harp.data.read(_reg_0_hint[0])["value"].iloc[0]
            )  # We read the first line of the file to get the WhoAmI value
            return _fetch_yml_bytes_from_who_am_i(device_hint, "main")

        case DeviceYmlByFile(path=path):
            # If a device.yml is provided we trivially pass it to the reader
            if path is None:
                path = Path(params.path) / "device.yml"
            return Path(path).read_bytes()

        case DeviceYmlByUrl(url=url):
            # If a device.yml URL is provided we fetch it and pass it to the reader
            return _fetch_yml_bytes_from_url(str(url))

        case _:
            raise ValueError("Invalid device yml hint")


def _harp_device_reader(
    params: HarpDeviceParams,
) -> Tuple[List[HarpRegister], DatasetReader, "DeviceModel"]:
    """Internal function to read Harp device data.

    Creates Harp register streams based on the provided parameters.

    Args:
        params: Parameters for Harp device reading configuration.

    Returns:
        tuple: The list of HarpRegister objects, the DatasetReader, and the parsed device schema.
    """
    yml = _resolve_device_yml(params)
    # The module drives reading; the schema is kept for metadata the module does not carry
    # (register descriptions, firmware version). Custom interfaceTypes decode as their native type.
    schema = parse_device_schema(yml)
    module = create_device_module(yml, require_converters=False)
    path = Path(params.path)
    reader = open_dataset(path if path.is_dir() else path.parent, module, epoch=params.epoch)
    register_params = HarpRegisterParams(
        keep_type=params.keep_type,
        decode_enums=params.decode_enums,
        demux_bit_masks=params.demux_bit_masks,
    )

    data_streams: List[HarpRegister] = []
    for address, register_class in sorted(module.REGISTER_MAP.items()):
        name = register_class.__name__
        is_core = core.REGISTER_MAP.get(address) is register_class
        # Private registers (emitted with an underscore-prefixed class name) are never exposed
        if name.startswith("_") or (is_core and not params.include_common_registers):
            continue
        description = register_class.__doc__ if is_core else schema.registers[name].description
        data_streams.append(
            HarpRegister.from_dataset_reader(
                name, reader, register_class, description=description, params=register_params
            )
        )
    return (data_streams, reader, schema)


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
    return io.BytesIO(_fetch_yml_bytes_from_who_am_i(who_am_i, release))


@cache
def _fetch_yml_bytes_from_who_am_i(who_am_i: int, release: str) -> bytes:
    """Fetch the content of a device YAML file based on its WhoAmI identifier.

    Results are cached per ``(who_am_i, release)`` for the lifetime of the process.
    Always call it with both arguments positionally, as ``functools.cache`` keys on
    the arguments as passed.
    Failed fetches raise and are therefore not cached.

    Args:
        who_am_i: WhoAmI identifier of the device.
        release: Git branch or tag to use for fetching the YAML file.

    Returns:
        bytes: The device YAML content.
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

    for hint in _repo_hint_paths:
        url = hint.format(repository_url=repository_url, release=release)
        if "github.com" in url:
            url = url.replace("github.com", "raw.githubusercontent.com")
        response = requests.get(url, allow_redirects=True, timeout=5)
        if response.status_code == 200:
            return response.content

    raise ValueError("device.yml not found in any repository")


@cache
def _fetch_yml_bytes_from_url(url: str) -> bytes:
    """Fetch the content of a device YAML file from a URL.

    Results are cached per URL for the lifetime of the process.
    Failed fetches raise and are therefore not cached.

    Args:
        url: HTTP URL to download the device YAML file from.

    Returns:
        bytes: The device YAML content.
    """
    response = requests.get(url, allow_redirects=True, timeout=5)
    response.raise_for_status()
    return response.content


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
    _dataset_reader: Optional[DatasetReader]
    _device_schema: Optional["DeviceModel"]

    @property
    def dataset_reader(self) -> DatasetReader:
        """Get the underlying ``harp.data.DatasetReader``.

        Returns:
            harp.data.DatasetReader: Reader over the device dataset folder.

        Raises:
            ValueError: If the dataset reader has not been initialized.
        """
        reader = getattr(self, "_dataset_reader", None)
        if reader is None:
            raise ValueError("Dataset reader is not set. Cannot read data.")
        return reader

    @property
    def device_schema(self) -> "DeviceModel":
        """Get the parsed device schema (``device.yml``).

        Returns:
            DeviceModel: The device schema, exposing e.g. ``whoAmI`` and ``firmwareVersion``.

        Raises:
            ValueError: If the device schema has not been loaded.
        """
        schema = getattr(self, "_device_schema", None)
        if schema is None:
            raise ValueError("Device schema is not set. Cannot read data.")
        return schema

    def _reader(self, params: HarpDeviceParams) -> List[HarpRegister]:
        """Create register data streams from Harp device data.

        Args:
            params: Parameters for Harp device reading configuration.

        Returns:
            List[HarpRegister]: List of data streams, one per device register.
        """
        regs, reader, schema = _harp_device_reader(params)
        self._dataset_reader = reader
        self._device_schema = schema
        return regs
