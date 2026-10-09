"""Tests for the harp contract module against an on-disk Harp dataset."""

import struct

import pandas as pd
import pytest

from contraqctor.contract import harp as harp_contract
from contraqctor.contract.harp import (
    DeviceYmlByFile,
    DeviceYmlByUrl,
    DeviceYmlByWhoAmI,
    HarpDevice,
    HarpDeviceParams,
    HarpRegister,
    HarpRegisterParams,
)

DEVICE_YML = """\
%YAML 1.1
---
device: Dev
whoAmI: 1234
firmwareVersion: "1.2"
hardwareTargets: "1.0"
registers:
  LickState:
    address: 32
    type: U8
    access: Event
    maskType: Channels
    description: Lick state bitmask
  RawVoltage:
    address: 34
    type: U16
    access: Event
    description: Raw ADC
  SensorData:
    address: 35
    type: U16
    length: 2
    access: Event
    payloadSpec:
      Temperature: {offset: 0}
      Humidity: {offset: 1}
  Reserved0:
    address: 36
    type: U8
    access: Read
    visibility: private
  NeverLogged:
    address: 37
    type: U8
    access: Read
bitMasks:
  Channels:
    bits:
      Channel0: 0x1
      Channel1: 0x2
"""

_READ, _WRITE, _EVENT = 1, 2, 3
_U8, _U16 = 0x01, 0x02


def _frame(message_type: int, address: int, payload_type: int, fmt: str, values: tuple, seconds: int) -> bytes:
    payload = struct.pack("<" + fmt, *values)
    body = bytes([address, 255, payload_type | 0x10]) + struct.pack("<IH", seconds, 0) + payload
    message = bytes([message_type, len(body) + 1]) + body
    return message + bytes([sum(message) & 0xFF])


@pytest.fixture
def harp_dataset(tmp_path):
    root = tmp_path / "Dev.harp"
    root.mkdir()
    (root / "device.yml").write_text(DEVICE_YML)

    def write(address, payload_type, fmt, rows):
        (root / f"Dev_{address}.bin").write_bytes(
            b"".join(_frame(mt, address, payload_type, fmt, v, s) for mt, v, s in rows)
        )

    write(0, _U16, "H", [(_READ, (1234,), 1)])
    write(32, _U8, "B", [(_EVENT, (1,), 1), (_EVENT, (3,), 2), (_EVENT, (0,), 3)])
    write(34, _U16, "H", [(_EVENT, (100,), 1), (_EVENT, (200,), 2)])
    write(35, _U16, "HH", [(_EVENT, (25, 60), 1), (_EVENT, (26, 61), 2)])
    write(36, _U8, "B", [(_READ, (7,), 1)])
    return root


def _load(root, **kwargs) -> HarpDevice:
    params = HarpDeviceParams(path=root, device_yml_hint=DeviceYmlByFile(), **kwargs)
    device = HarpDevice("Dev", reader_params=params)
    device.load()
    return device


class TestHarpDevice:
    def test_registers_and_schema(self, harp_dataset):
        device = _load(harp_dataset)
        names = [r.name for r in device]

        assert {"WhoAmI", "LickState", "RawVoltage", "SensorData"} <= set(names)
        assert "Reserved0" not in names
        assert device.device_schema.whoAmI == 1234
        assert device.device_schema.firmwareVersion == "1.2"
        assert device.dataset_reader.root == harp_dataset
        assert device["LickState"].description == "Lick state bitmask"
        assert device["WhoAmI"].description  # core registers carry their docstring
        assert isinstance(device["RawVoltage"], HarpRegister)
        assert device["RawVoltage"].register_class.address == 34

    def test_dataframe_layout(self, harp_dataset):
        device = _load(harp_dataset)

        who_am_i = device["WhoAmI"].data
        assert list(who_am_i.columns) == ["message_type", "value"]
        assert who_am_i.index.name == "Time"
        assert who_am_i["value"].iloc[-1] == 1234
        assert list(who_am_i["message_type"].astype(str)) == ["Read"]

        sensor = device["SensorData"].data
        assert list(sensor.columns) == ["message_type", "temperature", "humidity"]
        assert list(sensor["temperature"]) == [25, 26]

    def test_bit_masks_not_demuxed_by_default(self, harp_dataset):
        lick = _load(harp_dataset)["LickState"].data
        assert list(lick.columns) == ["message_type", "value"]
        assert list(lick["value"]) == [1, 3, 0]

    def test_bit_masks_demuxed(self, harp_dataset):
        lick = _load(harp_dataset, demux_bit_masks=True)["LickState"].data
        assert list(lick.columns) == ["message_type", "CHANNEL0", "CHANNEL1"]
        assert list(lick["CHANNEL1"]) == [False, True, False]

    def test_register_read_params_override(self, harp_dataset):
        register = _load(harp_dataset)["LickState"]
        df = register.read(HarpRegisterParams(keep_type=False, demux_bit_masks=True))
        assert list(df.columns) == ["CHANNEL0", "CHANNEL1"]

    def test_private_registers_are_never_exposed(self, harp_dataset):
        names = [r.name for r in _load(harp_dataset)]
        assert not any(name.lstrip("_") == "Reserved0" for name in names)

    def test_exclude_common_registers(self, harp_dataset):
        names = [r.name for r in _load(harp_dataset, include_common_registers=False)]
        assert "WhoAmI" not in names
        assert "LickState" in names

    def test_missing_register_file_is_load_error(self, harp_dataset):
        register = _load(harp_dataset)["NeverLogged"].load()
        assert register.has_error
        assert not register.has_data
        with pytest.raises(FileNotFoundError):
            register.read()

    def test_epoch(self, harp_dataset):
        device = _load(harp_dataset, epoch=pd.Timestamp("1904-01-01").to_pydatetime())
        assert isinstance(device["RawVoltage"].data.index, pd.DatetimeIndex)


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise harp_contract.requests.HTTPError(f"HTTP {self.status_code}")


class TestDeviceYmlCaching:
    @pytest.fixture(autouse=True)
    def _clear_caches(self):
        harp_contract._fetch_yml_bytes_from_who_am_i.cache_clear()
        harp_contract._fetch_yml_bytes_from_url.cache_clear()
        yield
        harp_contract._fetch_yml_bytes_from_who_am_i.cache_clear()
        harp_contract._fetch_yml_bytes_from_url.cache_clear()

    @pytest.fixture
    def fake_get(self, monkeypatch):
        calls = []
        responses = {}

        def get(url, **kwargs):
            calls.append(url)
            return responses.get(url, _FakeResponse(b"", 404))

        monkeypatch.setattr(harp_contract.requests, "get", get)
        monkeypatch.setattr(
            harp_contract,
            "fetch_who_am_i_list",
            lambda: {1234: {"repositoryUrl": "https://github.com/harp-tech/device.dev"}},
        )
        return calls, responses

    def test_who_am_i_yml_fetched_once(self, fake_get, harp_dataset):
        calls, responses = fake_get
        responses["https://raw.githubusercontent.com/harp-tech/device.dev/main/device.yml"] = _FakeResponse(
            DEVICE_YML.encode()
        )

        first = harp_contract.fetch_yml_from_who_am_i(1234)
        second = harp_contract.fetch_yml_from_who_am_i(1234)
        params = HarpDeviceParams(path=harp_dataset, device_yml_hint=DeviceYmlByWhoAmI(who_am_i=1234))
        HarpDevice("Dev", reader_params=params).load()

        assert len(calls) == 1
        assert first is not second
        assert first.getvalue() == second.getvalue() == DEVICE_YML.encode()

    def test_url_yml_fetched_once(self, fake_get, harp_dataset):
        calls, responses = fake_get
        url = "https://example.org/device.yml"
        responses[url] = _FakeResponse(DEVICE_YML.encode())

        for _ in range(2):
            params = HarpDeviceParams(path=harp_dataset, device_yml_hint=DeviceYmlByUrl(url=url))
            device = HarpDevice("Dev", reader_params=params).load()
            assert device.device_schema.whoAmI == 1234

        assert calls == [url]

    def test_failed_fetch_is_not_cached(self, fake_get):
        calls, responses = fake_get
        url = "https://example.org/device.yml"

        with pytest.raises(harp_contract.requests.HTTPError):
            harp_contract._fetch_yml_bytes_from_url(url)
        responses[url] = _FakeResponse(DEVICE_YML.encode())
        assert harp_contract._fetch_yml_bytes_from_url(url) == DEVICE_YML.encode()
        assert len(calls) == 2
