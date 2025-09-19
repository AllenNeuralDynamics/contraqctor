from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from contraqctor.qc.base import Status
from contraqctor.qc.harp.lickety_split import HarpLicketySplitTestSuite


class MockHarpRegister:
    def __init__(self, name, data=None, has_data=True):
        self.name = name
        self._data = data
        self._has_data = has_data

    @property
    def has_data(self):
        return self._has_data

    @property
    def data(self):
        if not self._has_data:
            raise ValueError(f"Register {self.name} has no data")
        return self._data

    def read(self):
        return self._data


class MockHarpDevice:
    def __init__(self, name="LicketySplit", registers=None, whoami=1400):
        self.whoami = whoami
        self.name = name
        self._registers = registers or {}
        self._create_default_registers()
        self.device_reader = Mock(
            device=Mock(
                whoAmI=self.whoami,
                firmwareVersion="1.2.3",
                name=name,
                registers={"WhoAmI": {}, "LickState": {}, "TimestampSeconds": {}},
            )
        )

    def _create_default_registers(self):
        if not self._registers:
            time_index = np.arange(0, 100, 0.1)
            lick_state = np.zeros(len(time_index))
            np.random.seed(0)
            lick_times = np.random.choice(np.arange(10, 990), size=100, replace=False)
            lick_state[lick_times] = 1
            lick_df = pd.DataFrame(
                {"Channel0": lick_state, "MessageType": ["EVENT"] * len(time_index)},
                index=pd.Index(time_index, name="Seconds"),
            )
            timestamp_df = pd.DataFrame(index=pd.Index(time_index, name="Seconds"))
            self._registers = {
                "LickState": MockHarpRegister("LickState", lick_df),
                "TimestampSeconds": MockHarpRegister("TimestampSeconds", timestamp_df),
            }

    def __getitem__(self, key):
        if key not in self._registers:
            raise KeyError(f"Register {key} not found")
        return self._registers[key]

    def __iter__(self):
        for reg in self._registers.values():
            yield reg


@pytest.fixture
def mock_lickety_split_device():
    return MockHarpDevice(whoami=1400)


@pytest.fixture
def mock_lickety_split_device_many_violations():
    device = MockHarpDevice(whoami=1400)
    time_index = np.linspace(0, 1, 1000)
    lick_df = pd.DataFrame(index=pd.Index(time_index, name="Seconds"))
    lick_signal = np.zeros(len(time_index))
    lick_signal[::2] = 1
    lick_df["Channel0"] = lick_signal
    lick_df["MessageType"] = ["EVENT"] * len(time_index)
    device._registers["LickState"] = MockHarpRegister("LickState", lick_df)
    return device


@pytest.fixture
def mock_lickety_split_device_low_rate():
    device = MockHarpDevice(whoami=1400)
    lick_data = device._registers["LickState"].data
    lick_df = lick_data.copy() if lick_data is not None else pd.DataFrame()
    if not lick_df.empty:
        lick_df["Channel0"] = 0  # No licks
    device._registers["LickState"] = MockHarpRegister("LickState", lick_df)
    return device


@pytest.fixture
def mock_lickety_split_device_duration_violations():
    device = MockHarpDevice(whoami=1400)
    lick_time = np.linspace(0, 1, 1000)
    lick_df = pd.DataFrame(index=pd.Index(lick_time, name="Seconds"))
    lick_signal = np.zeros(len(lick_time))
    lick_signal[::2] = 1
    lick_df["Channel0"] = lick_signal
    lick_df["MessageType"] = ["EVENT"] * len(lick_time)
    device._registers["LickState"] = MockHarpRegister("LickState", lick_df)
    return device


class TestHarpLicketySplitTestSuite:
    def test_init(self, mock_lickety_split_device):
        suite = HarpLicketySplitTestSuite(mock_lickety_split_device)
        assert suite.harp_device == mock_lickety_split_device
        assert "Channel0" in suite.data.columns

    def test_refractory_period_violations(self, mock_lickety_split_device, mock_lickety_split_device_many_violations):
        suite = HarpLicketySplitTestSuite(mock_lickety_split_device)
        result = suite.test_refractory_period_violations()
        assert result.status == Status.PASSED or result.status == Status.WARNING
        assert result.message is not None
        assert result.context is not None

        suite = HarpLicketySplitTestSuite(mock_lickety_split_device_many_violations)
        result = suite.test_refractory_period_violations()
        assert result.status in (Status.WARNING, Status.FAILED)
        assert result.message is not None
        assert result.context is not None

    def test_minimum_lick_rate(self, mock_lickety_split_device, mock_lickety_split_device_low_rate):
        suite = HarpLicketySplitTestSuite(mock_lickety_split_device)
        result = suite.test_minimum_lick_rate()
        assert result.status == Status.PASSED
        assert result.message is not None
        assert result.context is not None

        suite = HarpLicketySplitTestSuite(mock_lickety_split_device_low_rate)
        result = suite.test_minimum_lick_rate()
        assert result.status == Status.FAILED
        assert result.message is not None
        assert result.context is not None

    def test_lick_duration(self, mock_lickety_split_device, mock_lickety_split_device_duration_violations):
        suite = HarpLicketySplitTestSuite(mock_lickety_split_device)
        result = suite.test_lick_duration()
        assert result.status == Status.PASSED or result.status == Status.WARNING
        assert result.message is not None
        assert result.context is not None

        suite = HarpLicketySplitTestSuite(mock_lickety_split_device_duration_violations)
        result = suite.test_lick_duration()
        assert result.status == Status.WARNING
        assert result.message is not None
        assert result.context is not None
