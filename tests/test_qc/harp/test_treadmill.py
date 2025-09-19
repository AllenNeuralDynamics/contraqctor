from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from contraqctor.qc.base import Status
from contraqctor.qc.harp.treadmill import HarpTreadmillTestSuite


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
    def __init__(self, name="Treadmill", registers=None, whoami=1402):
        self.whoami = whoami
        self.name = name
        self._registers = registers or {}
        self._create_default_registers()
        self.device_reader = Mock(
            device=Mock(
                whoAmI=self.whoami,
                firmwareVersion="1.2.3",
                name=name,
                registers={"WhoAmI": {}, "SensorData": {}, "SensorDataDispatchRate": {}, "TorqueLimitState": {}},
            )
        )

    def _create_default_registers(self):
        if not self._registers:
            whoami_df = pd.DataFrame(
                {"WhoAmI": [self.whoami], "MessageType": ["READ"]}, index=pd.Index([0.0], name="Seconds")
            )
            time_index = np.linspace(0, 10, 1000)
            np.random.seed(0)
            encoder = np.cumsum(np.random.randint(1, 5, len(time_index)))
            torque = np.random.uniform(100, 2000, len(time_index))
            sensor_df = pd.DataFrame(
                {
                    "Encoder": encoder,
                    "Torque": torque,
                    "MessageType": ["EVENT"] * len(time_index),
                },
                index=pd.Index(time_index, name="Seconds"),
            )
            dispatch_rate_df = pd.DataFrame(
                {"SensorDataDispatchRate": [100], "MessageType": ["READ"]}, index=pd.Index([0.0], name="Seconds")
            )
            torque_limit_df = pd.DataFrame(
                {"TorqueLimitState": [0] * len(time_index), "MessageType": ["EVENT"] * len(time_index)},
                index=pd.Index(time_index, name="Seconds"),
            )
            self._registers = {
                "WhoAmI": MockHarpRegister("WhoAmI", whoami_df),
                "SensorData": MockHarpRegister("SensorData", sensor_df),
                "SensorDataDispatchRate": MockHarpRegister("SensorDataDispatchRate", dispatch_rate_df),
                "TorqueLimitState": MockHarpRegister("TorqueLimitState", torque_limit_df),
            }

    def __getitem__(self, key):
        if key not in self._registers:
            raise KeyError(f"Register {key} not found")
        return self._registers[key]

    def __iter__(self):
        for reg in self._registers.values():
            yield reg


@pytest.fixture
def mock_treadmill_device():
    return MockHarpDevice(whoami=1402)


@pytest.fixture
def mock_treadmill_device_bad_rate():
    device = MockHarpDevice(whoami=1402)
    time_index = np.cumsum(np.random.uniform(0.0005, 0.0015, 1000))
    encoder = np.cumsum(np.random.randint(1, 5, len(time_index)))
    torque = np.random.uniform(100, 2000, len(time_index))
    sensor_df = pd.DataFrame(
        {"Encoder": encoder, "Torque": torque, "MessageType": ["EVENT"] * len(time_index)},
        index=pd.Index(time_index, name="Seconds"),
    )
    device._registers["SensorData"] = MockHarpRegister("SensorData", sensor_df)
    return device


@pytest.fixture
def mock_treadmill_device_zero_ticks():
    device = MockHarpDevice(whoami=1402)
    time_index = np.linspace(0, 10, 1000)
    encoder = np.zeros(len(time_index))
    torque = np.random.uniform(100, 2000, len(time_index))
    sensor_df = pd.DataFrame(
        {"Encoder": encoder, "Torque": torque, "MessageType": ["EVENT"] * len(time_index)},
        index=pd.Index(time_index, name="Seconds"),
    )
    device._registers["SensorData"] = MockHarpRegister("SensorData", sensor_df)
    return device


@pytest.fixture
def mock_treadmill_device_bad_torque():
    device = MockHarpDevice(whoami=1402)
    time_index = np.linspace(0, 10, 1000)
    encoder = np.cumsum(np.random.randint(1, 5, len(time_index)))
    torque = np.concatenate(
        [
            np.random.uniform(0, 5, 500),  # too low
            np.random.uniform(4020, 4090, 500),  # too high
        ]
    )
    sensor_df = pd.DataFrame(
        {"Encoder": encoder, "Torque": torque, "MessageType": ["EVENT"] * 1000},
        index=pd.Index(time_index, name="Seconds"),
    )
    device._registers["SensorData"] = MockHarpRegister("SensorData", sensor_df)
    return device


@pytest.fixture
def mock_treadmill_device_tripwire():
    device = MockHarpDevice(whoami=1402)
    time_index = np.linspace(0, 10, 1000)
    torque_limit = np.zeros(len(time_index))
    torque_limit[100] = 1  # simulate tripwire triggered
    torque_limit_df = pd.DataFrame(
        {"TorqueLimitState": torque_limit, "MessageType": ["EVENT"] * len(time_index)},
        index=pd.Index(time_index, name="Seconds"),
    )
    device._registers["TorqueLimitState"] = MockHarpRegister("TorqueLimitState", torque_limit_df)
    return device


class TestHarpTreadmillTestSuite:
    def test_init(self, mock_treadmill_device):
        suite = HarpTreadmillTestSuite(mock_treadmill_device)
        assert suite.harp_device == mock_treadmill_device
        assert "Encoder" in suite.data.columns
        assert "Torque" in suite.data.columns

    def test_sampling_rate(self, mock_treadmill_device, mock_treadmill_device_bad_rate):
        suite = HarpTreadmillTestSuite(mock_treadmill_device)
        result = suite.test_sampling_rate()
        assert result.status == Status.PASSED
        assert result.message is not None and "Sampling rate is" in result.message

        suite = HarpTreadmillTestSuite(mock_treadmill_device_bad_rate)
        result = suite.test_sampling_rate()
        assert result.status == Status.FAILED
        assert result.message is not None and "not within nominal values" in result.message

    def test_encoder(self, mock_treadmill_device, mock_treadmill_device_zero_ticks):
        suite = HarpTreadmillTestSuite(mock_treadmill_device)
        result = suite.test_encoder()
        assert result.status == Status.PASSED
        assert result.message is not None and "Total ticks is" in result.message
        assert result.context is None or "total_ticks" in result.context

        suite = HarpTreadmillTestSuite(mock_treadmill_device_zero_ticks)
        result = suite.test_encoder()
        assert result.status == Status.FAILED
        assert result.message is not None and "Total ticks is zero" in result.message

    def test_torque(self, mock_treadmill_device, mock_treadmill_device_bad_torque):
        suite = HarpTreadmillTestSuite(mock_treadmill_device)
        result = suite.test_torque()
        assert result.status == Status.PASSED
        assert result.message is not None
        assert result.result is not None

        suite = HarpTreadmillTestSuite(mock_treadmill_device_bad_torque)
        result = suite.test_torque()
        assert result.status == Status.WARNING
        assert result.message is not None
        assert result.result is not None

    def test_torque_limit_tripwire(self, mock_treadmill_device, mock_treadmill_device_tripwire):
        suite = HarpTreadmillTestSuite(mock_treadmill_device)
        result = suite.test_torque_limit_tripwire()
        assert result.status == Status.PASSED
        assert result.message is not None and "Torque limit tripwire was never triggered" in result.message

        suite = HarpTreadmillTestSuite(mock_treadmill_device_tripwire)
        result = suite.test_torque_limit_tripwire()
        assert result.status == Status.FAILED
        assert result.message is not None and "Torque limit tripwire was triggered" in result.message
