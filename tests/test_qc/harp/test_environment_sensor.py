"""Tests for the HarpEnvironmentSensorTestSuite class."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from contraqctor.qc.base import Status
from contraqctor.qc.harp.environment_sensor import HarpEnvironmentSensorTestSuite


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


class MockHarpDevice:
    def __init__(self, name="EnvironmentSensor", registers=None, whoami=1405):
        self.whoami = whoami
        self.name = name
        self._registers = registers or {}
        self._create_default_registers()
        self.device_reader = Mock(
            device=Mock(
                whoAmI=self.whoami,
                firmwareVersion="1.2.3",
                name=name,
                registers={"WhoAmI": {}, "SensorData": {}, "SensorDataDispatchRate": {}},
            )
        )

    def _create_default_registers(self):
        if not self._registers:
            whoami_df = pd.DataFrame(
                {"WhoAmI": [self.whoami], "MessageType": ["READ"]}, index=pd.Index([0.0], name="Seconds")
            )
            time_index = np.linspace(0, 10, 1000)
            np.random.seed(0)
            temp = np.random.uniform(
                HarpEnvironmentSensorTestSuite.temperature_limit[0],
                HarpEnvironmentSensorTestSuite.temperature_limit[1],
                len(time_index),
            )
            humidity = np.random.uniform(
                HarpEnvironmentSensorTestSuite.humidity_limit[0],
                HarpEnvironmentSensorTestSuite.humidity_limit[1],
                len(time_index),
            )
            sensor_df = pd.DataFrame(
                {
                    "Temperature": temp,
                    "Humidity": humidity,
                    "MessageType": ["EVENT"] * len(time_index),
                },
                index=pd.Index(time_index, name="Seconds"),
            )
            self._registers = {
                "WhoAmI": MockHarpRegister("WhoAmI", whoami_df),
                "SensorData": MockHarpRegister("SensorData", sensor_df),
            }

    def __getitem__(self, key):
        if key not in self._registers:
            raise KeyError(f"Register {key} not found")
        return self._registers[key]

    def __iter__(self):
        for reg in self._registers.values():
            yield reg


@pytest.fixture
def mock_env_sensor_device():
    return MockHarpDevice(whoami=1405)


@pytest.fixture
def mock_env_sensor_device_bad_temp():
    device = MockHarpDevice(whoami=1405)
    time_index = np.linspace(0, 10, 1000)
    temp = np.concatenate(
        [
            np.random.uniform(
                HarpEnvironmentSensorTestSuite.temperature_limit[0] * 0.1,
                HarpEnvironmentSensorTestSuite.temperature_limit[0] * 0.9,
                500,
            ),  # too low
            np.random.uniform(
                HarpEnvironmentSensorTestSuite.temperature_limit[1] * 1.1,
                HarpEnvironmentSensorTestSuite.temperature_limit[1] * 1.2,
                500,
            ),  # too high
        ]
    )
    humidity = np.random.uniform(
        HarpEnvironmentSensorTestSuite.humidity_limit[0], HarpEnvironmentSensorTestSuite.humidity_limit[1], 1000
    )
    sensor_df = pd.DataFrame(
        {"Temperature": temp, "Humidity": humidity, "MessageType": ["EVENT"] * 1000},
        index=pd.Index(time_index, name="Seconds"),
    )
    device._registers["SensorData"] = MockHarpRegister("SensorData", sensor_df)
    return device


@pytest.fixture
def mock_env_sensor_device_bad_humidity():
    device = MockHarpDevice(whoami=1401)
    time_index = np.linspace(0, 10, 1000)
    temp = np.random.uniform(
        HarpEnvironmentSensorTestSuite.temperature_limit[0], HarpEnvironmentSensorTestSuite.temperature_limit[1], 1000
    )
    humidity = np.concatenate(
        [
            np.random.uniform(
                HarpEnvironmentSensorTestSuite.humidity_limit[0] * 0.1,
                HarpEnvironmentSensorTestSuite.humidity_limit[0] * 0.9,
                500,
            ),  # too low
            np.random.uniform(
                HarpEnvironmentSensorTestSuite.humidity_limit[1] * 1.1,
                HarpEnvironmentSensorTestSuite.humidity_limit[1] * 1.2,
                500,
            ),  # too high
        ]
    )
    sensor_df = pd.DataFrame(
        {"Temperature": temp, "Humidity": humidity, "MessageType": ["EVENT"] * 1000},
        index=pd.Index(time_index, name="Seconds"),
    )
    device._registers["SensorData"] = MockHarpRegister("SensorData", sensor_df)
    return device


class TestHarpEnvironmentSensorTestSuite:
    def test_init(self, mock_env_sensor_device):
        suite = HarpEnvironmentSensorTestSuite(mock_env_sensor_device)
        assert suite.harp_device == mock_env_sensor_device
        assert "Temperature" in suite.data.columns
        assert "Humidity" in suite.data.columns

    def test_sampling_rate(self, mock_env_sensor_device):
        suite = HarpEnvironmentSensorTestSuite(mock_env_sensor_device)
        result = suite.test_sampling_rate()
        assert result.status == Status.PASSED

    def test_temperature_within_expected_limits(self, mock_env_sensor_device, mock_env_sensor_device_bad_temp):
        suite = HarpEnvironmentSensorTestSuite(mock_env_sensor_device)
        result = suite.test_temperature_within_expected_limits()
        assert result.status == Status.PASSED
        assert result.context is not None
        assert "min" in result.context
        assert "max" in result.context
        assert "mean" in result.context

        suite = HarpEnvironmentSensorTestSuite(mock_env_sensor_device_bad_temp)
        result = suite.test_temperature_within_expected_limits()
        assert result.status == Status.WARNING

    def test_humidity_within_expected_limits(self, mock_env_sensor_device, mock_env_sensor_device_bad_humidity):
        suite = HarpEnvironmentSensorTestSuite(mock_env_sensor_device)
        result = suite.test_humidity_within_expected_limits()
        assert result.status == Status.PASSED
        assert result.context is not None
        assert "min" in result.context
        assert "max" in result.context
        assert "mean" in result.context

        suite = HarpEnvironmentSensorTestSuite(mock_env_sensor_device_bad_humidity)
        result = suite.test_humidity_within_expected_limits()
        assert result.status == Status.WARNING
