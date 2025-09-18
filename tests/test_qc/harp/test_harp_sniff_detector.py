"""Tests for the HarpSniffDetectorTestSuite class."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from contraqctor.qc.base import Status
from contraqctor.qc.harp import HarpSniffDetectorTestSuite


class MockHarpRegister:
    """Mock HarpRegister class for testing."""

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
    """Mock HarpDevice class for testing."""

    def __init__(self, name="SniffDetector", registers=None, whoami=1401):
        self.whoami = whoami
        self.name = name
        self._registers = registers or {}
        self._create_default_registers()
        self.device_reader = Mock(
            device=Mock(
                whoAmI=self.whoami,
                firmwareVersion="1.2.3",
                name=name,
                registers={"WhoAmI": {}, "RawVoltage": {}, "RawVoltageDispatchRate": {}},
            )
        )

    def _create_default_registers(self):
        if not self._registers:
            whoami_df = pd.DataFrame(
                {"WhoAmI": [self.whoami], "MessageType": ["READ"]}, index=pd.Index([0.0], name="Seconds")
            )

            time_index = np.linspace(0, 10, 1000)
            np.random.seed(0)
            signal = np.sin(2 * np.pi * 5 * time_index) + 0.1 * np.random.randn(len(time_index))

            raw_voltage_df = pd.DataFrame(
                {"RawVoltage": signal, "MessageType": ["EVENT"] * len(time_index)},
                index=pd.Index(time_index, name="Seconds"),
            )

            # Create dispatch rate register showing 100 Hz sampling
            dispatch_rate_df = pd.DataFrame(
                {"RawVoltageDispatchRate": [100], "MessageType": ["READ"]}, index=pd.Index([0.0], name="Seconds")
            )

            self._registers = {
                "WhoAmI": MockHarpRegister("WhoAmI", whoami_df),
                "RawVoltage": MockHarpRegister("RawVoltage", raw_voltage_df),
                "RawVoltageDispatchRate": MockHarpRegister("RawVoltageDispatchRate", dispatch_rate_df),
            }

    def __getitem__(self, key):
        """Get register by name."""
        if key not in self._registers:
            raise KeyError(f"Register {key} not found")
        return self._registers[key]

    def __iter__(self):
        """Generator for all data streams."""
        for reg in self._registers.values():
            yield reg


@pytest.fixture
def mock_sniff_device():
    """Fixture providing a mock sniff detector with good signal."""
    return MockHarpDevice(whoami=1401)


@pytest.fixture
def mock_sniff_device_wrong_whoami():
    """Fixture providing a mock sniff detector with incorrect WhoAmI."""
    return MockHarpDevice(whoami=1234)  # Not the correct sniff detector ID


@pytest.fixture
def mock_sniff_device_bad_rate():
    """Fixture providing a mock sniff detector with inconsistent sample rate."""
    device = MockHarpDevice(whoami=1401)

    np.random.seed(0)
    time_index = np.cumsum(np.random.uniform(0.0005, 0.0015, 1000))

    np.random.seed(0)
    signal = np.sin(2 * np.pi * 5 * time_index) + 0.1 * np.random.randn(len(time_index))

    raw_voltage_df = pd.DataFrame(
        {"RawVoltage": signal, "MessageType": ["EVENT"] * len(time_index)}, index=pd.Index(time_index, name="Seconds")
    )

    device._registers["RawVoltage"] = MockHarpRegister("RawVoltage", raw_voltage_df)
    return device


@pytest.fixture
def mock_sniff_device_bad_quality():
    """Fixture providing a mock sniff detector with poor signal quality."""
    device = MockHarpDevice(whoami=1401)

    time_index = np.linspace(0, 10, 1000)

    # Simulate clipping
    signal = np.sin(2 * np.pi * 5 * time_index)
    signal[signal > 0.8] = 0.8
    signal[signal < -0.8] = -0.8

    # Add clustering by quantizing
    signal = np.round(signal * 10) / 10

    raw_voltage_df = pd.DataFrame(
        {"RawVoltage": signal, "MessageType": ["EVENT"] * len(time_index)}, index=pd.Index(time_index, name="Seconds")
    )

    device._registers["RawVoltage"] = MockHarpRegister("RawVoltage", raw_voltage_df)
    return device


class TestHarpSniffDetectorTestSuite:
    """Tests for the HarpSniffDetectorTestSuite class."""

    def test_init(self, mock_sniff_device):
        """Test initializing the HarpSniffDetectorTestSuite."""
        suite = HarpSniffDetectorTestSuite(mock_sniff_device)

        # Check defaults
        assert suite.harp_device == mock_sniff_device
        assert suite.fs == 100  # From fixture
        assert suite.quantization_ratio_thr == 0.1
        assert suite.clustering_thr == 0.05
        assert suite.clipping_thr == 0.05
        assert suite.sudden_jumps_thr == 0.001
        assert suite.notch_filter_freq == 50

        custom_suite = HarpSniffDetectorTestSuite(
            mock_sniff_device,
            quantization_ratio_thr=0.2,
            clustering_thr=0.1,
            clipping_thr=0.1,
            sudden_jumps_thr=0.002,
            notch_filter_freq=60,
        )

        assert custom_suite.quantization_ratio_thr == 0.2
        assert custom_suite.clustering_thr == 0.1
        assert custom_suite.clipping_thr == 0.1
        assert custom_suite.sudden_jumps_thr == 0.002
        assert custom_suite.notch_filter_freq == 60

    def test_whoami(self, mock_sniff_device, mock_sniff_device_wrong_whoami):
        """Test test_whoami method."""
        suite = HarpSniffDetectorTestSuite(mock_sniff_device)
        result = suite.test_whoami()
        assert result.status == Status.PASSED
        assert result.result is True

        suite = HarpSniffDetectorTestSuite(mock_sniff_device_wrong_whoami)
        result = suite.test_whoami()
        assert result.status == Status.FAILED
        assert "Expected WhoAmI" in result.message

    def test_sniff_detector_sampling_rate(self, mock_sniff_device, mock_sniff_device_bad_rate):
        """Test test_sniff_detector_sampling_rate method."""
        suite = HarpSniffDetectorTestSuite(mock_sniff_device)
        result = suite.test_sniff_detector_sampling_rate()
        assert result.status == Status.PASSED
        assert "Sampling rate is" in result.message

        suite = HarpSniffDetectorTestSuite(mock_sniff_device_bad_rate)
        result = suite.test_sniff_detector_sampling_rate()
        assert result.status == Status.FAILED
        assert "not within nominal values" in result.message

    def test_sniff_detector_signal_quality(self, mock_sniff_device, mock_sniff_device_bad_quality):
        """Test test_sniff_detector_signal_quality method."""
        suite = HarpSniffDetectorTestSuite(mock_sniff_device)
        result = suite.test_sniff_detector_signal_quality()
        assert result.status == Status.PASSED
        assert "Signal quality is good" in result.message
        assert "context" in dir(result)

        suite = HarpSniffDetectorTestSuite(mock_sniff_device_bad_quality)
        result = suite.test_sniff_detector_signal_quality()
        assert result.status == Status.FAILED
        assert "Signal quality is not good" in result.message

        metrics = result.context
        assert "quantization_ratio" in metrics
        assert "clustering_ratio" in metrics
        assert "min_clipping" in metrics
        assert "max_clipping" in metrics
        assert "sudden_jumps_ratio" in metrics
