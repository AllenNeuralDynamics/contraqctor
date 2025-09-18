"""Tests for the QC harp module."""

from unittest.mock import Mock

import pandas as pd
import pytest
import semver

from contraqctor.qc.base import Status
from contraqctor.qc.harp import HarpDeviceTestSuite, HarpHubTestSuite


# Mock classes
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


class MockDeviceReader:
    """Mock DeviceReader class for testing."""

    def __init__(self, whoami=1234, firmware_version="1.2.3", device_name="MockDevice"):
        self.device = Mock(
            whoAmI=whoami,
            firmwareVersion=firmware_version,
            name=device_name,
            registers={
                "WhoAmI": {},
                "FirmwareVersionHigh": {},
                "FirmwareVersionLow": {},
                "CoreVersionHigh": {},
                "CoreVersionLow": {},
                "OperationControl": {},
                "ClockConfiguration": {},
            },
        )


class MockHarpDevice:
    """Mock HarpDevice class for testing."""

    def __init__(self, name="TestDevice", registers=None, whoami=1234):
        self.name = name
        self._registers = registers or {}
        self._create_default_registers()
        self.device_reader = MockDeviceReader(whoami=whoami)

    def _create_default_registers(self):
        """Create default registers if none are provided."""
        if not self._registers:
            whoami_df = pd.DataFrame({"WhoAmI": [1234], "MessageType": ["READ"]}, index=pd.Index([0.0], name="Seconds"))

            fw_high_df = pd.DataFrame(
                {"FirmwareVersionHigh": [1], "MessageType": ["READ"]}, index=pd.Index([0.1], name="Seconds")
            )

            fw_low_df = pd.DataFrame(
                {"FirmwareVersionLow": [2], "MessageType": ["READ"]}, index=pd.Index([0.2], name="Seconds")
            )

            core_high_df = pd.DataFrame(
                {"CoreVersionHigh": [3], "MessageType": ["READ"]}, index=pd.Index([0.3], name="Seconds")
            )

            core_low_df = pd.DataFrame(
                {"CoreVersionLow": [4], "MessageType": ["READ"]}, index=pd.Index([0.4], name="Seconds")
            )

            op_control_df = pd.DataFrame(
                {"OperationControl": [1, 2], "MessageType": ["READ", "WRITE"]},
                index=pd.Index([0.5, 0.6], name="Seconds"),
            )

            clock_config_df = pd.DataFrame(
                {"ClockGenerator": [False], "MessageType": ["READ"]}, index=pd.Index([0.7], name="Seconds")
            )

            self._registers = {
                "WhoAmI": MockHarpRegister("WhoAmI", whoami_df),
                "FirmwareVersionHigh": MockHarpRegister("FirmwareVersionHigh", fw_high_df),
                "FirmwareVersionLow": MockHarpRegister("FirmwareVersionLow", fw_low_df),
                "CoreVersionHigh": MockHarpRegister("CoreVersionHigh", core_high_df),
                "CoreVersionLow": MockHarpRegister("CoreVersionLow", core_low_df),
                "OperationControl": MockHarpRegister("OperationControl", op_control_df),
                "ClockConfiguration": MockHarpRegister("ClockConfiguration", clock_config_df),
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
def mock_harp_device():
    """Fixture for a mock HarpDevice."""
    return MockHarpDevice()


@pytest.fixture
def mock_harp_device_commands():
    """Fixture for mock HarpDevice commands."""
    op_control_df = pd.DataFrame(
        {"OperationControl": [1, 2], "MessageType": ["READ", "WRITE"]}, index=pd.Index([0.5, 0.6], name="Seconds")
    )

    test_reg_df = pd.DataFrame(
        {"TestReg": [1, 2, 3], "MessageType": ["WRITE", "WRITE", "WRITE"]},
        index=pd.Index([1.0, 1.1, 1.2], name="Seconds"),
    )

    registers = {
        "OperationControl": MockHarpRegister("OperationControl", op_control_df),
        "TestReg": MockHarpRegister("TestReg", test_reg_df),
    }

    return MockHarpDevice(name="CommandDevice", registers=registers)


@pytest.fixture
def clock_generator_device():
    """Fixture for a clock generator device."""
    clock_config_df = pd.DataFrame(
        {"ClockGenerator": [True], "MessageType": ["READ"]}, index=pd.Index([0.7], name="Seconds")
    )

    registers = {
        "ClockConfiguration": MockHarpRegister("ClockConfiguration", clock_config_df),
        "OperationControl": MockHarpRegister(
            "OperationControl",
            pd.DataFrame({"OperationControl": [1], "MessageType": ["WRITE"]}, index=pd.Index([1.0], name="Seconds")),
        ),
    }

    return MockHarpDevice(name="ClockGenerator", registers=registers)


@pytest.fixture
def subordinate_devices():
    """Fixture for subordinate devices."""
    devices = []

    for i in range(3):
        clock_config_df = pd.DataFrame(
            {"ClockGenerator": [False], "MessageType": ["READ"]}, index=pd.Index([0.7], name="Seconds")
        )

        op_control_df = pd.DataFrame(
            {"OperationControl": [1], "MessageType": ["WRITE"]}, index=pd.Index([1.01 + i * 0.01], name="Seconds")
        )

        registers = {
            "ClockConfiguration": MockHarpRegister("ClockConfiguration", clock_config_df),
            "OperationControl": MockHarpRegister("OperationControl", op_control_df),
        }

        devices.append(MockHarpDevice(name=f"Device{i}", registers=registers))

    return devices


class TestHarpDeviceTestSuite:
    """Tests for the HarpDeviceTestSuite class."""

    def test_init(self, mock_harp_device, mock_harp_device_commands):
        """Test initializing the HarpDeviceTestSuite."""
        suite = HarpDeviceTestSuite(mock_harp_device, mock_harp_device_commands, min_core_version="1.0.0")

        assert suite.harp_device == mock_harp_device
        assert suite.harp_device_commands == mock_harp_device_commands
        assert suite.min_core_version == "1.0.0"

    def test_get_whoami(self, mock_harp_device):
        """Test _get_whoami method."""
        suite = HarpDeviceTestSuite(mock_harp_device)

        whoami = suite._get_whoami(mock_harp_device)
        assert whoami == 1234

    def test_get_last_read(self, mock_harp_device):
        """Test _get_last_read method."""
        suite = HarpDeviceTestSuite(mock_harp_device)

        last_read = suite._get_last_read(mock_harp_device["WhoAmI"])
        assert last_read is not None
        assert last_read["WhoAmI"] == 1234

        op_control_df = pd.DataFrame(
            {"OperationControl": [1, 2], "MessageType": ["WRITE", "WRITE"]}, index=pd.Index([0.5, 0.6], name="Seconds")
        )

        mock_harp_device._registers["NoReadReg"] = MockHarpRegister("NoReadReg", op_control_df)
        assert suite._get_last_read(mock_harp_device["NoReadReg"]) is None

        mock_harp_device._registers["NoDataReg"] = MockHarpRegister("NoDataReg", None, has_data=False)
        with pytest.raises(ValueError):
            suite._get_last_read(mock_harp_device["NoDataReg"])

    def test_has_whoami(self, mock_harp_device):
        """Test test_has_whoami method."""
        suite = HarpDeviceTestSuite(mock_harp_device)

        result = suite.test_has_whoami()
        assert result.status == Status.PASSED
        assert result.result == 1234

        mock_harp_device._registers["WhoAmI"] = MockHarpRegister("WhoAmI", pd.DataFrame())
        result = suite.test_has_whoami()
        assert result.status == Status.FAILED
        assert "empty" in result.message.lower()

        mock_harp_device._registers["WhoAmI"] = MockHarpRegister("WhoAmI", None, has_data=False)
        result = suite.test_has_whoami()
        assert result.status == Status.FAILED
        assert "does not have loaded data" in result.message.lower()

        whoami_df = pd.DataFrame(
            {
                "WhoAmI": [10000],  # Outside valid range
                "MessageType": ["READ"],
            },
            index=pd.Index([0.0], name="Seconds"),
        )
        mock_harp_device._registers["WhoAmI"] = MockHarpRegister("WhoAmI", whoami_df)
        result = suite.test_has_whoami()
        assert result.status == Status.FAILED
        assert "not in the range" in result.message.lower()

    def test_match_whoami_to_yml(self, mock_harp_device):
        """Test test_match_whoami_to_yml method."""
        suite = HarpDeviceTestSuite(mock_harp_device)

        result = suite.test_match_whoami_to_yml()
        assert result.status == Status.PASSED

        mock_harp_device.device_reader.device.whoAmI = 5678
        result = suite.test_match_whoami_to_yml()
        assert result.status == Status.FAILED

    def test_read_dump_is_complete(self, mock_harp_device):
        """Test test_read_dump_is_complete method."""
        suite = HarpDeviceTestSuite(mock_harp_device)

        result = suite.test_read_dump_is_complete()
        assert result.status == Status.PASSED

        suite.harp_device.device_reader.device.registers["MissingReg"] = {}
        result = suite.test_read_dump_is_complete()
        assert result.status == Status.FAILED
        assert "missing_registers" in result.context

    def test_request_response(self, mock_harp_device, mock_harp_device_commands):
        """Test test_request_response method."""
        suite = HarpDeviceTestSuite(mock_harp_device, mock_harp_device_commands)

        test_reg_df = pd.DataFrame(
            {"TestReg": [1, 2, 3], "MessageType": ["WRITE", "WRITE", "WRITE"]},
            index=pd.Index([1.1, 1.2, 1.3], name="Seconds"),
        )
        mock_harp_device._registers["TestReg"] = MockHarpRegister("TestReg", test_reg_df)

        result = suite.test_request_response()
        assert result.status == Status.PASSED

        test_reg_df = pd.DataFrame(
            {
                "TestReg": [1],  # Only one response
                "MessageType": ["WRITE"],
            },
            index=pd.Index([1.1], name="Seconds"),
        )
        mock_harp_device._registers["TestReg"] = MockHarpRegister("TestReg", test_reg_df)

        result = suite.test_request_response()
        assert result.status == Status.FAILED
        assert "register_errors" in result.context

        suite = HarpDeviceTestSuite(mock_harp_device, None)
        result = suite.test_request_response()
        assert result.status == Status.SKIPPED

    def test_registers_are_monotonicity(self, mock_harp_device):
        """Test test_registers_are_monotonicity method."""
        suite = HarpDeviceTestSuite(mock_harp_device)

        result = suite.test_registers_are_monotonicity()
        assert result.status == Status.PASSED

        non_monotonic_df = pd.DataFrame(
            {"NonMonotonic": [1, 2, 3], "MessageType": ["READ", "READ", "READ"]},
            index=pd.Index([0.3, 0.2, 0.4], name="Seconds"),
        )  # Non-monotonic index
        mock_harp_device._registers["NonMonotonic"] = MockHarpRegister("NonMonotonic", non_monotonic_df)

        result = suite.test_registers_are_monotonicity()
        assert result.status == Status.FAILED
        assert "register_errors" in result.context

    def test_try_parse_semver(self):
        """Test _try_parse_semver method."""
        suite = HarpDeviceTestSuite(MockHarpDevice())

        version = suite._try_parse_semver("1.2.3")
        assert isinstance(version, semver.Version)
        assert str(version) == "1.2.3"

        version = suite._try_parse_semver("1.2")
        assert isinstance(version, semver.Version)
        assert str(version) == "1.2.0"

        version = suite._try_parse_semver("invalid")
        assert version is None

    def test_fw_version_matches_reader(self, mock_harp_device):
        """Test test_fw_version_matches_reader method."""
        mock_harp_device.device_reader.device.firmwareVersion = "1.2.0"

        fw_high_df = pd.DataFrame(
            {"FirmwareVersionHigh": [1], "MessageType": ["READ"]}, index=pd.Index([0.1], name="Seconds")
        )

        fw_low_df = pd.DataFrame(
            {"FirmwareVersionLow": [2], "MessageType": ["READ"]}, index=pd.Index([0.2], name="Seconds")
        )

        mock_harp_device._registers["FirmwareVersionHigh"] = MockHarpRegister("FirmwareVersionHigh", fw_high_df)
        mock_harp_device._registers["FirmwareVersionLow"] = MockHarpRegister("FirmwareVersionLow", fw_low_df)

        suite = HarpDeviceTestSuite(mock_harp_device)

        result = suite.test_fw_version_matches_reader()
        assert result.status == Status.PASSED

        mock_harp_device.device_reader.device.firmwareVersion = "1.3.0"
        result = suite.test_fw_version_matches_reader()
        assert result.status == Status.FAILED
        assert "Consider updating the device firmware" in result.message

        mock_harp_device.device_reader.device.firmwareVersion = "1.1.0"
        result = suite.test_fw_version_matches_reader()
        assert result.status == Status.WARNING
        assert "Consider updating interface package" in result.message

        mock_harp_device.device_reader.device.firmwareVersion = "invalid"
        result = suite.test_fw_version_matches_reader()
        assert result.status == Status.FAILED
        assert "not a valid semver version" in result.message

    def test_core_version(self, mock_harp_device):
        """Test test_core_version method."""
        core_high_df = pd.DataFrame(
            {"CoreVersionHigh": [2], "MessageType": ["READ"]}, index=pd.Index([0.3], name="Seconds")
        )

        core_low_df = pd.DataFrame(
            {"CoreVersionLow": [0], "MessageType": ["READ"]}, index=pd.Index([0.4], name="Seconds")
        )

        mock_harp_device._registers["CoreVersionHigh"] = MockHarpRegister("CoreVersionHigh", core_high_df)
        mock_harp_device._registers["CoreVersionLow"] = MockHarpRegister("CoreVersionLow", core_low_df)

        suite = HarpDeviceTestSuite(mock_harp_device)
        result = suite.test_core_version()
        assert result.status == Status.SKIPPED

        suite = HarpDeviceTestSuite(mock_harp_device, min_core_version="2.0.0")
        result = suite.test_core_version()
        assert result.status == Status.PASSED

        suite = HarpDeviceTestSuite(mock_harp_device, min_core_version="2.1.0")
        result = suite.test_core_version()
        assert result.status == Status.FAILED
        assert "Consider updating the device firmware" in result.message

        suite = HarpDeviceTestSuite(mock_harp_device, min_core_version="1.9.0")
        result = suite.test_core_version()
        assert result.status == Status.WARNING
        assert "is less than the device's version" in result.message

        suite = HarpDeviceTestSuite(mock_harp_device, min_core_version="invalid")
        result = suite.test_core_version()
        assert result.status == Status.SKIPPED


class TestHarpHubTestSuite:
    """Tests for the HarpHubTestSuite class."""

    def test_init(self, clock_generator_device, subordinate_devices):
        """Test initializing the HarpHubTestSuite."""
        suite = HarpHubTestSuite(clock_generator_device, subordinate_devices)

        assert suite.clock_generator_device == clock_generator_device
        assert len(suite.devices) == len(subordinate_devices)
        assert suite.read_dump_jitter_threshold_s == 0.05  # Default value

    def test_clock_generator_reg(self, clock_generator_device, subordinate_devices):
        """Test test_clock_generator_reg method."""
        suite = HarpHubTestSuite(clock_generator_device, subordinate_devices)

        result = suite.test_clock_generator_reg()
        assert result.status == Status.PASSED

        clock_config_df = pd.DataFrame(
            {"ClockGenerator": [False], "MessageType": ["READ"]}, index=pd.Index([0.7], name="Seconds")
        )

        clock_generator_device._registers["ClockConfiguration"] = MockHarpRegister(
            "ClockConfiguration", clock_config_df
        )
        result = suite.test_clock_generator_reg()
        assert result.status == Status.FAILED

        del clock_generator_device._registers["ClockConfiguration"]
        result = suite.test_clock_generator_reg()
        assert result.status == Status.FAILED

    def test_devices_are_subordinate(self, clock_generator_device, subordinate_devices):
        """Test test_devices_are_subordinate method."""
        suite = HarpHubTestSuite(clock_generator_device, subordinate_devices)

        results = list(suite.test_devices_are_subordinate())
        assert all(r.status == Status.PASSED for r in results)

        clock_config_df = pd.DataFrame(
            {"ClockGenerator": [True], "MessageType": ["READ"]}, index=pd.Index([0.7], name="Seconds")
        )

        subordinate_devices[0]._registers["ClockConfiguration"] = MockHarpRegister(
            "ClockConfiguration", clock_config_df
        )

        results = list(suite.test_devices_are_subordinate())
        assert any(r.status == Status.FAILED for r in results)
        assert any("is not subordinate" in r.message for r in results)

        del subordinate_devices[1]._registers["ClockConfiguration"]

        results = list(suite.test_devices_are_subordinate())
        assert any(r.status == Status.FAILED for r in results)
        assert any("not present" in r.message for r in results)

    def test_get_read_dump_time(self, clock_generator_device):
        """Test _get_read_dump_time method."""
        suite = HarpHubTestSuite(clock_generator_device, [])

        time = suite._get_read_dump_time(clock_generator_device)
        assert time == 1.0

        op_control_df = pd.DataFrame(
            {
                "OperationControl": [1],
                "MessageType": ["READ"],  # No WRITE
            },
            index=pd.Index([0.5], name="Seconds"),
        )

        clock_generator_device._registers["OperationControl"] = MockHarpRegister("OperationControl", op_control_df)
        time = suite._get_read_dump_time(clock_generator_device)
        assert time is None

    def test_is_read_dump_synchronized(self, clock_generator_device, subordinate_devices):
        """Test test_is_read_dump_synchronized method."""
        for i, device in enumerate(subordinate_devices):
            op_control_df = pd.DataFrame(
                {"OperationControl": [1], "MessageType": ["WRITE"]}, index=pd.Index([1.0 + i * 0.01], name="Seconds")
            )

            device._registers["OperationControl"] = MockHarpRegister("OperationControl", op_control_df)

        suite = HarpHubTestSuite(clock_generator_device, subordinate_devices)

        results = list(suite.test_is_read_dump_synchronized())
        assert all(r.status == Status.PASSED for r in results)

        op_control_df = pd.DataFrame(
            {"OperationControl": [1], "MessageType": ["WRITE"]}, index=pd.Index([1.1], name="Seconds")
        )

        subordinate_devices[0]._registers["OperationControl"] = MockHarpRegister("OperationControl", op_control_df)

        results = list(suite.test_is_read_dump_synchronized())
        assert any(r.status == Status.FAILED for r in results)
        assert any("not synchronized" in r.message for r in results)

        suite = HarpHubTestSuite(clock_generator_device, subordinate_devices, read_dump_jitter_threshold_s=None)
        result = list(suite.test_is_read_dump_synchronized())
        assert all([r.status == Status.SKIPPED for r in result])
