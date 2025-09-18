import abc
import typing as t

import pandas as pd
import semver

from ...contract.harp import HarpDevice, HarpRegister
from ..base import Suite


class HarpDeviceTestSuite(Suite):
    """Test suite for generic Harp devices.

    Provides a set of standard tests that all Harp devices are expected to pass,
    checking basic functionality and data integrity.

    Attributes:
        harp_device: The HarpDevice data stream to test.
        harp_device_commands: Optional HarpDevice data stream with device commands.
        min_core_version: Optional minimum required core version.

    Examples:
        ```python
        from contraqctor.contract.harp import HarpDevice
        from contraqctor.qc.harp import HarpDeviceTestSuite
        from contraqctor.qc.base import Runner

        # Create HarpDevice streams
        device = HarpDevice("behavior", reader_params=params).load()
        commands = HarpDevice("behavior_commands", reader_params=command_params).load()

        # Create and run test suite
        suite = HarpDeviceTestSuite(device, commands, min_core_version="1.2.0")
        runner = Runner().add_suite(suite)
        results = runner.run_all_with_progress()
        ```
    """

    def __init__(
        self,
        harp_device: HarpDevice,
        harp_device_commands: t.Optional[HarpDevice] = None,
        *,
        min_core_version: t.Optional[str] = None,
    ):
        """Initialize the Harp device test suite.

        Args:
            harp_device: The HarpDevice data stream to test.
            harp_device_commands: Optional HarpDevice data stream with command history.
                If None, tests requiring the commands will be skipped.
            min_core_version: Optional minimum required core version for validation.
        """
        self.harp_device = harp_device
        self.harp_device_commands = harp_device_commands
        self.min_core_version = min_core_version

    # Helper functions
    # ----------------
    @staticmethod
    def _get_whoami(device: HarpDevice) -> int:
        """Get the WhoAmI identifier of a Harp device.

        Args:
            device: The HarpDevice data stream.

        Returns:
            int: The WhoAmI value of the device.
        """
        return device["WhoAmI"].data.WhoAmI.iloc[-1]

    @staticmethod
    def _get_last_read(harp_register: HarpRegister) -> t.Optional[pd.DataFrame]:
        """Get the last READ message from a Harp register.

        Args:
            harp_register: The HarpRegister data stream.

        Returns:
            Optional[pd.DataFrame]: The last READ message, or None if no READ messages exist.

        Raises:
            ValueError: If the register does not have loaded data.
        """
        if not harp_register.has_data:
            raise ValueError(f"Harp register: <{harp_register.name}> does not have loaded data")
        reads = harp_register.data[harp_register.data["MessageType"] == "READ"]
        return reads.iloc[-1] if len(reads) > 0 else None

    # Tests
    # -----
    def test_has_whoami(self):
        """Check if the harp board data stream is present and return its value"""
        who_am_i_reg: HarpRegister = self.harp_device["WhoAmI"]
        if not who_am_i_reg.has_data:
            return self.fail_test(None, "WhoAmI does not have loaded data")
        if len(who_am_i_reg.data) == 0:
            return self.fail_test(None, "WhoAmI file is empty")
        who_am_i = self._get_whoami(self.harp_device)
        if not bool(0000 <= who_am_i <= 9999):
            return self.fail_test(who_am_i, "WhoAmI value is not in the range 0000-9999")
        return self.pass_test(int(who_am_i))

    def test_match_whoami_to_yml(self):
        """Check if the WhoAmI value matches the device's WhoAmI"""
        if self._get_whoami(self.harp_device) == self.harp_device.device_reader.device.whoAmI:
            return self.pass_test(True, "WhoAmI value matches the device's WhoAmI")
        else:
            return self.fail_test(False, "WhoAmI value does not match the device's WhoAmI")

    def test_read_dump_is_complete(self):
        """
        Check if the read dump from an harp device is complete
        """
        expected_regs = self.harp_device.device_reader.device.registers.keys()
        ds = [stream for stream in self.harp_device]
        missing_regs = [reg_name for reg_name in expected_regs if reg_name not in [r.name for r in ds]]
        if len(missing_regs) > 0:
            return self.fail_test(
                False,
                "Read dump is not complete. Some registers are missing.",
                context={"missing_registers": missing_regs},
            )

        def _try_get_last_read(r: HarpRegister) -> t.Optional[pd.DataFrame]:
            """We will assume that all data is loaded. If not we will return None
            Instead of raising an error, so that we can continue checking other registers."""
            try:
                return self._get_last_read(r)
            except ValueError:
                return None

        missing_read_dump = [
            r.name for r in ds if not (r.name in expected_regs and (_try_get_last_read(r) is not None))
        ]
        return (
            self.pass_test(True, "Read dump is complete")
            if len(missing_read_dump) == 0
            else self.fail_test(False, "Read dump is not complete", context={"missing_registers": missing_read_dump})
        )

    def test_request_response(self):
        """Check that each request to the device has a corresponding response"""
        if self.harp_device_commands is None:
            return self.skip_test("No harp device commands provided")

        op_ctr: pd.DataFrame = self.harp_device_commands["OperationControl"].data
        op_ctr = op_ctr[op_ctr["MessageType"] == "WRITE"]
        op_ctr = op_ctr.index.values[0]

        reg_error = []
        for req_reg in self.harp_device_commands:
            if req_reg.has_data:  # Only data streams with data can be checked
                # Only "Writes" will be considered, but in theory we could also check "Reads"
                requests: pd.DataFrame = req_reg.data[req_reg.data["MessageType"] == "WRITE"]
                rep_reg = self.harp_device[req_reg.name]
                replies: pd.DataFrame = rep_reg.data[rep_reg.data["MessageType"] == "WRITE"]

                # All responses must, by definition, be timestamped AFTER the request
                if len(requests) > 0:
                    requests = requests[requests.index >= op_ctr]
                    replies = replies[replies.index >= op_ctr]

                    if len(requests) != len(replies):
                        reg_error.append(
                            {"register": req_reg.name, "requests": len(requests), "responses": len(replies)}
                        )

        if len(reg_error) == 0:
            return self.pass_test(
                None,
                "Request/Response check passed. All requests have a corresponding response.",
            )
        else:
            return self.fail_test(
                None,
                "Request/Response check failed. Some requests do not have a corresponding response.",
                context={"register_errors": reg_error},
            )

    def test_registers_are_monotonicity(self):
        """
        Check that the all the harp device registers' timestamps are monotonic.
        This test will not check registers that error'ed out during loading.
        """
        reg_errors = []
        reg: HarpRegister
        for reg in self.harp_device:
            if not reg.has_data:
                continue
            for message_type, reg_type_data in reg.data.groupby("MessageType", observed=True):
                if not reg_type_data.index.is_monotonic_increasing:
                    reg_errors.append(
                        {
                            "register": reg.name,
                            "message_type": message_type,
                            "is_monotonic": reg.data.index.is_monotonic_increasing,
                        }
                    )
        if len(reg_errors) == 0:
            return self.pass_test(
                None,
                "Monotonicity check passed. All registers are monotonic.",
            )
        else:
            return self.fail_test(
                None,
                "Monotonicity check failed. Some registers are not monotonic.",
                context={"register_errors": reg_errors},
            )

    @staticmethod
    def _try_parse_semver(version: str) -> t.Optional[semver.Version]:
        """Try to parse a semantic version string.

        Args:
            version: The version string to parse.

        Returns:
            Optional[semver.Version]: The parsed Version object, or None if parsing fails.
        """
        if len(version.split(".")) < 3:
            version += ".0"
        try:
            return semver.Version.parse(version)
        except ValueError:
            return None

    def test_fw_version_matches_reader(self):
        """Check if the firmware version of the device matches the one in the reader"""
        reader = self.harp_device.device_reader

        fw = self._try_parse_semver(reader.device.firmwareVersion)
        device_fw = self._try_parse_semver(
            f"{self._get_last_read(self.harp_device['FirmwareVersionHigh']).iloc[0]}.{self._get_last_read(self.harp_device['FirmwareVersionLow']).iloc[0]}"
        )

        if (fw is None) or (device_fw is None):
            return self.fail_test(
                None, f"Firmware version is not a valid semver version. Expected {fw} and got {device_fw}"
            )
        if fw > device_fw:
            return self.fail_test(
                False,
                f"Expected version {fw} is greater than the device's version {device_fw}. Consider updating the device firmware.",
            )
        elif fw == device_fw:
            return self.pass_test(True, f"Expected version {fw} matches the device's version {device_fw}")
        else:
            return self.warn_test(
                False,
                f"Expected version {fw} is less than the device's version {device_fw}. Consider updating interface package.",
            )

    def test_core_version(self):
        """Check if the core version of the device matches the one provided"""
        core = self._try_parse_semver(self.min_core_version) if self.min_core_version else None
        device_core = self._try_parse_semver(
            f"{self._get_last_read(self.harp_device['CoreVersionHigh']).iloc[0]}.{self._get_last_read(self.harp_device['CoreVersionLow']).iloc[0]}"
        )

        if core is None:
            return self.skip_test("Core version not specified, skipping test.")
        if device_core is None:
            return self.fail_test("Core version is not a valid semver version.")

        if core > device_core:
            return self.fail_test(
                False,
                f"Core version {core} is greater than the device's version {device_core}. Consider updating the device firmware.",
            )
        elif core == device_core:
            return self.pass_test(True, f"Core version {core} matches the device's version {device_core}")
        else:
            return self.warn_test(False, f"Core version {core} is less than the device's version {device_core}")


class HarpHubTestSuite(Suite):
    """Test suite for a hub of Harp devices.

    Tests a collection of Harp devices that share the same clock generator source,
    verifying proper synchronization and configuration.

    Attributes:
        clock_generator_device: The Harp device acting as the clock generator.
        devices: List of subordinate Harp devices to test.
        read_dump_jitter_threshold_s: Maximum allowed time difference for read dumps.

    Examples:
        ```python
        from contraqctor.contract.harp import HarpDevice
        from contraqctor.qc.harp import HarpHubTestSuite
        from contraqctor.qc.base import Runner

        # Create HarpDevice streams
        clock_gen = HarpDevice("clock_gen", reader_params=clock_params).load()
        device1 = HarpDevice("device1", reader_params=params1).load()
        device2 = HarpDevice("device2", reader_params=params2).load()

        # Create and run hub test suite
        suite = HarpHubTestSuite(clock_gen, [device1, device2])
        runner = Runner().add_suite(suite)
        results = runner.run_all_with_progress()
        ```
    """

    def __init__(
        self,
        clock_generator_device: HarpDevice,
        devices: t.List[HarpDevice],
        *,
        read_dump_jitter_threshold_s: t.Optional[float] = 0.05,
    ):
        """Initialize the Harp hub test suite.

        Args:
            clock_generator_device: The Harp device acting as the clock generator.
            devices: List of Harp devices to test as part of the hub.
            read_dump_jitter_threshold_s: Maximum allowed time difference (in seconds)
                between devices' read dumps. Defaults to 0.05.
        """
        self.clock_generator_device = clock_generator_device
        self.devices = [device for device in devices if device is not clock_generator_device]
        self.read_dump_jitter_threshold_s = read_dump_jitter_threshold_s

    def test_clock_generator_reg(self):
        """Checks if the clock generator device is actually a clock generator"""
        if "ClockConfiguration" not in [x.name for x in self.clock_generator_device]:
            return self.fail_test(None, "ClockConfiguration data stream is not present")
        clock_reg = self.clock_generator_device["ClockConfiguration"].data.iloc[-1]
        if clock_reg["ClockGenerator"]:
            return self.pass_test(True, "Clock generator is a clock generator")
        else:
            return self.fail_test(False, "Clock generator is not a clock generator")

    def test_devices_are_subordinate(self):
        """Checks if the devices are subordinate to the clock generator"""
        for device in self.devices:
            if "ClockConfiguration" not in [x.name for x in device]:
                yield self.fail_test(None, f"ClockConfiguration data stream is not present in {device.name}")
            elif device["ClockConfiguration"].data.iloc[-1]["ClockGenerator"]:
                yield self.fail_test(False, f"Device {device.name} is not subordinate to the clock generator")
            else:
                yield self.pass_test(True, f"Device {device.name} is subordinate to the clock generator")

    @staticmethod
    def _get_read_dump_time(device: HarpDevice) -> t.Optional[float]:
        """Get the timestamp of the last read dump for a device.

        Args:
            device: The Harp device to check.

        Returns:
            Optional[float]: The timestamp of the last read dump, or None if no read dump exists.
        """
        try:
            read_dump: pd.DataFrame = device["OperationControl"].data.copy()
            read_dump = read_dump[read_dump["MessageType"] == "WRITE"]
        except KeyError:
            return None
        if len(read_dump) == 0:
            return None
        return read_dump.index[-1]

    def test_is_read_dump_synchronized(self):
        """Check if the read dump from the devices arrives are roughly the same time"""
        if self.read_dump_jitter_threshold_s is None:
            return self.skip_test("No read dump jitter threshold provided, skipping test.")
        clock_dump_time = self._get_read_dump_time(self.clock_generator_device)
        for device in self.devices:
            t_dump = self._get_read_dump_time(device)
            if t_dump is None:
                yield self.fail_test(None, f"Device {device.name} does not have a requested read dump")
            elif (dt := abs(t_dump - clock_dump_time)) > self.read_dump_jitter_threshold_s:
                yield self.fail_test(
                    False,
                    f"Device {device.name} read dump is not synchronized with the clock generator's. dt = {dt:.3f} s vs threshold {self.read_dump_jitter_threshold_s:.3f} s",
                )
            else:
                yield self.pass_test(True, f"Device {device.name} read dump is synchronized with the clock generator's")


class HarpDeviceTypeTestSuite(Suite, abc.ABC):
    """Base test suite for specific types of Harp devices.

    Abstract base class providing common functionality for testing
    specific Harp device types with known WhoAmI identifiers.

    Attributes:
        harp_device: The Harp device to test.
        _WHOAMI: Class variable defining the expected WhoAmI value for this device type.
    """

    _WHOAMI: int

    def __init__(self, harp_device: HarpDevice):
        """Initialize the device type test suite.

        Args:
            harp_device: The Harp device to test.
        """
        self.harp_device = harp_device

    @property
    def whoami(self) -> int:
        """Get the expected WhoAmI value for this device type.

        Returns:
            int: The expected WhoAmI identifier.
        """
        return self._WHOAMI

    def test_whoami(self):
        """Check if the WhoAmI value is correct"""
        try:
            who_am_i = self.harp_device["WhoAmI"].data["WhoAmI"].iloc[-1]
        except KeyError:
            return self.fail_test(None, "WhoAmI data stream is not present")
        if who_am_i != self.whoami:
            return self.fail_test(False, f"Expected WhoAmI value {self.whoami} but got {who_am_i}")
        return self.pass_test(True, f"WhoAmI value is {who_am_i} as expected")
