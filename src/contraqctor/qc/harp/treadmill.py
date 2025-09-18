import numpy as np
import pandas as pd
from typing_extensions import override

from ...contract.harp import HarpDevice
from .harp_device import HarpDeviceTypeTestSuite


class HarpTreadmillTestSuite(HarpDeviceTypeTestSuite):
    """Test suite for Harp Treadmill devices.

    Provides tests specific to the Treadmill device.

    Attributes:
        harp_device: The Harp Treadmill device to test.
        data: The data from the periodic sensor events.

    Examples:
        ```python
        from contraqctor.contract.harp import HarpDevice
        from contraqctor.qc.harp import HarpTreadmillTestSuite
        from contraqctor.qc.base import Runner

        # Create and load the treadmill device
        device = HarpDevice("treadmill", reader_params=params).load()

        # Create the test suite with custom thresholds
        suite = HarpTreadmillTestSuite(device)

        # Run tests
        runner = Runner().add_suite(suite)
        results = runner.run_all_with_progress()
        ```
    """

    _WHOAMI = 1402

    @override
    def __init__(
        self,
        harp_device: HarpDevice,
    ):
        """Initialize the Treadmill test suite.

        Args:
            harp_device: The Harp Treadmill device to test.
        """
        super().__init__(harp_device)
        self.harp_device = harp_device
        self.data: pd.DataFrame = self.harp_device["SensorData"].data.copy()
        self.data = self.data[self.data["MessageType"] == "EVENT"]

    def test_sampling_rate(self):
        """Tests if the sampling rate of the treadmill is within nominal values"""
        period = self.data.index.diff().dropna()
        mean_period = np.mean(period)
        fs: float = self.harp_device["SensorDataDispatchRate"].data.iloc[-1].values[0]
        if fs == 0:
            return self.fail_test(0, "Sampling rate is zero")

        if abs((dfps := (1.0 / mean_period)) - fs) > 0.1:
            return self.fail_test(
                dfps,
                f"Sampling rate is not within nominal values. Expected {fs} Hz but got {1.0 / mean_period:.2f} Hz",
            )
        return self.pass_test(dfps, f"Sampling rate is {dfps:.2f} Hz. Expected {fs} Hz")

    def test_encoder(self):
        """Tests the quality of the treadmill signal by calculating total distance and sudden jumps."""
        metrics = {}

        d = self.data["Encoder"].diff().dropna()
        # apply two's complement wrap for signed 32-bit
        mask = 0xFFFFFFFF
        d = d.astype(np.int64) & mask  # force 32-bit space
        d = d.astype(np.int64)

        # reinterpret as signed 32-bit
        d = np.where(d >= 0x80000000, d - 0x100000000, d)
        metrics["total_ticks"] = np.sum(d)
        if metrics["total_ticks"] == 0:
            return self.fail_test(metrics, "Total ticks is zero")
        else:
            return self.pass_test(metrics, f"Total ticks is {metrics['total_ticks']}")

    def test_torque(self):
        """Tests if the torque signal was within nominal values."""
        torque = self.data["Torque"].copy()
        metrics = {}
        metrics["min"] = torque.min()
        metrics["max"] = torque.max()
        metrics["mean"] = torque.mean()
        metrics["std"] = torque.std()
        if metrics["min"] < 10 or metrics["max"] > 4000:
            return self.warn_test(metrics, "Torque signal out of expected nominal range (10-4000)")
        return self.pass_test(metrics, "Torque signal within expected nominal range (10-4000)")

    def test_torque_limit_tripwire(self):
        """Tests if the torque limit tripwire was triggered."""
        tripwire = self.harp_device["TorqueLimitState"].read()
        tripwire = tripwire[tripwire["MessageType"] == "EVENT"]
        trips = tripwire["TorqueLimitState"] > 0
        if n := trips.sum() == 0:
            return self.pass_test(n, "Torque limit tripwire was never triggered during the session")
        return self.fail_test(n, f"Torque limit tripwire was triggered {n} times during the session")
