import numpy as np
import pandas as pd
from typing_extensions import override

from ...contract.harp import HarpDevice
from .harp_device import HarpDeviceTypeTestSuite


class HarpTreadmillTestSuite(HarpDeviceTypeTestSuite):
    """Test suite for Harp Treadmill devices.

    Provides tests specific to the Treadmill device.

    Attributes:
        _harp_device: The Harp Treadmill device to test.
        _data: The data from the periodic sensor events.

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
    def __init__(self, _harp_device: HarpDevice, *, adc_mid_tol_percent: float = 0.05, max_tick_jump: int = 410):
        """Initialize the Treadmill test suite.

        Args:
            _harp_device: The Harp Treadmill device to test.
            adc_mid_tol_percent: Tolerance percentage for the median torque value to be around the midpoint of the expected ADC range. Default is 10%.
            max_tick_jump: Maximum allowed jump in encoder ticks between consecutive readings. Default is 410 ticks, which corresponds to 5% of a full revolution for an encoder with 8192 pulses per revolution (PPR).
        """
        super().__init__(_harp_device)
        self._harp_device = _harp_device
        self._data: pd.DataFrame = self._harp_device["SensorData"].data.copy()
        self._data = self._data[self._data["MessageType"] == "EVENT"]
        self._adc_mid_tol_percent = adc_mid_tol_percent
        self._max_tick_jump = max_tick_jump

    def test_sampling_rate(self):
        """Tests if the sampling rate of the treadmill is within nominal values"""
        period = self._data.index.diff().dropna()
        mean_period = np.mean(period)
        fs: float = self._harp_device["SensorDataDispatchRate"].data.iloc[-1].values[0]
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

        d = self._data["Encoder"].diff().dropna()
        # apply two's complement wrap for signed 32-bit
        mask = 0xFFFFFFFF
        d = d.astype(np.int64) & mask  # force 32-bit space

        # reinterpret as signed 32-bit
        d = np.where(d >= 0x80000000, d - 0x100000000, d)
        metrics["total_ticks"] = np.sum(d)
        metrics["max_jump"] = np.max(np.abs(d))
        if metrics["total_ticks"] == 0:
            return self.fail_test(
                metrics, "Total ticks is zero, indicating the treadmill did not move during the session."
            )
        if metrics["max_jump"] > self._max_tick_jump:
            return self.warn_test(
                metrics,
                f"Maximum jump between consecutive encoder readings is {metrics['max_jump']} ticks (expected a maximum of {self._max_tick_jump}), which is unusually high and may indicate signal corruption or missed readings.",
            )
        return self.pass_test(metrics, "All encoder metrics are within expected limits.")

    def test_torque_range(self):
        """Tests if the torque signal is within expected nominal ADC range (10-4000)"""
        MIN, MAX = 95, 4000  # The ADC is 12-bit, but we add a small fudge factor on the edges.
        MID = (MIN + MAX) / 2  # The ADC is expected to be around mid-scale when the treadmill is stationary
        SOFT_MIN = MIN + (MAX - MIN) * 0.2  # 20% above the minimum
        SOFT_MAX = MAX - (MAX - MIN) * 0.2  # 20% below the maximum

        torque = self._data["Torque"].copy()
        metrics = {}
        metrics["min"] = torque.min()
        metrics["max"] = torque.max()
        metrics["mean"] = torque.mean()
        metrics["median"] = torque.median()
        metrics["std"] = torque.std()
        metrics["iqr"] = torque.quantile(0.75) - torque.quantile(0.25)

        if metrics["min"] < MIN or metrics["max"] > MAX:
            return self.fail_test(
                metrics,
                f"Torque signal out of expected nominal ADC range ({MIN} : {MAX}). This indicates the torque sensor was damaged during operation.",
            )
        if metrics["min"] < SOFT_MIN or metrics["max"] > SOFT_MAX:
            return self.warn_test(
                metrics,
                f"Torque signal out of expected soft ADC range ({SOFT_MIN} : {SOFT_MAX}). This MAY indicate the torque sensor was damaged during operation or it is not properly installed/calibrated.",
            )
        if abs(metrics["median"] - MID) > (MID * self._adc_mid_tol_percent):
            return self.warn_test(
                metrics,
                f"Torque signal median value {metrics['median']} is expected to be within {self._adc_mid_tol_percent * 100}% of mid-scale value: {MID}.",
            )

        return self.pass_test(metrics, "All metrics are within expected limits.")

    def test_torque_limit_tripwire(self):
        """Tests if the torque limit tripwire was triggered."""
        tripwire = self._harp_device["TorqueLimitState"].read()
        tripwire = tripwire[tripwire["MessageType"] == "EVENT"]
        trips = tripwire["TorqueLimitState"] > 0
        if n := trips.sum() == 0:
            return self.pass_test(n, "Torque limit tripwire was never triggered during the session")
        return self.fail_test(n, f"Torque limit tripwire was triggered {n} times during the session")
