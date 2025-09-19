import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing_extensions import override

from ...contract.harp import HarpDevice
from .._context_extensions import ContextExportableObj
from .harp_device import HarpDeviceTypeTestSuite


class HarpEnvironmentSensorTestSuite(HarpDeviceTypeTestSuite):
    """Test suite for Harp Environment Sensor devices.

    Provides tests specific to the Environment Sensor device, including
    signal quality analysis and breathing rate detection.

    Attributes:
        harp_device: The Harp Environment Sensor device to test.
        data: The raw voltage data from the device.
        fs: The sampling frequency of the device.

    Examples:
        ```python
        from contraqctor.contract.harp import HarpDevice
        from contraqctor.qc.harp import HarpEnvironmentSensorTestSuite
        from contraqctor.qc.base import Runner

        # Create and load the environment sensor device
        device = HarpDevice("environment", reader_params=params).load()

        # Create the test suite with custom thresholds
        suite = HarpEnvironmentSensorTestSuite(device)

        # Run tests
        runner = Runner().add_suite(suite)
        results = runner.run_all_with_progress()
        ```
    """

    _WHOAMI = 1405
    _FULL_BIT_DEPTH = 2**12
    # from https://grants.nih.gov/grants/olaw/guide-for-the-care-and-use-of-laboratory-animals.pdf
    temperature_limit = (20, 26)  # degrees Celsius
    humidity_limit = (30, 70)  # percent

    @override
    def __init__(
        self,
        harp_device: HarpDevice,
    ):
        """Initialize the Environment Sensor test suite.

        Args:
            harp_device: The Harp Environment Sensor device to test.
        """
        super().__init__(harp_device)
        self.harp_device = harp_device
        self.data: pd.DataFrame = self.harp_device["SensorData"].data.copy()
        self.data = self.data[self.data["MessageType"] == "EVENT"]

    def test_sampling_rate(self):
        """Tests if the sampling rate of the environment sensor is within nominal values"""
        if len(self.data) < 2:
            return self.fail_test(0, "Not enough data to compute sampling rate")
        period = self.data.index.diff().dropna()
        mean_hz = 1.0 / np.mean(period)
        return self.pass_test(mean_hz, f"Sampling rate is {mean_hz:.2f} Hz")

    def test_temperature_within_expected_limits(self):
        """Tests if the temperature sensor readings are within expected limits"""
        metrics = {}
        metrics["min"] = min_temp = self.data["Temperature"].min()
        metrics["max"] = max_temp = self.data["Temperature"].max()
        metrics["mean"] = self.data["Temperature"].mean()

        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.data.index, self.data["Temperature"], label="Temperature (°C)")
        plt.axhline(self.temperature_limit[0], color="r", linestyle="--", label="Min Expected Temp")
        plt.axhline(self.temperature_limit[1], color="g", linestyle="--", label="Max Expected Temp")
        plt.title("Temperature Over Time")
        plt.xlabel("Time")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        context = ContextExportableObj.as_context(fig)
        context.update(metrics)

        if (min_temp < self.temperature_limit[0]) or (max_temp > self.temperature_limit[1]):
            return self.warn_test(
                metrics,
                f"Temperature out of expected range ({self.temperature_limit[0]}-{self.temperature_limit[1]} °C)",
                context=context,
            )
        return self.pass_test(
            metrics,
            f"Temperature within expected range ({self.temperature_limit[0]}-{self.temperature_limit[1]} °C)",
            context=context,
        )

    def test_humidity_within_expected_limits(self):
        """Tests if the humidity sensor readings are within expected limits"""
        metrics = {}
        metrics["min"] = min_humidity = self.data["Humidity"].min()
        metrics["max"] = max_humidity = self.data["Humidity"].max()
        metrics["mean"] = self.data["Humidity"].mean()

        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.data.index, self.data["Humidity"], label="Humidity (%)", color="orange")
        plt.axhline(self.humidity_limit[0], color="r", linestyle="--", label="Min Expected Humidity")
        plt.axhline(self.humidity_limit[1], color="g", linestyle="--", label="Max Expected Humidity")
        plt.title("Humidity Over Time")
        plt.xlabel("Time")
        plt.ylabel("Humidity (%)")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        context = ContextExportableObj.as_context(fig)
        context.update(metrics)

        if (min_humidity < self.humidity_limit[0]) or (max_humidity > self.humidity_limit[1]):
            return self.warn_test(
                metrics,
                f"Humidity out of expected range ({self.humidity_limit[0]}-{self.humidity_limit[1]} %)",
                context=context,
            )
        return self.pass_test(
            metrics,
            f"Humidity within expected range ({self.humidity_limit[0]}-{self.humidity_limit[1]} %)",
            context=context,
        )
