import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing_extensions import override

from ...contract.harp import HarpDevice
from .._context_extensions import ContextExportableObj
from .harp_device import HarpDeviceTypeTestSuite


class HarpLicketySplitTestSuite(HarpDeviceTypeTestSuite):
    """Test suite for Harp Lickety Split devices.

    Provides tests specific to the Lickety Split device, including
    signal quality analysis and breathing rate detection.

    Attributes:
        harp_device: The Harp Lickety Split device to test.
        data: The event data from lick detection.
        lick_refractory_period: Minimum time between detected licks (in seconds).

    Examples:
        ```python
        from contraqctor.contract.harp import HarpDevice
        from contraqctor.qc.harp import HarpLicketySplitTestSuite
        from contraqctor.qc.base import Runner

        # Create and load the lickometer device
        device = HarpDevice("lickometer", reader_params=params).load()

        # Create the test suite with custom refractory period
        suite = HarpLicketySplitTestSuite(
            device,
            lick_refractory_period=0.05
            )

        # Run tests
        runner = Runner().add_suite(suite)
        results = runner.run_all_with_progress()
        ```
    """

    _WHOAMI = 1400
    _target_channel = "Channel0"

    @override
    def __init__(
        self,
        harp_device: HarpDevice,
        lick_refractory_period: float = 0.05,
    ):
        """Initialize the Sniff Detector test suite.

        Args:
            harp_device: The Harp Sniff Detector device to test.
            lick_refractory_period: Minimum time between detected licks (in seconds).
        """
        super().__init__(harp_device)
        self.harp_device = harp_device
        self.data: pd.DataFrame = self.harp_device["LickState"].data.copy()
        self.data = self.data[self.data["MessageType"] == "EVENT"]
        self.lick_refractory_period = lick_refractory_period

    @staticmethod
    def _get_distinct_from_channel(data: pd.DataFrame, channel: str):
        """Get distinct events from a specified channel."""
        lick_times = data[channel]
        mask = lick_times != lick_times.shift()
        return lick_times[mask]

    def test_refractory_period_violations(self):
        """Tests for violations where lick onsets occur within the refractory period."""
        lick_onsets = self._get_distinct_from_channel(self.data, self._target_channel).where(lambda x: x == 1).index
        inter_lick_intervals = np.diff(lick_onsets)
        violations = inter_lick_intervals < self.lick_refractory_period
        num_violations = np.sum(violations)

        metrics = {
            "total_licks": len(lick_onsets),
            "num_violations": int(num_violations),
            "percent_violations": float(num_violations / len(lick_onsets)) if len(lick_onsets) > 0 else 0.0,
        }

        fig = plt.figure(figsize=(10, 4))
        plt.hist(inter_lick_intervals, bins=np.arange(0, 0.5, 0.005), color="blue", alpha=0.7)
        plt.axvline(self.lick_refractory_period, color="red", linestyle="dashed", linewidth=1)
        plt.title("Inter-Lick Intervals (s)")
        plt.xlabel("Interval (s)")
        plt.ylabel("Count")
        plt.tight_layout()
        context = ContextExportableObj.as_context(fig)
        context.update(metrics)
        if metrics["percent_violations"] > 0.5:
            return self.fail_test(metrics, "Critical number of refractory period violations (>50%).", context=context)
        if metrics["percent_violations"] > 0.05:
            return self.warn_test(metrics, "High number of refractory period violations (>5%).", context=context)
        return self.pass_test(metrics, "Refractory period violations within acceptable range.", context=context)

    def test_minimum_lick_rate(self):
        """Tests if the lick rate meets a minimum threshold."""
        _min_lick_threshold = 0.1  # licks per second
        heartbeat = self.harp_device["TimestampSeconds"].read().index
        session_duration = heartbeat[-1] - heartbeat[0]
        lick_onsets = self._get_distinct_from_channel(self.data, self._target_channel).where(lambda x: x == 1).index
        total_licks = len(lick_onsets)
        lick_rate = total_licks / session_duration if session_duration > 0 else 0.0

        metrics = {"total_licks": total_licks, "lick_rate": lick_rate}

        if lick_rate < _min_lick_threshold:
            return self.fail_test(
                metrics, f"Minimum lick rate not met ({lick_rate:.2f} < {_min_lick_threshold:.2f}).", context=metrics
            )
        return self.pass_test(
            metrics,
            f"Lick rate meets minimum threshold ({lick_rate:.2f} >= {_min_lick_threshold:.2f}).",
            context=metrics,
        )

    def test_lick_duration(self):
        """Tests for licks that are shorter than the expected duration."""
        limits = (0.015, 1)  # in seconds
        lick = self._get_distinct_from_channel(self.data, self._target_channel)
        first_lick_onset = lick[lick == 1].index[0]
        lick = lick[first_lick_onset:]
        lick_durations = lick[lick == 0].index - lick[lick == 1].index

        fig = plt.figure(figsize=(10, 4))
        short = fig.add_subplot(121)
        short.hist(lick_durations, bins=np.arange(0, 0.2, 0.005), color="blue", alpha=0.7)
        short.axvline(limits[0], color="red", linestyle="dashed", linewidth=1)
        short.set_title("Short Inter-Lick Durations (s)")
        short.set_xlabel("Duration (s)")
        short.set_ylabel("Count")

        long = fig.add_subplot(122)
        long.hist(lick_durations, bins=np.linspace(0, np.max(lick_durations), 20), color="blue", alpha=0.7)
        long.axvline(limits[1], color="red", linestyle="dashed", linewidth=1)
        long.set_title("Long Inter-Lick Durations (s)")
        long.set_xlabel("Duration (s)")
        long.set_ylabel("Count")

        fig.tight_layout()
        context = ContextExportableObj.as_context(fig)

        metrics = {}
        metrics["mean"] = np.mean(lick_durations) if len(lick_durations) > 0 else None
        metrics["std"] = np.std(lick_durations) if len(lick_durations) > 0 else None
        metrics["percent_violations"] = (
            np.sum((lick_durations < limits[0]) | (lick_durations > limits[1])) / len(lick_durations)
            if len(lick_durations) > 0
            else 0.0
        )
        metrics["total_licks"] = len(lick_durations)
        metrics["num_violations"] = int(np.sum((lick_durations < limits[0]) | (lick_durations > limits[1])))
        metrics["max"] = np.max(lick_durations) if len(lick_durations) > 0 else None
        metrics["min"] = np.min(lick_durations) if len(lick_durations) > 0 else None
        metrics["num_long_violations"] = int(np.sum(lick_durations > limits[1]))
        metrics["num_short_violations"] = int(np.sum(lick_durations < limits[0]))
        context.update(metrics)

        if metrics["num_long_violations"] > 0:
            return self.warn_test(metrics, "Long lick duration violations detected.", context=context)
        if metrics["percent_violations"] > 0.05:
            return self.warn_test(metrics, "High number of lick duration violations (>5%).", context=context)
        return self.pass_test(metrics, "Lick duration distribution within expected range.", context=context)
