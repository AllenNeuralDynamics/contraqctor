import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks, iirnotch
from typing_extensions import override

from ...contract.harp import HarpDevice
from .._context_extensions import ContextExportableObj
from .harp_device import HarpDeviceTypeTestSuite


class HarpSniffDetectorTestSuite(HarpDeviceTypeTestSuite):
    """Test suite for Harp Sniff Detector devices.

    Provides tests specific to the Sniff Detector device, including
    signal quality analysis and breathing rate detection.

    Attributes:
        harp_device: The Harp Sniff Detector device to test.
        data: The raw voltage data from the device.
        fs: The sampling frequency of the device.
        quantization_ratio_thr: Threshold for the quantization ratio test.
        clustering_thr: Threshold for the clustering ratio test.
        clipping_thr: Threshold for the clipping detection test.
        sudden_jumps_thr: Threshold for the sudden jumps detection test.
        notch_filter_freq: Frequency (Hz) for the notch filter.

    Examples:
        ```python
        from contraqctor.contract.harp import HarpDevice
        from contraqctor.qc.harp import HarpSniffDetectorTestSuite
        from contraqctor.qc.base import Runner

        # Create and load the sniff detector device
        device = HarpDevice("sniff", reader_params=params).load()

        # Create the test suite with custom thresholds
        suite = HarpSniffDetectorTestSuite(
            device,
            quantization_ratio_thr=0.1,
            clustering_thr=0.05,
            notch_filter_freq=60  # For 60Hz power
        )

        # Run tests
        runner = Runner().add_suite(suite)
        results = runner.run_all_with_progress()
        ```
    """

    _WHOAMI = 1401
    _FULL_BIT_DEPTH = 2**12

    @override
    def __init__(
        self,
        harp_device: HarpDevice,
        quantization_ratio_thr: float = 0.1,
        clustering_thr: float = 0.05,
        clipping_thr: float = 0.05,
        sudden_jumps_thr: float = 0.001,
        notch_filter_freq: float = 50,
    ):
        """Initialize the Sniff Detector test suite.

        Args:
            harp_device: The Harp Sniff Detector device to test.
            quantization_ratio_thr: Threshold for the quantization ratio test. Defaults to 0.1.
            clustering_thr: Threshold for the clustering ratio test. Defaults to 0.05.
            clipping_thr: Threshold for the clipping detection test. Defaults to 0.05.
            sudden_jumps_thr: Threshold for the sudden jumps detection test. Defaults to 0.001.
            notch_filter_freq: Frequency (Hz) for the notch filter. Defaults to 50.
        """
        super().__init__(harp_device)
        self.harp_device = harp_device
        self.data: pd.DataFrame = self.harp_device["RawVoltage"].data.copy()
        self.data = self.data[self.data["MessageType"] == "EVENT"]["RawVoltage"]
        self.fs: float = self.harp_device["RawVoltageDispatchRate"].data.iloc[-1].values[0]
        self.quantization_ratio_thr = quantization_ratio_thr
        self.clustering_thr = clustering_thr
        self.clipping_thr = clipping_thr
        self.sudden_jumps_thr = sudden_jumps_thr
        self.notch_filter_freq = notch_filter_freq

    def test_sampling_rate(self):
        """Tests if the sampling rate of the sniff detector is within nominal values"""
        period = self.data.index.diff().dropna()
        mean_period = np.mean(period)
        if abs((dfps := (1.0 / mean_period)) - self.fs) > 0.1:
            return self.fail_test(
                dfps,
                f"Sampling rate is not within nominal values. Expected {self.fs} Hz but got {1.0 / mean_period:.2f} Hz",
            )
        return self.pass_test(dfps, f"Sampling rate is {dfps:.2f} Hz. Expected {self.fs} Hz")

    def test_signal_quality(self):
        """Tests the quality of the sniff detector signal by analyzing quantization, clustering, clipping, and sudden jumps."""
        metrics = {}
        TOTAL_SAMPLES = len(self.data)

        metrics["quantization_ratio"] = len(np.unique(self.data.values)) / self._FULL_BIT_DEPTH

        hist, _ = np.histogram(self.data.values, bins=self._FULL_BIT_DEPTH)
        metrics["clustering_ratio"] = np.max(hist) / TOTAL_SAMPLES

        # Check for clipping:
        tol = (np.max(self.data) - np.min(self.data)) * 0.01

        metrics["min_clipping"] = np.sum(np.abs(self.data - np.min(self.data)) < tol) / TOTAL_SAMPLES
        metrics["max_clipping"] = np.sum(np.abs(self.data - np.max(self.data)) < tol) / TOTAL_SAMPLES

        # Check for weird discontinuities
        derivative = np.diff(self.data.values) / np.diff(self.data.index)
        sudden_jumps_ratio = (np.sum(np.abs(derivative) > 3 * np.std(derivative))) / TOTAL_SAMPLES
        metrics["sudden_jumps_ratio"] = sudden_jumps_ratio

        is_ok = (
            metrics["quantization_ratio"] > self.quantization_ratio_thr
            and metrics["clustering_ratio"] < self.clustering_thr
            and metrics["min_clipping"] < self.clipping_thr
            and metrics["max_clipping"] < self.clipping_thr
            and metrics["sudden_jumps_ratio"] < self.sudden_jumps_thr
        )

        if is_ok:
            return self.pass_test(True, "Signal quality is good", context=metrics)
        else:
            return self.fail_test(
                False,
                "Signal quality is not good",
                context=metrics,
            )

    def test_physiological_relevance(self):
        """Tests if the sniff detector is actually detecting sniffs by analyzing peaks in the signal."""

        t = self.data.index.values
        signal = self.data.values
        dt = 1.0 / self.fs
        t_uniform = np.arange(t[0], t[-1], dt)

        interp_func = interp1d(t, signal, kind="linear", bounds_error=False, fill_value="extrapolate")
        y_uniform = interp_func(t_uniform)

        Q = 30.0
        b_notch, a_notch = iirnotch(self.notch_filter_freq, Q, self.fs)
        y_notch = filtfilt(b_notch, a_notch, y_uniform)

        b_high, a_high = butter(2, 0.2, "highpass", fs=self.fs)
        y_filtered = filtfilt(b_high, a_high, y_notch)

        b_low, a_low = butter(2, 15, "lowpass", fs=self.fs)
        y_filtered = filtfilt(b_low, a_low, y_filtered)

        peaks, _ = find_peaks(y_filtered, height=0.5 * np.std(y_filtered), prominence=2.5)

        # Create the asset and pass it in the context
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].plot(t_uniform, y_uniform, "b-")
        axes[0].plot(t_uniform[peaks], y_uniform[peaks], "ro")
        axes[0].set_title("Filtered Breathing Signal with Detected Peaks")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        middle_time = (t_uniform[0] + t_uniform[-1]) / 2
        axes[0].set_xlim(middle_time - 30 / 2, middle_time + 30 / 2)

        if len(peaks) >= 2:
            ipi = np.diff(peaks) * dt
            breathing_rate = 1.0 / np.mean(ipi)
            metrics = {
                "num_peaks": len(peaks),
                "mean_ipi": np.mean(ipi),
                "std_ipi": np.std(ipi),
                "breathing_rate_hz": breathing_rate,
                "perc99": 1.0 / np.percentile(ipi, 0.99),
                "perc01": 1.0 / np.percentile(ipi, 0.01),
            }

            axes[1].hist(ipi, bins=np.arange(0, 1, 0.025), alpha=0.7)
            axes[1].axvline(
                np.mean(ipi),
                color="r",
                linestyle="--",
                label=f"Mean: {np.mean(ipi):.3f}s ({breathing_rate * 60:.1f} BPM)",
            )
            axes[1].set_title("Histogram of Inter-Peak Intervals")
            axes[1].set_xlabel("Interval (s)")
            axes[1].set_ylabel("Count")
            axes[1].legend()

            fig.tight_layout()

            context = ContextExportableObj.as_context(fig)
            context.update(metrics)
            min_max_breathing_rate = (2, 10)  # in Hz
            if min_max_breathing_rate[0] <= breathing_rate <= min_max_breathing_rate[1]:
                return self.pass_test(metrics, f"Breathing rate is {breathing_rate} Hz", context=context)
            else:
                return self.warn_test(
                    metrics,
                    f"Breathing rate is {breathing_rate} Hz. Expected between {min_max_breathing_rate[0]} and {min_max_breathing_rate[1]} Hz",
                    context=context,
                )

        else:
            return self.fail_test(
                {"num_peaks": len(peaks)}, "Failed to detect sufficient peaks in the breathing signal", context=context
            )
