import typing as t

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..contract.camera import Camera
from ._context_extensions import ContextExportableObj
from .base import Suite


class CameraTestSuite(Suite):
    """Test suite for validating camera data integrity.

    Provides tests for validating video and metadata integrity according to the AIND
    file format specification for behavior videos.

    For more details, see:
    https://github.com/AllenNeuralDynamics/aind-file-standards/blob/ce0aa517a40064d1ac9764d42c9efe4ae5c61f7b/file_formats/behavior_videos.md

    Attributes:
        data_stream: The Camera data stream to test.
        expected_fps: Optional expected frames per second for validation.
        clock_jitter_s: Maximum allowed time difference between frame timestamps, in seconds.
        start_time_s: Optional expected start time for validation, in seconds.
        stop_time_s: Optional expected stop time for validation, in seconds.

    Examples:
        ```python
        from contraqctor.contract.camera import Camera, CameraParams
        from contraqctor.qc.camera import CameraTestSuite
        from contraqctor.qc.base import Runner

        # Create and load a camera data stream
        params = CameraParams(path="recordings/session1/")
        camera_stream = Camera("front_camera", reader_params=params).load()

        # Create test suite with validation parameters
        suite = CameraTestSuite(
            camera_stream,
            expected_fps=30,
            start_time_s=10.0,
            stop_time_s=310.0
        )

        # Run tests
        runner = Runner().add_suite(suite)
        results = runner.run_all_with_progress()
        ```
    """

    _expected_columns = {"ReferenceTime", "CameraFrameNumber", "CameraFrameTime"}

    def __init__(
        self,
        data_stream: Camera,
        *,
        expected_fps: t.Optional[int] = None,
        clock_jitter_s: float = 1e-4,
        start_time_s: t.Optional[float] = None,
        stop_time_s: t.Optional[float] = None,
        saturation_bounds: tuple[t.Optional[int], t.Optional[int]] = (5, 250),
    ):
        """Initialize the camera test suite.

        Args:
            data_stream: The Camera data stream to test.
            expected_fps: Optional expected frames per second for validation.
            clock_jitter_s: Maximum allowed time difference between frame timestamps, in seconds.
            start_time_s: Optional expected start time for validation, in seconds.
            stop_time_s: Optional expected stop time for validation, in seconds.
            saturation_bounds: Pixel intensity bounds to check for saturation (min, max).
        """
        self.data_stream: Camera = data_stream
        self.expected_fps = expected_fps
        self.clock_jitter_s = clock_jitter_s
        self.start_time_s = start_time_s
        self.stop_time_s = stop_time_s
        self.saturation_bounds = saturation_bounds

    def test_metadata_shape(self):
        """
        Checks if the metadata DataFrame has the expected shape. Including headers.
        """
        if not self.data_stream.has_data:
            return self.fail_test(None, "Data stream does not have loaded data")
        metadata = self.data_stream.data.metadata
        if not isinstance(metadata, pd.DataFrame):
            return self.fail_test(None, "Metadata is not a pandas DataFrame")

        (metadata_cols := list(metadata.columns)).append(metadata.index.name)
        if not all(col in metadata_cols for col in self._expected_columns):
            missing_columns = self._expected_columns - set(metadata_cols)
            return self.fail_test(None, f"Metadata columns do not match expected columns. Missing: {missing_columns}")
        if metadata.empty:
            return self.fail_test(None, "Metadata DataFrame is empty")
        return self.pass_test(None, "Metadata DataFrame has expected shape and columns")

    def test_check_dropped_frames(self):
        """
        Check if there are dropped frames in the metadata DataFrame.
        """
        metadata = (self.data_stream.data.metadata[list(self._expected_columns - {"ReferenceTime"})]).copy()
        metadata.loc[:, "ReferenceTime"] = metadata.index.values
        diff_metadata = metadata.diff()
        # Convert CameraFrameTime to seconds
        diff_metadata["CameraFrameTime"] = diff_metadata["CameraFrameTime"] * 1e-9

        if not all(diff_metadata["CameraFrameNumber"].dropna() == 1):
            return self.fail_test(
                None, f"Detected {sum(diff_metadata['CameraFrameNumber'].dropna() - 1)} dropped frames metadata."
            )

        inter_clock_diff = diff_metadata["CameraFrameTime"] - diff_metadata["ReferenceTime"]
        if not all(inter_clock_diff.dropna() < self.clock_jitter_s):
            return self.fail_test(
                None,
                f"Detected a difference between CameraFrameTime and ReferenceTime greater than the expected threshold: {self.clock_jitter_s} s.",
            )
        return self.pass_test(None, "No dropped frames detected in metadata.")

    def test_match_expected_fps(self):
        """
        Check if the frames per second (FPS) of the video metadata matches the expected FPS."""
        if self.expected_fps is None:
            return self.skip_test("No expected FPS provided, skipping test.")
        period = np.diff(self.data_stream.data.metadata.index.values)
        if np.std(period) > 1e-4:
            return self.fail_test(None, f"High std in frame period detected: {np.std(period)}")
        if abs(_mean := np.mean(period) - (_expected := (1.0 / self.expected_fps))) > (_expected * 0.01):
            return self.fail_test(None, f"Mean frame period ({_mean}) is different than expected: {_expected}")

        return self.pass_test(None, f"Mean frame period ({_mean}) is within expected range: {_expected}")

    def test_is_start_bounded(self):
        """
        Check if the start time of the video is bounded by the provided start time."""
        metadata = self.data_stream.data.metadata
        if self.start_time_s is not None:
            if metadata.index[0] < self.start_time_s:
                return self.fail_test(
                    None,
                    f"Start time is not bounded. First frame time: {metadata.index[0]}, expected start time: {self.start_time_s}",
                )
            else:
                return self.pass_test(
                    None,
                    f"Start time is bounded. First frame time: {metadata.index[0]}, expected start time: {self.start_time_s}",
                )
        else:
            return self.skip_test("No start time provided, skipping test.")

    def test_is_stop_bounded(self):
        """
        Check if the stop time of the video is bounded by the provided stop time."""
        metadata = self.data_stream.data.metadata
        if self.stop_time_s is not None:
            if metadata.index[-1] > self.stop_time_s:
                return self.fail_test(
                    None,
                    f"Stop time is not bounded. Last frame time: {metadata.index[-1]}, expected stop time: {self.stop_time_s}",
                )
            else:
                return self.pass_test(
                    None,
                    f"Stop time is bounded. Last frame time: {metadata.index[-1]}, expected stop time: {self.stop_time_s}",
                )
        else:
            return self.skip_test("No stop time provided, skipping test.")

    def test_video_frame_count(self):
        """
        Check if the number of frames in the video matches the number of rows in the metadata DataFrame.
        """
        data = self.data_stream.data
        if not data.has_video:
            return self.skip_test("No video data available. Skipping test.")

        if (n_frames := data.video_frame_count) != len(data.metadata):
            return self.fail_test(
                None,
                f"Number of frames in video ({n_frames}) does not match number of rows in metadata ({len(data.metadata)})",
            )
        else:
            return self.pass_test(
                None,
                f"Number of frames in video ({n_frames}) matches number of rows in metadata ({len(data.metadata)})",
            )

    def test_histogram_and_create_asset(self):
        """Checks the histogram of the video and ensures color is well distributed.
        It also saves an asset with a single frame of the video and color histogram."""

        data = self.data_stream.data
        if not data.has_video:
            return self.skip_test("No video data available. Skipping test.")

        with data.as_video_capture() as video:
            video.set(cv2.CAP_PROP_POS_FRAMES, video.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
            ret, frame = video.read()

            if not ret:
                return self.fail_test(None, "Failed to read a frame from the video")
            max_d = 2 ** (frame.dtype.itemsize * 8)

            fig, ax = plt.subplots(2, frame.shape[2], figsize=(15, 7))

            for channel in range(frame.shape[2]):
                hist = cv2.calcHist([frame], [channel], None, [max_d], [0, max_d])
                hist /= hist.sum()
                ax[0, channel].imshow(frame[:, :, channel], cmap="gray")
                ax[0, channel].axis("off")
                ax[1, channel].plot(hist, color="k", label=f"Channel-{channel}")
                ax[1, channel].set_xlim([0, max_d])
                ax[1, channel].set_xlabel("Pixel Value")
                ax[1, channel].set_ylabel("Normalized Frequency")
                ax[1, channel].set_title(f"Histogram channel-{channel}")
            fig.subplots_adjust(top=0.9)  # Leave space for suptitle
            fig.suptitle("Pixel value histogram")
            fig.tight_layout()

        return self.pass_test(
            None, "Histogram and asset created successfully.", context=ContextExportableObj.as_context(fig)
        )

    def test_create_pixel_saturation_visualizer(self):
        """Creates a visualization highlighting saturated and underexposed pixels in the video frame."""
        data = self.data_stream.data
        if not data.has_video:
            return self.skip_test("No video data available. Skipping test.")

        with data.as_video_capture() as video:
            video.set(cv2.CAP_PROP_POS_FRAMES, video.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
            ret, frame = video.read()

            if not ret:
                return self.fail_test(None, "Failed to read a frame from the video")

            lower_bound, upper_bound = self.saturation_bounds

            fig, ax = plt.subplots(1, frame.shape[2], figsize=(15, 5))

            for channel in range(frame.shape[2]):
                channel_data = frame[:, :, channel]

                channel_saturated = np.zeros(frame.shape[:2], dtype=bool)
                channel_underexposed = np.zeros(frame.shape[:2], dtype=bool)

                if upper_bound is not None:
                    channel_saturated = channel_data >= upper_bound
                if lower_bound is not None:
                    channel_underexposed = channel_data <= lower_bound

                # Create RGB image: grayscale with saturated pixels in red and underexposed in blue
                colored_frame = np.stack([channel_data, channel_data, channel_data], axis=-1)
                colored_frame[channel_saturated] = [255, 0, 0]  # Red for saturated
                colored_frame[channel_underexposed] = [0, 0, 255]  # Blue for underexposed

                ax[channel].imshow(colored_frame)
                ax[channel].axis("off")
                ax[channel].set_title(f"Channel-{channel}")

            fig.subplots_adjust(top=0.9)  # Leave space for suptitle
            fig.suptitle("Pixel Saturation Visualization (bounds: {})".format(self.saturation_bounds))
            fig.tight_layout()

            return self.pass_test(
                None, "Histogram and asset created successfully.", context=ContextExportableObj.as_context(fig)
            )
