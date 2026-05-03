import os
import typing as t
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from cv2 import CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, VideoCapture

from .base import DataStream, FilePathBaseParam


@dataclass(frozen=True)
class CameraData:
    """Container for camera data including metadata and video file reference.

    A frozen dataclass that holds both metadata about video recordings and
    a reference to the corresponding video file path.

    Attributes:
        metadata: DataFrame containing camera frame metadata such as timestamps and frame indices.
        video_path: Path to the video file associated with the metadata.
    """

    metadata: pd.DataFrame
    video_path: os.PathLike

    @property
    def has_video(self) -> bool:
        """Check if the referenced video file exists and can be opened.

        Returns:
            bool: True if the video file exists and can be opened, False otherwise.
        """
        if not (self.video_path is not None and os.path.exists(self.video_path)):
            return False
        # Not sure why this would fail, but I since its a check, lets make sure we catch it
        try:
            with self.as_video_capture() as video:
                return video.isOpened()
        except Exception:
            return False

    @contextmanager
    def as_video_capture(self):
        """Context manager for handling video capture resources.

        Opens the video file as a cv2.VideoCapture and ensures it's properly released
        after use, even if an exception occurs.

        Yields:
            VideoCapture: OpenCV VideoCapture object for the video file.

        Examples:
            ```python
            import cv2

            # Process video frames
            with camera_data.as_video_capture() as cap:
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite("first_frame.jpg", gray)
            ```
        """
        cap = VideoCapture(str(self.video_path))
        try:
            yield cap
        finally:
            cap.release()

    @property
    def video_frame_count(self) -> int:
        """Get the total number of frames in the video.

        Returns:
            int: The number of frames in the video file.
        """
        with self.as_video_capture() as video:
            return int(video.get(CAP_PROP_FRAME_COUNT))

    @property
    def video_frame_size(self) -> t.Tuple[int, int]:
        """Get the dimensions of the video frames.

        Returns:
            tuple: A tuple containing the width and height of video frames in pixels.
        """
        with self.as_video_capture() as video:
            return int(video.get(CAP_PROP_FRAME_WIDTH)), int(video.get(CAP_PROP_FRAME_HEIGHT))


@dataclass
class CameraParams(FilePathBaseParam):
    """Parameters for camera data processing.

    Extends the base file path parameters with camera-specific options.

    Attributes:
        metadata_name: Base filename of the CSV file containing frame metadata.
        video_name: Base filename of the video file (without extension).
    """

    metadata_name: str = "metadata"
    video_name: str = "video"


class Camera(DataStream[CameraData, CameraParams]):
    """Camera data stream provider.

    A data stream implementation for reading camera metadata and video files,
    combining them into a single CameraData object.

    Args:
        DataStream: Base class for data stream providers.

    Examples:
        ```python
        from contraqctor.contract.camera import Camera, CameraParams

        # Create and load a camera stream
        params = CameraParams(path="recordings/experiment_1/camera1/")
        cam_stream = Camera("front_view", reader_params=params).load()

        # Access the data
        camera_data = cam_stream.data
        metadata_df = camera_data.metadata

        # Check video properties
        if camera_data.has_video:
            print(f"Dimensions: {camera_data.video_frame_size}")
        ```
    """

    @staticmethod
    def _reader(params: CameraParams) -> CameraData:
        """Read camera metadata CSV and locate associated video file.

        Args:
            params: Parameters for camera data reading configuration.

        Returns:
            CameraData: Object containing metadata DataFrame and path to video file.

        Raises:
            ValueError: If metadata is missing required columns.
            FileNotFoundError: If no video file matching the specified name is found.
        """
        # Read the metadata and validate the required columns
        metadata = pd.read_csv(Path(params.path) / (params.metadata_name + ".csv"), header=0)
        required_columns = {"ReferenceTime", "CameraFrameNumber", "CameraFrameTime"}
        if not required_columns.issubset(metadata.columns):
            raise ValueError(f"Metadata is missing required columns: {required_columns - set(metadata.columns)}")
        metadata.set_index("ReferenceTime", inplace=True)

        candidates_path = list(Path(params.path).glob(f"{params.video_name}.*"))
        if len(candidates_path) == 0:
            raise FileNotFoundError(
                f"No video file found with name '{params.video_name}' and any extension in {params.path}"
            )
        else:
            video_path = candidates_path[0]

        return CameraData(metadata=metadata, video_path=video_path)

    make_params = CameraParams
