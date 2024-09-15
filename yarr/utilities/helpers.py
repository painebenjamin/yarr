import numpy as np

from typing import List, Optional
from typing_extensions import Literal

from moviepy.editor import ImageSequenceClip # type: ignore[import-untyped]
from moviepy.video.io.ffmpeg_writer import ffmpeg_write_video # type: ignore[import-untyped]
from PIL import Image

__all__ = [
    "write_video",
    "find_nearest_multiple",
    "get_normalized_dim"
]

def write_video(
    frames: List[Image.Image],
    path: str,
    fps: int,
    crf: Optional[int]=18,
    ffmpeg_params: List[str]=[],
) -> None:
    """
    Write a list of frames to a video file

    :param frames: The list of frames to be written
    :param path: The name of the video file
    :param fps: The frames per second of the video
    :param codec: The codec to be used for encoding the video. Default is "libx264"
    :param bitrate: The bitrate of the video. Default is "5M"
    :param preset: The preset of the video. Default is "medium"
    :param ffmpeg_params: Additional parameters to be passed to ffmpeg
    """
    if not path.endswith(".mp4"):
        path += ".mp4"

    if "-crf" not in ffmpeg_params and crf is not None:
        ffmpeg_params.extend(["-crf", str(crf)])

    clip = ImageSequenceClip(
        [np.array(frame) for frame in frames],
        fps=fps
    )
    ffmpeg_write_video(
        clip,
        path,
        fps,
        ffmpeg_params=ffmpeg_params,
        verbose=False,
        logger=None,
    )

def find_nearest_multiple(
    input_number: int,
    multiple: int,
    direction: Literal["up", "down"] = "up",
) -> int:
    """
    Find the nearest multiple of a number in a given direction

    >>> find_nearest_multiple(95, 8)
    96
    >>> find_nearest_multiple(100, 8)
    104

    :param input_number: The number for which the nearest multiple is to be found
    :param multiple: The number whose multiple is to be found
    :param direction: The direction in which the nearest multiple is to be found. Default is "up"
    :return: The nearest multiple of the input number
    """
    if input_number % multiple == 0:
        return input_number
    if direction == "down":
        return input_number - (input_number % multiple)
    return input_number + multiple - (input_number % multiple)

def get_normalized_dim(
    dim: int,
    multiple_of: int=8,
    down_ratio: float=2/3,
) -> int:
    """
    Gets a normalized dimension by first multiplying by a down ratio,
    then rounding up to the nearest multiple of a number.

    The default of 2/3 does a good job in keeping a balance of rounding
    down in the first step and up in the second step.

    >>> get_normalized_dim(76, 32)
    64
    >>> get_normalized_dim(100, 32)
    96
    >>> get_normalized_dim(106, 32)
    96
    >>> get_normalized_dim(146, 32)
    128

    :param dim: The dimension to be normalized
    :param down_ratio: The ratio by which the dimension is first multiplied by. Default is 2/3
    :param multiple_of: The number to which the dimension is rounded to. Default is 8
    :return: The normalized dimension
    """
    return find_nearest_multiple(
        int(dim * down_ratio),
        multiple_of,
        "up"
    )
