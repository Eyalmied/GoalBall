"""
Video handling: open it, describe it clearly, sample frames from it.

The pipeline only ever *reads* frames, and always at a fixed processing width
so that thresholds expressed as fractions of the frame stay comparable between
videos of different resolutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np

from . import logs


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def frame_of(self, seconds: float) -> int:
        return int(round(seconds * self.fps))

    def seconds_of(self, frame: int) -> float:
        return frame / self.fps if self.fps else 0.0

    def proc_height(self, proc_w: int) -> int:
        return max(1, int(round(proc_w * self.height / self.width)))

    def describe(self) -> None:
        logs.kv("resolution", f"{self.width} x {self.height}")
        logs.kv("frame rate", f"{self.fps:.2f} frames/second")
        logs.kv("length", f"{logs.human_duration(self.duration_s)} "
                          f"({self.frame_count:,} frames)")


def open_video(path: str):
    """Open a video, or stop with an explanation the user can act on."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logs.fail(f"cannot open the video file: {path}",
                  "check the path and that the file is a video your OpenCV build can read")
    info = VideoInfo(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280,
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720,
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 25.0,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0,
    )
    return cap, info


def downscale(frame: np.ndarray, max_w: int) -> np.ndarray:
    """Shrink a frame to the processing width (never enlarge)."""
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    scale = max_w / w
    return cv2.resize(frame, (max_w, max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def read_frame(cap, index: int, proc_w: Optional[int] = None):
    """Seek to a frame and return it, or None if it could not be read."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, index)))
    ok, raw = cap.read()
    if not ok:
        return None
    return downscale(raw, proc_w) if proc_w else raw


def sample_frames(cap, start: int, end: int, n: int,
                  proc_w: Optional[int] = None) -> Iterator[tuple]:
    """Yield (frame_index, frame) for n frames spread evenly over [start, end]."""
    if end < start:
        start, end = end, start
    for index in np.linspace(start, end, max(1, n), dtype=int):
        frame = read_frame(cap, int(index), proc_w)
        if frame is not None:
            yield int(index), frame


def read_span(cap, start: int, end: int, max_frames: int = 200,
              proc_w: Optional[int] = None) -> list:
    """
    Read a contiguous span for the review/labelling players, thinned so the
    loop stays smooth on long windows. Returns [(frame_index, frame), ...].
    """
    span = max(1, end - start + 1)
    stride = max(1, span // max(1, max_frames))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    out, index = [], start
    while index <= end:
        ok, frame = cap.read()
        if not ok:
            break
        if (index - start) % stride == 0:
            out.append((index, downscale(frame, proc_w) if proc_w else frame))
        index += 1
    return out
