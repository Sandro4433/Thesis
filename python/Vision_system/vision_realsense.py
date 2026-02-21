# vision_realsense.py
from __future__ import annotations

import numpy as np
import pyrealsense2 as rs


def capture_color_frame(
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    warmup_frames: int = 5,
) -> np.ndarray:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    pipeline.start(config)
    try:
        for _ in range(max(0, warmup_frames)):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("No color frame from RealSense.")
        img = np.asanyarray(color_frame.get_data())
        return img
    finally:
        pipeline.stop()