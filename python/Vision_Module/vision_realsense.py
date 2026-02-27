# vision_realsense.py
from __future__ import annotations

import numpy as np
import pyrealsense2 as rs


def _try_set(sensor: rs.sensor, option: rs.option, value: float) -> None:
    """
    Set a RealSense sensor option if supported; otherwise ignore.
    """
    try:
        if sensor.supports(option):
            sensor.set_option(option, value)
    except Exception:
        pass


def capture_color_frame(
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    warmup_frames: int = 5,
    # Camera color/brightness control (tune for your lighting)
    auto_exposure: bool = False,
    auto_white_balance: bool = False,
    exposure_us: float = 50.0,       # typical: 3000..15000
    gain: float = 64.0,                # model-dependent
    white_balance_k: float = 4500.0,   # typical: 2800..6500
) -> np.ndarray:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    pipeline.start(config)
    try:
        # Configure color sensor (exposure / white balance) early
        try:
            profile = pipeline.get_active_profile()
            dev = profile.get_device()
            color_sensor = dev.first_color_sensor()

            _try_set(color_sensor, rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
            _try_set(color_sensor, rs.option.enable_auto_white_balance, 1.0 if auto_white_balance else 0.0)

            # Only set manual values if autos are off
            if not auto_exposure:
                _try_set(color_sensor, rs.option.exposure, float(exposure_us))
                _try_set(color_sensor, rs.option.gain, float(gain))

            if not auto_white_balance:
                _try_set(color_sensor, rs.option.white_balance, float(white_balance_k))

        except Exception:
            # If anything about sensor access fails, still continue capturing
            pass

        # Warmup
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