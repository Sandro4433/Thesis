# vision_capture_worker.py
# ─────────────────────────────────────────────────────────────────────────────
# Standalone worker: captures ONE frame from RealSense and writes it to the
# path given as sys.argv[1].  Intentionally imports NOTHING from the rest of
# the Vision_Module so that libapriltag is never loaded into this process.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import sys
import numpy as np
import cv2
import pyrealsense2 as rs


def _try_set(sensor: rs.sensor, option: rs.option, value: float) -> None:
    try:
        if sensor.supports(option):
            sensor.set_option(option, value)
    except Exception:
        pass


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: vision_capture_worker.py <output_path> [width height fps warmup]", file=sys.stderr)
        sys.exit(1)

    out_path   = sys.argv[1]
    width      = int(sys.argv[2]) if len(sys.argv) > 2 else 1920
    height     = int(sys.argv[3]) if len(sys.argv) > 3 else 1080
    fps        = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    warmup     = int(sys.argv[5]) if len(sys.argv) > 5 else 5

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(cfg)

    try:
        profile      = pipeline.get_active_profile()
        color_sensor = profile.get_device().first_color_sensor()
        _try_set(color_sensor, rs.option.enable_auto_exposure,      0.0)
        _try_set(color_sensor, rs.option.enable_auto_white_balance, 0.0)
        _try_set(color_sensor, rs.option.exposure,                  50.0)
        _try_set(color_sensor, rs.option.gain,                      64.0)
        _try_set(color_sensor, rs.option.white_balance,             4500.0)
    except Exception as e:
        print(f"Warning: sensor settings: {e}", file=sys.stderr)

    for _ in range(max(0, warmup)):
        pipeline.wait_for_frames()

    frames      = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        pipeline.stop()
        print("ERROR: no color frame", file=sys.stderr)
        sys.exit(2)

    img = np.asanyarray(color_frame.get_data()).copy()
    pipeline.stop()

    ok = cv2.imwrite(out_path, img)
    if not ok:
        print(f"ERROR: cv2.imwrite failed → {out_path}", file=sys.stderr)
        sys.exit(3)

    print(f"OK {out_path}")


if __name__ == "__main__":
    main()