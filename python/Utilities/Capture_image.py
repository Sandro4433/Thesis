import numpy as np
import cv2
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


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    # RGB only at 1920x1080
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # Print active resolution (to confirm it actually streams 1920x1080)
    color_vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    print("Active COLOR:", color_vsp.width(), "x", color_vsp.height())

    # Apply the same camera settings as vision_realsense.py
    try:
        dev = profile.get_device()
        color_sensor = dev.first_color_sensor()

        _try_set(color_sensor, rs.option.enable_auto_exposure,      0.0)    # manual exposure
        _try_set(color_sensor, rs.option.enable_auto_white_balance, 0.0)    # manual white balance
        _try_set(color_sensor, rs.option.exposure,                  50.0)   # µs
        _try_set(color_sensor, rs.option.gain,                      64.0)
        _try_set(color_sensor, rs.option.white_balance,             4500.0) # Kelvin

    except Exception as e:
        print(f"Warning: could not apply sensor settings: {e}")

    # Warm up (5 frames, matching vision_realsense.py)
    for _ in range(5):
        pipeline.wait_for_frames()

    print("Capturing frame...")
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        pipeline.stop()
        raise RuntimeError("Could not get color frame.")

    color_image = np.asanyarray(color_frame.get_data()).copy()

    # Save
    cv2.imwrite("image.png", color_image)
    print("Saved image.png")

    # Display (scaled preview so it fits on screen)
    cv2.namedWindow("Color", cv2.WINDOW_NORMAL)
    preview = cv2.resize(color_image, (1280, 720), interpolation=cv2.INTER_AREA)
    cv2.imshow("Color", preview)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":
    main()