import numpy as np
import cv2
import pyrealsense2 as rs


def _try_set(sensor: rs.sensor, option: rs.option, value: float) -> None:
    try:
        if sensor.supports(option):
            sensor.set_option(option, value)
    except Exception:
        pass


def get_intrinsics_and_distortion(profile):
    color_vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_vsp.get_intrinsics()

    camera_matrix = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    dist_coeffs = np.array(intr.coeffs, dtype=np.float32).reshape(-1, 1)

    print("Active COLOR:", color_vsp.width(), "x", color_vsp.height())
    print("Intrinsics:")
    print(" fx =", intr.fx, " fy =", intr.fy)
    print(" ppx =", intr.ppx, " ppy =", intr.ppy)
    print(" dist =", intr.coeffs)
    print(" model =", intr.model)

    return camera_matrix, dist_coeffs, color_vsp.width(), color_vsp.height()


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    try:
        dev = profile.get_device()
        color_sensor = dev.first_color_sensor()

        _try_set(color_sensor, rs.option.enable_auto_exposure, 0.0)
        _try_set(color_sensor, rs.option.enable_auto_white_balance, 0.0)
        _try_set(color_sensor, rs.option.exposure, 50.0)
        _try_set(color_sensor, rs.option.gain, 64.0)
        _try_set(color_sensor, rs.option.white_balance, 4500.0)
    except Exception as e:
        print(f"Warning: could not apply sensor settings: {e}")

    for _ in range(5):
        pipeline.wait_for_frames()

    camera_matrix, dist_coeffs, w, h = get_intrinsics_and_distortion(profile)

    print("Capturing frame...")
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        pipeline.stop()
        raise RuntimeError("Could not get color frame.")

    color_image = np.asanyarray(color_frame.get_data()).copy()

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    undistorted = cv2.undistort(color_image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y+rh, x:x+rw]

    cv2.imwrite("image_original.png", color_image)
    cv2.imwrite("image_undistorted.png", undistorted)
    print("Saved image_original.png")
    print("Saved image_undistorted.png")

    orig_preview = cv2.resize(color_image, (960, 540), interpolation=cv2.INTER_AREA)
    und_preview = cv2.resize(undistorted, (960, 540), interpolation=cv2.INTER_AREA)

    combined = np.hstack((orig_preview, und_preview))
    cv2.imshow("Original | Undistorted", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pipeline.stop()


if __name__ == "__main__":
    main()