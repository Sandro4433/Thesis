import numpy as np
import cv2
import pyrealsense2 as rs


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    # RGB only at 1920x1080
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # Print active resolution (to confirm it actually streams 1920x1080)
    color_vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    print("Active COLOR:", color_vsp.width(), "x", color_vsp.height())

    # Warm up
    for _ in range(30):
        pipeline.wait_for_frames()

    print("Capturing frame...")
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        pipeline.stop()
        raise RuntimeError("Could not get color frame.")

    color_image = np.asanyarray(color_frame.get_data())

    # Save
    cv2.imwrite("image.png", color_image)

    # Display (scaled preview so it fits on screen)
    cv2.namedWindow("Color", cv2.WINDOW_NORMAL)
    preview = cv2.resize(color_image, (1280, 720), interpolation=cv2.INTER_AREA)
    cv2.imshow("Color", preview)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pipeline.stop()

if __name__ == "__main__":
    main()
