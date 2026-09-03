"""Live RGB preview utilities for capture scripts.

These helpers keep a live camera preview window open while you manually move a
robot, so you can confirm the arm stays inside the camera field of view before
each capture. pyrealsense2 cannot share a device with realsense-viewer, so the
preview is implemented with a background thread inside the same process.
"""

import threading
import time
from typing import Optional

import cv2
import numpy as np


class RealSensePreview:
    """Background-thread RGB preview window for a RealSense camera.

    The preview runs in a daemon thread and continuously grabs color frames.
    When another component needs to capture a frame exclusively, call
    :meth:`pause` before grabbing and :meth:`resume` afterwards. This avoids
    interleaved grabs between threads.
    """

    def __init__(
        self,
        serial_id: str = "none",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        window_name: str = "RealSense preview (press q in window to quit)",
        warmup_s: float = 1.0,
        use_depth: bool = False,
        exposure_time_us: Optional[int] = None,
    ):
        import pyrealsense2 as rs

        self.rs = rs
        self.window_name = window_name
        self._lock = threading.Lock()
        self._paused = False
        self._stopped = False

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found.")

        if serial_id == "none":
            serial_id = devices[0].get_info(rs.camera_info.serial_number)
            print("No RealSense serial id provided, using the first device found")

        self.serial_id = serial_id
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_id)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if use_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)

        if exposure_time_us is not None:
            self._set_color_exposure(profile.get_device(), exposure_time_us)

        if warmup_s > 0:
            print(f"Warming up RealSense {serial_id} for {warmup_s:.1f}s...")
            time.sleep(warmup_s)

        self._use_depth = use_depth
        if use_depth:
            self._align = rs.align(rs.stream.color)
        self._latest_depth = None

        self._latest_frame = None
        self._pending_keys = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _set_color_exposure(self, device, exposure_time_us: int):
        """Disable auto exposure on the color sensor and pin a manual exposure."""
        rs = self.rs
        for sensor in device.query_sensors():
            if not hasattr(sensor, "is_color_sensor") or not sensor.is_color_sensor():
                continue
            if not sensor.supports(rs.option.exposure):
                break
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 0.0)
            sensor.set_option(rs.option.exposure, float(exposure_time_us))
            print(f"Color sensor exposure set to {exposure_time_us} us (auto exposure off)")
            return
        print(
            "Warning: no color sensor supporting manual exposure found; "
            "exposure_time_us ignored."
        )

    def _run(self):
        while not self._stopped:
            if self._paused:
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    self._stopped = True
                time.sleep(0.02)
                continue
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue
            if self._use_depth:
                frames = self._align.process(frames)
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    with self._lock:
                        self._latest_depth = np.asanyarray(depth_frame.get_data())
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            with self._lock:
                self._latest_frame = np.asanyarray(color_frame.get_data())

            cv2.imshow(self.window_name, self._latest_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                self._stopped = True
            elif key != 255:
                with self._lock:
                    self._pending_keys.append(int(key))

    @property
    def running(self) -> bool:
        return not self._stopped and self._thread.is_alive()

    def pop_pending_key(self) -> Optional[int]:
        """Return and remove the oldest key pressed in the preview window, if any."""
        with self._lock:
            if self._pending_keys:
                return self._pending_keys.pop(0)
        return None

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def read_rgb(self, skip_frames: int = 3, settle_s: float = 0.2) -> np.ndarray:
        """Grab one fresh RGB frame (RGB channel order) with the preview paused."""
        self.pause()
        try:
            image = None
            for _ in range(skip_frames):
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                image = cv2.cvtColor(
                    np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB
                )
            time.sleep(settle_s)
        finally:
            self.resume()
        if image is None:
            raise RuntimeError("Failed to read RGB frame from RealSense camera.")
        return image

    def read_intrinsic(self) -> np.ndarray:
        profile = self.pipeline.get_active_profile()
        color_stream = profile.get_stream(self.rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        return np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def read_depth_aligned(self) -> Optional[np.ndarray]:
        """Return the latest depth frame aligned to color (uint16 mm), if enabled.

        Requires use_depth=True at construction. Returns the most recent depth
        frame; when alignment to the color stream is available it is applied so
        depth pixels share the color intrinsics.
        """
        if not getattr(self, "_use_depth", False):
            raise RuntimeError(
                "Depth stream was not enabled. Construct RealSensePreview with use_depth=True."
            )
        with self._lock:
            return None if self._latest_depth is None else self._latest_depth.copy()

    def join_until_closed(self):
        """Block while the preview runs (e.g. between manual captures)."""
        while self.running:
            time.sleep(0.1)

    def close(self):
        self._stopped = True
        self._thread.join(timeout=2.0)
        cv2.destroyWindow(self.window_name)
        try:
            self.pipeline.stop()
        except Exception:
            pass
