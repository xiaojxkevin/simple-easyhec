from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import tyro

from easyhec.examples.real.base import Args
from easyhec.optim.optimize import optimize
from easyhec.segmentation.interactive import InteractiveSegmentation
from easyhec.utils import visualization
from easyhec.utils.camera_conversions import opencv2ros


@dataclass
class RealPaperArgs(Args):
    """Calibrate a RealSense or ZED camera with a piece of standard sized paper."""
    output_dir: str = "results/paper"
    model_cfg: str = "/home/piper/data/xiaojx/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint: str = "/home/piper/data/xiaojx/sam2/checkpoints/sam2.1_hiera_large.pt"
    paper_type: str = "letter"
    """The type of paper to use to calibrate against. Options are 'letter' or 'a4'"""
    camera_type: str = "realsense"
    """Camera backend to use. Options are 'realsense' or 'zed'."""
    realsense_camera_serial_id: str = "none"
    """The serial id of the realsense camera to use for calibration"""
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    zed_camera_resolution: str = "HD720"
    """ZED resolution enum name, for example HD720, HD1080, HD2K, VGA."""
    # TODO (stao): A1, A2, A3, follow a nice structure, we can just generate the meshes for those.


paper_sizes = {
    "letter": {
        "width": 0.2159,  # 8.5 inches in mm
        "height": 0.2794,  # 11 inches in mm
    },
    "a4": {
        "width": 0.210,  # 8.27 inches in mm
        "height": 0.297,  # 11.69 inches in mm
    },
}


def import_realsense_api():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ImportError(
            "pyrealsense2 is required when --camera-type realsense. "
            "Install it with `conda run -n simplehec python -m pip install pyrealsense2`."
        ) from exc
    return rs


def import_zed_api():
    try:
        import pyzed.sl as sl
    except ImportError as exc:
        raise ImportError(
            "ZED Python API is required when --camera-type zed. Install it in the "
            "simplehec environment with "
            "`conda run -n simplehec python /usr/local/zed/get_python_api.py`."
        ) from exc
    return sl


def read_realsense_image_and_intrinsic(args: RealPaperArgs):
    rs = import_realsense_api()
    config = rs.config()
    pipeline = rs.pipeline()
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found.")

        if args.realsense_camera_serial_id == "none":
            print("No realsense camera serial id provided, using the first device found")
            realsense_camera_serial_id = devices[0].get_info(rs.camera_info.serial_number)
        else:
            realsense_camera_serial_id = args.realsense_camera_serial_id
        print(f"RealSense device id: {realsense_camera_serial_id}")
        config.enable_device(realsense_camera_serial_id)
        config.enable_stream(
            rs.stream.color,
            args.camera_width,
            args.camera_height,
            rs.format.bgr8,
            args.camera_fps,
        )
        profile = pipeline.start(config)
        color_stream = profile.get_stream(rs.stream.color)
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        intrinsic = np.array(
            [
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        image = None
        print("Starting RealSense camera and warming it up...")
        for _ in range(60):
            frames = pipeline.wait_for_frames()
            cframe = frames.get_color_frame()
            if not cframe:
                continue
            image = np.asanyarray(cframe.get_data())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            raise RuntimeError("Failed to read RGB frame from RealSense camera.")
        return image, intrinsic
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def read_zed_image_and_intrinsic(args: RealPaperArgs):
    sl = import_zed_api()
    camera = sl.Camera()
    image_mat = sl.Mat()
    resolution = getattr(sl.RESOLUTION, args.zed_camera_resolution, None)
    if resolution is None:
        valid = [
            name
            for name in dir(sl.RESOLUTION)
            if name.isupper() and not name.startswith("_")
        ]
        raise ValueError(
            f"Unknown ZED resolution {args.zed_camera_resolution!r}. "
            f"Known values include: {', '.join(valid)}"
        )

    init_params = sl.InitParameters()
    init_params.camera_resolution = resolution
    init_params.camera_fps = args.camera_fps
    if hasattr(sl, "DEPTH_MODE") and hasattr(sl.DEPTH_MODE, "NONE"):
        init_params.depth_mode = sl.DEPTH_MODE.NONE

    try:
        err = camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        info = camera.get_camera_information()
        camera_config = getattr(info, "camera_configuration", info)
        calibration = getattr(camera_config, "calibration_parameters")
        left_cam = calibration.left_cam
        intrinsic = np.array(
            [
                [left_cam.fx, 0.0, left_cam.cx],
                [0.0, left_cam.fy, left_cam.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        runtime_params = sl.RuntimeParameters()
        image = None
        print("Starting ZED camera and warming it up...")
        for _ in range(60):
            err = camera.grab(runtime_params)
            if err != sl.ERROR_CODE.SUCCESS:
                continue
            camera.retrieve_image(image_mat, sl.VIEW.LEFT)
            image = np.asarray(image_mat.get_data())

        if image is None:
            raise RuntimeError("Failed to read RGB frame from ZED camera.")
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(f"Unexpected ZED image shape: {image.shape}")
        return image, intrinsic
    finally:
        camera.close()


def read_camera_image_and_intrinsic(args: RealPaperArgs):
    if args.camera_type == "realsense":
        return read_realsense_image_and_intrinsic(args)
    if args.camera_type == "zed":
        return read_zed_image_and_intrinsic(args)
    raise ValueError("--camera-type must be either 'realsense' or 'zed'")


def main(args: RealPaperArgs):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    image, intrinsic = read_camera_image_and_intrinsic(args)
    camera_height, camera_width = image.shape[:2]

    print(f"Camera Intrinsics:\n {repr(intrinsic)}")
    images = [image]

    ### Make an initial guess for the extrinsic ###
    initial_extrinsic_guess = np.eye(4)

    # OpenCV optical convention:
    #   +X camera-right, +Y image-down, +Z forward along the optical axis.
    # The paper frame is centered on the paper, with +Z pointing up from the
    # tabletop. For the current ego-view setup, the paper center is about 20cm
    # to the right of the ZED optical center and about 70cm away along the view
    # direction. This initial guess is T^{cam}_{paper} in OpenCV convention.
    initial_extrinsic_guess[:3, :3] = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    initial_extrinsic_guess[:3, 3] = np.array([-0.20, 0.2, 0.70], dtype=np.float32)

    print("Initial extrinsic guess", initial_extrinsic_guess)

    # Create a box mesh representing the letter paper (in meters)
    paper_width = paper_sizes[args.paper_type]["width"]
    paper_height = paper_sizes[args.paper_type]["height"]
    paper_box = trimesh.creation.box(extents=(paper_width, paper_height, 1e-3))
    meshes = [paper_box]
    # We assume the world frame is centered at the paper and oriented to be perpendicular to the paper
    link_poses_dataset = np.stack(np.eye(4)).reshape(1, 1, 4, 4)

    camera_mount_poses = None

    interactive_segmentation = InteractiveSegmentation(
        segmentation_model="sam2",
        segmentation_model_cfg=dict(
            checkpoint=args.checkpoint, model_cfg=args.model_cfg
        ),
    )
    masks = interactive_segmentation.get_segmentation(images)

    ### run the optimization given the data ###
    predicted_camera_extrinsic_opencv = (
        optimize(
            camera_intrinsic=torch.from_numpy(intrinsic).float().to(device),
            masks=torch.from_numpy(masks).float().to(device),
            link_poses_dataset=torch.from_numpy(link_poses_dataset).float().to(device),
            initial_extrinsic_guess=torch.tensor(initial_extrinsic_guess)
            .float()
            .to(device),
            meshes=meshes,
            camera_width=camera_width,
            camera_height=camera_height,
            camera_mount_poses=(
                torch.from_numpy(camera_mount_poses).float().to(device)
                if camera_mount_poses is not None
                else None
            ),
            gt_camera_pose=None,
            iterations=args.train_steps,
            early_stopping_steps=args.early_stopping_steps,
        )
        .cpu()
        .numpy()
    )
    predicted_camera_extrinsic_ros = opencv2ros(predicted_camera_extrinsic_opencv)

    ### Print predicted results ###

    print(f"Predicted camera extrinsic")
    print(f"OpenCV:\n{repr(predicted_camera_extrinsic_opencv)}")
    print(f"ROS/SAPIEN/ManiSkill/Mujoco/Isaac:\n{repr(predicted_camera_extrinsic_ros)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    np.save(
        Path(args.output_dir) / "camera_extrinsic_opencv.npy",
        predicted_camera_extrinsic_opencv,
    )
    np.save(
        Path(args.output_dir) / "camera_extrinsic_ros.npy",
        predicted_camera_extrinsic_ros,
    )
    np.save(Path(args.output_dir) / "camera_intrinsic.npy", intrinsic)

    visualization.visualize_extrinsic_results(
        images=images,
        link_poses_dataset=link_poses_dataset,
        meshes=meshes,
        intrinsic=intrinsic,
        extrinsics=np.stack(
            [initial_extrinsic_guess, predicted_camera_extrinsic_opencv]
        ),
        masks=masks,
        labels=["Initial Extrinsic Guess", "Predicted Extrinsic"],
        output_dir=args.output_dir,
        frame_pose=np.eye(4, dtype=np.float32),
        frame_axis_length=min(paper_width, paper_height) * 0.3,
        frame_origin_radius=6,
    )
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(RealPaperArgs))
