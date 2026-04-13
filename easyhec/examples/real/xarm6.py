import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import tyro
from urchin import URDF

from easyhec.examples.real.base import Args
from easyhec.optim.optimize import optimize
from easyhec.segmentation.interactive import InteractiveSegmentation
from easyhec.utils import visualization
from easyhec.utils.camera_conversions import opencv2ros, ros2opencv
from easyhec.utils.utils_3d import merge_meshes


@dataclass
class XArm6Args(Args):
    """Calibrate a fixed external camera against an xArm6 + official gripper.

    This version is aligned to the provided ROS-generated URDF and uses a
    manual capture flow by default: you place the robot at each pose yourself,
    then the script records the current joint state and one RGB frame.

    The main items you may still need to tune are:

    1. The initial extrinsic guess.
    2. The gripper command/value mapping.
    3. The number of captured poses and their diversity.

    This script targets off-hand / eye-to-hand cameras. Eye-in-hand support
    would require wiring up camera_mount_poses.
    """

    output_dir: str = "results/xarm6"
    use_previous_captures: bool = False
    """Reuse previously captured images/link poses/masks if they exist."""

    xarm_ip: str = "192.168.1.1"
    """IP address of the xArm controller."""

    urdf_path: Optional[str] = None
    """Path to the xArm6 (+ gripper) URDF used for forward kinematics."""

    realsense_camera_serial_id: str = "none"
    """RealSense serial number. Uses the first device if set to 'none'."""

    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    motion_speed_rad_s: float = 0.4
    motion_acc_rad_s2: float = 1.0
    settle_time_s: float = 1.0

    manual_capture: bool = True
    """If True, capture the current arm pose after you manually reposition it."""

    num_manual_samples: int = 8
    """Number of manually positioned samples to capture."""

    use_gripper: bool = True
    """If True, also commands the gripper and inserts its state into the URDF cfg."""

    gripper_open_position: int = 800
    """Example xArm gripper command value for open. Adjust for your hardware."""

    gripper_closed_position: int = 0
    """Example xArm gripper command value for closed. Adjust for your hardware."""

    gripper_joint_open_value: float = 0.0
    """URDF joint value corresponding to gripper_open_position."""

    gripper_joint_closed_value: float = 0.85
    """URDF joint value corresponding to gripper_closed_position."""


ARM_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)

GRIPPER_JOINT_NAMES = ("drive_joint",)

QPOS_SAMPLES = [
    {
        "arm": np.deg2rad(np.array([0, -25, -15, 35, 0, 35], dtype=np.float32)),
        "gripper": "open",
    },
    {
        "arm": np.deg2rad(np.array([25, -35, -20, 55, -10, 10], dtype=np.float32)),
        "gripper": "closed",
    },
    {
        "arm": np.deg2rad(np.array([-30, -20, -30, 70, 20, -25], dtype=np.float32)),
        "gripper": "open",
    },
    {
        "arm": np.deg2rad(np.array([15, -50, -5, 60, 35, 40], dtype=np.float32)),
        "gripper": "closed",
    },
    {
        "arm": np.deg2rad(np.array([-15, -30, 10, 45, -30, 0], dtype=np.float32)),
        "gripper": "open",
    },
]


def import_xarm_api():
    try:
        from xarm.wrapper import XArmAPI
    except ImportError as exc:
        raise ImportError(
            "xArm Python SDK is required for this template. "
            "Install it first, for example with `pip install xarm-python-sdk`."
        ) from exc
    return XArmAPI


def create_realsense_pipeline(
    realsense_camera_serial_id: str,
    camera_width: int,
    camera_height: int,
    camera_fps: int,
):
    config = rs.config()
    pipeline = rs.pipeline()
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense devices found.")

    if realsense_camera_serial_id == "none":
        realsense_camera_serial_id = devices[0].get_info(rs.camera_info.serial_number)
        print("No realsense camera serial id provided, using the first device found")

    print(f"RealSense device id: {realsense_camera_serial_id}")
    config.enable_device(realsense_camera_serial_id)
    config.enable_stream(
        rs.stream.color,
        camera_width,
        camera_height,
        rs.format.bgr8,
        camera_fps,
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
    return pipeline, intrinsic


def warmup_and_read_rgb(pipeline, skip_frames: int = 30):
    image = None
    for _ in range(skip_frames):
        frames = pipeline.wait_for_frames()
        cframe = frames.get_color_frame()
        if not cframe:
            continue
        image = np.asanyarray(cframe.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        raise RuntimeError("Failed to read RGB frame from RealSense camera.")
    return image


def create_xarm(xarm_ip: str):
    XArmAPI = import_xarm_api()
    arm = XArmAPI(xarm_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    arm.clean_error()
    arm.clean_warn()
    return arm


def resolve_ros_package_urdf(urdf_path: Path, tmp_dir: Path) -> Path:
    """Rewrite package://xarm_description/meshes/... into local paths.

    The provided URDF is ROS-generated and still points at package-relative mesh
    assets. In this repository those assets live next to the URDF, so we rewrite
    the mesh paths into absolute local filesystem paths before loading with
    urchin.
    """
    urdf_text = urdf_path.read_text()
    mesh_root = urdf_path.parent.resolve()
    replacements = {
        "package://xarm_description/meshes/xarm6/": f"{mesh_root.as_posix()}/",
        "package://xarm_description/meshes/gripper/": f"{mesh_root.as_posix()}/gripper/",
        "package://xarm_description/meshes/end_tool/": f"{mesh_root.as_posix()}/end_tool/",
    }
    for src, dst in replacements.items():
        urdf_text = urdf_text.replace(src, dst)
    patched_urdf_path = tmp_dir / urdf_path.name
    patched_urdf_path.write_text(urdf_text)
    return patched_urdf_path


def get_arm_joint_values_radians(arm) -> np.ndarray:
    code, angles = arm.get_servo_angle(is_radian=True)
    if code != 0:
        raise RuntimeError(f"Failed to read xArm joint angles, error code={code}")
    return np.asarray(angles[: len(ARM_JOINT_NAMES)], dtype=np.float32)


def move_arm_joints(arm, qpos: np.ndarray, speed: float, mvacc: float):
    code = arm.set_servo_angle(
        angle=qpos.tolist(),
        is_radian=True,
        speed=speed,
        mvacc=mvacc,
        wait=True,
    )
    if code != 0:
        raise RuntimeError(f"Failed to move xArm joints, error code={code}")


def command_gripper(arm, gripper_state: str, args: XArm6Args):
    if not args.use_gripper:
        return
    target = (
        args.gripper_open_position
        if gripper_state == "open"
        else args.gripper_closed_position
    )
    code = arm.set_gripper_position(target, wait=True)
    if code != 0:
        raise RuntimeError(
            f"Failed to move xArm gripper to {gripper_state}, error code={code}"
        )


def gripper_state_to_urdf_value(gripper_state: str, args: XArm6Args) -> float:
    if gripper_state == "open":
        return args.gripper_joint_open_value
    return args.gripper_joint_closed_value


def build_robot_cfg(robot_urdf: URDF, arm_qpos: np.ndarray, gripper_state: str, args: XArm6Args):
    cfg = {joint_name: 0.0 for joint_name in robot_urdf.joint_map.keys()}
    for joint_name, joint_value in zip(ARM_JOINT_NAMES, arm_qpos):
        cfg[joint_name] = float(joint_value)
    if args.use_gripper:
        gripper_value = gripper_state_to_urdf_value(gripper_state, args)
        for joint_name in GRIPPER_JOINT_NAMES:
            cfg[joint_name] = float(gripper_value)
    return cfg


def load_meshes_from_urdf(robot_urdf: URDF):
    mesh_link_names = []
    meshes = []
    for link in robot_urdf.links:
        link_meshes = []
        for visual in link.visuals:
            if hasattr(visual.geometry, "mesh"):
                link_meshes += visual.geometry.mesh.meshes
        merged_mesh = merge_meshes(link_meshes)
        if merged_mesh is None:
            continue
        mesh_link_names.append(link.name)
        meshes.append(merged_mesh)
    return mesh_link_names, meshes


def normalize_sam2_model_cfg(model_cfg: str) -> str:
    """Make EasyHec's older SAM2 config path compatible with current SAM2."""
    if model_cfg.startswith("sam2/"):
        return model_cfg.removeprefix("sam2/")
    return model_cfg


def resolve_checkpoint_path(checkpoint: str) -> str:
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.exists():
        return str(checkpoint_path.resolve())

    local_sam2_root = Path("/media/sealab/data/xiaojx/sam2")
    candidate = local_sam2_root / checkpoint
    if candidate.exists():
        return str(candidate.resolve())

    return checkpoint


def wait_for_manual_capture(sample_idx: int, total_samples: int):
    prompt = (
        f"\nPlace the robot at manual pose {sample_idx}/{total_samples}, "
        "make sure the arm is stationary and visible, then press Enter to capture..."
    )
    input(prompt)


def capture_manual_samples(
    arm,
    pipeline,
    robot_urdf: URDF,
    mesh_link_names,
    meshes,
    args: XArm6Args,
):
    image_dataset = defaultdict(list)
    link_poses_dataset = np.zeros(
        (args.num_manual_samples, len(meshes), 4, 4), dtype=np.float32
    )

    print("Starting camera and warming it up...")
    warmup_and_read_rgb(pipeline, skip_frames=60)

    for sample_idx in range(args.num_manual_samples):
        wait_for_manual_capture(sample_idx + 1, args.num_manual_samples)
        time.sleep(args.settle_time_s)

        arm_qpos = get_arm_joint_values_radians(arm)
        image = warmup_and_read_rgb(pipeline, skip_frames=3)
        image_dataset["base_camera"].append(image)

        gripper_state = "open"
        cfg = build_robot_cfg(robot_urdf, arm_qpos, gripper_state, args)
        link_poses = robot_urdf.link_fk(cfg=cfg, use_names=True)
        for link_idx, link_name in enumerate(mesh_link_names):
            link_poses_dataset[sample_idx, link_idx] = link_poses[link_name]

        print(f"Captured sample {sample_idx + 1}/{args.num_manual_samples}")

    image_dataset["base_camera"] = np.stack(image_dataset["base_camera"])
    return link_poses_dataset, image_dataset


def align_loaded_link_poses_dataset(
    link_poses_dataset: np.ndarray,
    robot_urdf: URDF,
    mesh_link_names,
):
    """Adapt previously saved link poses to the current mesh-link subset."""
    expected_nlinks = len(mesh_link_names)
    if link_poses_dataset.shape[1] == expected_nlinks:
        return link_poses_dataset

    all_link_names = [link.name for link in robot_urdf.links]
    mesh_link_indices = [all_link_names.index(link_name) for link_name in mesh_link_names]

    if max(mesh_link_indices) >= link_poses_dataset.shape[1]:
        raise ValueError(
            "Cached link_poses_dataset does not match the current URDF layout. "
            "Please recapture data without --use-previous-captures."
        )

    print(
        "Aligning cached link poses to the current mesh-linked subset "
        f"({link_poses_dataset.shape[1]} -> {expected_nlinks} links)"
    )
    return link_poses_dataset[:, mesh_link_indices]


def resolve_initial_extrinsic_guess(args: XArm6Args) -> np.ndarray:
    """Return the initial guess in OpenCV convention expected by optimize()."""
    # Assumption from the paper marker setup:
    #   x_base = -y_paper, y_base = x_paper, z_base = z_paper
    # Start from the paper-calibrated Camera<-Paper pose, then convert it into
    # Camera<-Base using the assumed Paper->Base axis mapping.
    paper_calibrated_extrinsic_ros = np.array(
        [
            [0.88711198, -0.30247952, 0.34862352, -0.89565728],
            [0.2727517, 0.95288682, 0.1327145, -0.60423803],
            [-0.37234208, -0.02264498, 0.92781921, 0.45554754],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    paper_to_base_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    initial_extrinsic_guess_ros = np.eye(4, dtype=np.float32)
    initial_extrinsic_guess_ros[:3, :3] = (
        paper_to_base_rotation @ paper_calibrated_extrinsic_ros[:3, :3]
    )
    initial_extrinsic_guess_ros[:3, 3] = paper_to_base_rotation @ paper_calibrated_extrinsic_ros[:3, 3]
    initial_extrinsic_guess = ros2opencv(initial_extrinsic_guess_ros)
    print("Initial extrinsic guess from paper calibration + paper/base axis assumption")
    print(f"Camera<-Paper (ROS):\n{repr(paper_calibrated_extrinsic_ros)}")
    print(f"Camera<-Base (ROS):\n{repr(initial_extrinsic_guess_ros)}")
    print(f"Camera<-Base (OpenCV):\n{repr(initial_extrinsic_guess)}")
    return initial_extrinsic_guess


def main(args: XArm6Args):
    if args.urdf_path is None:
        raise ValueError(
            "Please provide --urdf-path pointing to your xArm6 + gripper URDF."
        )

    if len(ARM_JOINT_NAMES) != 6:
        raise ValueError("ARM_JOINT_NAMES must contain exactly 6 xArm6 joints.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    urdf_path = Path(args.urdf_path).expanduser().resolve()
    robot_name = urdf_path.stem
    output_root = Path(args.output_dir) / robot_name / "base_camera"
    output_root.mkdir(parents=True, exist_ok=True)

    arm = create_xarm(args.xarm_ip)
    pipeline, intrinsic = create_realsense_pipeline(
        args.realsense_camera_serial_id,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
    )
    initial_extrinsic_guess = resolve_initial_extrinsic_guess(args)

    link_poses_path = output_root.parent / "link_poses_dataset.npy"
    image_dataset_path = output_root.parent / "image_dataset.npy"
    mask_path = output_root / "mask.npy"

    with TemporaryDirectory(prefix="easyhec_xarm6_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        patched_urdf_path = resolve_ros_package_urdf(urdf_path, tmp_dir)
        robot_urdf = URDF.load(str(patched_urdf_path))
        mesh_link_names, meshes = load_meshes_from_urdf(robot_urdf)

        try:
            if args.use_previous_captures and link_poses_path.exists() and image_dataset_path.exists():
                print(f"Using previous captures from {output_root.parent}")
                link_poses_dataset = np.load(link_poses_path)
                image_dataset = np.load(image_dataset_path, allow_pickle=True).reshape(-1)[0]
                link_poses_dataset = align_loaded_link_poses_dataset(
                    link_poses_dataset=link_poses_dataset,
                    robot_urdf=robot_urdf,
                    mesh_link_names=mesh_link_names,
                )
                images = image_dataset["base_camera"]
            else:
                if args.manual_capture:
                    link_poses_dataset, image_dataset = capture_manual_samples(
                        arm=arm,
                        pipeline=pipeline,
                        robot_urdf=robot_urdf,
                        mesh_link_names=mesh_link_names,
                        meshes=meshes,
                        args=args,
                    )
                else:
                    image_dataset = defaultdict(list)
                    link_poses_dataset = np.zeros((len(QPOS_SAMPLES), len(meshes), 4, 4), dtype=np.float32)

                    print("Starting camera and warming it up...")
                    warmup_and_read_rgb(pipeline, skip_frames=60)

                    for i, sample in enumerate(QPOS_SAMPLES):
                        print(f"Capturing sample {i + 1}/{len(QPOS_SAMPLES)}")
                        move_arm_joints(
                            arm,
                            sample["arm"],
                            speed=args.motion_speed_rad_s,
                            mvacc=args.motion_acc_rad_s2,
                        )
                        command_gripper(arm, sample["gripper"], args)
                        time.sleep(args.settle_time_s)

                        arm_qpos = get_arm_joint_values_radians(arm)
                        image = warmup_and_read_rgb(pipeline, skip_frames=3)
                        image_dataset["base_camera"].append(image)

                        cfg = build_robot_cfg(robot_urdf, arm_qpos, sample["gripper"], args)
                        link_poses = robot_urdf.link_fk(cfg=cfg, use_names=True)
                        for link_idx, link_name in enumerate(mesh_link_names):
                            link_poses_dataset[i, link_idx] = link_poses[link_name]

                    image_dataset["base_camera"] = np.stack(image_dataset["base_camera"])

                images = image_dataset["base_camera"]
                np.save(link_poses_path, link_poses_dataset)
                np.save(image_dataset_path, image_dataset)

            if args.use_previous_captures and mask_path.exists():
                print(f"Using previous mask from {mask_path}")
                masks = np.load(mask_path)
            else:
                model_cfg = normalize_sam2_model_cfg(args.model_cfg)
                checkpoint = resolve_checkpoint_path(args.checkpoint)
                interactive_segmentation = InteractiveSegmentation(
                    segmentation_model="sam2",
                    segmentation_model_cfg=dict(
                        checkpoint=checkpoint,
                        model_cfg=model_cfg,
                    ),
                )
                masks = interactive_segmentation.get_segmentation(images)
                np.save(mask_path, masks)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            predicted_camera_extrinsic_opencv = (
                optimize(
                    camera_intrinsic=torch.from_numpy(intrinsic).float().to(device),
                    masks=torch.from_numpy(masks).float().to(device),
                    link_poses_dataset=torch.from_numpy(link_poses_dataset).float().to(device),
                    initial_extrinsic_guess=torch.from_numpy(initial_extrinsic_guess).float().to(device),
                    meshes=meshes,
                    camera_width=images.shape[2],
                    camera_height=images.shape[1],
                    camera_mount_poses=None,
                    gt_camera_pose=None,
                    iterations=args.train_steps,
                    batch_size=args.batch_size,
                    early_stopping_steps=args.early_stopping_steps,
                )
                .cpu()
                .numpy()
            )
            predicted_camera_extrinsic_ros = opencv2ros(predicted_camera_extrinsic_opencv)

            print("Predicted camera extrinsic")
            print(f"OpenCV:\n{repr(predicted_camera_extrinsic_opencv)}")
            print(f"ROS/SAPIEN/ManiSkill/Mujoco/Isaac:\n{repr(predicted_camera_extrinsic_ros)}")

            np.save(output_root / "camera_extrinsic_opencv.npy", predicted_camera_extrinsic_opencv)
            np.save(output_root / "camera_extrinsic_ros.npy", predicted_camera_extrinsic_ros)
            np.save(output_root / "camera_intrinsic.npy", intrinsic)

            visualization.visualize_extrinsic_results(
                images=images,
                link_poses_dataset=link_poses_dataset,
                meshes=meshes,
                intrinsic=intrinsic,
                extrinsics=np.stack([initial_extrinsic_guess, predicted_camera_extrinsic_opencv]),
                masks=masks,
                labels=["Initial Extrinsic Guess", "Predicted Extrinsic"],
                output_dir=output_root,
            )
            print(f"Visualizations saved to {output_root}")
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass
            try:
                arm.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main(tyro.cli(XArm6Args))
