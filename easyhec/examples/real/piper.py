import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import tyro
from transforms3d.euler import euler2mat
from urchin import URDF

from easyhec.examples.real.base import Args
from easyhec.optim.optimize import optimize
from easyhec.segmentation.interactive import InteractiveSegmentation
from easyhec.utils import visualization
from easyhec.utils.camera_conversions import opencv2ros, ros2opencv
from easyhec.utils.utils_3d import merge_meshes


@dataclass
class PiperArgs(Args):
    """Calibrate an external ZED camera against a Piper robot arm.

    This script is eye-to-hand only: the ZED is treated as a fixed external
    camera, and each sample records the current Piper joint feedback plus one
    ZED RGB frame while you manually place the robot in teaching mode.
    """

    output_dir: str = "results/piper"
    use_previous_captures: bool = False
    """Reuse previously captured images/link poses/masks if they exist."""

    can_name: str = "can0"
    """Activated CAN interface name used by piper_sdk."""

    camera_resolution: str = "HD720"
    """ZED resolution enum name, for example HD720, HD1080, HD2K, VGA."""

    camera_fps: int = 30
    settle_time_s: float = 1.0

    num_manual_samples: int = 8
    """Number of manually positioned samples to capture."""

    allow_mode_switch: bool = False
    """If True, call the SDK enable helper after connecting. This does not command motion."""

    paper_extrinsic_ros_path: Optional[str] = "results/paper_right/camera_extrinsic_ros.npy"
    """Paper calibration result used to initialize Camera<-Piper base in ROS convention."""

    paper_to_piper_base_x: float = 0.12
    paper_to_piper_base_y: float = -0.60
    paper_to_piper_base_z: float = 0.0
    """Paper-origin translation in the Piper base frame, in meters."""

    initial_x: float = -0.8
    initial_y: float = -0.5
    initial_z: float = 0.76
    initial_roll: float = 0.0
    initial_pitch: float = float(np.pi / 4)
    initial_yaw: float = 0.0
    """Initial Camera<-Base guess in ROS-style convention, using radians for RPY."""


ARM_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)

GRIPPER_JOINT_NAMES = ("joint7", "joint8")


def import_zed_api():
    try:
        import pyzed.sl as sl
    except ImportError as exc:
        raise ImportError(
            "ZED Python API is required for Piper calibration. Install it in the "
            "simplehec environment with something like "
            "`conda run -n simplehec python /usr/local/zed/get_python_api.py`."
        ) from exc
    return sl


def import_piper_api():
    try:
        from piper_sdk import C_PiperInterface_V2 as PiperInterface
    except ImportError:
        try:
            from piper_sdk import C_PiperInterface as PiperInterface
        except ImportError as exc:
            raise ImportError(
                "piper_sdk is required for Piper calibration. Install it with "
                "`conda run -n simplehec python -m pip install piper_sdk python-can`."
            ) from exc
    return PiperInterface


class ZedRgbCamera:
    def __init__(self, camera_resolution: str, camera_fps: int):
        self.sl = import_zed_api()
        self.camera = self.sl.Camera()
        self.image_mat = self.sl.Mat()

        resolution = getattr(self.sl.RESOLUTION, camera_resolution, None)
        if resolution is None:
            valid = [
                name
                for name in dir(self.sl.RESOLUTION)
                if name.isupper() and not name.startswith("_")
            ]
            raise ValueError(
                f"Unknown ZED resolution {camera_resolution!r}. "
                f"Known values include: {', '.join(valid)}"
            )

        init_params = self.sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.camera_fps = camera_fps
        if hasattr(self.sl, "DEPTH_MODE") and hasattr(self.sl.DEPTH_MODE, "NONE"):
            init_params.depth_mode = self.sl.DEPTH_MODE.NONE

        err = self.camera.open(init_params)
        if err != self.sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        self.runtime_params = self.sl.RuntimeParameters()
        self.intrinsic = self._read_left_intrinsic()

    def _read_left_intrinsic(self) -> np.ndarray:
        info = self.camera.get_camera_information()
        camera_config = getattr(info, "camera_configuration", info)
        calibration = getattr(camera_config, "calibration_parameters")
        left_cam = calibration.left_cam
        return np.array(
            [
                [left_cam.fx, 0.0, left_cam.cx],
                [0.0, left_cam.fy, left_cam.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def read_rgb(self, skip_frames: int = 1) -> np.ndarray:
        image = None
        for _ in range(skip_frames):
            err = self.camera.grab(self.runtime_params)
            if err != self.sl.ERROR_CODE.SUCCESS:
                continue
            self.camera.retrieve_image(self.image_mat, self.sl.VIEW.LEFT)
            image = np.asarray(self.image_mat.get_data())

        if image is None:
            raise RuntimeError("Failed to read RGB frame from ZED camera.")

        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raise RuntimeError(f"Unexpected ZED image shape: {image.shape}")

    def close(self):
        self.camera.close()


def create_piper(can_name: str, allow_mode_switch: bool):
    PiperInterface = import_piper_api()
    try:
        piper = PiperInterface(can_name=can_name)
    except TypeError:
        piper = PiperInterface(can_name)
    piper.ConnectPort()
    if allow_mode_switch and hasattr(piper, "EnableArm"):
        print("allow_mode_switch=True; enabling Piper arm feedback via EnableArm(7)")
        piper.EnableArm(7)
    return piper


def disconnect_piper(piper):
    for method_name in ("DisconnectPort", "disconnect", "Disconnect"):
        method = getattr(piper, method_name, None)
        if method is not None:
            method()
            return


def get_piper_joint_values_radians(piper) -> np.ndarray:
    arm_msgs = piper.GetArmJointMsgs()
    joint_state = getattr(arm_msgs, "joint_state", arm_msgs)
    raw_joint_values = [
        getattr(joint_state, "joint_1"),
        getattr(joint_state, "joint_2"),
        getattr(joint_state, "joint_3"),
        getattr(joint_state, "joint_4"),
        getattr(joint_state, "joint_5"),
        getattr(joint_state, "joint_6"),
    ]
    return np.deg2rad(np.asarray(raw_joint_values, dtype=np.float32) / 1000.0)


def build_robot_cfg(robot_urdf: URDF, arm_qpos: np.ndarray):
    cfg = {joint_name: 0.0 for joint_name in robot_urdf.joint_map.keys()}
    for joint_name, joint_value in zip(ARM_JOINT_NAMES, arm_qpos):
        cfg[joint_name] = float(joint_value)
    for joint_name in GRIPPER_JOINT_NAMES:
        if joint_name in cfg:
            cfg[joint_name] = 0.0
    return cfg


def load_meshes_from_urdf(robot_urdf: URDF):
    mesh_link_names = []
    meshes = []
    for link in robot_urdf.links:
        link_meshes = []
        for visual in link.visuals:
            mesh = getattr(visual.geometry, "mesh", None)
            if mesh is not None and mesh.meshes is not None:
                link_meshes += mesh.meshes
        merged_mesh = merge_meshes(link_meshes)
        if merged_mesh is None:
            continue
        mesh_link_names.append(link.name)
        meshes.append(merged_mesh)
    return mesh_link_names, meshes


def wait_for_manual_capture(sample_idx: int, total_samples: int):
    prompt = (
        f"\nPlace the Piper at manual pose {sample_idx}/{total_samples}, "
        "make sure the arm is stationary and visible, then press Enter to capture..."
    )
    input(prompt)


def capture_manual_samples(
    piper,
    camera: ZedRgbCamera,
    robot_urdf: URDF,
    mesh_link_names,
    meshes,
    args: PiperArgs,
):
    image_dataset = defaultdict(list)
    link_poses_dataset = np.zeros(
        (args.num_manual_samples, len(meshes), 4, 4), dtype=np.float32
    )

    print("Starting ZED camera and warming it up...")
    camera.read_rgb(skip_frames=60)

    for sample_idx in range(args.num_manual_samples):
        wait_for_manual_capture(sample_idx + 1, args.num_manual_samples)
        time.sleep(args.settle_time_s)

        arm_qpos = get_piper_joint_values_radians(piper)
        image = camera.read_rgb(skip_frames=3)
        image_dataset["base_camera"].append(image)

        cfg = build_robot_cfg(robot_urdf, arm_qpos)
        link_poses = robot_urdf.link_fk(cfg=cfg, use_names=True)
        for link_idx, link_name in enumerate(mesh_link_names):
            link_poses_dataset[sample_idx, link_idx] = link_poses[link_name]

        print(
            f"Captured sample {sample_idx + 1}/{args.num_manual_samples}; "
            f"qpos(rad)={np.array2string(arm_qpos, precision=4)}"
        )

    image_dataset["base_camera"] = np.stack(image_dataset["base_camera"])
    return link_poses_dataset, image_dataset


def align_loaded_link_poses_dataset(
    link_poses_dataset: np.ndarray,
    robot_urdf: URDF,
    mesh_link_names,
):
    expected_nlinks = len(mesh_link_names)
    if link_poses_dataset.shape[1] == expected_nlinks:
        return link_poses_dataset

    all_link_names = [link.name for link in robot_urdf.links]
    mesh_link_indices = [all_link_names.index(link_name) for link_name in mesh_link_names]

    if max(mesh_link_indices) >= link_poses_dataset.shape[1]:
        raise ValueError(
            "Cached link_poses_dataset does not match the current Piper URDF layout. "
            "Please recapture data without --use-previous-captures."
        )

    print(
        "Aligning cached link poses to the current mesh-linked subset "
        f"({link_poses_dataset.shape[1]} -> {expected_nlinks} links)"
    )
    return link_poses_dataset[:, mesh_link_indices]


def paper_to_piper_base_rotation() -> np.ndarray:
    # Piper base axes from the paper frame, estimated from the right-arm setup.
    # The paper center already places the robot roughly correctly, but the first
    # render showed the arm flipped in-plane. Compared with the original +90 deg
    # assumption, add another 180 deg around paper Z:
    #   x_base = y_paper, y_base = -x_paper, z_base = z_paper
    # This is a -90 degree rotation around the paper Z axis.
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def paper_to_piper_base_transform(args: PiperArgs) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = paper_to_piper_base_rotation()
    transform[:3, 3] = np.array(
        [
            args.paper_to_piper_base_x,
            args.paper_to_piper_base_y,
            args.paper_to_piper_base_z,
        ],
        dtype=np.float32,
    )
    return transform


def resolve_initial_extrinsic_guess(args: PiperArgs) -> np.ndarray:
    paper_extrinsic_path = (
        Path(args.paper_extrinsic_ros_path).expanduser()
        if args.paper_extrinsic_ros_path is not None
        else None
    )
    if paper_extrinsic_path is not None and paper_extrinsic_path.exists():
        paper_calibrated_extrinsic_ros = np.load(paper_extrinsic_path).astype(np.float32)
        paper_to_base_transform = paper_to_piper_base_transform(args)
        initial_extrinsic_guess_ros = (
            paper_to_base_transform @ paper_calibrated_extrinsic_ros
        ).astype(np.float32)
        print(
            "Initial extrinsic guess from paper_right calibration + "
            "paper/Piper base axis assumption"
        )
        print(f"Paper extrinsic path: {paper_extrinsic_path}")
        print(f"Camera<-Paper (ROS):\n{repr(paper_calibrated_extrinsic_ros)}")
        print(f"Paper->Piper base transform:\n{repr(paper_to_base_transform)}")
    else:
        initial_extrinsic_guess_ros = np.eye(4, dtype=np.float32)
        initial_extrinsic_guess_ros[:3, :3] = euler2mat(
            args.initial_roll,
            args.initial_pitch,
            args.initial_yaw,
        )
        initial_extrinsic_guess_ros[:3, 3] = np.array(
            [args.initial_x, args.initial_y, args.initial_z], dtype=np.float32
        )
        print("Initial extrinsic guess from manual PiperArgs values")
        if paper_extrinsic_path is not None:
            print(f"Paper extrinsic path not found: {paper_extrinsic_path}")

    initial_extrinsic_guess = ros2opencv(initial_extrinsic_guess_ros)
    print(f"Camera<-Piper base (ROS):\n{repr(initial_extrinsic_guess_ros)}")
    print(f"Camera<-Piper base (OpenCV):\n{repr(initial_extrinsic_guess)}")
    return initial_extrinsic_guess


def resolve_default_urdf_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "robot_definitions"
        / "piper"
        / "piper_no_gripper_description.urdf"
    )


def main(args: PiperArgs):
    if len(ARM_JOINT_NAMES) != 6:
        raise ValueError("ARM_JOINT_NAMES must contain exactly 6 Piper joints.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    urdf_path = (
        Path(args.urdf_path).expanduser().resolve()
        if args.urdf_path is not None
        else resolve_default_urdf_path().resolve()
    )
    robot_name = urdf_path.stem
    output_root = Path(args.output_dir) / robot_name / "base_camera"
    output_root.mkdir(parents=True, exist_ok=True)

    initial_extrinsic_guess = resolve_initial_extrinsic_guess(args)

    link_poses_path = output_root.parent / "link_poses_dataset.npy"
    image_dataset_path = output_root.parent / "image_dataset.npy"
    mask_path = output_root / "mask.npy"
    intrinsic_path = output_root / "camera_intrinsic.npy"
    piper = None
    camera = None

    robot_urdf = URDF.load(str(urdf_path))
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
            if intrinsic_path.exists():
                print(f"Using previous camera intrinsics from {intrinsic_path}")
                intrinsic = np.load(intrinsic_path)
            else:
                print(
                    "Previous captures found, but camera_intrinsic.npy is missing. "
                    "Connecting to the ZED to read intrinsics."
                )
                camera = ZedRgbCamera(args.camera_resolution, args.camera_fps)
                intrinsic = camera.intrinsic
        else:
            piper = create_piper(args.can_name, args.allow_mode_switch)
            camera = ZedRgbCamera(args.camera_resolution, args.camera_fps)
            intrinsic = camera.intrinsic
            link_poses_dataset, image_dataset = capture_manual_samples(
                piper=piper,
                camera=camera,
                robot_urdf=robot_urdf,
                mesh_link_names=mesh_link_names,
                meshes=meshes,
                args=args,
            )

            images = image_dataset["base_camera"]
            np.save(link_poses_path, link_poses_dataset)
            np.save(image_dataset_path, image_dataset)
            np.save(intrinsic_path, intrinsic)

        if args.use_previous_captures and mask_path.exists():
            print(f"Using previous mask from {mask_path}")
            masks = np.load(mask_path)
        else:
            interactive_segmentation = InteractiveSegmentation(
                segmentation_model="sam2",
                segmentation_model_cfg=dict(
                    checkpoint=args.checkpoint,
                    model_cfg=args.model_cfg,
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
        if camera is not None:
            camera.close()
        if piper is not None:
            disconnect_piper(piper)


if __name__ == "__main__":
    main(tyro.cli(PiperArgs))
