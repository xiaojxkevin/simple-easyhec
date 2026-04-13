import os.path as osp
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer


def visualize_extrinsic_results(
    images,
    link_poses_dataset,
    meshes,
    intrinsic: np.ndarray,
    extrinsics: np.ndarray,
    camera_mount_poses: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    labels: List[str] = [],
    output_dir="results/",
    return_rgb: bool = False,
    frame_pose: Optional[np.ndarray] = None,
    frame_axis_length: Optional[float] = None,
    frame_origin_radius: int = 6,
):
    """
    Visualizes a given list of extrinsic matrices and draws the mask cameras at those extrinsics would project on the original RGB images.

    Args:
        images (np.ndarray, shape (N, H, W, 3)): List of RGB images to visualize.
        link_poses_dataset (np.ndarray, shape (N, L, 4, 4)): Link poses relative to any frame (e.g. the robot base frame), where N is the number of samples, L is the number of links
        meshes (List[str | trimesh.Trimesh]): List of mesh paths or trimesh.Trimesh objects for each of the links
        intrinsic (np.ndarray, shape (3, 3)): Camera intrinsic matrix
        extrinsics (np.ndarray, shape (M, 4, 4)): Extrinsic matrices to visualize
        camera_mount_poses (np.ndarray, shape (N, 4, 4)): Camera mount poses relative to the robot base frame, where N is the number of samples. If none then camera is assumed to be fixed.
        masks (np.ndarray, shape (N, H, W)): If given, will also display an image showing the masks used for optimization on top of the original images.
        labels (List[str]): List of labels for each of the extrinsics
        output_dir (str): Directory to save the visualizations
        frame_pose (np.ndarray, shape (4, 4)): Optional frame pose in the object/world coordinate system to draw on top of each image.
        frame_axis_length (float): Optional axis length in meters for the visualized frame. If None, no frame is drawn.
        frame_origin_radius (int): Pixel radius used when drawing the frame origin.
    """
    ### visualization code for the predicted extrinsic ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera_height, camera_width = images[0].shape[:2]
    renderer = NVDiffrastRenderer(camera_height, camera_width)
    extrinsics = torch.from_numpy(extrinsics).float().to(device)
    if camera_mount_poses is not None:
        camera_mount_poses = torch.from_numpy(camera_mount_poses).float().to(device)
    link_poses_dataset = torch.from_numpy(link_poses_dataset).float().to(device)

    for i in range(len(meshes)):
        if isinstance(meshes[i], str):
            meshes[i] = trimesh.load(osp.expanduser(meshes[i]), force="mesh")
    link_vertices = [mesh.vertices.copy() for mesh in meshes]
    link_faces = [mesh.faces.copy() for mesh in meshes]
    link_vertices = [
        torch.from_numpy(mesh.vertices).float().to(device) for mesh in meshes
    ]
    link_faces = [torch.from_numpy(mesh.faces).int().to(device) for mesh in meshes]

    def get_mask_from_camera_pose(camera_pose):
        mask = torch.zeros((camera_height, camera_width), device=device)
        for j, link_pose in enumerate(link_poses_dataset[i]):
            link_mask = renderer.render_mask(
                link_vertices[j],
                link_faces[j],
                intrinsic,
                camera_pose @ link_pose,
            )
            link_mask = link_mask.detach()
            mask[link_mask > 0] = 1
        return mask

    def project_points(camera_pose: np.ndarray, points_world: np.ndarray):
        points_cam = (camera_pose[:3, :3] @ points_world.T).T + camera_pose[:3, 3]
        valid = points_cam[:, 2] > 1e-6
        pixels = np.full((len(points_world), 2), np.nan, dtype=np.float32)
        if np.any(valid):
            projected = (intrinsic @ points_cam[valid].T).T
            pixels[valid] = projected[:, :2] / projected[:, 2:3]
        return pixels, valid

    def draw_coordinate_frame(image: np.ndarray, camera_pose: np.ndarray):
        if frame_pose is None or frame_axis_length is None or frame_axis_length <= 0:
            return image

        origin = frame_pose[:3, 3]
        rotation = frame_pose[:3, :3]
        axis_points = np.stack(
            [
                origin,
                origin + rotation[:, 0] * frame_axis_length,
                origin + rotation[:, 1] * frame_axis_length,
                origin + rotation[:, 2] * frame_axis_length,
            ],
            axis=0,
        )
        pixels, valid = project_points(camera_pose, axis_points)
        if not valid[0]:
            return image

        origin_px = tuple(np.round(pixels[0]).astype(int))
        axis_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        ]
        for axis_idx, color in enumerate(axis_colors, start=1):
            if valid[axis_idx]:
                endpoint_px = tuple(np.round(pixels[axis_idx]).astype(int))
                cv2.arrowedLine(
                    image,
                    origin_px,
                    endpoint_px,
                    color=color,
                    thickness=3,
                    tipLength=0.15,
                )
        cv2.circle(
            image,
            origin_px,
            radius=frame_origin_radius,
            color=(255, 255, 255),
            thickness=-1,
        )
        cv2.circle(
            image,
            origin_px,
            radius=frame_origin_radius + 2,
            color=(0, 0, 0),
            thickness=1,
        )
        return image

    for i in tqdm(range(len(images))):
        overlaid_images = []
        for j in range(len(extrinsics)):
            if camera_mount_poses is not None:
                camera_pose = extrinsics[j] @ camera_mount_poses[i]
            else:
                camera_pose = extrinsics[j]
            mask = get_mask_from_camera_pose(camera_pose)
            mask = mask.cpu().numpy()
            overlaid_images.append(images[i].copy())
            overlaid_images[-1][mask > 0] = overlaid_images[-1][mask > 0] // 4
            overlaid_images[-1] = draw_coordinate_frame(
                overlaid_images[-1], camera_pose.detach().cpu().numpy()
            )

        num_subplots = len(extrinsics) + 1 if masks is not None else len(extrinsics)

        plt.rcParams.update({'font.size': 16})  # Increase font size for all text elements
        fig = plt.figure(figsize=(7 * num_subplots, 8))
        for j in range(len(extrinsics)):
            ax = fig.add_subplot(1, num_subplots, j + 1)
            ax.imshow(overlaid_images[j])
            ax.axis("off")
            ax.set_title(labels[j])
        
        if masks is not None:
            ax = fig.add_subplot(1, num_subplots, num_subplots)
            reference_mask = images[i].copy()
            reference_mask[masks[i] > 0] = reference_mask[masks[i] > 0] // 4
            ax.imshow(reference_mask)
            ax.axis("off")
            ax.set_title("Masks")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        fig.savefig(f"{output_dir}/{i}.png")
        plt.close()
        if return_rgb:
            return cv2.cvtColor(cv2.imread(f"{output_dir}/{i}.png"), cv2.COLOR_BGR2RGB)
