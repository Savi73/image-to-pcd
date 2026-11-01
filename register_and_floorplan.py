import open3d as o3d
import numpy as np
import argparse
import glob
import os
from pathlib import Path
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_and_preprocess_pcd(file_path, voxel_size=0.02):
    """Loads a point cloud, downsamples it, and estimates normals."""
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        raise ValueError(f"Point cloud {file_path} is empty or could not be read.")
    # Downsample for faster processing and uniform point density
    pcd_down = pcd.voxel_down_sample(voxel_size)
    # Estimate normals (required for point-to-plane ICP)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    return pcd, pcd_down

def execute_global_registration(source_down, target_down, voxel_size):
    """Rough alignment using FPFH features and RANSAC."""
    distance_threshold = voxel_size * 1.5

    # Compute FPFH features for feature matching
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration_icp(source, target, source_down, target_down, voxel_size, init_transform):
    """Precise alignment using Point-to-Plane ICP."""
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    return result

def generate_2d_floor_plan(combined_pcd, z_min_offset=0.2, z_max_offset=1.5, slice_thickness=0.1):
    """
    Slices a horizontal cross-section through the room point cloud
    at human-chest height to project a 2D floor plan.
    """
    points = np.asarray(combined_pcd.points)
    if points.size == 0:
        raise ValueError("The merged point cloud is empty; cannot create a floor plan.")

    # Find height boundaries of the room
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # Define slice window (e.g., a horizontal slice around 1 meter off the floor)
    slice_z_low = min_bound[2] + z_min_offset
    slice_z_high = min_bound[2] + z_max_offset

    # Filter points falling inside the height slice (walls and furniture outlines)
    mask = (points[:, 2] >= slice_z_low) & (points[:, 2] <= slice_z_high)
    sliced_points = points[mask]
    if not len(sliced_points):
        raise ValueError("No points fall within the selected floor-plan height slice.")

    floor_plan_points = sliced_points[:, :2]
    line_points = clean_floor_plan_outline(floor_plan_points)

    plan_pcd = o3d.geometry.PointCloud()
    plan_pcd.points = o3d.utility.Vector3dVector(
        np.column_stack((line_points, np.zeros(len(line_points))))
    )
    return plan_pcd


def clean_floor_plan_outline(floor_plan_points, resolution=0.01):
    """Convert a noisy horizontal point slice into a simplified perimeter line."""
    floor_plan_points = floor_plan_points[np.isfinite(floor_plan_points).all(axis=1)]
    if len(floor_plan_points) < 3:
        raise ValueError("The floor-plan slice has too few valid points to form an outline.")

    min_x, min_y = floor_plan_points.min(axis=0)
    max_x, max_y = floor_plan_points.max(axis=0)
    grid_width = int(np.ceil((max_x - min_x) / resolution)) + 1
    grid_height = int(np.ceil((max_y - min_y) / resolution)) + 1
    density = np.zeros((grid_height, grid_width), dtype=np.uint16)
    columns = np.floor((floor_plan_points[:, 0] - min_x) / resolution).astype(int)
    rows = np.floor((max_y - floor_plan_points[:, 1]) / resolution).astype(int)
    np.add.at(density, (rows, columns), 1)

    occupied = (density >= 2).astype(np.uint8) * 255
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    occupied = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, close_kernel)
    occupied = cv2.medianBlur(occupied, 5)
    contours, _ = cv2.findContours(occupied, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 100]
    if not contours:
        raise ValueError("Could not extract a continuous floor-plan outline from the scan.")

    outline = max(contours, key=cv2.contourArea)
    outline = cv2.approxPolyDP(outline, epsilon=resolution * 0.15, closed=True).reshape(-1, 2)
    outline_xy = np.column_stack((
        min_x + outline[:, 0] * resolution,
        max_y - outline[:, 1] * resolution,
    ))

    # Densify the simplified outline so the exported point cloud draws as a line.
    line_points = []
    for start, end in zip(outline_xy, np.roll(outline_xy, -1, axis=0)):
        segment_length = np.linalg.norm(end - start)
        segment_steps = max(1, int(np.ceil(segment_length / 0.02)))
        line_points.append(np.linspace(start, end, segment_steps, endpoint=False))
    return np.concatenate(line_points)


def save_floor_plan_preview(floor_plan_points, preview_path):
    """Render a cleaned room outline with CAD-style metric drafting details."""
    min_x, min_y = floor_plan_points.min(axis=0)
    max_x, max_y = floor_plan_points.max(axis=0)
    width = max_x - min_x
    length = max_y - min_y
    if width <= 0 or length <= 0:
        raise ValueError("The extracted floor-plan outline has no measurable area.")

    plan_xy = floor_plan_points - np.array([min_x, min_y])
    closed_plan = np.vstack((plan_xy, plan_xy[0]))
    margin = max(width, length) * 0.16
    dimension_offset = margin * 0.43
    tick_spacing = 0.5

    figure, axis = plt.subplots(figsize=(9, 8), facecolor="#f4f3ee")
    axis.set_facecolor("#f4f3ee")
    axis.plot(
        closed_plan[:, 0],
        closed_plan[:, 1],
        color="#202b24",
        linewidth=2.2,
        solid_joinstyle="miter",
        solid_capstyle="butt",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-margin, width + margin)
    axis.set_ylim(-margin, length + margin)
    axis.set_xticks(np.arange(0, width + tick_spacing, tick_spacing))
    axis.set_yticks(np.arange(0, length + tick_spacing, tick_spacing))
    axis.grid(color="#b9c0b9", linewidth=0.45, alpha=0.55)
    axis.tick_params(colors="#536057", labelsize=8, length=3)
    axis.set_xlabel("X  /  METRES", loc="right", fontsize=8, color="#536057", labelpad=8)
    axis.set_ylabel("Y  /  METRES", loc="top", fontsize=8, color="#536057", labelpad=8)
    axis.set_title(
        "ROOM PLAN   /   CLEANED SCAN OUTLINE",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color="#202b24",
        pad=18,
    )

    dimension_color = "#536057"
    axis.annotate(
        "",
        xy=(width, -dimension_offset),
        xytext=(0, -dimension_offset),
        arrowprops={"arrowstyle": "<->", "color": dimension_color, "linewidth": 0.8},
        annotation_clip=False,
    )
    axis.plot(
        [0, 0], [-dimension_offset * 0.35, -dimension_offset * 1.12],
        color=dimension_color, linewidth=0.65, clip_on=False,
    )
    axis.plot(
        [width, width], [-dimension_offset * 0.35, -dimension_offset * 1.12],
        color=dimension_color, linewidth=0.65, clip_on=False,
    )
    axis.text(
        width / 2,
        -dimension_offset * 1.32,
        f"{width:.2f} m",
        ha="center",
        va="top",
        fontsize=8,
        color=dimension_color,
    )
    axis.annotate(
        "",
        xy=(-dimension_offset, length),
        xytext=(-dimension_offset, 0),
        arrowprops={"arrowstyle": "<->", "color": dimension_color, "linewidth": 0.8},
        annotation_clip=False,
    )
    axis.plot(
        [-dimension_offset * 0.35, -dimension_offset * 1.12], [0, 0],
        color=dimension_color, linewidth=0.65, clip_on=False,
    )
    axis.plot(
        [-dimension_offset * 0.35, -dimension_offset * 1.12], [length, length],
        color=dimension_color, linewidth=0.65, clip_on=False,
    )
    axis.text(
        -dimension_offset * 1.38,
        length / 2,
        f"{length:.2f} m",
        ha="center",
        va="center",
        rotation=90,
        fontsize=8,
        color=dimension_color,
    )
    axis.text(
        1,
        -0.15,
        "SCAN OUTLINE ONLY  /  VERIFY DIMENSIONS BEFORE CONSTRUCTION",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#68746b",
        clip_on=False,
    )
    for spine in axis.spines.values():
        spine.set_color("#78847b")
        spine.set_linewidth(0.65)

    figure.tight_layout(pad=2)
    figure.savefig(preview_path, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

def main(input_dir, output_dir, visualize=True):
    voxel_size = 0.02  # 2cm resolution
    os.makedirs(output_dir, exist_ok=True)
    point_cloud_paths = sorted(glob.glob(os.path.join(input_dir, "*.ply")))
    if len(point_cloud_paths) < 2:
        raise ValueError(f"At least two point clouds are required in {input_dir}.")

    print(f"Loading {len(point_cloud_paths)} point clouds...")
    clouds = [load_and_preprocess_pcd(path, voxel_size) for path in point_cloud_paths]
    target, target_down = clouds[0]
    complete_room = target
    for source, source_down in clouds[1:]:
        print("Performing global rough registration...")
        global_result = execute_global_registration(source_down, target_down, voxel_size)
        print("Refining alignment with Point-to-Plane ICP...")
        icp_result = refine_registration_icp(
            source, target, source_down, target_down, voxel_size, global_result.transformation
        )
        print(f"ICP Fitness: {icp_result.fitness:.4f}, Inlier RMSE: {icp_result.inlier_rmse:.4f}")
        source.transform(icp_result.transformation)
        complete_room += source

    merged_path = os.path.join(output_dir, "complete_room_merged.ply")
    if not o3d.io.write_point_cloud(merged_path, complete_room):
        raise OSError(f"Could not write merged point cloud: {merged_path}")
    print(f"Saved merged point cloud as '{merged_path}'")

    print("Slicing point cloud to extract 2D floor plan...")
    floor_plan = generate_2d_floor_plan(complete_room)
    floor_plan_path = os.path.join(output_dir, "extracted_floor_plan.ply")
    if not o3d.io.write_point_cloud(floor_plan_path, floor_plan):
        raise OSError(f"Could not write floor plan: {floor_plan_path}")
    print(f"Saved floor plan as '{floor_plan_path}'")

    preview_path = os.path.join(output_dir, "floor_plan_preview.png")
    save_floor_plan_preview(np.asarray(floor_plan.points)[:, :2], preview_path)
    print(f"Saved floor plan preview as '{preview_path}'")

    if visualize:
        o3d.visualization.draw_geometries([complete_room], window_name="Merged Room Point Cloud")


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Register point clouds and extract a floor plan.")
    parser.add_argument("--input-dir", default=str(project_dir / "my_test" / "output" / "indoor"))
    parser.add_argument("--output-dir", default=str(project_dir))
    parser.add_argument("--no-visualization", action="store_true")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, visualize=not args.no_visualization)