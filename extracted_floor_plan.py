import numpy as np
import open3d as o3d
from register_and_floorplan import clean_floor_plan_outline, save_floor_plan_preview

floor_plan = o3d.io.read_point_cloud("extracted_floor_plan.ply")
pts = np.asarray(floor_plan.points)
if len(pts) == 0:
    raise ValueError("The extracted floor plan point cloud is empty.")

outline = clean_floor_plan_outline(pts[:, :2])
width = outline[:, 0].max() - outline[:, 0].min()
length = outline[:, 1].max() - outline[:, 1].min()
print(f"Estimated Room Dimensions: {width:.2f}m x {length:.2f}m")
save_floor_plan_preview(outline, "floor_plan_meters.png")
print("Saved CAD-style scan outline as 'floor_plan_meters.png'")