"""Interactive Open3D visualization of CARLA semantic LiDAR sequences.

Colors points by CARLA semantic class (moving objects in red, static
background in gray) to sanity-check collected simulation data.
"""
import os
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.io import read_bin


# Moving-object semantic classes (CARLA 0.9.14+ standard).
MOVING_OBJECTS = {
    12: 'Pedestrian',
    13: 'Rider',
    14: 'Car',
    15: 'Truck',
    16: 'Bus',
    17: 'Train',
    18: 'Motorcycle',
    19: 'Bicycle',
}

# Full semantic label set, used by visualize_motion_analysis for the breakdown printout.
SEMANTIC_LABELS = {
    0: 'Unlabeled', 1: 'Roads', 2: 'SideWalks', 3: 'Building',
    4: 'Wall', 5: 'Fence', 6: 'Pole', 7: 'TrafficLight',
    8: 'TrafficSign', 9: 'Vegetation', 10: 'Terrain', 11: 'Sky',
    12: 'Pedestrian', 13: 'Rider', 14: 'Car', 15: 'Truck',
    16: 'Bus', 17: 'Train', 18: 'Motorcycle', 19: 'Bicycle',
    20: 'Static', 21: 'Dynamic', 22: 'Other', 23: 'Water',
    24: 'RoadLine', 25: 'Ground', 26: 'Bridge', 27: 'RailTrack',
    28: 'GuardRail'
}


def create_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def apply_motion_colors(pcd, semantic_ids):
    """
    Color a point cloud by moving-object vs. static-background semantic class.

    Args:
        pcd: Open3D point cloud.
        semantic_ids: per-point semantic id array.

    Returns:
        dict with moving/static point counts and a per-class breakdown.
    """
    num_points = len(semantic_ids)
    colors = np.zeros((num_points, 3))

    moving_mask = np.zeros(num_points, dtype=bool)
    for semantic_id in MOVING_OBJECTS.keys():
        moving_mask |= (semantic_ids == semantic_id)

    colors[moving_mask] = [1.0, 0.0, 0.0]  # Red: moving objects.
    static_mask = ~moving_mask
    colors[static_mask] = [0.4, 0.4, 0.4]  # Gray: static background.
    pcd.colors = o3d.utility.Vector3dVector(colors)

    stats = {}
    for semantic_id, label in MOVING_OBJECTS.items():
        stats[label.lower()] = int(np.sum(semantic_ids == semantic_id))

    total_moving = int(np.sum(moving_mask))
    total_static = int(np.sum(static_mask))
    moving_ratio = total_moving / num_points * 100 if num_points > 0 else 0

    return {
        'total_moving': total_moving,
        'total_static': total_static,
        'moving_ratio': moving_ratio,
        'breakdown': stats
    }


def visualize_pcd_sequence_realtime_motion(bin_folder, fps=10, point_size=1.0):
    """Play a folder of .bin frames as a live-updating point cloud, moving objects highlighted red."""
    bin_files = sorted([f for f in os.listdir(bin_folder) if f.endswith('.bin')])
    if not bin_files:
        print("No .bin files found!")
        return

    print(f"Found {len(bin_files)} .bin files")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Semantic Point Cloud - Motion Highlighting", width=1200, height=800)
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = point_size

    first_bin = os.path.join(bin_folder, bin_files[0])
    pcd_array = read_bin(first_bin, 8)  # Columns: x, y, z, range, intensity, cosine, semantic_id, object_id.
    pcd_array[:, 1] = -1 * pcd_array[:, 1]  # Flip Y axis.

    pcd = create_pcd(pcd_array[:, :3])
    semantic_ids = pcd_array[:, 6].astype(int)
    motion_stats = apply_motion_colors(pcd, semantic_ids)

    vis.add_geometry(pcd)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])  # Look down.
    view_control.set_up([1, 0, 0])

    frame_interval = 1.0 / fps
    print(f"Playing {len(bin_files)} frames at {fps} FPS")
    print("Moving objects: red. Static background: gray.")
    print("Moving classes: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle.")
    print("Close the window to stop.")
    print(f"First frame moving-object points: {motion_stats['total_moving']} ({motion_stats['moving_ratio']:.1f}%)")
    for obj_type, count in motion_stats['breakdown'].items():
        if count > 0:
            print(f"  - {obj_type.capitalize()}: {count} points")

    try:
        frame_idx = 0
        while True:
            start_time = time.time()

            bin_file = os.path.join(bin_folder, bin_files[frame_idx])
            pcd_array = read_bin(bin_file, 8)
            pcd_array[:, 1] = -1 * pcd_array[:, 1]

            new_points = pcd_array[:, :3]
            semantic_ids = pcd_array[:, 6].astype(int)
            pcd.points = o3d.utility.Vector3dVector(new_points)
            motion_stats = apply_motion_colors(pcd, semantic_ids)

            if frame_idx % 20 == 0:
                breakdown = motion_stats['breakdown']
                active_objects = [f"{k}:{v}" for k, v in breakdown.items() if v > 0]
                objects_str = " ".join(active_objects) if active_objects else "none"
                print(f"Frame {frame_idx + 1}/{len(bin_files)}: moving points "
                      f"{motion_stats['total_moving']} ({motion_stats['moving_ratio']:.1f}%) [{objects_str}]")

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            if not vis.poll_events():
                break

            elapsed = time.time() - start_time
            time.sleep(max(0, frame_interval - elapsed))

            frame_idx = (frame_idx + 1) % len(bin_files)
            if frame_idx == 0:
                print("Playback loop complete, restarting...")

    except KeyboardInterrupt:
        print("Playback interrupted by user")
    finally:
        vis.destroy_window()


def visualize_motion_analysis(bin_folder, frame_id=0):
    """Print a detailed per-class breakdown for one frame and show it as a static point cloud."""
    bin_files = sorted([f for f in os.listdir(bin_folder) if f.endswith('.bin')])
    if frame_id >= len(bin_files):
        print(f"Frame id {frame_id} out of range (total {len(bin_files)} frames)")
        return

    bin_file = os.path.join(bin_folder, bin_files[frame_id])
    pcd_array = read_bin(bin_file, 8)
    pcd_array[:, 1] = -1 * pcd_array[:, 1]
    semantic_ids = pcd_array[:, 6].astype(int)

    moving_objects = set(MOVING_OBJECTS.keys())
    unique_ids, counts = np.unique(semantic_ids, return_counts=True)
    total_points = len(semantic_ids)

    print(f"\nFrame {frame_id} motion analysis:")
    print("=" * 60)

    moving_points = 0
    static_points = 0

    print("Moving objects (shown in red):")
    for semantic_id, count in zip(unique_ids, counts):
        if semantic_id in moving_objects:
            label = SEMANTIC_LABELS.get(semantic_id, f'Unknown_{semantic_id}')
            percentage = count / total_points * 100
            print(f"  {label:15s}: {count:6d} points ({percentage:5.1f}%)")
            moving_points += count

    print("\nStatic background (shown in gray):")
    for semantic_id, count in zip(unique_ids, counts):
        if semantic_id not in moving_objects:
            label = SEMANTIC_LABELS.get(semantic_id, f'Unknown_{semantic_id}')
            percentage = count / total_points * 100
            print(f"  {label:15s}: {count:6d} points ({percentage:5.1f}%)")
            static_points += count

    print("\nSummary:")
    print(f"  Total points: {total_points:,}")
    print(f"  Moving:  {moving_points:,} ({moving_points / total_points * 100:.1f}%)")
    print(f"  Static:  {static_points:,} ({static_points / total_points * 100:.1f}%)")

    pcd = create_pcd(pcd_array[:, :3])
    apply_motion_colors(pcd, semantic_ids)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Frame {frame_id} - Motion Analysis", width=1200, height=800)
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = 1.5
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def visualize_moving_objects_only(bin_folder, fps=10, point_size=2.0):
    """Play back only the moving-object points (all others filtered out)."""
    bin_files = sorted([f for f in os.listdir(bin_folder) if f.endswith('.bin')])
    if not bin_files:
        print("No .bin files found!")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Moving Objects Only", width=1200, height=800)
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = point_size

    moving_ids = set(MOVING_OBJECTS.keys())

    first_bin = os.path.join(bin_folder, bin_files[0])
    pcd_array = read_bin(first_bin, 8)
    pcd_array[:, 1] = -1 * pcd_array[:, 1]
    semantic_ids = pcd_array[:, 6].astype(int)
    moving_mask = np.isin(semantic_ids, list(moving_ids))
    moving_points = pcd_array[moving_mask, :3]

    if len(moving_points) == 0:
        print("No moving-object points in the first frame!")
        return

    pcd = create_pcd(moving_points)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])
    vis.add_geometry(pcd)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    frame_interval = 1.0 / fps
    print(f"Playing moving-object points only at {fps} FPS")
    print(f"First frame moving-object points: {len(moving_points)}")

    try:
        frame_idx = 0
        while True:
            start_time = time.time()

            bin_file = os.path.join(bin_folder, bin_files[frame_idx])
            pcd_array = read_bin(bin_file, 8)
            pcd_array[:, 1] = -1 * pcd_array[:, 1]
            semantic_ids = pcd_array[:, 6].astype(int)
            moving_mask = np.isin(semantic_ids, list(moving_ids))
            moving_points = pcd_array[moving_mask, :3]

            if len(moving_points) > 0:
                pcd.points = o3d.utility.Vector3dVector(moving_points)
                pcd.paint_uniform_color([1.0, 0.0, 0.0])
            else:
                pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))

            if frame_idx % 20 == 0:
                print(f"Frame {frame_idx + 1}: moving-object points {len(moving_points)}")

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            if not vis.poll_events():
                break

            elapsed = time.time() - start_time
            time.sleep(max(0, frame_interval - elapsed))
            frame_idx = (frame_idx + 1) % len(bin_files)

    except KeyboardInterrupt:
        print("Playback interrupted by user")
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    bin_folder = 'data/simulation/base_scenario'

    print("Select visualization mode:")
    print("1. Motion-highlighted playback (moving objects red, static gray)")
    print("2. Single-frame motion analysis")
    print("3. Moving objects only")
    choice = input("Mode (1/2/3): ").strip()

    if choice == "1":
        visualize_pcd_sequence_realtime_motion(bin_folder, fps=10, point_size=1.5)
    elif choice == "2":
        frame_id = int(input("Frame id to analyze (starting from 0): "))
        visualize_motion_analysis(bin_folder, frame_id)
    elif choice == "3":
        visualize_moving_objects_only(bin_folder, fps=10, point_size=2.0)
    else:
        print("Defaulting to motion-highlighted playback")
        visualize_pcd_sequence_realtime_motion(bin_folder, fps=10, point_size=1.5)
