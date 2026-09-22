"""Real-world object-level detection: clustering, classification, and bbox generation.

Takes the fused, foreground-labeled point clouds produced by
run_realworld_preprocessing.py (or relabel_realworld_baseline.py for a
baseline method) and runs the "Improved DBSCAN" reliability-weighted
density-adaptive clustering stage, a lightweight geometric classifier, and
oriented bounding-box fitting on each frame's foreground points. The
resulting per-frame bounding boxes are saved to a CSV consumed by
evaluate_detection_ap.py to compute AP@IoU against the annotated ground
truth.

Note: the exact clustering/classification configuration used to reproduce
the specific AP numbers reported in the paper's Table 4 was not fully
recorded; the defaults below reflect the best-available reconstruction of
that pipeline. Re-tune `--rho_min`/`--alpha` if you need to match a
particular published number exactly.
"""
import os
import sys
import argparse
import time
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.detection.clustering import clustering_stage_1
from src.detection.classifying import classifying_stage_1, classifying_stage_2
from src.detection.compute_bbox import compute_bbox
from src.utils.io import read_bin, write_bin, ensure_directory
from src.utils.visualization import visualize_frame, visualize_frame_3d, create_point_cloud_video

CLASSES = ['person', 'small vehicle', 'large vehicle', 'other']


def process_frame(frame_id, frame_name, pcd_array, filtered_pcd_array, arg):
    """Cluster, classify, and fit bounding boxes for one frame's foreground points."""
    clusters, clusters_coords, eps = clustering_stage_1(
        filtered_pcd_array, arg.columns, rho_min=arg.params['rho_min'],
        alpha=arg.params['alpha'], eps_min=0.5, eps_max=1.0)
    coarse_classes = classifying_stage_1(clusters_coords)
    cluster_classes, class_probabilities = classifying_stage_2(clusters_coords, coarse_classes)

    bboxes = []
    reliability = filtered_pcd_array[:, arg.columns.index('reliability')]
    cluster_id = np.ones(len(pcd_array)) * -1  # -1: not clustered.
    if clusters:
        tree = cKDTree(pcd_array[:, :arg.columns.index('z') + 1])
        for i, cluster in enumerate(clusters):
            clustered_points = clusters_coords[i]
            dist, idx = tree.query(clustered_points, k=1)
            for j, point_idx in enumerate(idx):
                if dist[j] < 1e-6:
                    cluster_id[point_idx] = i
            bbox = compute_bbox(clusters_coords[i])
            bboxes.append({
                "frame_id": frame_id,
                "frame_name": frame_name,
                'cluster_id': i,
                "label": cluster_classes[i],
                'confidence': class_probabilities[i],
                "x": bbox[0], "y": bbox[1], "z": bbox[2],
                "w": bbox[3], "l": bbox[4], "h": bbox[5], "yaw": bbox[6],
                "avg_eps": np.mean([eps[cluster]]),
                "avg_reliability": np.mean([reliability[cluster]]),
                "num_points": len(cluster)
            })

    pcd_array = np.column_stack([pcd_array, cluster_id])
    if arg.visualize:
        visualize_frame(pcd_array, cluster_id, bboxes, frame_id, CLASSES)
    return bboxes, pcd_array


def main(arg, min_height=0.1, max_height=3):
    print("=== Starting real-world detection pipeline ===")
    pcd_file_paths = sorted(glob(os.path.join(arg.input_folder, '*.bin')))
    print(f"Found {len(pcd_file_paths)} point cloud files")

    if arg.create_video:
        video_path = os.path.join(arg.output_folder, 'point_cloud_video.mp4')
        create_point_cloud_video(pcd_file_paths, video_path, process_frame, arg, fps=arg.fps)
        return

    all_bboxes = []
    columns_len = len(arg.columns)
    for i, pcd_file_path in tqdm(enumerate(pcd_file_paths)):
        frame_name = os.path.basename(pcd_file_path)
        pcd_array = read_bin(pcd_file_path, columns_len)
        filter_mask = (pcd_array[:, arg.columns.index('z')] >= min_height) & (
                pcd_array[:, arg.columns.index('z')] <= max_height) & (
                              pcd_array[:, arg.columns.index('dynamic_mask')] == 1)
        filtered_pcd_array = pcd_array[filter_mask]

        frame_bboxes, pcd_array = process_frame(i, frame_name, pcd_array, filtered_pcd_array, arg)
        all_bboxes.extend(frame_bboxes)

        if arg.save_bin:
            save_folder = os.path.join(arg.output_folder, 'detection')
            ensure_directory(save_folder)
            output_path = os.path.join(save_folder, os.path.basename(pcd_file_path))
            write_bin(pcd_array, output_path)

    print("\nSaving results...")
    bbox_path = os.path.join(arg.output_folder, arg.output_csv_name)
    bbox_df = pd.DataFrame(all_bboxes, columns=["frame_id", "frame_name", "cluster_id", "label", 'confidence',
                                                 "x", "y", "z", "w", "l", "h", "yaw",
                                                 "avg_eps", "avg_reliability", "num_points"])
    bbox_df.to_csv(bbox_path, index=False)
    print(f"Saved bounding box information to {bbox_path}")
    print("\n=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-world FRGB3D detection: clustering + bounding boxes')
    parser.add_argument("--output_folder", default="results/realworld_detection", help="Folder to save output files")
    parser.add_argument("--input_folder", default="bin/merge_1000", help="Folder of fused, foreground-labeled point clouds")
    parser.add_argument("--output_csv_name", default="bboxes.csv", help="Output bounding-box CSV file name")
    params = {
        'alpha': 0.019,     # Adaptive-radius decay rate, [0, 1].
        'rho_min': 30,      # Minimum reliability-weighted density threshold, [10, 30].
    }
    parser.add_argument("--params", default=params, help="Clustering parameters")
    parser.add_argument("--visualize", action="store_true", default=False, help="Visualize results per frame")
    parser.add_argument("--save_bin", action="store_true", default=False, help="Save processed point clouds with cluster ids")
    columns = ['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'dynamic_mask', 'reliability', 'lidar_source']
    parser.add_argument("--columns", default=columns, help="Columns to read from PCD files")
    parser.add_argument("--create_video", action="store_true", default=False, help="Render a point cloud video instead of computing bounding boxes")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the video")
    args = parser.parse_args()
    main(args)
