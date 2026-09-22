"""Re-label the fused real-world benchmark with a baseline foreground mask.

Takes the fused point clouds produced by run_realworld_preprocessing.py
(which uses FRGB3D for foreground detection) and overwrites the
'dynamic_mask' column using a baseline background-modeling method (SD, MA,
or GBM) instead, while keeping everything else (registration, fusion,
'reliability' column) unchanged. This produces the input used to generate
the baseline rows of the real-world object-level AP table
(e.g. "SD+Improved DBSCAN", "MA+Improved DBSCAN", "GBM+Improved DBSCAN")
so that the downstream clustering/bounding-box/AP pipeline is identical
across methods and only the foreground mask varies.

Example:
    python scripts/relabel_realworld_baseline.py --method gbm \\
        --source_pcd_folder_path bin/merge_1000 --output_pcd_folder_path bin/merge_1000_gbm
"""
import os
import sys
import argparse
from glob import glob

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.frgb3d.baselines import (
    build_sd_background, detect_sd_foreground,
    build_ma_background, detect_ma_foreground,
    build_nr_background, detect_nr_foreground,
)
from src.utils.io import ensure_directory, read_bin, write_bin

METHODS = {
    'sd': (build_sd_background, detect_sd_foreground),
    'ma': (build_ma_background, detect_ma_foreground),
    'gbm': (build_nr_background, detect_nr_foreground),  # Standard GBM = FRGB3D without reliability weighting.
}


def main(args):
    build_background, detect_foreground = METHODS[args.method]

    ensure_directory(args.output_folder_path)
    bg_model_path = os.path.join(args.output_folder_path, 'bg_model.npy')
    pcd_files = sorted(glob(os.path.join(args.source_pcd_folder_path, "*.bin")))

    if os.path.exists(bg_model_path):
        bg_model = np.load(bg_model_path)
    else:
        bg_model = build_background(pcd_files, sample_size=1000, columns=args.columns)
        np.save(bg_model_path, bg_model)

    start_idx = args.start_frame if args.start_frame >= 0 else 0
    end_idx = args.end_frame if args.end_frame >= 0 else len(pcd_files)
    eval_pcd_files = pcd_files[start_idx:end_idx:args.frame_step]

    ensure_directory(args.output_pcd_folder_path)
    for pcd_file in tqdm(eval_pcd_files):
        pcd_array = read_bin(pcd_file, len(args.columns))
        foreground_mask, _ = detect_foreground(pcd_array, bg_model, columns=args.columns, visualize=False)
        pcd_array[:, args.columns.index('dynamic_mask')] = foreground_mask.astype(np.float32)
        output_pcd_file = os.path.join(args.output_pcd_folder_path, os.path.basename(pcd_file))
        write_bin(pcd_array, output_pcd_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-label fused point clouds with a baseline foreground mask")
    parser.add_argument("--method", choices=list(METHODS.keys()), default="gbm")
    parser.add_argument("--output_folder_path", default="results/baseline_relabel", help="Folder to save the background model")
    parser.add_argument("--output_pcd_folder_path", default="bin/merge_1000_baseline", help="Folder to save re-labeled point clouds")
    parser.add_argument("--source_pcd_folder_path", default="bin/merge_1000")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame number (inclusive)")
    parser.add_argument("--end_frame", type=int, default=-1,
                         help="Ending frame number (exclusive). Use -1 for all remaining frames")
    parser.add_argument("--frame_step", type=int, default=1, help="Process every nth frame")
    columns = ['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'dynamic_mask', 'reliability', 'lidar_source']
    parser.add_argument("--columns", default=columns, help="Columns to read from PCD files")
    args = parser.parse_args()
    main(args)
