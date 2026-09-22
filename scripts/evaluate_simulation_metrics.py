"""Point-level foreground detection evaluation on CARLA simulation data.

Builds a background model with the selected method, runs foreground
detection frame by frame, and compares against CARLA's semantic-LiDAR
ground truth (foreground = pedestrian/rider/car/truck/bus/train/
motorcycle/bicycle/dynamic) to report Precision / Recall / F1 / Accuracy,
matching the simulation experiments in the FRGB3D paper (Table: base /
adverse weather / high-density scenarios).

Example:
    python scripts/evaluate_simulation_metrics.py \\
        --method frgb3d \\
        --source_pcd_folder_path data/simulation/base_scenario \\
        --output_folder_path results/base_scenario_frgb3d
"""
import os
import sys
import json
import argparse
import time
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.frgb3d.background_model import build_background as build_frgb3d, detect_foreground as detect_frgb3d
from src.frgb3d.baselines import (
    build_sd_background, detect_sd_foreground,
    build_ma_background, detect_ma_foreground,
    build_nr_background, detect_nr_foreground,
)
from src.utils.io import ensure_directory, read_bin

METHODS = {
    'frgb3d': (build_frgb3d, detect_frgb3d),
    'sd': (build_sd_background, detect_sd_foreground),
    'ma': (build_ma_background, detect_ma_foreground),
    'gbm': (build_nr_background, detect_nr_foreground),  # Standard GBM = FRGB3D without reliability weighting.
}

# CARLA semantic segmentation labels (CARLA 0.9.14+) treated as foreground
# (moving objects), matching the paper's simulation ground-truth definition.
FOREGROUND_SEMANTIC_CLASSES = [
    12,  # Pedestrian
    13,  # Rider
    14,  # Car
    15,  # Truck
    16,  # Bus
    17,  # Train
    18,  # Motorcycle
    19,  # Bicycle
    21,  # Dynamic (generic moving object)
]


def get_ground_truth_foreground_mask(semantic_ids):
    """Build a point-level foreground ground-truth mask from CARLA semantic labels."""
    return np.isin(semantic_ids, FOREGROUND_SEMANTIC_CLASSES)


def calculate_f1_metrics(y_true, y_pred):
    """Compute precision/recall/F1/accuracy and the confusion matrix counts."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    }


def print_metrics_summary(all_metrics):
    """Print an aggregate precision/recall/F1/accuracy summary across all frames."""
    if not all_metrics:
        print("No metrics to summarize.")
        return

    total_tp = sum(m['tp'] for m in all_metrics)
    total_fp = sum(m['fp'] for m in all_metrics)
    total_tn = sum(m['tn'] for m in all_metrics)
    total_fn = sum(m['fn'] for m in all_metrics)

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)
                  if (overall_precision + overall_recall) > 0 else 0)
    overall_accuracy = ((total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
                         if (total_tp + total_tn + total_fp + total_fn) > 0 else 0)

    print("\n" + "=" * 60)
    print("Foreground detection evaluation summary")
    print("=" * 60)
    print(f"Frames processed: {len(all_metrics)}")
    print(f"Ground-truth foreground points: {total_tp + total_fn:,}")
    print(f"Predicted foreground points: {total_tp + total_fp:,}")
    print(f"True positives: {total_tp:,}")
    print("-" * 60)
    print(f"Overall precision: {overall_precision:.4f}")
    print(f"Overall recall:    {overall_recall:.4f}")
    print(f"Overall F1:        {overall_f1:.4f}")
    print(f"Overall accuracy:  {overall_accuracy:.4f}")
    print("-" * 60)
    print(f"Mean per-frame precision: {np.mean([m['precision'] for m in all_metrics]):.4f}")
    print(f"Mean per-frame recall:    {np.mean([m['recall'] for m in all_metrics]):.4f}")
    print(f"Mean per-frame F1:        {np.mean([m['f1'] for m in all_metrics]):.4f}")
    print(f"Mean per-frame accuracy:  {np.mean([m['accuracy'] for m in all_metrics]):.4f}")
    print("=" * 60)


def main(args):
    ensure_directory(args.output_folder_path)
    build_background, detect_foreground = METHODS[args.method]

    bg_model_path = os.path.join(args.output_folder_path, 'bg_model.npy')
    pcd_files = sorted(glob(os.path.join(args.source_pcd_folder_path, "*.bin")))
    print(f"Found {len(pcd_files)} point cloud files")

    if os.path.exists(bg_model_path):
        print("Loading existing background model...")
        bg_model = np.load(bg_model_path)
    else:
        print(f"Building background model ({args.method})...")
        bg_model = build_background(pcd_files, sample_size=args.sample_size, columns=args.columns)
        np.save(bg_model_path, bg_model)
        print(f"Background model saved to: {bg_model_path}")

    start_idx = args.start_frame if args.start_frame >= 0 else 0
    end_idx = args.end_frame if args.end_frame >= 0 else len(pcd_files)
    eval_pcd_files = pcd_files[start_idx:end_idx:args.frame_step]
    print(f"Evaluating {len(eval_pcd_files)} frames "
          f"(index {start_idx} to {end_idx - 1}, step {args.frame_step})")

    if 'semantic_id' not in args.columns:
        raise ValueError("'semantic_id' must be included in --columns for F1 evaluation")
    semantic_id_index = args.columns.index('semantic_id')

    all_metrics = []
    all_processing_times = []
    results_df = None
    detailed_results_path = None

    for i, pcd_file in tqdm(enumerate(eval_pcd_files), desc="Processing frames", total=len(eval_pcd_files)):
        pcd_array = read_bin(pcd_file, len(args.columns))
        semantic_ids = pcd_array[:, semantic_id_index].astype(int)
        ground_truth_mask = get_ground_truth_foreground_mask(semantic_ids)

        start_time = time.time()
        predicted_foreground_mask, _ = detect_foreground(
            pcd_array, bg_model, columns=args.columns, visualize=args.visualize
        )
        processing_time = (time.time() - start_time) * 1000  # ms
        all_processing_times.append(processing_time)

        metrics = calculate_f1_metrics(ground_truth_mask, predicted_foreground_mask)
        all_metrics.append(metrics)

        if args.verbose and (i + 1) % 50 == 0:
            current_avg_f1 = np.mean([m['f1'] for m in all_metrics])
            current_avg_time = np.mean(all_processing_times)
            print(f"\nProgress {i + 1}/{len(eval_pcd_files)}: "
                  f"avg F1={current_avg_f1:.4f}, avg time={current_avg_time:.2f}ms")

        if args.save_detailed_results:
            detailed_result = {
                'frame_idx': i,
                'file_name': os.path.basename(pcd_file),
                'processing_time_ms': processing_time,
                **metrics
            }
            if i == 0:
                results_df = pd.DataFrame([detailed_result])
                detailed_results_path = os.path.join(args.output_folder_path, 'detailed_f1_results.csv')
            else:
                results_df = pd.concat([results_df, pd.DataFrame([detailed_result])], ignore_index=True)

    if args.save_detailed_results:
        results_df.to_csv(detailed_results_path, index=False)
        print(f"\nDetailed results saved to: {detailed_results_path}")

    print_metrics_summary(all_metrics)

    avg_time = np.mean(all_processing_times)
    std_time = np.std(all_processing_times)
    fps = 1000 / avg_time

    print("\nProcessing time statistics:")
    print(f"Average: {avg_time:.2f} +/- {std_time:.2f} ms")
    print(f"Throughput: {fps:.1f} FPS")
    print(f"Total time: {sum(all_processing_times) / 1000:.2f} s")

    if args.save_summary:
        total_tp = sum(m['tp'] for m in all_metrics)
        total_fp = sum(m['fp'] for m in all_metrics)
        total_tn = sum(m['tn'] for m in all_metrics)
        total_fn = sum(m['fn'] for m in all_metrics)
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)
                      if (overall_precision + overall_recall) > 0 else 0)

        summary = {
            'method': args.method,
            'total_frames': len(all_metrics),
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'avg_frame_f1': np.mean([m['f1'] for m in all_metrics]),
            'avg_processing_time_ms': avg_time,
            'fps': fps,
            'total_tp': total_tp, 'total_fp': total_fp, 'total_tn': total_tn, 'total_fn': total_fn,
        }
        summary_path = os.path.join(args.output_folder_path, 'f1_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point-level foreground detection F1 evaluation on CARLA data")

    parser.add_argument("--method", choices=list(METHODS.keys()), default="frgb3d",
                         help="Background modeling method to evaluate")
    parser.add_argument("--output_folder_path", default="results/simulation_eval", help="Output folder")
    parser.add_argument("--source_pcd_folder_path", default="data/simulation/base_scenario",
                         help="Folder of CARLA .bin point cloud frames")
    parser.add_argument("--start_frame", type=int, default=400, help="Starting frame index")
    parser.add_argument("--end_frame", type=int, default=-1, help="Ending frame index (-1: all remaining frames)")
    parser.add_argument("--frame_step", type=int, default=20, help="Evaluate every nth frame")
    parser.add_argument("--sample_size", type=int, default=5000, help="Number of frames used to build the background model")

    parser.add_argument("--verbose", action="store_true", default=False, help="Print periodic progress")
    parser.add_argument("--visualize", action="store_true", default=False, help="Visualize detection results")
    parser.add_argument("--save_detailed_results", action="store_true", help="Save per-frame metrics to CSV")
    parser.add_argument("--save_summary", action="store_true", help="Save aggregate metrics to JSON")

    columns = ['x', 'y', 'z', 'range', 'reflectivity', 'cosine', 'semantic_id', 'object_id']
    parser.add_argument("--columns", default=columns, help="Point cloud columns; must include 'semantic_id'")

    args = parser.parse_args()
    main(args)
