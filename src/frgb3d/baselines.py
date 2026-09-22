"""Baseline background modeling methods compared against FRGB3D in the paper:

  - Simple Difference (SD)
  - Moving Average (MA)
  - FRGB3D-NR: standard Gaussian Background Modeling (GBM), i.e. FRGB3D with
    reliability weighting removed (No-Reliability ablation / GBM baseline).
"""
import numpy as np
from numba import jit, prange
from tqdm import tqdm

from ..utils.io import read_bin
from ..utils.visualization import visualize_dynamic_points


# ===== Simple Difference (SD) =====
@jit(nopython=True, parallel=True)
def update_sd_background_model(mean, counts, indices, ranges, total_points):
    """Simple Difference background model update (running mean)."""
    for i in prange(len(indices)):
        idx = indices[i]
        if idx >= total_points:
            continue
        r = ranges[i]
        counts[idx] += 1
        if counts[idx] == 1:
            mean[idx] = r
        else:
            mean[idx] = (mean[idx] * (counts[idx] - 1) + r) / counts[idx]
    return mean, counts


@jit(nopython=True, parallel=True)
def detect_sd_foreground_points(indices, ranges, mean, counts, min_count,
                                 fixed_threshold, foreground_mask):
    """Simple Difference foreground detection (fixed threshold)."""
    for i in prange(len(indices)):
        idx = indices[i]
        if idx >= len(mean):
            continue
        r = ranges[i]
        if counts[idx] >= min_count and mean[idx] > 0:
            diff = abs(mean[idx] - r)
            if diff > fixed_threshold:
                foreground_mask[indices[i]] = True
    return foreground_mask


def build_sd_background(pcd_files, min_range=0.5, max_range=200, sample_size=1000,
                         columns=['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'intensity', 'flags']):
    """Build a Simple Difference background model."""
    total_points = len(read_bin(pcd_files[0], len(columns)))
    mean = np.zeros(total_points, dtype=np.float32)
    counts = np.zeros(total_points, dtype=np.int32)

    interval = max(1, len(pcd_files) // sample_size)
    for i in tqdm(range(0, len(pcd_files), interval), desc="Building SD background model"):
        pcd_array = read_bin(pcd_files[i], len(columns))
        ranges = pcd_array[:, columns.index('range')]
        indices = np.arange(len(pcd_array))
        mask = (ranges > min_range) & (ranges < max_range)
        valid_ranges = ranges[mask]
        valid_indices = indices[mask]
        mean, counts = update_sd_background_model(mean, counts, valid_indices, valid_ranges, total_points)

    bg_model = np.column_stack([mean, counts])
    valid_points = counts > 0
    print(f"SD background model built: {np.sum(valid_points)}/{total_points} valid points "
          f"({np.sum(valid_points) / total_points:.2%})")
    return bg_model


def detect_sd_foreground(pcd_array, bg_model, fixed_threshold=1.0, min_count=10,
                          min_range=0.5, max_range=200, visualize=False,
                          bg_model_columns=['mean', 'counts'],
                          columns=['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'intensity', 'flags']):
    """Simple Difference foreground detection."""
    mean = bg_model[:, bg_model_columns.index('mean')]
    counts = bg_model[:, bg_model_columns.index('counts')]
    ranges = pcd_array[:, columns.index('range')]
    indices = np.arange(len(pcd_array))
    mask = (ranges > min_range) & (ranges < max_range)
    valid_ranges = ranges[mask]
    valid_indices = indices[mask]

    foreground_mask = np.zeros(len(pcd_array), dtype=np.bool_)
    foreground_mask = detect_sd_foreground_points(
        valid_indices, valid_ranges, mean, counts, min_count, fixed_threshold, foreground_mask
    )
    if visualize:
        dynamic_points = pcd_array[foreground_mask]
        static_points = pcd_array[~foreground_mask]
        visualize_dynamic_points(dynamic_points, static_points)
    return foreground_mask, []


# ===== Moving Average (MA) =====
@jit(nopython=True, parallel=True)
def update_ma_background_model(mean, variance, counts, indices, ranges,
                                alpha, total_points):
    """Moving Average background model update (exponential moving average)."""
    for i in prange(len(indices)):
        idx = indices[i]
        if idx >= total_points:
            continue
        r = ranges[i]
        counts[idx] += 1
        if counts[idx] == 1:
            mean[idx] = r
            variance[idx] = 1.0
        else:
            old_mean = mean[idx]
            mean[idx] = alpha * r + (1 - alpha) * old_mean
            diff = r - old_mean
            variance[idx] = alpha * (diff ** 2) + (1 - alpha) * variance[idx]
    return mean, variance, counts


@jit(nopython=True, parallel=True)
def detect_ma_foreground_points(indices, ranges, mean, variance, counts,
                                 min_count, threshold_factor, foreground_mask):
    """Moving Average foreground detection (variance-based adaptive threshold)."""
    for i in prange(len(indices)):
        idx = indices[i]
        if idx >= len(mean):
            continue
        r = ranges[i]
        if counts[idx] >= min_count and mean[idx] > 0:
            std = np.sqrt(variance[idx])
            threshold = threshold_factor * std
            diff = abs(mean[idx] - r)
            if diff > threshold:
                foreground_mask[indices[i]] = True
    return foreground_mask


def build_ma_background(pcd_files, min_range=0.5, max_range=200, alpha=0.01,
                         sample_size=1000,
                         columns=['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'intensity', 'flags']):
    """Build a Moving Average background model."""
    total_points = len(read_bin(pcd_files[0], len(columns)))
    mean = np.zeros(total_points, dtype=np.float32)
    variance = np.ones(total_points, dtype=np.float32)
    counts = np.zeros(total_points, dtype=np.int32)

    interval = max(1, len(pcd_files) // sample_size)
    for i in tqdm(range(0, len(pcd_files), interval), desc="Building MA background model"):
        pcd_array = read_bin(pcd_files[i], len(columns))
        ranges = pcd_array[:, columns.index('range')]
        indices = np.arange(len(pcd_array))
        mask = (ranges > min_range) & (ranges < max_range)
        valid_ranges = ranges[mask]
        valid_indices = indices[mask]
        mean, variance, counts = update_ma_background_model(
            mean, variance, counts, valid_indices, valid_ranges, alpha, total_points
        )

    bg_model = np.column_stack([mean, variance, counts])
    valid_points = counts > 0
    print(f"MA background model built: {np.sum(valid_points)}/{total_points} valid points "
          f"({np.sum(valid_points) / total_points:.2%})")
    return bg_model


def detect_ma_foreground(pcd_array, bg_model, threshold_factor=2.5, min_count=10,
                          min_range=0.5, max_range=200, visualize=False,
                          bg_model_columns=['mean', 'variance', 'counts'],
                          columns=['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'intensity', 'flags']):
    """Moving Average foreground detection."""
    mean = bg_model[:, bg_model_columns.index('mean')]
    variance = bg_model[:, bg_model_columns.index('variance')]
    counts = bg_model[:, bg_model_columns.index('counts')]
    ranges = pcd_array[:, columns.index('range')]
    indices = np.arange(len(pcd_array))
    mask = (ranges > min_range) & (ranges < max_range)
    valid_ranges = ranges[mask]
    valid_indices = indices[mask]

    foreground_mask = np.zeros(len(pcd_array), dtype=np.bool_)
    foreground_mask = detect_ma_foreground_points(
        valid_indices, valid_ranges, mean, variance, counts, min_count, threshold_factor, foreground_mask
    )
    if visualize:
        dynamic_points = pcd_array[foreground_mask]
        static_points = pcd_array[~foreground_mask]
        visualize_dynamic_points(dynamic_points, static_points)
    return foreground_mask, []


# ===== FRGB3D-NR: standard Gaussian Background Modeling (GBM baseline) =====
@jit(nopython=True, parallel=True)
def update_nr_background_model(mean, std, weight, counts, indices, ranges,
                                learning_rate, total_points):
    """Background model update with reliability weighting removed (uniform weight)."""
    for i in prange(len(indices)):
        idx = indices[i]
        if idx >= total_points:
            continue
        r = ranges[i]
        counts[idx] += 1
        if weight[idx] == 0:
            mean[idx] = r
            weight[idx] = 1.0  # Uniform weight (no reliability term).
        else:
            diff = r - mean[idx]
            weight[idx] += learning_rate
            mean[idx] += learning_rate * diff / weight[idx]
            std[idx] = np.sqrt((1 - learning_rate) * (std[idx] ** 2) + learning_rate * (diff ** 2))
    return mean, std, weight, counts


@jit(nopython=True, parallel=True)
def detect_nr_foreground_points(indices, ranges, mean, std, counts,
                                 min_count, base_threshold, foreground_mask):
    """Foreground detection with reliability weighting removed (fixed threshold)."""
    for i in prange(len(indices)):
        idx = indices[i]
        if idx >= len(mean):
            continue
        r = ranges[i]
        if counts[idx] >= min_count and mean[idx] > 0:
            diff = abs(mean[idx] - r)
            threshold = base_threshold + std[idx]
            if diff > threshold:
                foreground_mask[indices[i]] = True
    return foreground_mask


def build_nr_background(pcd_files, min_range=0.5, max_range=200, learning_rate=0.01,
                         init_std=1, sample_size=1000,
                         columns=['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'intensity', 'flags']):
    """Build a standard Gaussian Background Model (GBM), i.e. FRGB3D without reliability weighting."""
    total_points = len(read_bin(pcd_files[0], len(columns)))
    mean = np.zeros(total_points, dtype=np.float32)
    std = np.ones(total_points, dtype=np.float32) * init_std
    weight = np.zeros(total_points, dtype=np.float32)
    counts = np.zeros(total_points, dtype=np.int32)

    interval = max(1, len(pcd_files) // sample_size)
    for i in tqdm(range(0, len(pcd_files), interval), desc="Building GBM (FRGB3D-NR) background model"):
        pcd_array = read_bin(pcd_files[i], len(columns))
        ranges = pcd_array[:, columns.index('range')]
        indices = np.arange(len(pcd_array))
        mask = (ranges > min_range) & (ranges < max_range)
        valid_ranges = ranges[mask]
        valid_indices = indices[mask]
        mean, std, weight, counts = update_nr_background_model(
            mean, std, weight, counts, valid_indices, valid_ranges, learning_rate, total_points
        )

    bg_model = np.column_stack([mean, std, counts])
    valid_points = counts > 0
    print(f"GBM (FRGB3D-NR) background model built: {np.sum(valid_points)}/{total_points} valid points "
          f"({np.sum(valid_points) / total_points:.2%})")
    return bg_model


def detect_nr_foreground(pcd_array, bg_model, base_threshold=1, min_count=10,
                          min_range=0.5, max_range=200, visualize=False,
                          bg_model_columns=['mean', 'std', 'counts'],
                          columns=['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'intensity', 'flags']):
    """GBM (FRGB3D-NR) foreground detection."""
    mean = bg_model[:, bg_model_columns.index('mean')]
    std = bg_model[:, bg_model_columns.index('std')]
    counts = bg_model[:, bg_model_columns.index('counts')]
    ranges = pcd_array[:, columns.index('range')]
    indices = np.arange(len(pcd_array))
    mask = (ranges > min_range) & (ranges < max_range)
    valid_ranges = ranges[mask]
    valid_indices = indices[mask]

    foreground_mask = np.zeros(len(pcd_array), dtype=np.bool_)
    foreground_mask = detect_nr_foreground_points(
        valid_indices, valid_ranges, mean, std, counts, min_count, base_threshold, foreground_mask
    )
    if visualize:
        dynamic_points = pcd_array[foreground_mask]
        static_points = pcd_array[~foreground_mask]
        visualize_dynamic_points(dynamic_points, static_points)
    return foreground_mask, []
