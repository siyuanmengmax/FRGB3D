import numpy as np
from scipy.spatial import cKDTree


def clustering_stage_1(points, columns, rho_min=10, alpha=0.5, eps_min=0.5, eps_max=1, density_unit=1):
    """Reliability-weighted, density-adaptive DBSCAN-style clustering.

    Extends standard DBSCAN with a per-point adaptive search radius that
    shrinks in high-utility (well-observed, high-density) regions and grows
    towards eps_max elsewhere, and with a reliability-weighted density
    estimate (points are weighted by their FRGB3D reliability score rather
    than counted uniformly). Used as the "Improved DBSCAN" clustering stage
    for turning FRGB3D foreground points into object clusters.

    Args:
        points: (N, M) point array, expected to include a 'reliability' column.
        columns: column name list used to index into `points`.
        rho_min: minimum reliability-weighted density threshold for a core point.
        alpha: adaptive-radius decay rate in [0, 1].
        eps_min, eps_max: bounds for the adaptive search radius.
        density_unit: scaling factor applied to the density estimate.

    Returns:
        clusters: list of per-cluster point index lists.
        clusters_coord: list of per-cluster point coordinate arrays.
        adaptive_radius: per-point adaptive radius used during clustering.
    """
    n_points = len(points)
    point_coord = points[:, :columns.index('z') + 1]
    tree = cKDTree(point_coord)
    point_reliability = points[:, columns.index('reliability')]

    # Base neighborhood (radius eps_max) used only to estimate local density.
    basic_point_neighbor_id = tree.query_ball_point(point_coord, r=eps_max)
    basic_point_density = np.array(
        [density_unit * np.sum(point_reliability[ids]) for ids in basic_point_neighbor_id])
    utility = 1 - np.clip(rho_min / basic_point_density, 0, 1)
    adaptive_radius = np.clip(eps_max * (1.0 - alpha * utility), eps_min, eps_max)

    clusters = []
    clusters_coord = []
    processed = np.zeros(n_points, dtype=bool)
    for i in range(n_points):
        if not processed[i]:
            adaptive_point_neighbor_id = tree.query_ball_point(point_coord[i], r=adaptive_radius[i])
            adaptive_point_density = density_unit * np.sum(point_reliability[adaptive_point_neighbor_id])
            if adaptive_point_density >= rho_min:
                cluster_point_id = [i]
                points_to_process = list(adaptive_point_neighbor_id)
                processed[i] = True
                while points_to_process:
                    j = points_to_process.pop()
                    if not processed[j]:
                        processed[j] = True
                        new_adaptive_point_neighbor_id = tree.query_ball_point(point_coord[j], r=adaptive_radius[j])
                        new_adaptive_point_density = np.sum(point_reliability[new_adaptive_point_neighbor_id])
                        if new_adaptive_point_density >= rho_min:
                            new_points = [k for k in new_adaptive_point_neighbor_id if not processed[k]]
                            points_to_process.extend(new_points)
                        cluster_point_id.append(j)
                clusters.append(cluster_point_id)
                clusters_coord.append(point_coord[cluster_point_id])
    return clusters, clusters_coord, adaptive_radius
