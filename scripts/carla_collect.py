#!/usr/bin/env python3
"""
CARLA semantic LiDAR data collection script.

Spawns a semantic LiDAR sensor (configured to approximate an Ouster OS1-128)
plus simulated traffic in a CARLA town, and records structured point cloud
frames to disk. Each frame is saved as a fixed-size .bin file with columns
[x, y, z, range, intensity, cosine, semantic_id, object_id], where
semantic_id follows CARLA's semantic segmentation tag set and can be used
directly as foreground/background ground truth (see
scripts/evaluate_simulation_metrics.py).

Requires a running CARLA simulator (tested against CARLA 0.9.15) and the
`carla` Python API on PYTHONPATH.
"""

import carla
import numpy as np
import os
import time
import weakref
from datetime import datetime
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LidarSemanticCollector:
    def __init__(self, host='localhost', port=2000, timeout=30.0):
        """Set up the CARLA client connection and LiDAR collector state."""
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = None
        self.semantic_lidar = None
        self.vehicle_actors = []
        self.pedestrian_actors = []
        self.frame_count = 0
        self.data_dir = None
        self.semantic_stats = Counter()
        self.total_points_processed = 0
        # Ouster OS1-128 sensor parameters.
        self.semantic_params = {
            'channels': '128',
            'range': '200',
            'points_per_second': '1310720',  # 1024 * 128 * 10 Hz = 1310720
            'rotation_frequency': '10.0',
            'upper_fov': '22.5',
            'lower_fov': '-22.5',
            'horizontal_fov': '360.0',
            'sensor_tick': '0.0',
        }

    def setup_world(self, town='Town03'):
        """Load the CARLA map, set weather, and enable synchronous mode."""
        logger.info(f"Loading map: {town}")
        self.world = self.client.load_world(town)

        weather = carla.WeatherParameters(
            cloudiness=20.0,
            precipitation=0.0,
            sun_altitude_angle=70.0
        )
        self.world.set_weather(weather)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # 10 FPS
        self.world.apply_settings(settings)

    def spawn_traffic(self, num_vehicles=25, num_pedestrians=15):
        """Spawn autopilot vehicles and randomly-walking pedestrians."""
        logger.info(f"Spawning traffic: {num_vehicles} vehicles, {num_pedestrians} pedestrians")
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        pedestrian_blueprints = blueprint_library.filter('walker.pedestrian.*')
        spawn_points = self.world.get_map().get_spawn_points()

        for i in range(min(num_vehicles, len(spawn_points))):
            vehicle_bp = np.random.choice(vehicle_blueprints.filter('vehicle.*'))
            try:
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[i])
                vehicle.set_autopilot(True)
                self.vehicle_actors.append(vehicle)
            except Exception as e:
                logger.warning(f"Failed to spawn vehicle {i}: {e}")

        pedestrian_spawn_points = []
        for _ in range(num_pedestrians):
            spawn_point = self.world.get_random_location_from_navigation()
            if spawn_point is not None:
                pedestrian_spawn_points.append(spawn_point)
        for i, spawn_point in enumerate(pedestrian_spawn_points):
            pedestrian_bp = np.random.choice(pedestrian_blueprints)
            try:
                pedestrian = self.world.spawn_actor(pedestrian_bp, carla.Transform(spawn_point))
                self.pedestrian_actors.append(pedestrian)
                pedestrian.apply_control(carla.WalkerControl(
                    direction=carla.Vector3D(np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0),
                    speed=1.4
                ))
            except Exception as e:
                logger.warning(f"Failed to spawn pedestrian {i}: {e}")
        logger.info(f"Spawned {len(self.vehicle_actors)} vehicles and {len(self.pedestrian_actors)} pedestrians")

    def create_lidar_sensor(self, location, rotation):
        """Spawn the semantic LiDAR sensor and register the data callback."""
        blueprint_library = self.world.get_blueprint_library()
        transform = carla.Transform(location, rotation)
        semantic_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
        for param, value in self.semantic_params.items():
            semantic_bp.set_attribute(param, value)
        self.semantic_lidar = self.world.spawn_actor(semantic_bp, transform)
        weak_self = weakref.ref(self)
        self.semantic_lidar.listen(lambda data: LidarSemanticCollector._on_semantic_data(weak_self, data))

    @staticmethod
    def _on_semantic_data(weak_self, data):
        self = weak_self()
        if self is not None:
            self.process_semantic_data(data)

    def process_semantic_data(self, data):
        """Convert one raw CARLA semantic LiDAR frame into the structured point cloud format."""
        detections = list(data)
        xyz_list = []
        cosine_list = []
        semantic_tags_list = []
        object_indices_list = []
        for detection in detections:
            xyz_list.append([detection.point.x, detection.point.y, detection.point.z])
            cosine_list.append(detection.cos_inc_angle)
            semantic_tags_list.append(detection.object_tag)
            object_indices_list.append(detection.object_idx)

        xyz0 = np.array(xyz_list, dtype=np.float32)
        xyz = self.add_noise(xyz0)
        cosines = np.array(cosine_list, dtype=np.float32)
        semantic_ids = np.array(semantic_tags_list, dtype=int)
        object_ids = np.array(object_indices_list, dtype=int)
        ranges = np.linalg.norm(xyz, axis=1)
        intensities = self.calculate_intensity_from_semantic(ranges, cosine_angles=cosines)
        structured_points = self.generate_semantic_pointcloud(
            xyz, ranges, intensities, cosines, semantic_ids, object_ids
        )
        self.save_pointcloud(structured_points, self.frame_count)
        self.frame_count += 1
        self.total_points_processed += len(xyz)

    def calculate_intensity_from_semantic(self, ranges, cosine_angles=None, attenuation_rate=0.004):
        """Simple range- and incidence-angle-based intensity model (CARLA has no native intensity)."""
        intensity = np.exp(-attenuation_rate * ranges)
        if cosine_angles is not None:
            intensity *= np.abs(cosine_angles)
        return intensity

    def add_noise(self, xyz, base_noise=0.01, distance_factor=0.0001):
        """
        Add range-dependent measurement noise along each point's ray direction.

        base_noise: base noise standard deviation (near range).
        distance_factor: how much noise grows with range: sigma = base_noise + distance_factor * distance.
        """
        distances = np.sqrt(np.sum(xyz ** 2, axis=1, keepdims=True))
        adaptive_noise = base_noise + distance_factor * distances.flatten()
        directions = xyz / (distances + 1e-8)
        noise = np.random.normal(0, adaptive_noise.reshape(-1, 1), xyz.shape)
        # Only add noise along the ray direction (more physically realistic).
        radial_noise = np.sum(noise * directions, axis=1, keepdims=True)
        noisy_xyz = xyz + directions * radial_noise
        return noisy_xyz

    def generate_semantic_pointcloud(self, xyz, ranges, intensities, cosines, semantic_ids, object_ids):
        """
        Pack per-point arrays into the fixed-size structured point cloud.
        Output columns: [x, y, z, range, intensity, cosine, semantic_id, object_id].
        """
        horizontal_points = 1024
        vertical_points = 128
        total_points = horizontal_points * vertical_points  # 131072
        structured_cloud = np.zeros((total_points, 8), dtype=np.float32)
        num_detected = len(xyz)
        points_to_fill = min(num_detected, total_points)
        for i in range(points_to_fill):
            x, y, z = xyz[i]
            structured_cloud[i] = [x, y, z, ranges[i], intensities[i], cosines[i], semantic_ids[i], object_ids[i]]
        return structured_cloud

    def save_pointcloud(self, points, frame_id):
        """
        Save one frame to a .bin file.
        points: array of shape (131072, 8), columns [x, y, z, range, intensity, cosine, semantic_id, object_id].
        """
        if not self.data_dir:
            return
        filename = os.path.join(self.data_dir, f"frame_{frame_id:06d}.bin")
        if isinstance(points, list):
            points_array = np.array(points, dtype=np.float32)
        else:
            points_array = points.astype(np.float32)

        expected_shape = (131072, 8)
        if points_array.shape != expected_shape:
            logger.warning(f"Point cloud shape mismatch: expected {expected_shape}, got {points_array.shape}")
            standard_array = np.zeros(expected_shape, dtype=np.float32)
            min_rows = min(points_array.shape[0], expected_shape[0])
            min_cols = min(points_array.shape[1], expected_shape[1])
            standard_array[:min_rows, :min_cols] = points_array[:min_rows, :min_cols]
            points_array = standard_array

        points_array.tofile(filename)
        expected_size = 131072 * 8 * 4  # 131072 points x 8 fields x 4 bytes
        actual_size = os.path.getsize(filename)
        if actual_size != expected_size:
            logger.warning(f"File size mismatch: expected {expected_size} bytes, got {actual_size} bytes")

    def load_pointcloud(self, bin_file):
        """Load one frame back. Returns an (131072, 8) array, same columns as save_pointcloud."""
        points = np.fromfile(bin_file, dtype=np.float32).reshape(131072, 8)
        return points

    def setup_data_directory(self):
        """Create a timestamped output directory and write a config summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = f"carla_semantic_data_{timestamp}"
        os.makedirs(self.data_dir, exist_ok=True)

        config_file = os.path.join(self.data_dir, "config.txt")
        with open(config_file, 'w') as f:
            f.write("CARLA semantic LiDAR data collection configuration\n")
            f.write("=" * 50 + "\n")
            f.write("Sensor model: Ouster OS1-128\n")
            f.write("Sensor type: Semantic LiDAR\n")
            f.write(f"Channels: {self.semantic_params['channels']}\n")
            f.write(f"Rotation frequency: {self.semantic_params['rotation_frequency']} Hz\n")
            f.write(f"Points per second: {self.semantic_params['points_per_second']}\n")
            f.write("Horizontal resolution: 1024 points\n")
            f.write("Vertical resolution: 128 points\n")
            f.write("Points per frame: 131072 (fixed)\n")
            f.write("Point format: x, y, z, range, intensity, cosine, semantic_id, object_id\n")
            f.write("Data type: float32\n")
            f.write("Bytes per point: 32 (8 * 4)\n")
            f.write(f"Bytes per frame: {131072 * 8 * 4} (~4.19MB)\n")

        logger.info(f"Data will be saved to: {self.data_dir}")

    def collect_data(self, duration_hours=0.5):
        """Advance the simulation and collect data for the given duration."""
        logger.info(f"Starting data collection for {duration_hours} hours")
        start_time = time.time()
        duration_seconds = duration_hours * 3600
        try:
            while time.time() - start_time < duration_seconds:
                self.world.tick()
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Data collection interrupted by user")
        logger.info("Data collection complete")
        logger.info(f"Total frames: {self.frame_count}")
        logger.info(f"Total points processed: {self.total_points_processed}")

    def cleanup(self):
        """Destroy spawned actors and restore asynchronous simulation mode."""
        logger.info("Cleaning up...")
        if self.semantic_lidar:
            self.semantic_lidar.stop()
            self.semantic_lidar.destroy()
        for vehicle in self.vehicle_actors:
            vehicle.destroy()
        for pedestrian in self.pedestrian_actors:
            pedestrian.destroy()
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        logger.info("Cleanup complete")


def main():
    collector = LidarSemanticCollector()
    try:
        collector.setup_world('Town05')
        collector.setup_data_directory()
        lidar_location = carla.Location(x=27.18, y=-0.37, z=3)
        lidar_rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        collector.create_lidar_sensor(lidar_location, lidar_rotation)
        collector.spawn_traffic(num_vehicles=25, num_pedestrians=15)
        time.sleep(3.0)  # Let the sensor initialize.
        collector.collect_data(duration_hours=0.5)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()
