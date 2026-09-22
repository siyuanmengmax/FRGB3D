#!/usr/bin/env python3
"""
Interactive CARLA sensor-placement explorer.

Helps identify candidate roadside LiDAR mounting locations (map center,
near intersections, near high vehicle-spawn-density areas) in a CARLA
town, visualizes them with in-simulator markers and debug text, spawns
some traffic for context, and lets the user interactively pick a location
to use with scripts/carla_collect.py.
"""

import carla
import numpy as np


class LidarPositionExplorer:
    def __init__(self, client):
        self.client = client
        self.world = None
        self.spectator = None
        self.lidar_markers = []  # Placed LiDAR position markers.
        self.vehicle_actors = []  # Spawned vehicles.

    def setup_world(self, town='Town03'):
        """Load the CARLA map and enable synchronous mode."""
        print(f"Loading map: {town}")
        self.world = self.client.load_world(town)
        self.spectator = self.world.get_spectator()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(settings)

    def create_lidar_marker(self, location, name="LiDAR", color=(255, 0, 0)):
        """
        Place a visible marker at a candidate LiDAR location.

        Args:
            location: carla.Location.
            name: marker label.
            color: RGB color tuple.
        """
        try:
            blueprint_library = self.world.get_blueprint_library()

            # Preferred: a static prop mesh.
            marker_bp = blueprint_library.find('static.prop.streetbarrier')
            if marker_bp:
                transform = carla.Transform(
                    carla.Location(x=location.x, y=location.y, z=location.z + 1.0),
                    carla.Rotation(pitch=0, yaw=0, roll=0)
                )
                marker = self.world.spawn_actor(marker_bp, transform)
                marker_info = {'actor': marker, 'name': name, 'location': location, 'color': color}
                self.lidar_markers.append(marker_info)
                print(f"Created LiDAR marker: {name} at ({location.x:.1f}, {location.y:.1f}, {location.z:.1f})")
                return marker

        except Exception as e:
            print(f"Failed to create marker: {e}")

        # Fallback: use a vehicle as a visible marker.
        try:
            vehicle_bp = blueprint_library.find('vehicle.audi.a2')
            if vehicle_bp:
                vehicle_bp.set_attribute('color', '255,0,0')
                transform = carla.Transform(
                    carla.Location(x=location.x, y=location.y, z=location.z + 0.5),
                    carla.Rotation(pitch=0, yaw=0, roll=0)
                )
                marker = self.world.spawn_actor(vehicle_bp, transform)
                marker_info = {'actor': marker, 'name': name, 'location': location, 'color': color}
                self.lidar_markers.append(marker_info)
                print(f"Created vehicle marker: {name} at ({location.x:.1f}, {location.y:.1f}, {location.z:.1f})")
                return marker

        except Exception as e:
            print(f"Failed to create vehicle marker: {e}")

        return None

    def create_debug_text(self, location, text, color=(255, 255, 255)):
        """Draw a floating 3D debug text label above a location."""
        try:
            debug = self.world.debug
            debug.draw_string(
                location=carla.Location(x=location.x, y=location.y, z=location.z + 3.0),
                text=text,
                draw_shadow=True,
                color=carla.Color(r=color[0], g=color[1], b=color[2]),
                life_time=30.0,
                persistent_lines=True
            )
            print(f"Created debug text: {text}")
        except Exception as e:
            print(f"Failed to create debug text: {e}")

    def explore_town03_with_visualization(self):
        """Explore Town03 and visualize a set of candidate LiDAR locations."""
        self.setup_world('Town03')
        carla_map = self.world.get_map()

        print("=== Exploring Town03 and visualizing candidate LiDAR locations ===")

        spawn_points = carla_map.get_spawn_points()
        print(f"Vehicle spawn points: {len(spawn_points)}")

        x_coords = [sp.location.x for sp in spawn_points]
        y_coords = [sp.location.y for sp in spawn_points]
        print(f"X range: {min(x_coords):.1f} to {max(x_coords):.1f}")
        print(f"Y range: {min(y_coords):.1f} to {max(y_coords):.1f}")

        print("\n=== Locating intersections ===")
        waypoints = carla_map.generate_waypoints(distance=2.0)
        junctions = []
        for wp in waypoints:
            if wp.is_junction:
                junction = wp.get_junction()
                if junction and junction.id not in [j['id'] for j in junctions]:
                    junctions.append({
                        'id': junction.id,
                        'location': junction.bounding_box.location,
                        'extent': junction.bounding_box.extent
                    })
        print(f"Found {len(junctions)} intersections")

        print("\n=== Creating candidate LiDAR locations ===")

        # Candidate 1: map center.
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        center_location = carla.Location(x=center_x, y=center_y, z=4.0)
        self.create_lidar_marker(center_location, "Center", (255, 0, 0))
        self.create_debug_text(center_location, "LiDAR-Center", (255, 255, 0))

        # Candidates 2-4: near intersections.
        for i, junction in enumerate(junctions[:3]):
            loc = junction['location']
            offset_location = carla.Location(x=loc.x + 15, y=loc.y + 15, z=5.0)  # Offset to avoid blocking traffic.
            self.create_lidar_marker(offset_location, f"Intersection {i + 1}", (0, 255, 0))
            self.create_debug_text(offset_location, f"LiDAR-Intersection{i + 1}", (0, 255, 255))

        # Candidates 5-6: near high vehicle-spawn-density areas.
        density_positions = self.calculate_density_positions(spawn_points)
        for i, pos in enumerate(density_positions[:2]):
            marker_location = carla.Location(x=pos[0], y=pos[1], z=4.0)
            self.create_lidar_marker(marker_location, f"High-density {i + 1}", (0, 0, 255))
            self.create_debug_text(marker_location, f"LiDAR-HighDensity{i + 1}", (255, 0, 255))

        self.spawn_traffic_for_visualization(num_vehicles=20)
        self.set_spectator_overview()

        print("\n=== Visualization complete ===")
        print("Red markers: map center")
        print("Green markers: near intersections")
        print("Blue markers: high vehicle-spawn-density areas")
        print("\nSpectator set to an overhead view showing all candidate markers")

        return self.lidar_markers

    def calculate_density_positions(self, spawn_points, grid_size=50):
        """Bin vehicle spawn points on a grid and return the highest-density cell centers."""
        if not spawn_points:
            return []

        x_coords = [sp.location.x for sp in spawn_points]
        y_coords = [sp.location.y for sp in spawn_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        x_bins = np.linspace(x_min, x_max, grid_size // 10)
        y_bins = np.linspace(y_min, y_max, grid_size // 10)
        density_map = np.zeros((len(x_bins) - 1, len(y_bins) - 1))

        for sp in spawn_points:
            x_idx = np.digitize(sp.location.x, x_bins) - 1
            y_idx = np.digitize(sp.location.y, y_bins) - 1
            if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1:
                density_map[x_idx, y_idx] += 1

        high_density_positions = []
        for _ in range(3):
            max_idx = np.unravel_index(np.argmax(density_map), density_map.shape)
            x_pos = (x_bins[max_idx[0]] + x_bins[max_idx[0] + 1]) / 2
            y_pos = (y_bins[max_idx[1]] + y_bins[max_idx[1] + 1]) / 2
            high_density_positions.append((x_pos, y_pos))
            density_map[max_idx] = 0  # Avoid picking the same cell twice.

        return high_density_positions

    def spawn_traffic_for_visualization(self, num_vehicles=20):
        """Spawn a small amount of traffic so the candidate locations can be assessed visually."""
        print(f"\n=== Spawning {num_vehicles} vehicles for visualization ===")
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()

        for i in range(min(num_vehicles, len(spawn_points))):
            vehicle_bp = np.random.choice(vehicle_blueprints)
            if vehicle_bp.has_attribute('color'):
                colors = ['255,255,255', '0,0,255', '255,255,0', '0,255,0']
                vehicle_bp.set_attribute('color', np.random.choice(colors))
            try:
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[i])
                vehicle.set_autopilot(True)
                self.vehicle_actors.append(vehicle)
            except Exception as e:
                print(f"Failed to spawn vehicle {i}: {e}")

        print(f"Spawned {len(self.vehicle_actors)} vehicles")

    def set_spectator_overview(self):
        """Move the spectator camera to an overhead view of all placed markers."""
        if self.lidar_markers:
            avg_x = np.mean([m['location'].x for m in self.lidar_markers])
            avg_y = np.mean([m['location'].y for m in self.lidar_markers])
            overview_location = carla.Location(x=avg_x, y=avg_y, z=50.0)
            overview_rotation = carla.Rotation(pitch=-60, yaw=0, roll=0)
            self.spectator.set_transform(carla.Transform(overview_location, overview_rotation))
            print(f"Spectator set to overhead view: ({avg_x:.1f}, {avg_y:.1f}, 50.0)")

    def interactive_position_selection(self):
        """Prompt the user to pick one of the placed candidate markers."""
        print("\n=== Interactive location selection ===")
        print("You should now see all candidate LiDAR location markers.")
        print("Observe the scene and choose the most suitable location.")

        for i, marker in enumerate(self.lidar_markers):
            loc = marker['location']
            print(f"{i + 1}. {marker['name']}: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

        try:
            choice = int(input("\nSelect a LiDAR location (enter number): ")) - 1
            if 0 <= choice < len(self.lidar_markers):
                selected = self.lidar_markers[choice]
                print(f"Selected: {selected['name']}")

                selected_loc = selected['location']
                nearby_location = carla.Location(
                    x=selected_loc.x + 20, y=selected_loc.y + 20, z=selected_loc.z + 10
                )
                self.spectator.set_transform(
                    carla.Transform(nearby_location, carla.Rotation(pitch=-30, yaw=-45, roll=0))
                )
                print("Spectator moved near the selected location")
                return selected
            else:
                print("Invalid selection")
                return None

        except ValueError:
            print("Please enter a valid number")
            return None

    def cleanup(self):
        """Destroy all placed markers and spawned vehicles, and restore asynchronous mode."""
        print("\n=== Cleaning up ===")
        for marker in self.lidar_markers:
            try:
                marker['actor'].destroy()
                print(f"Removed marker: {marker['name']}")
            except Exception:
                pass
        for vehicle in self.vehicle_actors:
            try:
                vehicle.destroy()
            except Exception:
                pass
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        print("Cleanup complete")

    def run_exploration_session(self):
        """Run the full explore -> observe -> select workflow."""
        try:
            self.explore_town03_with_visualization()
            input("\nPress Enter to continue to interactive selection...")
            selected = self.interactive_position_selection()
            input("\nPress Enter to end the exploration session...")
            return selected

        except KeyboardInterrupt:
            print("\nExploration interrupted by user")
            return None
        except Exception as e:
            print(f"Exploration failed: {e}")
            return None
        finally:
            self.cleanup()


def main():
    print("=== LiDAR Position Explorer ===")
    print("Helps find a good roadside LiDAR mounting location in Town03,")
    print("visualized with in-simulator markers.")

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(30.0)

        explorer = LidarPositionExplorer(client)
        selected_position = explorer.run_exploration_session()

        if selected_position:
            loc = selected_position['location']
            print("\nFinal selected location:")
            print(f"Name: {selected_position['name']}")
            print(f"Coordinates: x={loc.x:.1f}, y={loc.y:.1f}, z={loc.z:.1f}")
            print("\nUse this location in scripts/carla_collect.py:")
            print(f"lidar_location = carla.Location(x={loc.x:.1f}, y={loc.y:.1f}, z={loc.z:.1f})")

    except Exception as e:
        print(f"Failed to connect to CARLA: {e}")
        print("Make sure the CARLA server is running.")


if __name__ == "__main__":
    main()
