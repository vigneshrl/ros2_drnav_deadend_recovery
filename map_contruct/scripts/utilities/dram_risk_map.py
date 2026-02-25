#!/usr/bin/env python3

"""
DRaM Risk Map — merged cost-layer + heatmap node

Replaces the two-node pipeline:
  cost_layer_processor  (sectors + recovery points)
  dram_heatmap_viz      (Bayesian heatmap)

Single node, single callback, no inter-node topic dependency.

Subscribes:
  /dead_end_detection/path_status  (Float32MultiArray from infer_vis)
  /map                             (OccupancyGrid)

Publishes:
  /dram_exploration_map            — Bayesian risk grid, read by goal_generator for EDE
  /dead_end_detection/recovery_points — spatial recovery points for goal_generator
  /cost_layer                      — real-time sector visualization for RViz
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformListener, Buffer
import math
import time
import numpy as np

from map_contruct.scripts.utilities.recovery_points import RecoveryPointManager


class DRaMRiskMap(Node):
    def __init__(self):
        super().__init__('dram_risk_map')

        # ── Subscribers ──────────────────────────────────────────────────
        self.path_sub = self.create_subscription(
            Float32MultiArray,
            '/dead_end_detection/path_status',
            self.path_status_callback,
            10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # ── Publishers ───────────────────────────────────────────────────
        # Risk heatmap — goal_generator reads this for EDE
        self.risk_map_pub = self.create_publisher(
            MarkerArray, '/dram_exploration_map', 10
        )
        # Recovery points — goal_generator navigates back to these
        self.recovery_points_pub = self.create_publisher(
            Float32MultiArray, '/dead_end_detection/recovery_points', 10
        )
        # Sector overlay — RViz debug only
        self.cost_layer_pub = self.create_publisher(
            MarkerArray, '/cost_layer', 10
        )

        # ── TF ───────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Recovery point tracking ───────────────────────────────────────
        self.recovery_manager = RecoveryPointManager(
            confidence_threshold=0.56,
            max_stored_points=50,
            min_distance_m=1.0,
            max_age_s=60.0
        )

        # ── Bayesian safety grid (for /dram_exploration_map) ─────────────
        # grid_pos (int, int) → {'safety': float, 'timestamp': float}
        self.explored_grid = {}
        self.grid_resolution = 0.3        # metres per cell
        self.exploration_radius = 3.0     # metres around robot to update

        # ── Sector visualisation parameters ──────────────────────────────
        self.sector_radius = 3.0
        self.sector_angle = math.pi / 3   # 60 degrees per direction
        self.max_history_age = 30.0
        self.cost_history = {}            # for historical faded sectors

        # ── Map ───────────────────────────────────────────────────────────
        self.current_map = None
        self.threshold = 0.56             # single threshold used everywhere

        self.get_logger().info('DRaM Risk Map initialized')

    # ─────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────

    def map_callback(self, msg: OccupancyGrid):
        self.current_map = msg

    def path_status_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 3:
            return

        robot_x, robot_y, robot_yaw, frame_id = self._get_robot_position()
        current_time = time.time()

        probs = list(msg.data[:3])   # [front, left, right]
        path_binary = [1 if p > self.threshold else 0 for p in probs]
        front_open, left_open, right_open = path_binary
        open_count = sum(path_binary)
        is_dead_end = (open_count == 0)

        # 1. Update RecoveryPointManager ──────────────────────────────────
        new_rp = self.recovery_manager.process_probabilities(probs, robot_x, robot_y)
        if new_rp:
            self.get_logger().info(
                f'Recovery point rank={new_rp.rank} at '
                f'({robot_x:.2f}, {robot_y:.2f})'
            )

        # 2. Update Bayesian safety grid ──────────────────────────────────
        if open_count > 0:
            safety_level = 1.0   # any path open → safe
        elif is_dead_end:
            safety_level = 0.0   # all blocked → dead end
        else:
            safety_level = None  # uncertain — skip grid update

        if safety_level is not None:
            self._update_explored_grid(robot_x, robot_y, safety_level, current_time)

        # 3. Publish cost_layer (sector visualisation) ────────────────────
        sector_markers = self._build_sector_markers(
            robot_x, robot_y, robot_yaw, path_binary, frame_id, current_time
        )
        self.cost_layer_pub.publish(sector_markers)

        # 4. Publish recovery points ──────────────────────────────────────
        self._publish_recovery_points()

        # 5. Publish dram_exploration_map ─────────────────────────────────
        self._publish_exploration_map(frame_id)

        # Clean old cost history
        cutoff = current_time - self.max_history_age
        self.cost_history = {k: v for k, v in self.cost_history.items()
                             if v['timestamp'] > cutoff}

    # ─────────────────────────────────────────────────────────────────────
    # Bayesian grid helpers
    # ─────────────────────────────────────────────────────────────────────

    def _update_explored_grid(self, robot_x, robot_y, safety_level, timestamp):
        for dx in np.arange(-self.exploration_radius,
                            self.exploration_radius + self.grid_resolution,
                            self.grid_resolution):
            for dy in np.arange(-self.exploration_radius,
                                self.exploration_radius + self.grid_resolution,
                                self.grid_resolution):
                dist = math.hypot(dx, dy)
                if dist > self.exploration_radius:
                    continue

                wx = robot_x + dx
                wy = robot_y + dy
                if not self._is_navigable(wx, wy):
                    continue

                grid_pos = (int(math.floor(wx / self.grid_resolution)),
                            int(math.floor(wy / self.grid_resolution)))

                distance_weight = max(0.1, 1.0 - dist / self.exploration_radius)
                weighted_safety = safety_level * distance_weight

                if grid_pos in self.explored_grid:
                    old_safety = self.explored_grid[grid_pos]['safety']
                    # Only update the cell the robot is directly at;
                    # leave surrounding cells with their existing value
                    new_safety = weighted_safety if dist <= self.grid_resolution else old_safety
                else:
                    new_safety = weighted_safety

                self.explored_grid[grid_pos] = {
                    'safety': new_safety,
                    'timestamp': timestamp
                }

    def _is_navigable(self, wx, wy):
        """True if world point falls in a free map cell."""
        if self.current_map is None:
            return True  # no map yet — allow update

        info = self.current_map.info
        mx = int((wx - info.origin.position.x) / info.resolution)
        my = int((wy - info.origin.position.y) / info.resolution)

        for ddx in [-1, 0, 1]:
            for ddy in [-1, 0, 1]:
                cx, cy = mx + ddx, my + ddy
                if cx < 0 or cx >= info.width or cy < 0 or cy >= info.height:
                    continue
                idx = cy * info.width + cx
                if idx < len(self.current_map.data) and self.current_map.data[idx] != 0:
                    return False
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Sector visualisation (cost_layer)
    # ─────────────────────────────────────────────────────────────────────

    def _build_sector_markers(self, rx, ry, ryaw, path_binary,
                              frame_id, current_time):
        marker_array = MarkerArray()
        directions = ['front', 'left', 'right']
        angle_offsets = [0.0, math.pi / 2, -math.pi / 2]
        mid = int(1e9 * current_time) % 1000000  # unique id base

        for i, (direction, ao, binary) in enumerate(
                zip(directions, angle_offsets, path_binary)):
            central_angle = ryaw + ao
            sector_key = f'{direction}_{rx:.1f}_{ry:.1f}_{current_time}'
            self.cost_history[sector_key] = {
                'cost_value': binary, 'timestamp': current_time,
                'direction': direction,
                'robot_x': rx, 'robot_y': ry,
                'robot_yaw': ryaw, 'angle_offset': ao
            }

            m = self._make_sector_marker(
                rx, ry, central_angle, self.sector_radius,
                self.sector_angle, frame_id,
                ns=f'sector_{direction}', marker_id=i,
                is_open=(binary == 1), alpha=0.6,
                num_triangles=10, lifetime_sec=5
            )
            marker_array.markers.append(m)

        # Status sphere at robot centre
        sphere = Marker()
        sphere.header.frame_id = frame_id
        sphere.header.stamp = self.get_clock().now().to_msg()
        sphere.ns = 'status_center'
        sphere.id = 3
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = rx
        sphere.pose.position.y = ry
        sphere.pose.position.z = 0.05
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = 0.3
        sphere.scale.z = 0.1
        open_count = sum(path_binary)
        sphere.color.r = 0.0 if open_count >= 2 else (1.0 if open_count == 0 else 0.0)
        sphere.color.g = 1.0 if open_count >= 2 else (0.0 if open_count == 0 else 1.0)
        sphere.color.b = 0.0
        sphere.color.a = 1.0
        sphere.lifetime.sec = 3
        marker_array.markers.append(sphere)

        # Historical faded sectors
        faded_id = 10
        for sk, data in self.cost_history.items():
            if abs(data['timestamp'] - current_time) < 1.0:
                continue
            m = self._make_sector_marker(
                data['robot_x'], data['robot_y'],
                data['robot_yaw'] + data['angle_offset'],
                self.sector_radius * 0.8, self.sector_angle, frame_id,
                ns=f'history_{data["direction"]}', marker_id=faded_id,
                is_open=(data['cost_value'] == 1), alpha=0.25,
                num_triangles=8, lifetime_sec=20
            )
            marker_array.markers.append(m)
            faded_id += 1

        return marker_array

    def _make_sector_marker(self, cx, cy, central_angle, radius, angle_width,
                            frame_id, ns, marker_id, is_open, alpha,
                            num_triangles, lifetime_sec):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = marker_id
        m.type = Marker.TRIANGLE_LIST
        m.action = Marker.ADD
        m.pose.position.z = 0.1
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color.r = 0.0 if is_open else 1.0
        m.color.g = 1.0 if is_open else 0.0
        m.color.b = 0.0
        m.color.a = alpha
        m.lifetime.sec = lifetime_sec

        a_start = central_angle - angle_width / 2
        a_end = central_angle + angle_width / 2
        center = Point()
        center.x = cx
        center.y = cy
        center.z = 0.1

        for j in range(num_triangles):
            a1 = a_start + (a_end - a_start) * j / num_triangles
            a2 = a_start + (a_end - a_start) * (j + 1) / num_triangles
            p1 = Point()
            p1.x = cx + radius * math.cos(a1)
            p1.y = cy + radius * math.sin(a1)
            p1.z = 0.1
            p2 = Point()
            p2.x = cx + radius * math.cos(a2)
            p2.y = cy + radius * math.sin(a2)
            p2.z = 0.1
            m.points.extend([center, p1, p2])

        return m

    # ─────────────────────────────────────────────────────────────────────
    # Recovery points publishing
    # ─────────────────────────────────────────────────────────────────────

    def _publish_recovery_points(self):
        points = self.recovery_manager.get_all_points()
        if not points:
            return

        data = []
        for rp in points:
            rp_type = 1.0 if rp.rank >= 2 else 2.0
            data.extend([rp_type, float(rp.x), float(rp.y)])

        msg = Float32MultiArray()
        msg.data = data
        self.recovery_points_pub.publish(msg)

    # ─────────────────────────────────────────────────────────────────────
    # Exploration heatmap publishing (/dram_exploration_map)
    # ─────────────────────────────────────────────────────────────────────

    def _publish_exploration_map(self, frame_id):
        marker_array = MarkerArray()
        marker_id = 0

        # ── Heatmap POINTS marker ─────────────────────────────────────────
        # goal_generator reads this: ns="exploration_heatmap", type=POINTS
        if self.explored_grid:
            hm = Marker()
            hm.header.frame_id = frame_id
            hm.header.stamp = self.get_clock().now().to_msg()
            hm.ns = 'exploration_heatmap'       # ← goal_generator reads this exact ns
            hm.id = marker_id
            hm.type = Marker.POINTS
            hm.action = Marker.ADD
            hm.scale.x = self.grid_resolution * 1.5
            hm.scale.y = self.grid_resolution * 1.5
            hm.pose.orientation.w = 1.0
            hm.lifetime.sec = 0  # persistent

            for (gx, gy), data in self.explored_grid.items():
                wx = (gx + 0.5) * self.grid_resolution
                wy = (gy + 0.5) * self.grid_resolution
                p = Point()
                p.x = wx
                p.y = wy
                p.z = 0.03
                hm.points.append(p)

                safety = data['safety']
                c = ColorRGBA()
                c.r = 0.0 if safety >= 0.5 else 1.0
                c.g = 1.0 if safety >= 0.5 else 0.0
                c.b = 0.0
                c.a = 0.9
                hm.colors.append(c)

            if hm.points:
                marker_array.markers.append(hm)
            marker_id += 1

        # ── Recovery point cylinders + labels ────────────────────────────
        for i, rp in enumerate(self.recovery_manager.get_all_points()):
            # Cylinder
            pin = Marker()
            pin.header.frame_id = frame_id
            pin.header.stamp = self.get_clock().now().to_msg()
            pin.ns = 'recovery_pins'
            pin.id = marker_id
            pin.type = Marker.CYLINDER
            pin.action = Marker.ADD
            pin.pose.position.x = rp.x
            pin.pose.position.y = rp.y
            pin.pose.position.z = 0.1
            pin.pose.orientation.w = 1.0
            pin.scale.x = pin.scale.y = 0.3
            pin.scale.z = 0.2
            # colour by rank: purple=3, dark blue=2, light blue=1
            if rp.rank >= 3:
                pin.color.r, pin.color.g, pin.color.b = 0.5, 0.0, 1.0
            elif rp.rank == 2:
                pin.color.r, pin.color.g, pin.color.b = 0.0, 0.0, 0.8
            else:
                pin.color.r, pin.color.g, pin.color.b = 0.3, 0.6, 1.0
            pin.color.a = 0.9
            pin.lifetime.sec = 0
            marker_array.markers.append(pin)
            marker_id += 1

            # Text label
            txt = Marker()
            txt.header.frame_id = frame_id
            txt.header.stamp = self.get_clock().now().to_msg()
            txt.ns = 'recovery_labels'
            txt.id = marker_id
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = rp.x
            txt.pose.position.y = rp.y
            txt.pose.position.z = 0.35
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.15
            txt.text = f'{rp.rank} open'
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.lifetime.sec = 0
            marker_array.markers.append(txt)
            marker_id += 1

        self.risk_map_pub.publish(marker_array)

    # ─────────────────────────────────────────────────────────────────────
    # TF helper
    # ─────────────────────────────────────────────────────────────────────

    def _get_robot_position(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'body', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            return (
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.rotation.z,
                'map'
            )
        except Exception:
            return (0.0, 0.0, 0.0, 'body')


def main(args=None):
    rclpy.init(args=args)
    node = DRaMRiskMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
