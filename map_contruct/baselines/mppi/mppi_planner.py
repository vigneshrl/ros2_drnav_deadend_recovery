#!/usr/bin/env python3
"""
MPPI Planner — baseline
========================
Model Predictive Path Integral control following the information-theoretic
formulation (G. Williams et al.).

Corrected from the previous point-to-point version:
  - Tracks the A* global path (nearest waypoint stage cost).
  - Stage cost = position error to nearest path point + heading alignment.
  - Terminal cost = goal distance.
  - Collision cost = 1e6 per step (large, from reference).

Global path integration
-----------------------
  Subscribes to /global_path (nav_msgs/Path).
  Carrot point 1.5 m ahead used as the immediate tracking target.

Local costmap
-------------
  Publishes /local_costmap (nav_msgs/OccupancyGrid) from laser scan.
  Same format as DWA for fair RViz2 comparison.

Topics
------
  Subscribes:
    /global_path      nav_msgs/Path
    /goal_pose        PoseStamped
    /map              nav_msgs/OccupancyGrid
    /scan             LaserScan
  Publishes:
    /cmd_vel          Twist
    /local_costmap    nav_msgs/OccupancyGrid
"""

import math
import time
import json
import os

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformListener, Buffer
from scipy.ndimage import binary_dilation


class MppiPlannerNode(Node):
    def __init__(self):
        super().__init__('mppi_planner_node')

        # ── MPPI parameters (from reference) ─────────────────────────────────
        self.max_speed      = 0.5     # [m/s]
        self.max_omega      = 1.0     # [rad/s]
        self.dt             = 0.1     # [s]
        self.horizon        = 20      # time steps
        self.num_samples    = 200     # sampled trajectories (K)
        self.param_lambda   = 10.0    # temperature λ
        self.sigma_v        = 0.15    # linear velocity noise std
        self.sigma_omega    = 0.4     # angular velocity noise std

        # Stage cost weights [pos_x, pos_y, heading, speed]
        self.w_pos          = 2.0     # position error to path waypoint
        self.w_heading      = 0.5     # heading alignment
        self.collision_cost = 1.0e5   # large per-step collision penalty

        # ── Navigation parameters ─────────────────────────────────────────────
        self.goal_tol        = 0.5
        self.carrot_dist     = 1.5
        self.prox_stop_dist  = 0.4
        self.inflation_radius = 0.35

        # ── Local costmap parameters ──────────────────────────────────────────
        self.lc_size       = 10.0
        self.lc_resolution = 0.1

        # ── State ────────────────────────────────────────────────────────────
        self.current_goal   = None
        self.global_path    = []
        self.occupancy_grid = None
        self.inflated_mask  = None
        self.last_scan      = None
        self.front_min_range = 999.0
        # Rolling control sequence [v, omega] × horizon
        self.control_sequence = np.zeros((self.horizon, 2))

        # Metrics
        self.metrics = {
            'method_name': 'mppi',
            'start_time': time.time(),
            'total_distance': 0.0,
            'completion_time': 0.0,
        }
        self.last_pose  = None
        self.output_dir = self._create_output_dir()

        # TF
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(Path,          '/global_path', self._path_cb, 10)
        self.create_subscription(PoseStamped,   '/goal_pose',   self._goal_cb, 10)
        self.create_subscription(OccupancyGrid, '/map',         self._map_cb,  10)
        self.create_subscription(LaserScan,     '/scan',        self._scan_cb, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist,         '/cmd_vel',       10)
        self.lc_pub  = self.create_publisher(OccupancyGrid, '/local_costmap', 10)

        self.create_timer(0.1, self.plan_and_publish)
        self.get_logger().info('MPPI Planner initialized (path-tracking, K=%d)' % self.num_samples)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _path_cb(self, msg: Path):
        self.global_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def _goal_cb(self, msg: PoseStamped):
        self.current_goal = (msg.pose.position.x, msg.pose.position.y)
        self.control_sequence = np.zeros((self.horizon, 2))
        self.get_logger().info(f'Goal: ({self.current_goal[0]:.2f}, {self.current_goal[1]:.2f})')

    def _map_cb(self, msg: OccupancyGrid):
        self.occupancy_grid = msg
        info  = msg.info
        raw   = np.array(msg.data, dtype=np.int8).reshape((info.height, info.width))
        cells = max(1, int(math.ceil(self.inflation_radius / info.resolution)))
        self.inflated_mask = binary_dilation(raw > 50, iterations=cells)

    def _scan_cb(self, msg: LaserScan):
        self.last_scan = msg
        front = [r for i, r in enumerate(msg.ranges)
                 if (abs(msg.angle_min + i * msg.angle_increment) < math.pi / 6
                     and msg.range_min < r < msg.range_max)]
        self.front_min_range = min(front) if front else 999.0

    # ── Main loop ─────────────────────────────────────────────────────────────

    def plan_and_publish(self):
        rx, ry, rtheta, ok = self._get_robot_pose()
        if not ok:
            return

        if self.current_goal is None:
            self.cmd_pub.publish(Twist())
            return

        if self.front_min_range < self.prox_stop_dist:
            self.cmd_pub.publish(Twist())
            return

        gx, gy = self.current_goal
        if np.hypot(rx - gx, ry - gy) < self.goal_tol:
            self.get_logger().info('Goal reached')
            self.current_goal = None
            self.cmd_pub.publish(Twist())
            return

        # Carrot point from global path
        local_gx, local_gy = self._get_carrot(rx, ry, gx, gy)

        best_v, best_omega = self._mppi_plan(rx, ry, rtheta, local_gx, local_gy, gx, gy)

        cmd = Twist()
        cmd.linear.x  = best_v
        cmd.angular.z = best_omega
        self.cmd_pub.publish(cmd)

        if self.last_scan is not None:
            self._publish_local_costmap(rx, ry, rtheta)

        self._update_metrics(rx, ry)
        self.last_pose = (rx, ry)

    # ── Carrot extraction ─────────────────────────────────────────────────────

    def _get_carrot(self, rx, ry, gx, gy):
        if not self.global_path:
            return gx, gy
        min_d, nearest_i = float('inf'), 0
        for i, (wx, wy) in enumerate(self.global_path):
            d = math.hypot(rx - wx, ry - wy)
            if d < min_d:
                min_d, nearest_i = d, i
        accumulated = 0.0
        for i in range(nearest_i, len(self.global_path) - 1):
            wx0, wy0 = self.global_path[i]
            wx1, wy1 = self.global_path[i + 1]
            seg = math.hypot(wx1 - wx0, wy1 - wy0)
            if accumulated + seg >= self.carrot_dist:
                t = (self.carrot_dist - accumulated) / seg
                return wx0 + t * (wx1 - wx0), wy0 + t * (wy1 - wy0)
            accumulated += seg
        return gx, gy

    # ── MPPI ──────────────────────────────────────────────────────────────────

    def _mppi_plan(self, rx, ry, rtheta, local_gx, local_gy, final_gx, final_gy):
        """
        Information-theoretic MPPI.
        Stage cost: position error to carrot + heading.
        Terminal cost: distance to final goal.
        """
        K = self.num_samples
        T = self.horizon

        # Sample noise — shape (K, T)
        eps_v     = np.random.normal(0, self.sigma_v,     (K, T))
        eps_omega = np.random.normal(0, self.sigma_omega, (K, T))

        base   = self.control_sequence.copy()
        costs  = np.zeros(K)
        # Store perturbed sequences for weighted update
        v_seqs = np.zeros((K, T))
        w_seqs = np.zeros((K, T))

        for k in range(K):
            v_seq = np.clip(base[:, 0] + eps_v[k],     0.0,              self.max_speed)
            w_seq = np.clip(base[:, 1] + eps_omega[k], -self.max_omega,  self.max_omega)
            v_seqs[k] = v_seq
            w_seqs[k] = w_seq
            costs[k]  = self._evaluate(rx, ry, rtheta, v_seq, w_seq,
                                       local_gx, local_gy, final_gx, final_gy)

        # Information-theoretic weights (Algorithm 2 from Williams et al.)
        rho     = costs.min()
        eta     = np.sum(np.exp(-(costs - rho) / self.param_lambda))
        weights = np.exp(-(costs - rho) / self.param_lambda) / eta

        # Weighted average of perturbed sequences
        new_controls = np.zeros((T, 2))
        for k in range(K):
            new_controls[:, 0] += weights[k] * v_seqs[k]
            new_controls[:, 1] += weights[k] * w_seqs[k]

        # Shift rolling horizon
        self.control_sequence[:-1] = new_controls[1:]
        self.control_sequence[-1]  = new_controls[-1]

        return float(new_controls[0, 0]), float(new_controls[0, 1])

    def _evaluate(self, rx, ry, rtheta, v_seq, w_seq, local_gx, local_gy, gx, gy):
        x, y, theta = rx, ry, rtheta
        total = 0.0
        for t in range(self.horizon):
            x     += v_seq[t] * math.cos(theta) * self.dt
            y     += v_seq[t] * math.sin(theta) * self.dt
            theta += w_seq[t] * self.dt

            if v_seq[t] > 1e-9 and self._is_occupied(x, y):
                return self.collision_cost * (self.horizon - t)

            # Stage cost: position error to carrot + heading to carrot
            dist_to_carrot = math.hypot(x - local_gx, y - local_gy)
            heading_to_carrot = math.atan2(local_gy - y, local_gx - x)
            heading_err = abs(self._normalize(theta - heading_to_carrot))
            total += self.w_pos * dist_to_carrot + self.w_heading * heading_err

        # Terminal: distance to final goal
        total += 2.0 * self.w_pos * math.hypot(x - gx, y - gy)
        return total

    def _normalize(self, angle):
        while angle >  math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def _is_occupied(self, x, y):
        if self.inflated_mask is None or self.occupancy_grid is None:
            return False
        info = self.occupancy_grid.info
        gx   = int((x - info.origin.position.x) / info.resolution)
        gy   = int((y - info.origin.position.y) / info.resolution)
        if gx < 0 or gx >= info.width or gy < 0 or gy >= info.height:
            return False  # unknown = free (allow planning near/beyond map edge)
        return bool(self.inflated_mask[gy, gx])

    # ── Local costmap ──────────────────────────────────────────────────────────

    def _publish_local_costmap(self, rx, ry, rtheta):
        scan = self.last_scan
        n    = int(self.lc_size / self.lc_resolution)
        grid = np.zeros((n, n), dtype=np.int8)
        infl = int(math.ceil(0.35 / self.lc_resolution))

        for i, r in enumerate(scan.ranges):
            if not (scan.range_min < r < scan.range_max):
                continue
            ang = scan.angle_min + i * scan.angle_increment + rtheta
            ox  = rx + r * math.cos(ang)
            oy  = ry + r * math.sin(ang)
            lgx = int((ox - rx + self.lc_size / 2) / self.lc_resolution)
            lgy = int((oy - ry + self.lc_size / 2) / self.lc_resolution)
            if not (0 <= lgx < n and 0 <= lgy < n):
                continue
            grid[lgy, lgx] = 100
            for dx in range(-infl, infl + 1):
                for dy in range(-infl, infl + 1):
                    nx, ny = lgx + dx, lgy + dy
                    if 0 <= nx < n and 0 <= ny < n and grid[ny, nx] < 100:
                        grid[ny, nx] = 50

        msg = OccupancyGrid()
        msg.header.frame_id          = 'map'
        msg.header.stamp             = self.get_clock().now().to_msg()
        msg.info.resolution          = self.lc_resolution
        msg.info.width               = n
        msg.info.height              = n
        msg.info.origin.position.x   = rx - self.lc_size / 2
        msg.info.origin.position.y   = ry - self.lc_size / 2
        msg.info.origin.orientation.w = 1.0
        msg.data                     = grid.flatten().tolist()
        self.lc_pub.publish(msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            x     = t.transform.translation.x
            y     = t.transform.translation.y
            qz    = t.transform.rotation.z
            qw    = t.transform.rotation.w
            theta = 2.0 * math.atan2(qz, qw)
            return x, y, theta, True
        except Exception:
            return 0.0, 0.0, 0.0, False

    def _update_metrics(self, rx, ry):
        if self.last_pose is not None:
            self.metrics['total_distance'] += math.hypot(
                rx - self.last_pose[0], ry - self.last_pose[1])
        self.metrics['completion_time'] = time.time() - self.metrics['start_time']

    def _create_output_dir(self):
        d = f"mppi_metrics_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(d, exist_ok=True)
        return d

    def save_metrics(self):
        path = os.path.join(self.output_dir, 'mppi_metrics.json')
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.get_logger().info(f'Metrics saved to {path}')


def main(args=None):
    rclpy.init(args=args)
    node = MppiPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_metrics()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
