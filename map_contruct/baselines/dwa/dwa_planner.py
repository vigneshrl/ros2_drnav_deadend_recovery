#!/usr/bin/env python3
"""
DWA Planner — baseline
=======================
True Dynamic Window Approach following the reference (Atsushi Sakai et al.).

Key differences from a naive velocity sampler
----------------------------------------------
  - Dynamic window: constrains (v, omega) to what is physically reachable from
    the current velocity given acceleration limits (max_accel, max_delta_yaw).
  - Obstacle cost: 1 / min_obstacle_dist  (smooth, not binary).
  - Goal cost: heading-angle error to goal  (from reference calc_to_goal_cost).
  - Speed cost: max_speed − final_speed    (reward higher speed).

Global path integration
-----------------------
  Subscribes to /global_path (nav_msgs/Path) published by global_planner.py.
  Extracts a carrot point 1.5 m ahead on the path as the local goal.
  Falls back to the original goal if the path is empty.

Local costmap
-------------
  Publishes /local_costmap (nav_msgs/OccupancyGrid) built from the laser scan:
  obstacle cells marked 100, inflation ring 50, free cells 0.
  Used by RViz2 to show the local obstacle picture for each method.

Topics
------
  Subscribes:
    /global_path      nav_msgs/Path          from global_planner
    /goal_pose        PoseStamped            original goal (fallback)
    /map              nav_msgs/OccupancyGrid for global collision
    /scan             LaserScan              for local costmap + obstacle cost
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
import tf2_geometry_msgs  # noqa: F401 — registers PoseStamped transform support
from scipy.ndimage import binary_dilation


class DwaPlannerNode(Node):
    def __init__(self):
        super().__init__('dwa_planner_node')

        # ── Robot / DWA parameters (from reference) ──────────────────────────
        self.max_speed          = 1.0     # [m/s]
        self.min_speed          = 0.0     # [m/s]  allow stop/rotate-in-place
        self.max_omega          = 1.0     # [rad/s]
        self.max_accel          = 0.5     # [m/s²]  higher → faster ramp-up
        self.max_delta_yaw      = 10.0    # [rad/s²] — effectively unconstrained so full omega available immediately
        self.robot_radius       = 0.5    # [m] for collision check
        self.dt                 = 0.1     # [s] control timestep
        self.predict_time       = 2.0     # [s] trajectory simulation horizon
        # DWA scoring gains (reference values — to_goal_gain deliberately small
        # so obstacle avoidance is not overwhelmed by heading error)
        self.to_goal_gain       = 1.5     # heading-error cost weight
        self.speed_gain         = 2.0     # reward forward speed
        self.obstacle_gain      = 1.0     # obstacle cost weight
        self.robot_stuck_flag   = 0.001   # minimum v/omega to prevent freeze
        # Velocity sampling resolution inside dynamic window
        self.v_samples          = 6       # number of v samples in window
        self.omega_samples      = 15      # number of omega samples in window

        # ── Navigation parameters ─────────────────────────────────────────────
        self.goal_tol           = 0.5     # [m] goal-reached tolerance
        self.carrot_dist        = 1.5     # [m] lookahead on A* path
        self.prox_stop_dist     = 0.4     # [m] hard stop if obstacle this close in front
        self.inflation_radius   = 0.5    # [m] obstacle inflation for map checks

        # ── Local costmap parameters ──────────────────────────────────────────
        self.lc_size            = 10.0    # [m] local costmap window side length
        self.lc_resolution      = 0.1     # [m/cell]

        # ── State ────────────────────────────────────────────────────────────
        self.current_goal       = None    # (x, y) in map frame
        self.global_path        = []      # list of (x, y) from global_planner
        self.occupancy_grid     = None
        self.inflated_mask      = None
        self.last_scan          = None
        self.front_min_range    = 999.0
        self.current_v          = 0.0     # tracks actual velocity for dynamic window
        self.current_omega      = 0.0

        # Metrics
        self.metrics = {
            'method_name': 'dwa',
            'start_time': time.time(),
            'total_distance': 0.0,
            'completion_time': 0.0,
        }
        self.last_pose      = None
        self.output_dir     = self._create_output_dir()

        # TF
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        # self.create_subscription(Path,          '/global_path', self._path_cb,  10)
        self.create_subscription(PoseStamped,   '/goal_pose',   self._goal_cb,  10)
        self.create_subscription(OccupancyGrid, '/map',         self._map_cb,   10)
        self.create_subscription(LaserScan,     '/scan',        self._scan_cb,  10)

        # Publishers
        self.cmd_pub      = self.create_publisher(Twist,          '/cmd_vel',       10)
        self.lc_pub       = self.create_publisher(OccupancyGrid,  '/local_costmap', 10)

        self.create_timer(0.1, self.plan_and_publish)
        self.get_logger().info('DWA Planner initialized (true dynamic window)')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    # def _path_cb(self, msg: Path):
    #     self.global_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def _goal_cb(self, msg: PoseStamped):
        self.current_goal  = (msg.pose.position.x, msg.pose.position.y)
        self.current_v     = 0.0
        self.current_omega = 0.0
        self.get_logger().info(f'Goal (map): ({self.current_goal[0]:.2f}, {self.current_goal[1]:.2f})')

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

    # ── Main control loop ──────────────────────────────────────────────────────

    def plan_and_publish(self):
        robot_x, robot_y, robot_theta, ok = self._get_robot_pose()
        if not ok:
            return

        if self.current_goal is None:
            self.cmd_pub.publish(Twist())
            return

        # Hard stop if wall too close in front
        if self.front_min_range < self.prox_stop_dist:
            self.get_logger().warn(
                f'PROX STOP: front={self.front_min_range:.2f} m < {self.prox_stop_dist} m',
                throttle_duration_sec=1.0)
            self.cmd_pub.publish(Twist())
            self.current_v = 0.0
            return

        gx, gy = self.current_goal
        dist_to_goal = np.hypot(robot_x - gx, robot_y - gy)

        if dist_to_goal < self.goal_tol:
            self.get_logger().info('Goal reached')
            self.current_goal = None
            self.cmd_pub.publish(Twist())
            self.current_v = 0.0
            return

        # Extract carrot point from A* path (1.5 m lookahead)
        # local_gx, local_gy = self._get_carrot(robot_x, robot_y, gx, gy)

        # Pre-compute scan obstacle world positions once per cycle
        obstacle_pts = self._scan_to_world(robot_x, robot_y, robot_theta)

        # True DWA planning
        best_v, best_omega = self._dwa_plan(
            robot_x, robot_y, robot_theta, gx, gy, obstacle_pts)

        # Update tracked velocities
        self.current_v     = best_v
        self.current_omega = best_omega

        cmd = Twist()
        cmd.linear.x  = best_v
        cmd.angular.z = best_omega
        self.cmd_pub.publish(cmd)

        # Publish local costmap
        if self.last_scan is not None:
            self._publish_local_costmap(robot_x, robot_y, robot_theta)

        self._update_metrics(robot_x, robot_y)
        self.last_pose = (robot_x, robot_y)

    # ── Carrot extraction ─────────────────────────────────────────────────────

    def _get_carrot(self, rx, ry, gx, gy):
        """Find the point on the global path 1.5 m ahead of the robot."""
        if not self.global_path:
            return gx, gy

        # Find nearest path point
        min_d, nearest_i = float('inf'), 0
        for i, (wx, wy) in enumerate(self.global_path):
            d = math.hypot(rx - wx, ry - wy)
            if d < min_d:
                min_d, nearest_i = d, i

        # Walk forward along path until carrot_dist accumulated
        accumulated = 0.0
        for i in range(nearest_i, len(self.global_path) - 1):
            wx0, wy0 = self.global_path[i]
            wx1, wy1 = self.global_path[i + 1]
            seg = math.hypot(wx1 - wx0, wy1 - wy0)
            if accumulated + seg >= self.carrot_dist:
                t = (self.carrot_dist - accumulated) / seg
                return wx0 + t * (wx1 - wx0), wy0 + t * (wy1 - wy0)
            accumulated += seg

        # End of path — return goal directly
        return gx, gy

    # ── True DWA ──────────────────────────────────────────────────────────────

    def _calc_dynamic_window(self):
        """Compute the dynamic window: velocity range reachable in one dt step."""
        v_min  = max(self.min_speed,  self.current_v     - self.max_accel    * self.dt)
        v_max  = min(self.max_speed,  self.current_v     + self.max_accel    * self.dt)
        om_min = max(-self.max_omega, self.current_omega - self.max_delta_yaw * self.dt)
        om_max = min( self.max_omega, self.current_omega + self.max_delta_yaw * self.dt)
        return v_min, v_max, om_min, om_max

    def _scan_to_world(self, rx, ry, rtheta):
        """Convert current scan to obstacle (x, y) positions in world frame."""
        pts = []
        if self.last_scan is None:
            return pts
        scan = self.last_scan
        for i, r in enumerate(scan.ranges):
            if not (scan.range_min < r < scan.range_max):
                continue
            ang = scan.angle_min + i * scan.angle_increment + rtheta
            pts.append((rx + r * math.cos(ang), ry + r * math.sin(ang)))
        return pts

    def _dwa_plan(self, rx, ry, rtheta, gx, gy, obstacle_pts):
        """
        Sample velocities within the dynamic window, simulate trajectories,
        score each, and return the best (v, omega).
        """
        v_min, v_max, om_min, om_max = self._calc_dynamic_window()

        v_samples  = np.linspace(v_min,  v_max,  self.v_samples)
        om_samples = np.linspace(om_min, om_max, self.omega_samples)

        best_cost = float('inf')
        best_v, best_omega = 0.0, 0.0

        for v in v_samples:
            for omega in om_samples:
                traj = self._predict_trajectory(rx, ry, rtheta, v, omega)

                # Obstacle cost (collision → inf)
                ob_cost = self._obstacle_cost(traj, obstacle_pts)
                if ob_cost == float('inf'):
                    continue

                # Goal cost: heading error at trajectory endpoint (reference formula)
                goal_cost = self._to_goal_cost(traj, gx, gy)

                # Speed cost: reward high speed
                speed_cost = self.max_speed - traj[-1, 3]

                total = (self.to_goal_gain   * goal_cost +
                         self.speed_gain     * speed_cost +
                         self.obstacle_gain  * ob_cost)

                if total < best_cost:
                    best_cost  = total
                    best_v     = v
                    best_omega = omega

        # Anti-freeze: if no valid trajectory, stop and spin
        if best_cost == float('inf'):
            self.get_logger().warn('DWA: all trajectories blocked — forcing turn',
                                   throttle_duration_sec=1.0)
            best_v     = 0.0
            best_omega = self.max_omega
        elif abs(best_v) < self.robot_stuck_flag and abs(best_omega) < self.robot_stuck_flag:
            best_v     = 0.0
            best_omega = self.max_omega

        self.get_logger().debug(
            f'DWA: v={best_v:.3f} ω={best_omega:.3f}  '
            f'dyn_win=[{v_min:.3f},{v_max:.3f}]  '
            f'obstacles={len(obstacle_pts)}',
            throttle_duration_sec=0.5)

        return best_v, best_omega

    def _predict_trajectory(self, x, y, theta, v, omega):
        """Simulate robot motion for predict_time seconds. Returns array [x,y,θ,v,ω]."""
        state = np.array([x, y, theta, v, omega])
        traj  = [state.copy()]
        t     = 0.0
        while t <= self.predict_time:
            state[2] += omega * self.dt
            state[0] += v * math.cos(state[2]) * self.dt
            state[1] += v * math.sin(state[2]) * self.dt
            state[3]  = v
            state[4]  = omega
            traj.append(state.copy())
            t += self.dt
        return np.array(traj)

    def _to_goal_cost(self, traj, gx, gy):
        """Heading-angle error to goal at trajectory endpoint (reference formula)."""
        dx = gx - traj[-1, 0]
        dy = gy - traj[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle  = error_angle - traj[-1, 2]
        return abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    def _obstacle_cost(self, traj, obstacle_pts):
        """
        1 / min_obstacle_distance (reference formula).
        Returns inf if any moving trajectory point collides.
        obstacle_pts: pre-computed obstacle world positions from _scan_to_world().

        NOTE: scan-based distance is computed for ALL states (including v=0) so
        that rotation-in-place is not artificially free — it pays the same 1/dist
        cost as forward motion, preventing the DWA from preferring v=0.
        """
        if not obstacle_pts:
            return 0.0

        min_r = float('inf')
        for i, state in enumerate(traj):
            x, y = state[0], state[1]
            moving = state[3] >= 1e-9

            if i == 0:
                # Current position: only track distance for continuous cost,
                # never hard-reject (robot is already here).
                for ox, oy in obstacle_pts:
                    d = math.hypot(x - ox, y - oy)
                    if d < min_r:
                        min_r = d
                continue

            if moving:
                # Moving state: hard collision check
                if self._is_occupied(x, y):
                    return float('inf')
                for ox, oy in obstacle_pts:
                    d = math.hypot(x - ox, y - oy)
                    if d < self.robot_radius + 0.1:   # +0.1 m safety clearance
                        return float('inf')
                    if d < min_r:
                        min_r = d
            else:
                # v=0 (spinning in place): robot doesn't move, no new collision risk.
                # Only track min_r for continuous cost — do NOT hard-reject because
                # robot is already at this position (close to wall is expected).
                for ox, oy in obstacle_pts:
                    d = math.hypot(x - ox, y - oy)
                    if d < min_r:
                        min_r = d

        return 1.0 / min_r

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
        """
        Build a rolling OccupancyGrid around the robot from the laser scan.
        Obstacle cells = 100, inflation ring = 50, free = 0.
        """
        scan = self.last_scan
        n    = int(self.lc_size / self.lc_resolution)
        grid = np.zeros((n, n), dtype=np.int8)

        infl_cells = int(math.ceil(self.robot_radius / self.lc_resolution))

        for i, r in enumerate(scan.ranges):
            if not (scan.range_min < r < scan.range_max):
                continue
            ang  = scan.angle_min + i * scan.angle_increment + rtheta
            ox   = rx + r * math.cos(ang)
            oy   = ry + r * math.sin(ang)
            # Convert obstacle world pos → local grid index (centred at robot)
            lgx  = int((ox - rx + self.lc_size / 2) / self.lc_resolution)
            lgy  = int((oy - ry + self.lc_size / 2) / self.lc_resolution)
            if not (0 <= lgx < n and 0 <= lgy < n):
                continue
            grid[lgy, lgx] = 100
            # Inflate
            for dx in range(-infl_cells, infl_cells + 1):
                for dy in range(-infl_cells, infl_cells + 1):
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
        d = f"dwa_metrics_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(d, exist_ok=True)
        return d

    def save_metrics(self):
        path = os.path.join(self.output_dir, 'dwa_metrics.json')
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.get_logger().info(f'Metrics saved to {path}')


def main(args=None):
    rclpy.init(args=args)
    node = DwaPlannerNode()
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
