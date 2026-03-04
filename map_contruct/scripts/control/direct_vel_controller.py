#!/usr/bin/env python3
"""
DR.Nav DWA Controller
=====================
DR.Nav local planner: DWA + DRAM Bayesian risk costmap + dead-end recovery.

Compared to the baseline DWA:
  - Obstacle cost from DRAM /local_costmap (Bayesian dead-end predictions)
    instead of laser scan — lets the model "see" dead ends early.
  - Dead-end detection from consecutive /dead_end_detection/path_status.
  - Recovery: navigate to the best saved recovery point from
    /dead_end_detection/recovery_points, then penalise the dead-end
    direction in DWA scoring so the robot explores other routes.

Topics
------
  Subscribes:
    /global_path                        nav_msgs/Path
    /goal_pose                          PoseStamped
    /local_costmap                      nav_msgs/OccupancyGrid  (from dram_risk_map)
    /dead_end_detection/path_status     Float32MultiArray  [F, L, R]
    /dead_end_detection/recovery_points Float32MultiArray  [type, x, y, ...]
    /scan                               LaserScan  (proximity safety only)
  Publishes:
    /cmd_vel                            Twist
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from tf2_ros import TransformListener, Buffer


class DrNavDWAController(Node):
    def __init__(self):
        super().__init__('dr_nav_dwa_controller')

        # ── DWA parameters ───────────────────────────────────────────────────
        self.max_speed        = 1.0
        self.min_speed        = 0.2
        self.max_omega        = 1.0
        self.max_accel        = 0.5
        self.max_delta_yaw    = 1.0
        self.dt               = 0.1
        self.predict_time     = 2.0
        self.v_resolution     = 0.05
        self.omega_resolution = 0.1

        # DWA cost weights
        self.w_heading  = 0.15
        self.w_dist     = 1.0
        self.w_vel      = 0.1
        self.penalty_w  = 2.0    # dead-end direction penalty weight

        # ── Navigation parameters ────────────────────────────────────────────
        self.goal_tol        = 0.5
        self.carrot_dist     = 1.5
        self.prox_stop_dist  = 0.4
        self.blocked_thr     = 0.56
        self.consecutive_thr = 5
        self.robot_radius    = 0.5   # inflate obstacle check — increase to make robot more conservative

        # ── State ─────────────────────────────────────────────────────────────
        self.nav_state           = 'navigating'  # 'navigating' | 'recovering'
        self.consecutive_blocked = 0
        self.current_goal        = None
        self.original_goal       = None
        self.global_path         = []
        self.dram_costmap        = None   # OccupancyGrid from dram_risk_map
        self.recovery_points     = []     # list of (x, y)
        self.dead_end_yaw        = None   # robot heading when dead end was confirmed
        self.current_v           = 0.0
        self.current_omega       = 0.0
        self.front_min_range     = 999.0

        # TF
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        # self.create_subscription(Path,             '/global_path',
        #                          self._path_cb,            10)
        self.create_subscription(PoseStamped,      '/goal_pose',
                                 self._goal_cb,            10)
        self.create_subscription(OccupancyGrid,    '/local_costmap',
                                 self._costmap_cb,         10)
        self.create_subscription(Float32MultiArray,
                                 '/dead_end_detection/path_status',
                                 self._path_status_cb,     10)
        self.create_subscription(Float32MultiArray,
                                 '/dead_end_detection/recovery_points',
                                 self._recovery_points_cb, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info('DR.Nav DWA Controller initialized')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    # def _path_cb(self, msg: Path):
    #     self.global_path = [(p.pose.position.x, p.pose.position.y)
    #                         for p in msg.poses]

    def _goal_cb(self, msg: PoseStamped):
        self.current_goal        = (msg.pose.position.x, msg.pose.position.y)
        self.original_goal       = None
        self.nav_state           = 'navigating'
        self.consecutive_blocked = 0
        self.dead_end_yaw        = None
        self.get_logger().info(
            f'Goal: ({self.current_goal[0]:.2f}, {self.current_goal[1]:.2f})')

    def _costmap_cb(self, msg: OccupancyGrid):
        self.dram_costmap = msg

    def _path_status_cb(self, msg: Float32MultiArray):
        if len(msg.data) < 3 or self.nav_state != 'navigating':
            return
        F, L, R = msg.data[0], msg.data[1], msg.data[2]
        all_blocked = (F < self.blocked_thr and
                       L < self.blocked_thr and
                       R < self.blocked_thr)
        if all_blocked:
            self.consecutive_blocked += 1
        else:
            self.consecutive_blocked = 0

    def _recovery_points_cb(self, msg: Float32MultiArray):
        data = list(msg.data)
        pts  = []
        i = 0
        while i + 2 < len(data):
            pts.append((float(data[i + 1]), float(data[i + 2])))
            i += 3
        self.recovery_points = pts

    def _scan_cb(self, msg: LaserScan):
        front = [r for idx, r in enumerate(msg.ranges)
                 if (abs(msg.angle_min + idx * msg.angle_increment) < math.pi / 6
                     and msg.range_min < r < msg.range_max)]
        self.front_min_range = min(front) if front else 999.0

    # ── Main control loop ──────────────────────────────────────────────────────

    def _control_loop(self):
        rx, ry, ryaw, ok = self._get_robot_pose()
        if not ok or self.current_goal is None:
            self.cmd_pub.publish(Twist())
            return

        # Proximity hard stop
        if self.front_min_range < self.prox_stop_dist:
            self.cmd_pub.publish(Twist())
            return

        gx, gy = self.current_goal
        dist_to_goal = math.hypot(gx - rx, gy - ry)

        # Goal / recovery point reached
        if dist_to_goal < self.goal_tol:
            if self.nav_state == 'recovering':
                self.get_logger().info(
                    'Reached recovery point. Resuming toward original goal.')
                self.current_goal        = self.original_goal
                self.original_goal       = None
                self.nav_state           = 'navigating'
                self.consecutive_blocked = 0
                # dead_end_yaw persists → penalisation continues
            else:
                self.get_logger().info('Goal reached.')
                self.current_goal = None
                self.dead_end_yaw = None
            self.cmd_pub.publish(Twist())
            return

        # Dead-end confirmation → trigger recovery
        if (self.nav_state == 'navigating' and
                self.consecutive_blocked >= self.consecutive_thr):
            self.dead_end_yaw = ryaw  # direction robot was heading into dead end
            target = self._best_recovery_point(rx, ry)
            if target is None:
                # Fallback: 1.5 m directly behind current heading
                target = (rx - 1.5 * math.cos(ryaw),
                          ry - 1.5 * math.sin(ryaw))
            self.get_logger().warn(
                f'Dead end confirmed (×{self.consecutive_blocked}). '
                f'Recovering to ({target[0]:.2f}, {target[1]:.2f})')
            self.original_goal       = self.current_goal
            self.current_goal        = target
            self.nav_state           = 'recovering'
            self.consecutive_blocked = 0
            gx, gy = self.current_goal

        # DWA — use goal directly as carrot (no global path)
        best_v, best_omega = self._dwa(rx, ry, ryaw, (gx, gy))

        cmd = Twist()
        cmd.linear.x  = best_v
        cmd.angular.z = best_omega
        self.cmd_pub.publish(cmd)

        self.current_v     = best_v
        self.current_omega = best_omega

    # ── Carrot extraction ──────────────────────────────────────────────────────

    def _get_carrot(self, rx, ry, gx, gy):
        # During recovery go straight to recovery point
        if self.nav_state == 'recovering' or not self.global_path:
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

    # ── DWA ───────────────────────────────────────────────────────────────────

    def _dwa(self, rx, ry, ryaw, carrot):
        v_min  = max(self.min_speed,  self.current_v     - self.max_accel     * self.dt)
        v_max  = min(self.max_speed,  self.current_v     + self.max_accel     * self.dt)
        om_min = max(-self.max_omega, self.current_omega - self.max_delta_yaw * self.dt)
        om_max = min(self.max_omega,  self.current_omega + self.max_delta_yaw * self.dt)

        best_cost         = float('inf')
        best_v, best_omega = 0.0, 0.0

        v_steps  = max(1, int(round((v_max  - v_min)  / self.v_resolution)))
        om_steps = max(1, int(round((om_max - om_min) / self.omega_resolution)))

        for iv in range(v_steps + 1):
            v = v_min + iv * (v_max - v_min) / v_steps
            for iw in range(om_steps + 1):
                omega = om_min + iw * (om_max - om_min) / om_steps
                traj  = self._predict(rx, ry, ryaw, v, omega)
                if traj is None:
                    continue

                goal_cost  = self._heading_cost(traj, carrot)
                obs_cost   = self._obstacle_cost(traj)
                speed_cost = self.max_speed - traj[-1, 3]

                # Penalise trajectories heading back toward the dead-end direction
                penalty = 0.0
                if self.dead_end_yaw is not None:
                    final_heading = traj[-1, 2]
                    delta = math.atan2(
                        math.sin(final_heading - self.dead_end_yaw),
                        math.cos(final_heading - self.dead_end_yaw))
                    penalty = self.penalty_w * max(0.0, math.cos(delta))

                cost = (self.w_heading * goal_cost +
                        self.w_dist   * obs_cost   +
                        self.w_vel    * speed_cost +
                        penalty)

                if cost < best_cost:
                    best_cost  = cost
                    best_v     = v
                    best_omega = omega

        # Anti-freeze: if no valid trajectory found, rotate in place
        if best_cost == float('inf'):
            best_v, best_omega = 0.0, self.max_omega * 0.5

        return best_v, best_omega

    def _predict(self, x, y, theta, v, omega):
        """Simulate trajectory; returns np.array shape (steps, 4) [x,y,θ,v]."""
        steps = int(self.predict_time / self.dt)
        traj  = np.zeros((steps, 4))
        for i in range(steps):
            x     += v * math.cos(theta) * self.dt
            y     += v * math.sin(theta) * self.dt
            theta += omega * self.dt
            traj[i] = [x, y, theta, v]
        return traj

    # ── Cost functions ─────────────────────────────────────────────────────────

    def _heading_cost(self, traj, carrot):
        fx, fy, ftheta = traj[-1, 0], traj[-1, 1], traj[-1, 2]
        dx, dy         = carrot[0] - fx, carrot[1] - fy
        error_angle    = math.atan2(dy, dx)
        cost_angle     = error_angle - ftheta
        return abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    def _obstacle_cost(self, traj):
        """Max DRAM risk score along the trajectory, normalised to [0, 1].
        Samples a circle of robot_radius around each trajectory point so the
        robot treats nearby high-cost cells as obstacles (inflation in DWA space).
        """
        if self.dram_costmap is None:
            return 0.0
        res = self.dram_costmap.info.resolution
        n   = max(1, int(self.robot_radius / res))
        offsets = [(dx * res, dy * res)
                   for dx in range(-n, n + 1)
                   for dy in range(-n, n + 1)
                   if math.hypot(dx, dy) <= n]
        max_val = 0
        for row in traj:
            for ox, oy in offsets:
                c = self._dram_cost_at(row[0] + ox, row[1] + oy)
                if c > max_val:
                    max_val = c
        return max_val / 100.0

    def _dram_cost_at(self, wx, wy):
        """Return DRAM OccupancyGrid value at world (wx, wy).
        Returns 0 for unknown or out-of-bounds (no penalty for unexplored areas).
        """
        if self.dram_costmap is None:
            return 0
        info = self.dram_costmap.info
        col  = int((wx - info.origin.position.x) / info.resolution)
        row  = int((wy - info.origin.position.y) / info.resolution)
        if col < 0 or col >= info.width or row < 0 or row >= info.height:
            return 0
        idx = row * info.width + col
        if idx >= len(self.dram_costmap.data):
            return 0
        val = int(self.dram_costmap.data[idx])
        return max(0, val)  # treat -1 (unknown) as 0

    # ── Recovery helpers ───────────────────────────────────────────────────────

    def _best_recovery_point(self, rx, ry):
        """Return the closest saved recovery point, or None if none available."""
        if not self.recovery_points:
            return None
        return min(self.recovery_points,
                   key=lambda p: math.hypot(p[0] - rx, p[1] - ry))

    # ── TF helper ─────────────────────────────────────────────────────────────

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


def main(args=None):
    rclpy.init(args=args)
    node = DrNavDWAController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
