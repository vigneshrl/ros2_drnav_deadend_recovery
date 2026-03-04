#!/usr/bin/env python3
"""
Direct Velocity Controller for DR.Nav
======================================
Converts infer_vis path scores (F, L, R) directly to cmd_vel,
blended with goal-directed heading from /goal_pose.

No waypoint generation or local planner needed.
Goal is given via /goal_pose (RViz2 Nav2 Goal button).

Behaviour
---------
  Normal   : desired_heading = blend(model_heading, goal_heading)
             goal_weight=3.0 makes goal dominant over model's forward bias
             speed scaled by LiDAR proximity (slows 0.8 m, stops 0.4 m)
             v   = max_v * confidence * prox_scale * cos(heading_error)
             ω   = Kp * heading_error
  Proximity: front LiDAR min-dist < prox_stop_dist → zero forward, rotate to goal
  Recovery : all paths below blocked_threshold → reverse + rotate toward goal
           : OR stuck (< 0.05 m movement in 3 s) → same recovery action
  Reached  : dist to goal < goal_tol → stop and wait for next goal

Topic I/O
---------
  Subscribes:
    /dead_end_detection/path_status  Float32MultiArray  [F, L, R] from infer_vis
    /lidar/front/points              PointCloud2        front sector from pointcloud_segmenter
    /goal_pose                       PoseStamped        goal from RViz2
  Publishes:
    /cmd_vel                         Twist
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
from tf2_ros import TransformListener, Buffer
from collections import deque


class DirectVelController(Node):
    def __init__(self):
        super().__init__('direct_vel_controller')

        # ── Tunable parameters ──────────────────────────────────────────────
        self.max_v          = 0.4    # max forward speed (m/s)
        self.max_omega      = 1.2    # max angular speed (rad/s)
        self.Kp             = 1.5    # proportional heading gain
        self.goal_weight    = 3.0    # goal pulls 6× stronger than model (was 0.5)
        self.blocked_thr    = 0.35   # all paths below this → recovery mode
        self.goal_tol       = 0.5    # distance (m) at which goal is "reached"
        self.recovery_v     = -0.15  # reverse speed in recovery (m/s)
        self.recovery_omega = 0.8    # rotation speed in recovery (rad/s)
        self.stuck_window   = 30     # ticks at 10 Hz = 3 s to detect stuck
        self.stuck_dist     = 0.05   # metres — movement below this = stuck
        self.prox_stop_dist = 0.4    # metres — hard stop forward motion
        self.prox_slow_dist = 0.8    # metres — begin slowing down
        # ────────────────────────────────────────────────────────────────────

        # State
        self.path_status       = None   # [F, L, R] latest model scores
        self.current_goal      = None   # (gx, gy) in map frame
        self.in_recovery       = False
        self.recovery_dir      = 1.0    # +1 left / -1 right rotation in recovery
        self.recovery_cooldown = 0      # callbacks before recovery can re-trigger
        self.pos_history       = deque(maxlen=self.stuck_window)  # (x, y) ring buffer
        self.front_min_dist    = 999.0  # latest min distance from front LiDAR (metres)

        # TF
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(
            Float32MultiArray,
            '/dead_end_detection/path_status',
            self._path_status_cb, 10)
        self.create_subscription(
            PointCloud2,
            '/lidar/front/points',
            self._front_lidar_cb, 10)
        self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self._goal_cb, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control loop at 10 Hz
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info('DR.Nav Direct Velocity Controller initialized')
        self.get_logger().info('Waiting for goal via RViz2 2D Nav Goal...')

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _path_status_cb(self, msg):
        if len(msg.data) >= 3:
            self.path_status = [float(msg.data[0]),
                                float(msg.data[1]),
                                float(msg.data[2])]

    def _front_lidar_cb(self, msg: PointCloud2):
        """Compute minimum horizontal distance from front PointCloud2."""
        try:
            n = msg.width * msg.height
            if n == 0:
                return
            raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
            x = raw[:, 0:4].copy().view(np.float32).flatten()
            y = raw[:, 4:8].copy().view(np.float32).flatten()
            z = raw[:, 8:12].copy().view(np.float32).flatten()
            # Ignore ground (z < -0.1 m) and ceiling returns
            valid = np.isfinite(x) & np.isfinite(y) & (z > -0.1) & (z < 1.5)
            if not valid.any():
                self.front_min_dist = 999.0
                return
            self.front_min_dist = float(np.min(np.sqrt(x[valid]**2 + y[valid]**2)))
        except Exception:
            self.front_min_dist = 999.0

    def _goal_cb(self, msg):
        self.current_goal      = (msg.pose.position.x, msg.pose.position.y)
        self.in_recovery       = False
        self.recovery_cooldown = 0
        self.pos_history.clear()
        self.get_logger().info(
            f'New goal: ({self.current_goal[0]:.2f}, {self.current_goal[1]:.2f})')

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            x   = t.transform.translation.x
            y   = t.transform.translation.y
            qz  = t.transform.rotation.z
            qw  = t.transform.rotation.w
            yaw = 2.0 * math.atan2(qz, qw)
            return x, y, yaw, True
        except Exception:
            return 0.0, 0.0, 0.0, False

    def _normalize(self, angle):
        while angle >  math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    # ── Main control loop ─────────────────────────────────────────────────

    def _control_loop(self):
        rx, ry, ryaw, ok = self._get_robot_pose()
        cmd = Twist()

        if not ok or self.current_goal is None:
            self.cmd_pub.publish(cmd)
            return

        gx, gy = self.current_goal
        dist_to_goal = math.hypot(gx - rx, gy - ry)

        # ── Goal reached ─────────────────────────────────────────────────
        if dist_to_goal < self.goal_tol:
            self.get_logger().info(
                f'Goal reached ({dist_to_goal:.2f} m). Waiting for next goal.')
            self.current_goal = None
            self.cmd_pub.publish(cmd)
            return

        # Goal angle in robot frame
        goal_world_angle = math.atan2(gy - ry, gx - rx)
        goal_robot_angle = self._normalize(goal_world_angle - ryaw)

        # Track position for stuck detection
        self.pos_history.append((rx, ry))

        # Tick recovery cooldown
        if self.recovery_cooldown > 0:
            self.recovery_cooldown -= 1

        # ── Decide recovery ───────────────────────────────────────────────
        if self.path_status is not None:
            F, L, R = self.path_status
            all_blocked = (F < self.blocked_thr and
                           L < self.blocked_thr and
                           R < self.blocked_thr)
        else:
            F, L, R = 0.5, 0.5, 0.5
            all_blocked = False

        # Stuck detection: if we have a full window and haven't moved, force recovery
        if (not self.in_recovery and self.recovery_cooldown == 0
                and len(self.pos_history) == self.stuck_window):
            xs = [p[0] for p in self.pos_history]
            ys = [p[1] for p in self.pos_history]
            spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if spread < self.stuck_dist:
                all_blocked = True
                self.get_logger().warn(
                    f'Stuck detected: moved only {spread:.3f} m in '
                    f'{self.stuck_window / 10:.0f} s — forcing recovery')

        if all_blocked and self.recovery_cooldown == 0:
            if not self.in_recovery:
                self.in_recovery  = True
                # Rotate in the direction that faces the goal when reversing
                self.recovery_dir = math.copysign(1.0, goal_robot_angle)
                self.get_logger().info(
                    f'Recovery: all paths blocked '
                    f'(F={F:.2f} L={L:.2f} R={R:.2f}), reversing')

            cmd.linear.x  = self.recovery_v
            cmd.angular.z = self.recovery_dir * self.recovery_omega
            self.cmd_pub.publish(cmd)
            return

        if self.in_recovery and not all_blocked:
            self.in_recovery       = False
            self.recovery_cooldown = 15   # 1.5 s cooldown at 10 Hz
            self.get_logger().info('Recovery complete, resuming normal navigation')

        # ── LiDAR proximity: hard stop forward if wall too close ──────────
        if self.front_min_dist < self.prox_stop_dist:
            cmd.angular.z = float(max(-self.max_omega,
                                      min(self.max_omega, self.Kp * goal_robot_angle)))
            self.cmd_pub.publish(cmd)
            return

        # ── Normal navigation: blend model + goal ─────────────────────────
        #
        # Model heading vector (robot frame):
        #   F maps to  0°  (straight ahead)
        #   L maps to +90° (left)
        #   R maps to -90° (right)
        #
        #   dx_model = F·cos(0) + L·cos(90°) + R·cos(-90°) = F
        #   dy_model = F·sin(0) + L·sin(90°) + R·sin(-90°) = L - R
        #
        dx_model = F
        dy_model = L - R

        # Goal heading vector (robot frame)
        dx_goal = self.goal_weight * math.cos(goal_robot_angle)
        dy_goal = self.goal_weight * math.sin(goal_robot_angle)

        # Blended desired heading
        dx = dx_model + dx_goal
        dy = dy_model + dy_goal
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            desired_heading = goal_robot_angle
        else:
            desired_heading = math.atan2(dy, dx)

        heading_error = self._normalize(desired_heading)
        confidence    = max(F, L, R)

        # Proximity scale: 1.0 when clear, ramps to 0.0 at prox_stop_dist
        prox_scale = min(1.0, max(0.0,
            (self.front_min_dist - self.prox_stop_dist) /
            (self.prox_slow_dist - self.prox_stop_dist)))

        # Forward speed: full when aligned, tapers to zero at 90°
        v     = self.max_v * confidence * prox_scale * max(0.0, math.cos(heading_error))
        omega = float(max(-self.max_omega,
                          min(self.max_omega, self.Kp * heading_error)))

        cmd.linear.x  = float(v)
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = DirectVelController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
