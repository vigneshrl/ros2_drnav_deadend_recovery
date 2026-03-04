#!/usr/bin/env python3
"""
Direct Velocity Controller for DR.Nav
======================================
Three-input fusion: model F/L/R + LiDAR proximity + goal direction.

State machine
-------------
  NAVIGATING  : normal blended navigation toward goal.
                Tracks consecutive dead-end model detections and the last
                junction (position where ≥2 paths were open).

  RECOVERING  : dead end confirmed (≥5 consecutive model updates all-blocked).
                Makes the saved junction the temporary sub-goal and navigates
                back to it.  On arrival, restores the original goal and
                activates dead-end-direction avoidance so the robot picks a
                different route.

Immediate safety
----------------
  LiDAR proximity  < 0.4 m → zero forward speed, rotate toward goal
  LiDAR proximity  0.4–0.8 m → linearly scale forward speed to zero

Topic I/O
---------
  Subscribes:
    /dead_end_detection/path_status  Float32MultiArray  [F, L, R]
    /lidar/front/points              PointCloud2        front sector
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
        self.max_v            = 0.4    # max forward speed (m/s)
        self.max_omega        = 1.2    # max angular speed (rad/s)
        self.Kp               = 1.5    # proportional heading gain
        self.goal_weight      = 3.0    # goal dominates model's forward bias
        self.blocked_thr      = 0.65   # path score below this → blocked
        self.goal_tol         = 0.5    # goal-reached radius (m)
        self.recovery_v       = -0.15  # reverse speed during short recovery (m/s)
        self.recovery_omega   = 0.8    # rotation speed during short recovery (rad/s)
        self.stuck_window     = 30     # ticks (10 Hz) for stuck detection = 3 s
        self.stuck_dist       = 0.05   # metres — spread below this = stuck
        self.prox_stop_dist   = 0.4    # metres — hard stop forward motion
        self.prox_slow_dist   = 0.8    # metres — begin slowing
        self.consecutive_thr  = 5      # model updates confirming dead end
        self.avoid_weight     = 2.0    # heading penalty for dead-end direction
        self.avoid_clear_dist = 1.5    # metres from junction to clear avoidance
        # ────────────────────────────────────────────────────────────────────

        # ── Navigation state machine ─────────────────────────────────────────
        self.nav_state           = 'navigating'  # 'navigating' | 'recovering'
        self.consecutive_blocked = 0
        self.junction_point      = None   # (x, y) — last pos with ≥2 open paths
        self.original_goal       = None   # saved while recovering to junction
        self.dead_end_yaw        = None   # world-frame yaw when dead end confirmed
        self.avoid_active        = False  # penalise dead-end dir after returning

        # ── Short recovery state (physical wall, not confirmed dead end) ─────
        self.in_recovery         = False
        self.recovery_dir        = 1.0
        self.recovery_cooldown   = 0

        # ── Sensor state ─────────────────────────────────────────────────────
        self.path_status         = None
        self.current_goal        = None
        self.front_min_dist      = 999.0
        self.pos_history         = deque(maxlen=self.stuck_window)

        # TF
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(
            Float32MultiArray, '/dead_end_detection/path_status',
            self._path_status_cb, 10)
        self.create_subscription(
            PointCloud2, '/lidar/front/points',
            self._front_lidar_cb, 10)
        self.create_subscription(
            PoseStamped, '/goal_pose',
            self._goal_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info('DR.Nav Direct Velocity Controller initialized')
        self.get_logger().info('Waiting for goal via RViz2 Nav2 Goal...')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _path_status_cb(self, msg):
        if len(msg.data) < 3:
            return
        F, L, R = float(msg.data[0]), float(msg.data[1]), float(msg.data[2])
        self.path_status = [F, L, R]
        # Count consecutive all-blocked model updates (only while navigating)
        all_blocked = (F < self.blocked_thr and
                       L < self.blocked_thr and
                       R < self.blocked_thr)
        if all_blocked and self.nav_state == 'navigating':
            self.consecutive_blocked += 1
        else:
            self.consecutive_blocked = 0

    def _front_lidar_cb(self, msg: PointCloud2):
        try:
            n = msg.width * msg.height
            if n == 0:
                return
            raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
            x = raw[:, 0:4].copy().view(np.float32).flatten()
            y = raw[:, 4:8].copy().view(np.float32).flatten()
            z = raw[:, 8:12].copy().view(np.float32).flatten()
            valid = np.isfinite(x) & np.isfinite(y) & (z > -0.1) & (z < 1.5)
            if not valid.any():
                self.front_min_dist = 999.0
                return
            self.front_min_dist = float(np.min(np.sqrt(x[valid]**2 + y[valid]**2)))
        except Exception:
            self.front_min_dist = 999.0

    def _goal_cb(self, msg):
        self.current_goal        = (msg.pose.position.x, msg.pose.position.y)
        self.nav_state           = 'navigating'
        self.original_goal       = None
        self.consecutive_blocked = 0
        self.in_recovery         = False
        self.recovery_cooldown   = 0
        self.avoid_active        = False
        self.dead_end_yaw        = None
        self.pos_history.clear()
        self.get_logger().info(
            f'New goal: ({self.current_goal[0]:.2f}, {self.current_goal[1]:.2f})')

    # ── Helpers ───────────────────────────────────────────────────────────────

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

    # ── Main control loop ──────────────────────────────────────────────────

    def _control_loop(self):
        rx, ry, ryaw, ok = self._get_robot_pose()
        cmd = Twist()

        if not ok or self.current_goal is None:
            self.cmd_pub.publish(cmd)
            return

        gx, gy = self.current_goal
        dist_to_goal = math.hypot(gx - rx, gy - ry)

        # ── Goal / recovery-point reached ─────────────────────────────────
        if dist_to_goal < self.goal_tol:
            if self.nav_state == 'recovering':
                self.get_logger().info(
                    f'Reached recovery point ({gx:.2f}, {gy:.2f}). '
                    f'Resuming toward original goal — dead-end direction penalised.')
                self.current_goal        = self.original_goal
                self.original_goal       = None
                self.nav_state           = 'navigating'
                self.avoid_active        = True   # steer away from dead-end dir
                self.consecutive_blocked = 0
                self.pos_history.clear()
            else:
                self.get_logger().info(
                    f'Goal reached ({dist_to_goal:.2f} m). Waiting for next goal.')
                self.current_goal = None
                self.avoid_active = False
                self.dead_end_yaw = None
            self.cmd_pub.publish(cmd)
            return

        # Goal angle in robot frame
        goal_world_angle = math.atan2(gy - ry, gx - rx)
        goal_robot_angle = self._normalize(goal_world_angle - ryaw)

        # Position history for stuck detection
        self.pos_history.append((rx, ry))

        # Cooldown tick
        if self.recovery_cooldown > 0:
            self.recovery_cooldown -= 1

        # ── Path status ───────────────────────────────────────────────────
        if self.path_status is not None:
            F, L, R = self.path_status
            open_count = sum(1 for s in [F, L, R] if s > self.blocked_thr)
            all_blocked = (open_count == 0)
        else:
            F, L, R = 0.5, 0.5, 0.5
            open_count = 3
            all_blocked = False

        # ── NAVIGATING: junction tracking + dead-end confirmation ─────────
        if self.nav_state == 'navigating':

            # Save junction whenever ≥2 paths are open
            if open_count >= 2:
                self.junction_point = (rx, ry)

            # Confirmed dead end after consecutive model detections
            if self.consecutive_blocked >= self.consecutive_thr:
                self.dead_end_yaw = ryaw

                if self.junction_point is None:
                    # No junction seen yet — synthesise 1.5 m directly behind
                    self.junction_point = (
                        rx - 1.5 * math.cos(ryaw),
                        ry - 1.5 * math.sin(ryaw))

                self.get_logger().warn(
                    f'Dead end confirmed ({self.consecutive_blocked} consecutive detections) '
                    f'F={F:.2f} L={L:.2f} R={R:.2f}')
                self.get_logger().info(
                    f'Navigating to recovery point '
                    f'({self.junction_point[0]:.2f}, {self.junction_point[1]:.2f})')

                self.original_goal       = self.current_goal
                self.current_goal        = self.junction_point
                self.nav_state           = 'recovering'
                self.consecutive_blocked = 0
                self.in_recovery         = False
                self.recovery_cooldown   = 0
                self.pos_history.clear()

                # Recalculate goal angle toward new sub-goal
                gx, gy = self.current_goal
                goal_world_angle = math.atan2(gy - ry, gx - rx)
                goal_robot_angle = self._normalize(goal_world_angle - ryaw)

            # Stuck detection (backup: physical wall, model uncertain)
            elif (self.recovery_cooldown == 0 and not self.in_recovery and
                  len(self.pos_history) == self.stuck_window):
                xs = [p[0] for p in self.pos_history]
                ys = [p[1] for p in self.pos_history]
                spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                if spread < self.stuck_dist:
                    all_blocked = True
                    self.get_logger().warn(
                        f'Stuck detected: moved only {spread:.3f} m in '
                        f'{self.stuck_window / 10:.0f} s — short recovery')

        # ── Short recovery (stuck against wall, not a confirmed dead end) ──
        if all_blocked and self.recovery_cooldown == 0 and self.nav_state == 'navigating':
            if not self.in_recovery:
                self.in_recovery  = True
                self.recovery_dir = math.copysign(1.0, goal_robot_angle)
                self.get_logger().info(
                    f'Short recovery: reversing (F={F:.2f} L={L:.2f} R={R:.2f})')
            cmd.linear.x  = self.recovery_v
            cmd.angular.z = self.recovery_dir * self.recovery_omega
            self.cmd_pub.publish(cmd)
            return

        if self.in_recovery and not all_blocked:
            self.in_recovery       = False
            self.recovery_cooldown = 15
            self.get_logger().info('Short recovery complete, resuming navigation')

        # ── LiDAR proximity (skip during recovery — junction may be near wall) ─
        if self.front_min_dist < self.prox_stop_dist and self.nav_state != 'recovering':
            cmd.angular.z = float(max(-self.max_omega,
                                      min(self.max_omega, self.Kp * goal_robot_angle)))
            self.cmd_pub.publish(cmd)
            return

        # ── Heading blend: model + goal (+ dead-end avoidance) ───────────
        dx_model = F
        dy_model = L - R

        dx_goal = self.goal_weight * math.cos(goal_robot_angle)
        dy_goal = self.goal_weight * math.sin(goal_robot_angle)

        dx = dx_model + dx_goal
        dy = dy_model + dy_goal

        # After returning from dead end: penalise the direction that led there
        if self.avoid_active and self.dead_end_yaw is not None:
            dead_end_robot_angle = self._normalize(self.dead_end_yaw - ryaw)
            dx -= self.avoid_weight * math.cos(dead_end_robot_angle)
            dy -= self.avoid_weight * math.sin(dead_end_robot_angle)
            # Clear once robot has moved away from junction
            if (self.junction_point is not None and
                    math.hypot(rx - self.junction_point[0],
                               ry - self.junction_point[1]) > self.avoid_clear_dist):
                self.avoid_active = False
                self.dead_end_yaw = None
                self.get_logger().info(
                    'Dead-end avoidance cleared — moved far enough from junction')

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            desired_heading = goal_robot_angle
        else:
            desired_heading = math.atan2(dy, dx)

        heading_error = self._normalize(desired_heading)
        confidence    = max(F, L, R)

        prox_scale = min(1.0, max(0.0,
            (self.front_min_dist - self.prox_stop_dist) /
            (self.prox_slow_dist - self.prox_stop_dist)))

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
