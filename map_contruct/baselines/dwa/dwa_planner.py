#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid
from tf2_ros import TransformListener, Buffer
import numpy as np
import math
import time
import json
import os
from scipy.ndimage import binary_dilation


class DwaPlannerNode(Node):
    def __init__(self):
        super().__init__('dwa_planner_node')

        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # State
        self.current_goal = None
        self.occupancy_grid = None
        self.inflated_mask = None          # precomputed inflated occupancy (numpy bool)
        self.inflation_radius = 0.35       # Jackal half-diagonal (0.33 m) + 2 cm margin

        # DWA parameters
        self.max_speed = 0.5
        self.max_omega = 1.0
        self.dt = 0.1             # simulation step size
        self.sim_steps = 10       # simulate 1 second ahead
        self.v_samples = 5
        self.omega_samples = 9

        # Scoring weights
        self.goal_weight = 3.0
        self.safety_weight = 5.0
        self.turn_weight = 0.1

        # Metrics
        self.metrics = {
            'method_name': 'dwa_lidar',
            'start_time': time.time(),
            'total_distance': 0.0,
            'completion_time': 0.0
        }
        self.last_pose = None
        self.output_dir = self._create_output_dir()

        self.create_timer(0.1, self.plan_and_publish)
        self.get_logger().info('DWA Planner (geometric baseline) initialized')

    def goal_callback(self, msg):
        self.current_goal = {'x': msg.pose.position.x, 'y': msg.pose.position.y}
        self.get_logger().info(f'New goal: ({self.current_goal["x"]:.2f}, {self.current_goal["y"]:.2f})')

    def map_callback(self, msg):
        self.occupancy_grid = msg
        self._rebuild_inflated_grid()

    def _rebuild_inflated_grid(self):
        """Dilate every occupied cell by the robot radius once per map update.

        After dilation a single centre-point check is equivalent to the old
        9-point footprint loop, but is continuous (no angular gaps between
        sample points) and cheaper to evaluate during trajectory sampling.
        """
        info = self.occupancy_grid.info
        raw = np.array(self.occupancy_grid.data, dtype=np.int8).reshape(
            (info.height, info.width))
        occupied = raw > 50
        cells = int(math.ceil(self.inflation_radius / info.resolution))
        self.inflated_mask = binary_dilation(occupied, iterations=cells)

    def get_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            theta = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            return x, y, theta, True
        except Exception:
            return 0.0, 0.0, 0.0, False

    def is_occupied(self, x, y):
        """Return True if the inflated map marks (x, y) as no-go.

        The inflated mask was built by dilating every occupied cell by
        inflation_radius (0.35 m) at map-update time, so a single centre-point
        lookup here is equivalent to checking the full robot footprint — with
        no angular gaps between sample points.
        """
        if self.inflated_mask is None or self.occupancy_grid is None:
            return False
        info = self.occupancy_grid.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        if gx < 0 or gx >= info.width or gy < 0 or gy >= info.height:
            return True
        return bool(self.inflated_mask[gy, gx])

    def plan_and_publish(self):
        robot_x, robot_y, robot_theta, ok = self.get_robot_pose()
        if not ok:
            return

        if not self.current_goal:
            self.cmd_pub.publish(Twist())
            return

        target_x = self.current_goal['x']
        target_y = self.current_goal['y']

        # Check if goal reached
        if np.hypot(robot_x - target_x, robot_y - target_y) < 1.0:
            self.get_logger().info('Goal reached')
            self.current_goal = None
            self.cmd_pub.publish(Twist())
            return

        # DWA: sample velocities, simulate, score
        best_score = -float('inf')
        best_v, best_omega = 0.0, 0.0

        # Precompute goal direction for heading alignment
        goal_angle = math.atan2(target_y - robot_y, target_x - robot_x)

        for v in np.linspace(0.0, self.max_speed, self.v_samples):
            for omega in np.linspace(-self.max_omega, self.max_omega, self.omega_samples):
                x, y, theta = robot_x, robot_y, robot_theta
                collision = False

                for _ in range(self.sim_steps):
                    x += v * math.cos(theta) * self.dt
                    y += v * math.sin(theta) * self.dt
                    theta += omega * self.dt
                    # For v=0 (rotation-in-place) position never changes, so
                    # is_occupied would wrongly flag nearby walls at every step.
                    # Skip the check — the robot is already safely at that spot.
                    if v > 1e-9 and self.is_occupied(x, y):
                        collision = True
                        break

                if collision:
                    continue

                goal_dist = np.hypot(x - target_x, y - target_y)
                goal_score = -goal_dist * self.goal_weight

                # Heading alignment: reward the trajectory that ends with the
                # robot facing the goal. Without this, all v=0 trajectories have
                # the same goal_dist, so omega=0 always wins (lowest turn_penalty)
                # and the robot freezes instead of rotating toward the goal.
                heading_diff = abs(self._normalize_angle(theta - goal_angle))
                heading_score = -heading_diff * 1.5

                turn_penalty = abs(omega) * self.turn_weight
                total_score = goal_score + heading_score - turn_penalty

                if total_score > best_score:
                    best_score = total_score
                    best_v = v
                    best_omega = omega

        cmd = Twist()
        cmd.linear.x = best_v
        cmd.angular.z = best_omega
        self.cmd_pub.publish(cmd)

        self._update_metrics(robot_x, robot_y)
        self.last_pose = (robot_x, robot_y)

    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _update_metrics(self, robot_x, robot_y):
        if self.last_pose is not None:
            self.metrics['total_distance'] += math.hypot(
                robot_x - self.last_pose[0], robot_y - self.last_pose[1]
            )
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
