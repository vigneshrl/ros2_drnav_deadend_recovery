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


class MppiPlannerNode(Node):
    def __init__(self):
        super().__init__('mppi_planner_node')

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

        # MPPI parameters
        self.max_speed = 0.5
        self.max_omega = 1.0
        self.dt = 0.1
        self.horizon = 20         # time steps per trajectory
        self.num_samples = 100    # number of sampled trajectories
        self.temperature = 1.0   # softmax temperature
        self.sigma_v = 0.2        # linear velocity noise std
        self.sigma_omega = 0.5    # angular velocity noise std

        # Cost weights
        self.goal_weight = 1.0
        self.control_weight = 0.1
        self.smooth_weight = 0.05
        self.collision_cost = 1000.0  # large cost for hitting an obstacle

        # Rolling control sequence [v, omega] x horizon
        self.control_sequence = np.zeros((self.horizon, 2))

        # Metrics
        self.metrics = {
            'method_name': 'mppi_lidar',
            'start_time': time.time(),
            'total_distance': 0.0,
            'completion_time': 0.0
        }
        self.last_pose = None
        self.output_dir = self._create_output_dir()

        self.create_timer(0.1, self.plan_and_publish)
        self.get_logger().info('MPPI Planner (geometric baseline) initialized')

    def goal_callback(self, msg):
        self.current_goal = {'x': msg.pose.position.x, 'y': msg.pose.position.y}
        self.get_logger().info(f'New goal: ({self.current_goal["x"]:.2f}, {self.current_goal["y"]:.2f})')

    def map_callback(self, msg):
        self.occupancy_grid = msg

    def get_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'odom', rclpy.time.Time(),
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
        """Check if a world-frame point is occupied in the occupancy grid."""
        if self.occupancy_grid is None:
            return False
        info = self.occupancy_grid.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        if gx < 0 or gx >= info.width or gy < 0 or gy >= info.height:
            return True
        idx = gy * info.width + gx
        return self.occupancy_grid.data[idx] > 50

    def plan_and_publish(self):
        robot_x, robot_y, robot_theta, ok = self.get_robot_pose()
        if not ok:
            return

        if not self.current_goal:
            self.cmd_pub.publish(Twist())
            return

        target_x = self.current_goal['x']
        target_y = self.current_goal['y']

        if np.hypot(robot_x - target_x, robot_y - target_y) < 1.0:
            self.get_logger().info('Goal reached')
            self.current_goal = None
            self.cmd_pub.publish(Twist())
            return

        best_v, best_omega = self.mppi_planning(robot_x, robot_y, robot_theta, target_x, target_y)

        cmd = Twist()
        cmd.linear.x = best_v
        cmd.angular.z = best_omega
        self.cmd_pub.publish(cmd)

        self._update_metrics(robot_x, robot_y)
        self.last_pose = (robot_x, robot_y)

    def mppi_planning(self, start_x, start_y, start_theta, goal_x, goal_y):
        """MPPI: sample trajectories, compute weights, return optimal first control."""
        noise_v = np.random.normal(0, self.sigma_v, (self.num_samples, self.horizon))
        noise_omega = np.random.normal(0, self.sigma_omega, (self.num_samples, self.horizon))

        base = self.control_sequence.copy()
        costs = np.zeros(self.num_samples)

        for i in range(self.num_samples):
            v_seq = np.clip(base[:, 0] + noise_v[i], 0.0, self.max_speed)
            w_seq = np.clip(base[:, 1] + noise_omega[i], -self.max_omega, self.max_omega)
            costs[i] = self._evaluate_trajectory(
                start_x, start_y, start_theta, v_seq, w_seq, goal_x, goal_y
            )

        # Softmax weights (lower cost = higher weight)
        weights = np.exp(-costs / self.temperature)
        weights /= np.sum(weights)

        # Weighted average over all perturbed sequences
        new_controls = np.zeros((self.horizon, 2))
        for i in range(self.num_samples):
            v_seq = np.clip(base[:, 0] + noise_v[i], 0.0, self.max_speed)
            w_seq = np.clip(base[:, 1] + noise_omega[i], -self.max_omega, self.max_omega)
            new_controls[:, 0] += weights[i] * v_seq
            new_controls[:, 1] += weights[i] * w_seq

        # Shift rolling horizon
        self.control_sequence[:-1] = new_controls[1:]
        self.control_sequence[-1] = new_controls[-1]

        return float(new_controls[0, 0]), float(new_controls[0, 1])

    def _evaluate_trajectory(self, start_x, start_y, start_theta, v_seq, w_seq, goal_x, goal_y):
        x, y, theta = start_x, start_y, start_theta
        total_cost = 0.0

        for t in range(self.horizon):
            x += v_seq[t] * math.cos(theta) * self.dt
            y += v_seq[t] * math.sin(theta) * self.dt
            theta += w_seq[t] * self.dt

            # Collision cost
            if self.is_occupied(x, y):
                return self.collision_cost * (self.horizon - t)  # early exit, large penalty

            # Goal cost
            total_cost += np.hypot(x - goal_x, y - goal_y) * self.goal_weight

            # Control effort cost
            total_cost += (v_seq[t] ** 2 + w_seq[t] ** 2) * self.control_weight

            # Smoothness cost
            if t > 0:
                total_cost += ((v_seq[t] - v_seq[t-1]) ** 2 +
                               (w_seq[t] - w_seq[t-1]) ** 2) * self.smooth_weight

        # Terminal cost (heavier weight on final position)
        total_cost += np.hypot(x - goal_x, y - goal_y) * self.goal_weight * 2.0
        return total_cost

    def _update_metrics(self, robot_x, robot_y):
        if self.last_pose is not None:
            self.metrics['total_distance'] += math.hypot(
                robot_x - self.last_pose[0], robot_y - self.last_pose[1]
            )
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
