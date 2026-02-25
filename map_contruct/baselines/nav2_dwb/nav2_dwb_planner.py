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


class Nav2DwbPlannerNode(Node):
    def __init__(self):
        super().__init__('nav2_dwb_planner_node')

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
        self.current_vel = {'x': 0.0, 'theta': 0.0}

        # Nav2 DWB parameters
        self.max_vel_x = 0.5
        self.min_vel_x = 0.0
        self.max_vel_theta = 1.0
        self.min_vel_theta = -1.0
        self.acc_lim_x = 2.5
        self.acc_lim_theta = 3.2
        self.decel_lim_x = -2.5
        self.decel_lim_theta = -3.2

        # Simulation
        self.sim_time = 3.0           # seconds to simulate forward
        self.sim_granularity = 0.05   # step size in seconds
        self.vx_samples = 6
        self.vtheta_samples = 20

        # Scoring weights (Nav2 DWB style)
        self.path_distance_bias = 32.0
        self.goal_distance_bias = 24.0
        self.forward_point_distance = 0.325

        # Metrics
        self.metrics = {
            'method_name': 'nav2_dwb_lidar',
            'start_time': time.time(),
            'total_distance': 0.0,
            'completion_time': 0.0
        }
        self.last_pose = None
        self.output_dir = self._create_output_dir()

        self.create_timer(0.1, self.plan_and_publish)
        self.get_logger().info('Nav2 DWB Planner (geometric baseline) initialized')

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
            self.current_vel = {'x': 0.0, 'theta': 0.0}
            return

        target_x = self.current_goal['x']
        target_y = self.current_goal['y']

        if np.hypot(robot_x - target_x, robot_y - target_y) < 1.0:
            self.get_logger().info('Goal reached')
            self.current_goal = None
            self.cmd_pub.publish(Twist())
            self.current_vel = {'x': 0.0, 'theta': 0.0}
            return

        best_v, best_omega = self.dwb_planning(robot_x, robot_y, robot_theta, target_x, target_y)

        cmd = Twist()
        cmd.linear.x = best_v
        cmd.angular.z = best_omega
        self.cmd_pub.publish(cmd)

        self.current_vel = {'x': best_v, 'theta': best_omega}
        self._update_metrics(robot_x, robot_y)
        self.last_pose = (robot_x, robot_y)

    def dwb_planning(self, start_x, start_y, start_theta, goal_x, goal_y):
        """Nav2 DWB: sample within acceleration-limited velocity window, score each trajectory."""
        dt = 0.1

        # Achievable velocity ranges given current velocity and acceleration limits
        max_vx = min(self.max_vel_x, self.current_vel['x'] + self.acc_lim_x * dt)
        min_vx = max(self.min_vel_x, self.current_vel['x'] + self.decel_lim_x * dt)
        max_vt = min(self.max_vel_theta, self.current_vel['theta'] + self.acc_lim_theta * dt)
        min_vt = max(self.min_vel_theta, self.current_vel['theta'] + self.decel_lim_theta * dt)

        vx_samples = np.linspace(min_vx, max_vx, self.vx_samples)
        vt_samples = np.linspace(min_vt, max_vt, self.vtheta_samples)

        best_score = -float('inf')
        best_vx, best_vt = 0.0, 0.0

        for vx in vx_samples:
            if vx < 0:
                continue
            for vt in vt_samples:
                score = self._evaluate_trajectory(
                    start_x, start_y, start_theta, vx, vt, goal_x, goal_y
                )
                if score > best_score:
                    best_score = score
                    best_vx = vx
                    best_vt = vt

        return best_vx, best_vt

    def _evaluate_trajectory(self, start_x, start_y, start_theta, vx, vtheta, goal_x, goal_y):
        """Simulate trajectory and compute Nav2 DWB score. Returns -inf if collision."""
        x, y, theta = start_x, start_y, start_theta
        sim_steps = int(self.sim_time / self.sim_granularity)

        for _ in range(sim_steps):
            x += vx * math.cos(theta) * self.sim_granularity
            y += vx * math.sin(theta) * self.sim_granularity
            theta += vtheta * self.sim_granularity
            if self.is_occupied(x, y):
                return -float('inf')  # discard colliding trajectories

        # Path distance score: how close the trajectory end is to the goal
        path_dist = np.hypot(x - goal_x, y - goal_y)
        path_score = -path_dist * self.path_distance_bias

        # Goal distance score: current distance to goal
        goal_dist = np.hypot(start_x - goal_x, start_y - goal_y)
        goal_score = -goal_dist * self.goal_distance_bias

        # Goal alignment score: penalise being turned away from goal
        goal_angle = math.atan2(goal_y - start_y, goal_x - start_x)
        angle_diff = abs(self._normalize_angle(start_theta - goal_angle))
        alignment_score = -angle_diff * 10.0

        # Speed preference: encourage moving forward
        speed_score = vx * 2.0

        # Oscillation penalty
        oscillation_penalty = -abs(vtheta) * 1.0

        return path_score + goal_score + alignment_score + speed_score + oscillation_penalty

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
        d = f"nav2_dwb_metrics_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(d, exist_ok=True)
        return d

    def save_metrics(self):
        path = os.path.join(self.output_dir, 'nav2_dwb_metrics.json')
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.get_logger().info(f'Metrics saved to {path}')


def main(args=None):
    rclpy.init(args=args)
    node = Nav2DwbPlannerNode()
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
