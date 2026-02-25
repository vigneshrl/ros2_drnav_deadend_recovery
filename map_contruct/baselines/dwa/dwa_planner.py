#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformListener, Buffer
import numpy as np
import math
import time
import json
import os

class DwaPlannerNode(Node):
    def __init__(self):
        super().__init__('dwa_planner_node')

        # Subscriptions
        self.recovery_sub = self.create_subscription(
            Float32MultiArray,
            '/dead_end_detection/recovery_points',
            self.recovery_callback,
            10
        )
        self.costmap_sub = self.create_subscription(
            MarkerArray,
            '/dram_exploration_map',
            self.costmap_callback,
            10
        )

        #single camera abalation study
        # self.dead_end_sub = self.create_subscription(
        #     Bool,
        #     '/single_camera/is_dead_end',
        #     self.dead_end_callback,
        #     10
        # )

        #multi camera my method 
        self.dead_end_sub = self.create_subscription(
            Bool,
            '/dead_end_detection/is_dead_end',
            self.dead_end_callback,
            10
        )
        
        # Subscribe to goals from Goal Generator
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )
        
        # Publisher
        self.cmd_pub = self.create_publisher(
            Twist,
            '/mcu/command/manual_twist',  # Changed back to real robot control
            10
        )

        # Transform listener for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Internal state
        self.recovery_points = []  # List of {'x': float, 'y': float, 'type': int, 'open_paths': int}
        self.costmap = {}          # (x, y) -> {'safety': float, 'is_recovery': bool, 'open_paths': int}
        self.current_goal = None   # Current goal from Goal Generator

        # Dead-end tracking
        self.dead_end_count = 0
        self.consecutive_dead_ends = 0
        self.max_consecutive_dead_ends = 4  # Go to recovery point after 4 consecutive dead-ends
        self.is_in_recovery_mode = False
        self.target_recovery_point = None

        # Robot parameters (tune for your robot)
        self.max_speed = 0.5
        self.max_omega = 1.0
        self.dt = 0.1

        # Recovery point preferences
        self.max_recovery_distance = 15.0  # Maximum distance to consider recovery points

        # Timer for planning loop
        self.create_timer(0.1, self.plan_and_publish)

        # Metrics collection
        self.metrics = {
            'start_time': time.time(),
            'method_name': 'dram_method',
            'total_distance': 0.0,
            'total_energy': 0.0,
            'dead_end_detections': 0,
            'false_positive_dead_ends': 0,
            'false_negatives': 0,
            'recovery_point_detections': 0,
            'recovery_activations': 0,
            'detection_lead_distances': [],
            'distance_to_first_recovery': 0.0,
            'freezes': 0,
            'time_trapped': 0.0,
            'ede_integral': 0.0,
            'completion_time': 0.0
        }
        self.last_pose = None
        self.last_cmd = Twist()
        
        # Create output directory
        self.output_dir = self.create_output_directory()

        self.get_logger().info('DWA Planner Node initialized - using multi-camera data')

    def recovery_callback(self, msg):
        # Parse recovery points from Float32MultiArray
        # Format: [type1, x1, y1, type2, x2, y2, ...]
        data = np.array(msg.data)
        if len(data) % 3 == 0:
            self.recovery_points = []
            for i in range(0, len(data), 3):
                point_type = int(data[i])
                # Determine number of open paths from point type
                open_paths = 2 if point_type == 1 else 1

                self.recovery_points.append({
                    'type': point_type,
                    'x': float(data[i+1]),
                    'y': float(data[i+2]),
                    'open_paths': open_paths
                })
            self.get_logger().info(f'📍 Updated {len(self.recovery_points)} recovery points')
        else:
            self.get_logger().warn(f'Recovery points array length {len(data)} is not divisible by 3!')

    def costmap_callback(self, msg):
        # Parse costmap from MarkerArray (heatmap)
        self.costmap = {}
        for marker in msg.markers:
            if marker.ns == "exploration_heatmap":
                # Process heatmap points
                for i, point in enumerate(marker.points):
                    x = point.x
                    y = point.y

                    # Get color information
                    if i < len(marker.colors):
                        color = marker.colors[i]
                        # Check if this is a recovery point (different color scheme)
                        is_recovery = (color.b > 0.5)  # Recovery points have blue component

                        if is_recovery:
                            # Determine open paths from color
                            if color.r > 0.4:  # Purple (3+ paths)
                                open_paths = 3
                            elif color.b > 0.7:  # Dark blue (2 paths)
                                open_paths = 2
                            else:  # Light blue (1 path)
                                open_paths = 1
                        else:
                            open_paths = 0

                        self.costmap[(x, y)] = {
                            'safety': 1.0 if color.g > 0.5 else 0.0,  # Green = safe, Red = unsafe
                            'is_recovery': is_recovery,
                            'open_paths': open_paths
                        }

    def dead_end_callback(self, msg):
        # Track consecutive dead-ends
        if msg.data:
            self.dead_end_count += 1
            self.consecutive_dead_ends += 1
            self.metrics['dead_end_detections'] += 1
            self.get_logger().warn(f'🚨 Dead-end detected! Count: {self.dead_end_count}, Consecutive: {self.consecutive_dead_ends}')
        else:
            # Reset consecutive counter when we escape dead-end
            self.consecutive_dead_ends = 0

    def goal_callback(self, msg):
        """Receive goals from Goal Generator"""
        self.current_goal = {
            'x': msg.pose.position.x,
            'y': msg.pose.position.y
        }
        self.get_logger().info(f'🎯 New goal received: ({self.current_goal["x"]:.2f}, {self.current_goal["y"]:.2f})')

    def plan_and_publish(self):
        # Get current robot pose
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
            )
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            # Extract yaw from quaternion
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            robot_theta = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        except Exception:
            # Fallback to origin if transform not available
            robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0

        # Check if we need to go to recovery point
        if self.consecutive_dead_ends >= self.max_consecutive_dead_ends:
            if not self.is_in_recovery_mode:
                self.is_in_recovery_mode = True
                self.target_recovery_point = self.select_best_recovery_point(robot_x, robot_y)
                self.metrics['recovery_activations'] += 1
                if self.target_recovery_point:
                    self.get_logger().warn(f'🎯 Entering RECOVERY MODE - heading to recovery point with {self.target_recovery_point["open_paths"]} open paths')

        # If we're in recovery mode, navigate to recovery point
        if self.is_in_recovery_mode and self.target_recovery_point:
            # Check if we've reached the recovery point
            distance_to_recovery = np.hypot(robot_x - self.target_recovery_point['x'],
                                          robot_y - self.target_recovery_point['y'])

            if distance_to_recovery < 3.0:  # Within 1m of recovery point
                self.get_logger().info(f'✅ Reached recovery point! Exiting recovery mode')
                self.is_in_recovery_mode = False
                self.target_recovery_point = None
                self.consecutive_dead_ends = 0  # Reset dead-end counter

                # Stop briefly to re-evaluate
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                return

            # Navigate to recovery point
            target_x, target_y = self.target_recovery_point['x'], self.target_recovery_point['y']
            self.get_logger().debug(f'🎯 Recovery mode: heading to ({target_x:.2f}, {target_y:.2f}), dist={distance_to_recovery:.2f}m')

        elif self.current_goal:
            # Use goal from Goal Generator
            target_x, target_y = self.current_goal['x'], self.current_goal['y']
            self.get_logger().debug(f'🎯 Following goal: ({target_x:.2f}, {target_y:.2f})')
        elif self.recovery_points:
            # Fallback: use recovery points if available
            target_recovery = self.select_best_recovery_point(robot_x, robot_y)
            if target_recovery:
                target_x, target_y = target_recovery['x'], target_recovery['y']
            else:
                # No valid recovery points - stop
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                self.get_logger().warn('🚨 No recovery points within range')
                return
        else:
            # No goals or recovery points available - stop
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.get_logger().debug('🔄 Waiting for goals from Goal Generator...')
            return

        # DWA: Sample velocities and find best path to target
        best_score = -float('inf')
        best_v = 0.0
        best_omega = 0.0

        for v in np.linspace(0, self.max_speed, 5):
            for omega in np.linspace(-self.max_omega, self.max_omega, 9):
                # Simulate forward
                x, y, theta = robot_x, robot_y, robot_theta
                for _ in range(10):  # Simulate 1 second ahead
                    x += v * np.cos(theta) * self.dt
                    y += v * np.sin(theta) * self.dt
                    theta += omega * self.dt

                # Score: prefer safe areas, close to target, smooth motion
                safety_score = self.get_safety_score(x, y)
                goal_distance = np.hypot(x - target_x, y - target_y)
                goal_score = -goal_distance  # Closer is better

                # Prefer straight motion over turning
                turn_penalty = abs(omega) * 0.1

                # Total score
                total_score = safety_score * 5.0 + goal_score * 3.0 - turn_penalty

                if total_score > best_score:
                    best_score = total_score
                    best_v = v
                    best_omega = omega

        # Publish best command
        cmd = Twist()
        cmd.linear.x = best_v
        cmd.angular.z = best_omega
        self.cmd_pub.publish(cmd)

        # Update metrics
        self.update_metrics(robot_x, robot_y, cmd)

        # Log planning result
        mode = "RECOVERY" if self.is_in_recovery_mode else "EXPLORATION"
        distance_to_target = np.hypot(robot_x - target_x, robot_y - target_y)
        self.get_logger().debug(f'🚀 DWA [{mode}]: v={best_v:.2f}, ω={best_omega:.2f} → '
                               f'Target ({target_x:.2f}, {target_y:.2f}) dist={distance_to_target:.2f}m')

    def get_safety_score(self, x, y):
        """Get safety score for a position (higher = safer)"""
        # Find nearest costmap cell
        min_dist = float('inf')
        min_safety = 0.0

        for (cx, cy), data in self.costmap.items():
            dist = np.hypot(x - cx, y - cy)
            if dist < min_dist:
                min_dist = dist
                min_safety = data['safety']

        return min_safety

    def create_output_directory(self):
        """Create output directory for metrics"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"dram_metrics_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def update_metrics(self, robot_x, robot_y, cmd):
        """Update performance metrics"""
        current_pose = (robot_x, robot_y)
        
        if self.last_pose is not None:
            # Calculate distance traveled
            dx = robot_x - self.last_pose[0]
            dy = robot_y - self.last_pose[1]
            distance = math.hypot(dx, dy)
            self.metrics['total_distance'] += distance
            
            # Calculate energy consumption (simplified)
            v = abs(cmd.linear.x)
            w = abs(cmd.angular.z)
            energy = (v * v + w * w) * 0.1
            self.metrics['total_energy'] += energy
        
        # Update completion time
        self.metrics['completion_time'] = time.time() - self.metrics['start_time']
        
        self.last_pose = current_pose
        self.last_cmd = cmd

    def save_metrics(self):
        """Save performance metrics to file"""
        metrics_file = os.path.join(self.output_dir, 'dram_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.get_logger().info(f'📊 Metrics saved to {metrics_file}')

def main(args=None):
    rclpy.init(args=args)
    node = DwaPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save metrics before shutdown
        node.save_metrics()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()