#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image
from tf2_ros import TransformListener, Buffer
import numpy as np
import math
import time
import json
import os
import pickle
from typing import Dict, List

class MppiRosbagPlannerNode(Node):
    def __init__(self):
        super().__init__('mppi_rosbag_planner_node')

        # Subscribe to goals from Goal Generator
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )
        
        # Subscribe to front camera images to get timestamps
        # Use reliable QoS to avoid dropping frames
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Use best effort for camera data
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100  # Larger queue to handle high-frequency camera data
        )
        
        self.front_camera_sub = self.create_subscription(
            Image,
            '/argus/ar0234_front_left/image_raw',
            self.front_camera_callback,
            camera_qos
        )
        
        # Publisher for commands
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Transform listener for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Internal state
        self.current_goal = None

        # MPPI parameters
        self.max_speed = 0.5
        self.max_omega = 1.0
        self.dt = 0.1
        self.horizon = 20  # Number of time steps to predict
        self.num_samples = 100  # Number of trajectory samples
        self.temperature = 1.0  # Temperature for sampling
        self.sigma_v = 0.2  # Noise standard deviation for linear velocity
        self.sigma_omega = 0.5  # Noise standard deviation for angular velocity
        
        # Cost weights
        self.goal_weight = 1.0
        self.control_weight = 0.1
        self.smooth_weight = 0.05

        # Control sequence (rolling horizon)
        self.control_sequence = np.zeros((self.horizon, 2))  # [v, omega] for each time step

        # Timer for planning loop
        self.create_timer(0.1, self.plan_and_publish)

        # Action logging for pickle export
        self.action_log = []
        self.start_time = time.time()
        
        # Camera-action synchronization
        self.camera_action_log = {}
        self.current_action = {'v': 0.0, 'omega': 0.0}
        self.current_robot_pose = (0.0, 0.0, 0.0)
        
        # Camera frame tracking
        self.camera_frame_count = 0
        self.camera_dropped_frames = 0
        self.last_camera_timestamp = None
        self.camera_callback_active = False
        self.camera_callback_start_time = None
        
        # Metrics collection
        self.metrics = {
            'start_time': self.start_time,
            'method_name': 'mppi_method',
            'total_distance': 0.0,
            'total_energy': 0.0,
            'completion_time': 0.0,
            'camera_frames_processed': 0,
            'camera_frames_dropped': 0,
            'camera_frames_synced': 0
        }
        self.last_pose = None
        self.last_cmd = Twist()
        
        # Create output directory
        self.output_dir = self.create_output_directory()

        self.get_logger().info('MPPI Rosbag Planner Node initialized')
        self.get_logger().info(f'Output directory: {self.output_dir}')

    def goal_callback(self, msg):
        """Receive goals from Goal Generator"""
        self.current_goal = {
            'x': msg.pose.position.x,
            'y': msg.pose.position.y
        }
        self.get_logger().info(f'🎯 New goal received: ({self.current_goal["x"]:.2f}, {self.current_goal["y"]:.2f})')

    def front_camera_callback(self, msg):
        """Callback for front camera images - sync current action with camera timestamp"""
        # Mark camera callback as active
        if not self.camera_callback_active:
            self.camera_callback_active = True
            self.camera_callback_start_time = time.time()
            self.get_logger().info('📸 Camera callback activated - starting frame capture')
        
        # Convert ROS timestamp to float
        camera_timestamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
        
        # Track frame count and detect dropped frames
        self.camera_frame_count += 1
        
        # Check for dropped frames (timestamps should be increasing)
        if self.last_camera_timestamp is not None:
            time_diff = camera_timestamp - self.last_camera_timestamp
            # If time difference is too large, we might have dropped frames
            if time_diff > 0.1:  # More than 100ms gap suggests dropped frames
                self.camera_dropped_frames += 1
        
        self.last_camera_timestamp = camera_timestamp
        
        # Store current action with this camera timestamp
        self.camera_action_log[camera_timestamp] = {
            'v': self.current_action['v'],
            'omega': self.current_action['omega'],
            'robot_pose': self.current_robot_pose,
            'ros_timestamp_sec': msg.header.stamp.sec,
            'ros_timestamp_nanosec': msg.header.stamp.nanosec,
            'frame_number': self.camera_frame_count
        }
        
        # Log periodically to avoid spam
        if self.camera_frame_count % 100 == 0:
            self.get_logger().info(f'📸 Processed {self.camera_frame_count} camera frames, '
                                 f'dropped: {self.camera_dropped_frames}, '
                                 f'synced: {len(self.camera_action_log)}')

    def get_robot_pose(self):
        """Get current robot pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'odom', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
            )
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            # Extract yaw from quaternion
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            robot_theta = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            return robot_x, robot_y, robot_theta, True
        except Exception as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return 0.0, 0.0, 0.0, False

    def plan_and_publish(self):
        # Get current robot pose
        robot_x, robot_y, robot_theta, tf_success = self.get_robot_pose()
        
        if not tf_success:
            return

        # Check if we have a goal
        if not self.current_goal:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            
            self.current_action = {'v': 0.0, 'omega': 0.0}
            self.current_robot_pose = (float(robot_x), float(robot_y), float(robot_theta))
            self.log_action(cmd, robot_x, robot_y, robot_theta)
            return

        target_x, target_y = self.current_goal['x'], self.current_goal['y']

        # Check if goal is reached
        goal_distance = np.hypot(robot_x - target_x, robot_y - target_y)
        if goal_distance < 1.0:  # Goal reached
            self.get_logger().info(f'✅ Goal reached! Distance: {goal_distance:.2f}m')
            self.current_goal = None
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            
            self.current_action = {'v': 0.0, 'omega': 0.0}
            self.current_robot_pose = (float(robot_x), float(robot_y), float(robot_theta))
            self.log_action(cmd, robot_x, robot_y, robot_theta)
            return

        # MPPI: Sample trajectories and compute optimal control
        best_v, best_omega = self.mppi_planning(robot_x, robot_y, robot_theta, target_x, target_y)

        # Publish best command
        cmd = Twist()
        cmd.linear.x = best_v
        cmd.angular.z = best_omega
        self.cmd_pub.publish(cmd)

        # Update current action and robot pose for camera synchronization
        self.current_action = {'v': float(best_v), 'omega': float(best_omega)}
        self.current_robot_pose = (float(robot_x), float(robot_y), float(robot_theta))

        # Log action with timestamp
        self.log_action(cmd, robot_x, robot_y, robot_theta)

        # Update metrics
        self.update_metrics(robot_x, robot_y, cmd)

        # Log planning result
        self.get_logger().debug(f'🚀 MPPI: v={best_v:.2f}, ω={best_omega:.2f} → '
                               f'Goal ({target_x:.2f}, {target_y:.2f}) dist={goal_distance:.2f}m')

    def mppi_planning(self, start_x, start_y, start_theta, goal_x, goal_y):
        """MPPI algorithm for trajectory optimization"""
        
        # Generate noise samples for control perturbations
        noise_v = np.random.normal(0, self.sigma_v, (self.num_samples, self.horizon))
        noise_omega = np.random.normal(0, self.sigma_omega, (self.num_samples, self.horizon))
        
        # Current control sequence (from previous iteration)
        current_controls = self.control_sequence.copy()
        
        costs = np.zeros(self.num_samples)
        
        # Evaluate each sampled trajectory
        for i in range(self.num_samples):
            # Perturbed control sequence
            perturbed_v = np.clip(current_controls[:, 0] + noise_v[i], 0, self.max_speed)
            perturbed_omega = np.clip(current_controls[:, 1] + noise_omega[i], -self.max_omega, self.max_omega)
            
            # Simulate trajectory with perturbed controls
            cost = self.evaluate_trajectory_mppi(start_x, start_y, start_theta, 
                                                perturbed_v, perturbed_omega, goal_x, goal_y)
            costs[i] = cost
        
        # Compute weights using softmax (temperature-based)
        weights = np.exp(-costs / self.temperature)
        weights = weights / np.sum(weights)
        
        # Update control sequence using weighted average
        new_controls = np.zeros((self.horizon, 2))
        for i in range(self.num_samples):
            perturbed_v = np.clip(current_controls[:, 0] + noise_v[i], 0, self.max_speed)
            perturbed_omega = np.clip(current_controls[:, 1] + noise_omega[i], -self.max_omega, self.max_omega)
            
            new_controls[:, 0] += weights[i] * perturbed_v
            new_controls[:, 1] += weights[i] * perturbed_omega
        
        # Shift control sequence for next iteration (rolling horizon)
        self.control_sequence[:-1] = new_controls[1:]
        self.control_sequence[-1] = new_controls[-1]  # Repeat last control
        
        # Return first control in optimized sequence
        return new_controls[0, 0], new_controls[0, 1]

    def evaluate_trajectory_mppi(self, start_x, start_y, start_theta, v_sequence, omega_sequence, goal_x, goal_y):
        """Evaluate cost of a trajectory for MPPI"""
        x, y, theta = start_x, start_y, start_theta
        total_cost = 0.0
        
        for t in range(self.horizon):
            # Simulate one step
            x += v_sequence[t] * np.cos(theta) * self.dt
            y += v_sequence[t] * np.sin(theta) * self.dt
            theta += omega_sequence[t] * self.dt
            
            # Goal cost (distance to goal)
            goal_dist = np.hypot(x - goal_x, y - goal_y)
            goal_cost = goal_dist * self.goal_weight
            
            # Control cost (energy)
            control_cost = (v_sequence[t]**2 + omega_sequence[t]**2) * self.control_weight
            
            # Smoothness cost (change in control)
            if t > 0:
                smooth_cost = ((v_sequence[t] - v_sequence[t-1])**2 + 
                             (omega_sequence[t] - omega_sequence[t-1])**2) * self.smooth_weight
            else:
                smooth_cost = 0.0
            
            total_cost += goal_cost + control_cost + smooth_cost
        
        # Terminal cost (final distance to goal)
        final_goal_dist = np.hypot(x - goal_x, y - goal_y)
        total_cost += final_goal_dist * self.goal_weight * 2.0  # Higher weight for final position
        
        return total_cost

    def log_action(self, cmd: Twist, robot_x: float, robot_y: float, robot_theta: float):
        """Log action data for pickle export"""
        current_time = time.time()
        relative_time = current_time - self.start_time
        
        action_data = {
            'timestamp': current_time,
            'relative_time': relative_time,
            'v': float(cmd.linear.x),
            'omega': float(cmd.angular.z),
            'x': float(robot_x),
            'y': float(robot_y),
            'theta': float(robot_theta)
        }
        
        self.action_log.append(action_data)
        
        if len(self.action_log) % 50 == 0:
            self.get_logger().info(f'📊 Logged {len(self.action_log)} actions')

    def create_output_directory(self):
        """Create output directory for metrics and pickle files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"/home/mrvik/dram_ws/mppi_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def update_metrics(self, robot_x, robot_y, cmd):
        """Update performance metrics"""
        current_pose = (robot_x, robot_y)
        
        if self.last_pose is not None:
            dx = robot_x - self.last_pose[0]
            dy = robot_y - self.last_pose[1]
            distance = math.hypot(dx, dy)
            self.metrics['total_distance'] += distance
            
            v = abs(cmd.linear.x)
            w = abs(cmd.angular.z)
            energy = (v * v + w * w) * 0.1
            self.metrics['total_energy'] += energy
        
        self.metrics['completion_time'] = time.time() - self.metrics['start_time']
        
        # Update camera frame metrics
        self.metrics['camera_frames_processed'] = self.camera_frame_count
        self.metrics['camera_frames_dropped'] = self.camera_dropped_frames
        self.metrics['camera_frames_synced'] = len(self.camera_action_log)
        
        self.last_pose = current_pose
        self.last_cmd = cmd

    def save_action_pickle(self):
        """Save action log to pickle file"""
        pickle_file = os.path.join(self.output_dir, 'action_log.pkl')
        
        data_to_save = {
            'actions': self.action_log,
            'camera_actions': self.camera_action_log,
            'metrics': self.metrics,
            'session_info': {
                'start_time': self.start_time,
                'end_time': time.time(),
                'total_actions': len(self.action_log),
                'total_camera_frames': len(self.camera_action_log),
                'node_name': self.get_name(),
                'output_dir': self.output_dir
            }
        }
        
        try:
            with open(pickle_file, 'wb') as f:
                pickle.dump(data_to_save, f)
            
            self.get_logger().info(f'💾 Action log saved to {pickle_file}')
            self.get_logger().info(f'📊 Total actions logged: {len(self.action_log)}')
            self.get_logger().info(f'📸 Camera frames processed: {self.camera_frame_count}')
            self.get_logger().info(f'📸 Camera frames dropped: {self.camera_dropped_frames}')
            self.get_logger().info(f'📸 Camera-action pairs synced: {len(self.camera_action_log)}')
            if self.camera_frame_count > 0:
                capture_rate = (len(self.camera_action_log) / self.camera_frame_count) * 100
                self.get_logger().info(f'📸 Camera frame capture rate: {capture_rate:.1f}%')
            
            json_file = os.path.join(self.output_dir, 'action_log.json')
            with open(json_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            self.get_logger().info(f'📝 Action log also saved as JSON: {json_file}')
            
        except Exception as e:
            self.get_logger().error(f'❌ Failed to save action log: {e}')

    def save_camera_actions_pickle(self):
        """Save camera-action synchronization data as separate pickle file"""
        camera_pickle_file = os.path.join(self.output_dir, 'camera_actions.pkl')
        
        # Create simple format: {timestamp: {'v': float, 'omega': float}}
        simple_camera_data = {}
        for timestamp, action_data in self.camera_action_log.items():
            simple_camera_data[timestamp] = {
                'v': action_data['v'],
                'omega': action_data['omega']
            }
        
        try:
            with open(camera_pickle_file, 'wb') as f:
                pickle.dump(simple_camera_data, f)
            
            self.get_logger().info(f'📸 Camera-action data saved to {camera_pickle_file} (simple format)')
            self.get_logger().info(f'📸 Format: {{timestamp: {{\'v\': float, \'omega\': float}}}}')
            
        except Exception as e:
            self.get_logger().error(f'❌ Failed to save camera-action data: {e}')

    def save_metrics(self):
        """Save performance metrics to file"""
        metrics_file = os.path.join(self.output_dir, 'mppi_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.get_logger().info(f'📊 Metrics saved to {metrics_file}')

def main(args=None):
    rclpy.init(args=args)
    node = MppiRosbagPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('🛑 Shutting down - saving data...')
        
        # Log final camera statistics
        if node.camera_callback_active:
            duration = time.time() - node.camera_callback_start_time
            node.get_logger().info(f'📸 Camera active for {duration:.1f}s, processed {node.camera_frame_count} frames')
        
        node.save_action_pickle()
        node.save_camera_actions_pickle()
        node.save_metrics()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
