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

class Nav2DwbRosbagPlannerNode(Node):
    def __init__(self):
        super().__init__('nav2_dwb_rosbag_planner_node')

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

        # Nav2 DWB parameters (based on Nav2 DWB planner)
        self.max_vel_x = 0.5
        self.min_vel_x = 0.0
        self.max_vel_theta = 1.0
        self.min_vel_theta = -1.0
        self.acc_lim_x = 2.5
        self.acc_lim_theta = 3.2
        self.decel_lim_x = -2.5
        self.decel_lim_theta = -3.2
        
        # Simulation parameters
        self.sim_time = 3.0  # Forward simulation time
        self.sim_granularity = 0.05  # Simulation step size
        self.vx_samples = 6  # Number of velocity samples
        self.vtheta_samples = 20  # Number of angular velocity samples
        
        # Scoring parameters (Nav2 style)
        self.path_distance_bias = 32.0
        self.goal_distance_bias = 24.0
        self.occdist_scale = 0.1
        self.forward_point_distance = 0.325
        self.stop_time_buffer = 0.2
        
        # Current velocity (for acceleration limits)
        self.current_vel = {'x': 0.0, 'theta': 0.0}

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
            'method_name': 'nav2_dwb_method',
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

        self.get_logger().info('Nav2 DWB Rosbag Planner Node initialized')
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
            self.current_vel = {'x': 0.0, 'theta': 0.0}
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
            self.current_vel = {'x': 0.0, 'theta': 0.0}
            self.log_action(cmd, robot_x, robot_y, robot_theta)
            return

        # Nav2 DWB: Generate velocity samples and find best trajectory
        best_v, best_omega = self.dwb_planning(robot_x, robot_y, robot_theta, target_x, target_y)

        # Publish best command
        cmd = Twist()
        cmd.linear.x = best_v
        cmd.angular.z = best_omega
        self.cmd_pub.publish(cmd)

        # Update current velocities and action for camera synchronization
        self.current_vel = {'x': best_v, 'theta': best_omega}
        self.current_action = {'v': float(best_v), 'omega': float(best_omega)}
        self.current_robot_pose = (float(robot_x), float(robot_y), float(robot_theta))

        # Log action with timestamp
        self.log_action(cmd, robot_x, robot_y, robot_theta)

        # Update metrics
        self.update_metrics(robot_x, robot_y, cmd)

        # Log planning result
        self.get_logger().debug(f'🚀 Nav2 DWB: v={best_v:.2f}, ω={best_omega:.2f} → '
                               f'Goal ({target_x:.2f}, {target_y:.2f}) dist={goal_distance:.2f}m')

    def dwb_planning(self, start_x, start_y, start_theta, goal_x, goal_y):
        """Nav2 DWB algorithm for trajectory optimization"""
        
        # Generate velocity samples considering acceleration limits
        dt = 0.1  # Control period
        
        # Calculate achievable velocity ranges based on current velocity and acceleration limits
        max_vel_x = min(self.max_vel_x, self.current_vel['x'] + self.acc_lim_x * dt)
        min_vel_x = max(self.min_vel_x, self.current_vel['x'] + self.decel_lim_x * dt)
        
        max_vel_theta = min(self.max_vel_theta, self.current_vel['theta'] + self.acc_lim_theta * dt)
        min_vel_theta = max(self.min_vel_theta, self.current_vel['theta'] + self.decel_lim_theta * dt)
        
        # Sample velocities
        if self.vx_samples > 1:
            vx_samples = np.linspace(min_vel_x, max_vel_x, self.vx_samples)
        else:
            vx_samples = [max_vel_x]
            
        if self.vtheta_samples > 1:
            vtheta_samples = np.linspace(min_vel_theta, max_vel_theta, self.vtheta_samples)
        else:
            vtheta_samples = [0.0]
        
        best_score = -float('inf')
        best_vx = 0.0
        best_vtheta = 0.0
        
        # Evaluate each velocity combination
        for vx in vx_samples:
            for vtheta in vtheta_samples:
                # Skip invalid combinations
                if vx < 0:
                    continue
                
                # Evaluate trajectory
                score = self.evaluate_trajectory_dwb(start_x, start_y, start_theta, 
                                                   vx, vtheta, goal_x, goal_y)
                
                if score > best_score:
                    best_score = score
                    best_vx = vx
                    best_vtheta = vtheta
        
        return best_vx, best_vtheta

    def evaluate_trajectory_dwb(self, start_x, start_y, start_theta, vx, vtheta, goal_x, goal_y):
        """Evaluate trajectory using Nav2 DWB scoring"""
        
        # Simulate trajectory
        x, y, theta = start_x, start_y, start_theta
        trajectory_points = []
        
        sim_steps = int(self.sim_time / self.sim_granularity)
        for _ in range(sim_steps):
            x += vx * np.cos(theta) * self.sim_granularity
            y += vx * np.sin(theta) * self.sim_granularity
            theta += vtheta * self.sim_granularity
            trajectory_points.append((x, y, theta))
        
        if not trajectory_points:
            return -float('inf')
        
        # Calculate Nav2 DWB scores
        
        # 1. Path Distance Score (distance from end of trajectory to goal)
        final_x, final_y, _ = trajectory_points[-1]
        path_distance = np.hypot(final_x - goal_x, final_y - goal_y)
        path_distance_score = -path_distance * self.path_distance_bias
        
        # 2. Goal Distance Score (distance from robot to goal)
        goal_distance = np.hypot(start_x - goal_x, start_y - goal_y)
        goal_distance_score = -goal_distance * self.goal_distance_bias
        
        # 3. Goal Alignment Score (how well aligned the robot is with goal direction)
        goal_angle = math.atan2(goal_y - start_y, goal_x - start_x)
        angle_diff = abs(self.normalize_angle(start_theta - goal_angle))
        goal_alignment_score = -angle_diff * 10.0
        
        # 4. Forward Point Distance (encourage forward motion)
        forward_point_x = start_x + self.forward_point_distance * np.cos(start_theta)
        forward_point_y = start_y + self.forward_point_distance * np.sin(start_theta)
        forward_dist = np.hypot(forward_point_x - goal_x, forward_point_y - goal_y)
        forward_score = -forward_dist * 5.0
        
        # 5. Speed preference (prefer higher speeds when safe)
        speed_score = vx * 2.0
        
        # 6. Oscillation penalty (penalize high angular velocities)
        oscillation_penalty = -abs(vtheta) * 1.0
        
        # Total score
        total_score = (path_distance_score + 
                      goal_distance_score + 
                      goal_alignment_score + 
                      forward_score + 
                      speed_score + 
                      oscillation_penalty)
        
        return total_score

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

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
        output_dir = f"/home/mrvik/dram_ws/nav2_dwb_results_{timestamp}"
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
        metrics_file = os.path.join(self.output_dir, 'nav2_dwb_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.get_logger().info(f'📊 Metrics saved to {metrics_file}')

def main(args=None):
    rclpy.init(args=args)
    node = Nav2DwbRosbagPlannerNode()

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
