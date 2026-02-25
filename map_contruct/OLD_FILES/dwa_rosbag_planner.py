#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from visualization_msgs.msg import MarkerArray
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
from collections import deque

class DwaRosbagPlannerNode(Node):
    def __init__(self):
        super().__init__('dwa_rosbag_planner_node')

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

        # Multi camera dead end detection
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
            '/argus/ar0234_front_left/image_raw',  # Adjust topic name as needed
            self.front_camera_callback,
            camera_qos
        )
        
        # Publisher for commands
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',  # Standard topic for rosbag compatibility
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

        # Action logging for pickle export
        self.action_log = []  # List of {'timestamp': float, 'v': float, 'omega': float, 'x': float, 'y': float, 'theta': float}
        self.start_time = time.time()
        
        # Camera-action synchronization
        self.camera_action_log = {}  # Dictionary: {camera_timestamp: {'v': float, 'omega': float, 'robot_pose': tuple}}
        self.current_action = {'v': 0.0, 'omega': 0.0}  # Most recent action
        self.current_robot_pose = (0.0, 0.0, 0.0)  # Most recent robot pose (x, y, theta)
        
        # Camera frame tracking
        self.camera_frame_count = 0
        self.camera_dropped_frames = 0
        self.last_camera_timestamp = None
        self.camera_callback_active = False
        self.camera_callback_start_time = None
        
        # Metrics collection
        self.metrics = {
            'start_time': self.start_time,
            'method_name': 'dram_rosbag_method',
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
            'completion_time': 0.0,
            'camera_frames_processed': 0,
            'camera_frames_dropped': 0,
            'camera_frames_synced': 0
        }
        self.last_pose = None
        self.last_cmd = Twist()
        
        # Create output directory
        self.output_dir = self.create_output_directory()

        self.get_logger().info('DWA Rosbag Planner Node initialized - logging actions to pickle')
        self.get_logger().info(f'Output directory: {self.output_dir}')

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
            # Fallback to origin if transform not available
            self.get_logger().debug(f'TF lookup failed: {e}')
            return 0.0, 0.0, 0.0, False

    def plan_and_publish(self):
        # Get current robot pose
        robot_x, robot_y, robot_theta, tf_success = self.get_robot_pose()
        
        if not tf_success:
            # Don't publish commands if we don't have valid pose
            return

        # Check if we need to go to recovery point
        if self.consecutive_dead_ends >= self.max_consecutive_dead_ends:
            if not self.is_in_recovery_mode:
                self.is_in_recovery_mode = True
                self.target_recovery_point = self.select_best_recovery_point(robot_x, robot_y)
                self.metrics['recovery_activations'] += 1
                if self.target_recovery_point:
                    self.get_logger().warn(f'🎯 Entering RECOVERY MODE - heading to recovery point with {self.target_recovery_point["open_paths"]} open paths')

        # Determine target position
        target_x, target_y = self.determine_target(robot_x, robot_y)
        
        if target_x is None or target_y is None:
            # No valid target - stop
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            
            # Update current action for camera synchronization
            self.current_action = {'v': 0.0, 'omega': 0.0}
            self.current_robot_pose = (float(robot_x), float(robot_y), float(robot_theta))
            
            self.log_action(cmd, robot_x, robot_y, robot_theta)
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

        # Update current action and robot pose for camera synchronization
        self.current_action = {'v': float(best_v), 'omega': float(best_omega)}
        self.current_robot_pose = (float(robot_x), float(robot_y), float(robot_theta))

        # Log action with timestamp
        self.log_action(cmd, robot_x, robot_y, robot_theta)

        # Update metrics
        self.update_metrics(robot_x, robot_y, cmd)

        # Log planning result
        mode = "RECOVERY" if self.is_in_recovery_mode else "EXPLORATION"
        distance_to_target = np.hypot(robot_x - target_x, robot_y - target_y)
        self.get_logger().debug(f'🚀 DWA [{mode}]: v={best_v:.2f}, ω={best_omega:.2f} → '
                               f'Target ({target_x:.2f}, {target_y:.2f}) dist={distance_to_target:.2f}m')

    def determine_target(self, robot_x, robot_y):
        """Determine target position based on current mode"""
        # If we're in recovery mode, navigate to recovery point
        if self.is_in_recovery_mode and self.target_recovery_point:
            # Check if we've reached the recovery point
            distance_to_recovery = np.hypot(robot_x - self.target_recovery_point['x'],
                                          robot_y - self.target_recovery_point['y'])

            if distance_to_recovery < 3.0:  # Within 3m of recovery point
                self.get_logger().info(f'✅ Reached recovery point! Exiting recovery mode')
                self.is_in_recovery_mode = False
                self.target_recovery_point = None
                self.consecutive_dead_ends = 0  # Reset dead-end counter
                return None, None  # Stop briefly to re-evaluate

            # Navigate to recovery point
            return self.target_recovery_point['x'], self.target_recovery_point['y']

        elif self.current_goal:
            # Use goal from Goal Generator
            return self.current_goal['x'], self.current_goal['y']
            
        elif self.recovery_points:
            # Fallback: use recovery points if available
            target_recovery = self.select_best_recovery_point(robot_x, robot_y)
            if target_recovery:
                return target_recovery['x'], target_recovery['y']
            else:
                self.get_logger().warn('🚨 No recovery points within range')
                return None, None
        else:
            # No goals or recovery points available
            self.get_logger().debug('🔄 Waiting for goals from Goal Generator...')
            return None, None

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
            'theta': float(robot_theta),
            'is_recovery_mode': self.is_in_recovery_mode,
            'dead_end_count': self.dead_end_count,
            'consecutive_dead_ends': self.consecutive_dead_ends
        }
        
        self.action_log.append(action_data)
        
        # Log every 50 actions to avoid spam
        if len(self.action_log) % 50 == 0:
            self.get_logger().info(f'📊 Logged {len(self.action_log)} actions')

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

    def select_best_recovery_point(self, robot_x, robot_y):
        """
        Select best recovery point using the new logic:
        1. Always choose nearest point first
        2. If there are multiple points at same distance, choose one with most open paths
        3. Return the selected recovery point
        """
        if not self.recovery_points:
            return None

        # Find all recovery points within max distance
        valid_points = []
        for rp in self.recovery_points:
            distance = np.hypot(rp['x'] - robot_x, rp['y'] - robot_y)
            if distance <= self.max_recovery_distance:
                valid_points.append({
                    'point': rp,
                    'distance': distance
                })

        if not valid_points:
            return None

        # Sort by distance (nearest first)
        valid_points.sort(key=lambda x: x['distance'])

        # Find the minimum distance
        min_distance = valid_points[0]['distance']

        # Get all points at minimum distance
        nearest_points = [vp for vp in valid_points if vp['distance'] == min_distance]

        # Among nearest points, choose the one with most open paths
        best_point = max(nearest_points, key=lambda x: x['point']['open_paths'])

        selected_point = best_point['point']
        self.get_logger().info(f'🎯 Selected recovery point: {selected_point["open_paths"]} open paths '
                              f'at ({selected_point["x"]:.2f}, {selected_point["y"]:.2f}) '
                              f'distance: {best_point["distance"]:.2f}m')

        return selected_point

    def create_output_directory(self):
        """Create output directory for metrics and pickle files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"/home/mrvik/dram_ws/dram_rosbag_results_{timestamp}"
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
        
        # Update camera frame metrics
        self.metrics['camera_frames_processed'] = self.camera_frame_count
        self.metrics['camera_frames_dropped'] = self.camera_dropped_frames
        self.metrics['camera_frames_synced'] = len(self.camera_action_log)
        
        self.last_pose = current_pose
        self.last_cmd = cmd

    def save_action_pickle(self):
        """Save action log to pickle file"""
        pickle_file = os.path.join(self.output_dir, 'action_log.pkl')
        
        # Create comprehensive data structure
        data_to_save = {
            'actions': self.action_log,
            'camera_actions': self.camera_action_log,  # NEW: Camera-synchronized actions
            'metrics': self.metrics,
            'session_info': {
                'start_time': self.start_time,
                'end_time': time.time(),
                'total_actions': len(self.action_log),
                'total_camera_frames': len(self.camera_action_log),  # NEW: Camera frame count
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
            
            # Also save as JSON for human readability
            json_file = os.path.join(self.output_dir, 'action_log.json')
            with open(json_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            self.get_logger().info(f'📝 Action log also saved as JSON: {json_file}')
            
        except Exception as e:
            self.get_logger().error(f'❌ Failed to save action log: {e}')

    def save_camera_actions_pickle(self):
        """Save camera-action synchronization data as separate pickle file"""
        camera_pickle_file = os.path.join(self.output_dir, 'camera_actions.pkl')
        
        # Create data structure: {timestamp: {'v': float, 'omega': float, 'robot_pose': tuple}}
        camera_data = {}
        for timestamp, action_data in self.camera_action_log.items():
            camera_data[timestamp] = {
                'v': action_data['v'],
                'omega': action_data['omega'],
                'robot_pose': action_data['robot_pose']
            }
        
        try:
            with open(camera_pickle_file, 'wb') as f:
                pickle.dump(camera_data, f)
            
            self.get_logger().info(f'📸 Camera-action data saved to {camera_pickle_file}')
            
        except Exception as e:
            self.get_logger().error(f'❌ Failed to save camera-action data: {e}')

    def save_metrics(self):
        """Save performance metrics to file"""
        metrics_file = os.path.join(self.output_dir, 'dram_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.get_logger().info(f'📊 Metrics saved to {metrics_file}')

def main(args=None):
    rclpy.init(args=args)
    node = DwaRosbagPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save all data before shutdown
        node.get_logger().info('🛑 Shutting down - saving data...')
        
        # Log final camera statistics
        if node.camera_callback_active:
            duration = time.time() - node.camera_callback_start_time
            node.get_logger().info(f'📸 Camera active for {duration:.1f}s, processed {node.camera_frame_count} frames')
        
        node.save_action_pickle()
        node.save_camera_actions_pickle()  # NEW: Save camera-action sync data
        node.save_metrics()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

