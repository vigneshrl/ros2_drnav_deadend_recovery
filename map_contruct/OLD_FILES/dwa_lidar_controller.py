#!/usr/bin/env python3

"""
DWA Controller with LiDAR-based Dead-End Detection

This controller uses only LiDAR data to detect dead-ends and navigate,
providing a baseline for comparison against vision-based models.
It implements:
1. LiDAR-based obstacle detection in 3 directions (front, left, right)
2. Dead-end detection using distance thresholds
3. DWA local path planning for navigation
4. Recovery point generation and navigation
5. Performance metrics collection
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformListener, Buffer
import numpy as np
import math
import time
import json
import os
from typing import Dict, List, Optional

class DWALidarController(Node):
    def __init__(self):
        super().__init__('dwa_lidar_controller')
        
        # LiDAR subscriber
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        # Goal subscriber (from goal_generator)
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.path_status_pub = self.create_publisher(Float32MultiArray, '/dwa_lidar/path_status', 10)
        self.recovery_points_pub = self.create_publisher(Float32MultiArray, '/dwa_lidar/recovery_points', 10)
        self.dead_end_pub = self.create_publisher(Bool, '/dwa_lidar/is_dead_end', 10)
        self.planned_path_pub = self.create_publisher(Path, '/dwa_lidar/planned_path', 10)
        
        # Transform listener for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # LiDAR data storage
        self.current_scan = None
        
        # Detection parameters
        self.obstacle_threshold = 1.0  # Distance threshold for obstacles (meters)
        self.dead_end_threshold = 3   # Consecutive dead-end detections to trigger recovery
        self.path_width = 0.8  # Required path width for navigation
        
        # DWA parameters
        self.max_speed = 0.8
        self.max_omega = 1.2
        self.dt = 0.1
        self.prediction_time = 2.0  # Seconds to predict ahead
        self.velocity_samples = 10
        self.angular_samples = 20
        
        # Recovery points storage and management
        self.recovery_points = []
        self.consecutive_dead_ends = 0
        self.is_in_recovery_mode = False
        self.target_recovery_point = None
        self.recovery_distance_threshold = 2.0
        
        # Control state
        self.is_dead_end = False
        self.path_probabilities = [0.5, 0.5, 0.5]  # [front, left, right]
        self.robot_pose = None
        
        # Goal management
        self.current_goal = None
        self.goal_tolerance = 1.0  # meters
        
        # Performance metrics
        self.metrics = {
            'start_time': time.time(),
            'total_distance': 0.0,
            'total_energy': 0.0,
            'dead_end_detections': 0,
            'false_positive_dead_ends': 0,
            'recovery_point_detections': 0,
            'recovery_activations': 0,
            'success': False,
            'completion_time': 0.0
        }
        self.last_pose = None
        self.last_cmd = Twist()
        
        # Create output directory for results
        self.output_dir = self.create_output_directory()
        
        # Create timer for main control loop
        self.create_timer(0.1, self.control_loop)  # 10Hz
        
        self.get_logger().info('🚀 DWA LiDAR Controller initialized')
        self.get_logger().info(f'📡 Detection thresholds: obstacle={self.obstacle_threshold}m, dead-end={self.dead_end_threshold} consecutive')
        self.get_logger().info('🎯 Subscribing to /scan and /move_base_simple/goal')
        self.get_logger().info('✅ Ready to receive goals from goal_generator')
    
    def create_output_directory(self):
        """Create timestamped output directory"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"/home/mrvik/dram_ws/evaluation_results/dwa_lidar_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def goal_callback(self, msg: PoseStamped):
        """Handle new goal from goal_generator"""
        self.current_goal = {
            'x': msg.pose.position.x,
            'y': msg.pose.position.y,
            'timestamp': time.time()
        }
        self.get_logger().info(f'🎯 New goal received: ({self.current_goal["x"]:.1f}, {self.current_goal["y"]:.1f})')

    def scan_callback(self, msg: LaserScan):
        """Process LiDAR scan data"""
        self.current_scan = msg
        
        # Analyze scan for dead-end detection
        self.analyze_scan_for_paths(msg)

    def analyze_scan_for_paths(self, scan: LaserScan):
        """Analyze LiDAR scan to detect path availability in 3 directions"""
        if not scan.ranges:
            return
        
        num_rays = len(scan.ranges)
        
        # Define angular sectors for front, left, right
        # Assuming 0 degrees is front, positive angles are left
        front_start = int(num_rays * 0.4)  # -36 degrees
        front_end = int(num_rays * 0.6)    # +36 degrees
        left_start = int(num_rays * 0.75)  # +45 degrees  
        left_end = int(num_rays * 0.95)    # +135 degrees
        right_start = int(num_rays * 0.05) # -135 degrees
        right_end = int(num_rays * 0.25)   # -45 degrees
        
        # Check path availability in each direction
        front_clear = self.check_sector_clear(scan, front_start, front_end)
        left_clear = self.check_sector_clear(scan, left_start, left_end)
        right_clear = self.check_sector_clear(scan, right_start, right_end)
        
        # Convert to probabilities (1.0 = clear, 0.0 = blocked)
        self.path_probabilities = [
            1.0 if front_clear else 0.0,
            1.0 if left_clear else 0.0, 
            1.0 if right_clear else 0.0
        ]
        
        # Determine dead-end status
        open_paths = sum(self.path_probabilities)
        current_dead_end = (open_paths == 0)
        is_recovery_point = (open_paths >= 2)
        
        # Update dead-end tracking
        if current_dead_end:
            self.consecutive_dead_ends += 1
            self.metrics['dead_end_detections'] += 1
            self.is_dead_end = True
        else:
            self.consecutive_dead_ends = 0
            self.is_dead_end = False
        
        # Store recovery point if detected
        if is_recovery_point and self.robot_pose:
            self.add_recovery_point(self.robot_pose[0], self.robot_pose[1], int(open_paths))
        
        # Publish path status
        path_msg = Float32MultiArray()
        path_msg.data = self.path_probabilities
        self.path_status_pub.publish(path_msg)
        
        # Publish dead-end status
        dead_end_msg = Bool()
        dead_end_msg.data = self.is_dead_end
        self.dead_end_pub.publish(dead_end_msg)
        
        self.get_logger().debug(f'LiDAR Analysis: F={front_clear}, L={left_clear}, R={right_clear} | '
                               f'Dead-end: {current_dead_end} | Recovery: {is_recovery_point} | '
                               f'Consecutive: {self.consecutive_dead_ends}')

    def check_sector_clear(self, scan: LaserScan, start_idx: int, end_idx: int) -> bool:
        """Check if a sector is clear of obstacles"""
        if start_idx > end_idx:  # Handle wrap-around
            ranges = scan.ranges[start_idx:] + scan.ranges[:end_idx]
        else:
            ranges = scan.ranges[start_idx:end_idx]
        
        # Filter out invalid readings
        valid_ranges = [r for r in ranges if not math.isnan(r) and not math.isinf(r) and r > 0]
        
        if not valid_ranges:
            return False
        
        # Check if minimum distance in sector is above threshold
        min_distance = min(valid_ranges)
        return min_distance > self.obstacle_threshold

    def add_recovery_point(self, x: float, y: float, open_paths: int):
        """Add a recovery point if not already exists nearby"""
        # Check for duplicates within 1.5m
        for rp in self.recovery_points:
            if math.hypot(rp['x'] - x, rp['y'] - y) < 1.5:
                return  # Already exists
        
        recovery_point = {
            'x': x,
            'y': y,
            'open_paths': open_paths,
            'timestamp': time.time()
        }
        self.recovery_points.append(recovery_point)
        self.metrics['recovery_point_detections'] += 1
        
        # Publish recovery points
        self.publish_recovery_points()
        
        self.get_logger().info(f'🎯 New recovery point: ({x:.2f}, {y:.2f}) with {open_paths} open paths')

    def publish_recovery_points(self):
        """Publish all recovery points"""
        recovery_data = []
        for rp in self.recovery_points:
            # Type: 1 for 2+ openings, 2 for 1 opening
            point_type = 1 if rp['open_paths'] >= 2 else 2
            recovery_data.extend([float(point_type), float(rp['x']), float(rp['y'])])
        
        msg = Float32MultiArray()
        msg.data = recovery_data
        self.recovery_points_pub.publish(msg)

    def get_robot_pose(self):
        """Get current robot pose"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'odom', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
            )
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            # Extract yaw from quaternion
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            
            return (x, y, yaw)
        except Exception:
            return None

    def select_recovery_point(self):
        """Select best recovery point based on distance and open paths"""
        if not self.recovery_points or not self.robot_pose:
            return None
        
        robot_x, robot_y, _ = self.robot_pose
        
        # Calculate scores for each recovery point
        best_point = None
        best_score = -float('inf')
        
        for rp in self.recovery_points:
            distance = math.hypot(rp['x'] - robot_x, rp['y'] - robot_y)
            
            # Skip if too far
            if distance > 15.0:
                continue
            
            # Score: prefer closer points with more open paths
            distance_score = 1.0 / (1.0 + distance)  # Closer = higher score
            openness_score = rp['open_paths'] / 3.0   # More open paths = higher score
            
            total_score = distance_score * 0.7 + openness_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_point = rp
        
        return best_point

    def dwa_planning(self, goal_x: float, goal_y: float) -> Twist:
        """DWA trajectory planning"""
        if not self.robot_pose or not self.current_scan:
            return Twist()
        
        robot_x, robot_y, robot_yaw = self.robot_pose
        
        best_cmd = Twist()
        best_score = -float('inf')
        
        # Sample velocity space
        for v in np.linspace(0, self.max_speed, self.velocity_samples):
            for w in np.linspace(-self.max_omega, self.max_omega, self.angular_samples):
                # Simulate trajectory
                x, y, yaw = robot_x, robot_y, robot_yaw
                trajectory_valid = True
                
                # Check trajectory for collisions
                for t in range(int(self.prediction_time / self.dt)):
                    x += v * math.cos(yaw) * self.dt
                    y += v * math.sin(yaw) * self.dt
                    yaw += w * self.dt
                    
                    # Simple collision check (in real implementation, use proper obstacle checking)
                    if self.check_collision(x, y):
                        trajectory_valid = False
                        break
                
                if not trajectory_valid:
                    continue
                
                # Calculate score
                # Goal distance (closer is better)
                goal_dist = math.hypot(x - goal_x, y - goal_y)
                goal_score = 1.0 / (1.0 + goal_dist)
                
                # Velocity preference (prefer forward motion)
                velocity_score = v / self.max_speed
                
                # Smoothness (prefer low angular velocity)
                smooth_score = 1.0 - abs(w) / self.max_omega
                
                # Total score
                total_score = goal_score * 0.6 + velocity_score * 0.3 + smooth_score * 0.1
                
                if total_score > best_score:
                    best_score = total_score
                    best_cmd.linear.x = v
                    best_cmd.angular.z = w
        
        return best_cmd

    def check_collision(self, x: float, y: float) -> bool:
        """Simple collision check - in practice, would use proper obstacle detection"""
        if not self.current_scan or not self.robot_pose:
            return False
        
        # Simple check: if too close to robot's current position, assume no collision
        # In real implementation, transform scan data to check specific positions
        robot_x, robot_y, _ = self.robot_pose
        dist_to_robot = math.hypot(x - robot_x, y - robot_y)
        
        # Very simple heuristic - assume collision if trajectory goes too far from current safe area
        return dist_to_robot > 3.0

    def control_loop(self):
        """Main control loop"""
        self.robot_pose = self.get_robot_pose()
        
        if not self.robot_pose:
            return
        
        robot_x, robot_y, robot_yaw = self.robot_pose
        
        # Update metrics
        self.update_metrics()
        
        # Check if we need recovery mode
        if self.consecutive_dead_ends >= self.dead_end_threshold:
            if not self.is_in_recovery_mode:
                self.is_in_recovery_mode = True
                self.target_recovery_point = self.select_recovery_point()
                self.metrics['recovery_activations'] += 1
                
                if self.target_recovery_point:
                    self.get_logger().warn(f'🚨 Entering RECOVERY MODE - heading to ({self.target_recovery_point["x"]:.2f}, {self.target_recovery_point["y"]:.2f})')
                else:
                    self.get_logger().error('🚨 No recovery points available!')
        
        # Determine goal
        if self.is_in_recovery_mode and self.target_recovery_point:
            goal_x = self.target_recovery_point['x']
            goal_y = self.target_recovery_point['y']
            
            # Check if reached recovery point
            distance_to_recovery = math.hypot(robot_x - goal_x, robot_y - goal_y)
            if distance_to_recovery < self.recovery_distance_threshold:
                self.get_logger().info('✅ Reached recovery point - exiting recovery mode')
                self.is_in_recovery_mode = False
                self.target_recovery_point = None
                self.consecutive_dead_ends = 0
        elif self.current_goal:
            # Use goal from goal_generator (MAIN MODE)
            goal_x = self.current_goal['x']
            goal_y = self.current_goal['y']
            
            # Check if reached goal
            distance_to_goal = math.hypot(robot_x - goal_x, robot_y - goal_y)
            if distance_to_goal < self.goal_tolerance:
                self.get_logger().info(f'✅ Reached goal ({goal_x:.1f}, {goal_y:.1f})')
                self.current_goal = None  # Clear completed goal
        else:
            # No goal available - wait or explore
            goal_x = robot_x + 1.0 * math.cos(robot_yaw)
            goal_y = robot_y + 1.0 * math.sin(robot_yaw)
            
            # Log waiting status occasionally
            if not hasattr(self, 'last_waiting_log') or time.time() - self.last_waiting_log > 5.0:
                self.get_logger().info('⏳ Waiting for goals from goal_generator...')
                self.last_waiting_log = time.time()
        
        # Plan trajectory using DWA
        cmd = self.dwa_planning(goal_x, goal_y)
        
        # If no valid trajectory found, try emergency maneuver
        if cmd.linear.x == 0 and cmd.angular.z == 0:
            if not self.is_dead_end:
                # Try backing up and turning
                cmd.linear.x = -0.2
                cmd.angular.z = 0.5
            else:
                # Complete stop in dead-end
                cmd = Twist()
        
        # Publish command
        self.cmd_pub.publish(cmd)
        self.last_cmd = cmd
        
        # Log status
        mode = "RECOVERY" if self.is_in_recovery_mode else "EXPLORATION"
        self.get_logger().debug(f'DWA [{mode}]: v={cmd.linear.x:.2f}, w={cmd.angular.z:.2f} | '
                               f'Goal: ({goal_x:.1f}, {goal_y:.1f}) | Dead-ends: {self.consecutive_dead_ends}')

    def update_metrics(self):
        """Update performance metrics"""
        if not self.robot_pose or not self.last_pose:
            self.last_pose = self.robot_pose
            return
        
        # Calculate distance traveled
        dx = self.robot_pose[0] - self.last_pose[0]
        dy = self.robot_pose[1] - self.last_pose[1]
        distance = math.hypot(dx, dy)
        self.metrics['total_distance'] += distance
        
        # Calculate energy consumption (simplified)
        v = abs(self.last_cmd.linear.x)
        w = abs(self.last_cmd.angular.z)
        energy = (v * v + w * w) * 0.1  # Simplified energy model
        self.metrics['total_energy'] += energy
        
        # Update completion time
        self.metrics['completion_time'] = time.time() - self.metrics['start_time']
        
        self.last_pose = self.robot_pose

    def save_metrics(self):
        """Save performance metrics to file"""
        metrics_file = os.path.join(self.output_dir, 'dwa_lidar_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.get_logger().info(f'📊 Metrics saved to {metrics_file}')

def main(args=None):
    rclpy.init(args=args)
    node = DWALidarController()
    
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