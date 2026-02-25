#!/usr/bin/env python3

"""
DRAM-Aware Goal Generator

This goal generator uses the same basic exploration logic for all methods,
but gives DRAM the advantage of using its costmap intelligence while 
baselines use only basic laser-based obstacle avoidance.

Fair comparison with DRAM's natural advantages intact!
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray
from tf2_ros import TransformListener, Buffer
import numpy as np
import math
import time
from typing import List, Tuple, Optional

class DramAwareGoalGenerator(Node):
    def __init__(self):
        super().__init__('dram_aware_goal_generator')
        
        # Declare parameters to detect which method is running
        self.declare_parameter('method_type', 'dram')  # 'dram', 'vanilla_dwa', 'mppi', 'nav2_dwb'
        self.method_type = self.get_parameter('method_type').get_parameter_value().string_value
        
        # Parameters
        self.goal_distance = 3.0  # Distance to generate goals (meters)
        self.goal_generation_rate = 5.0  # Hz
        self.laser_range_max = 10.0  # Maximum laser range to consider
        self.min_obstacle_distance = 1.5  # Minimum distance from obstacles
        
        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
        
        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        
        # DRAM-specific subscriber (only for DRAM method)
        self.dram_costmap = None
        if self.method_type == 'dram':
            self.costmap_sub = self.create_subscription(
                MarkerArray,
                '/dram_exploration_map',
                self.costmap_callback,
                10
            )
            self.get_logger().info('🧠 DRAM mode: Using costmap intelligence')
        else:
            self.get_logger().info(f'🔧 Baseline mode ({self.method_type}): Using laser-only navigation')
        
        # Transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Internal state
        self.laser_data = None
        self.current_goal = None
        self.goal_start_time = None
        self.exploration_angle = 0.0  # Current exploration direction
        
        # Timer for goal generation
        self.create_timer(1.0 / self.goal_generation_rate, self.generate_goal)
        
        self.get_logger().info(f'DRAM-Aware Goal Generator initialized for {self.method_type}')
        self.get_logger().info(f'Goal distance: {self.goal_distance}m, Rate: {self.goal_generation_rate}Hz')

    def laser_callback(self, msg):
        """Store laser scan data"""
        self.laser_data = msg

    def costmap_callback(self, msg):
        """Store DRAM costmap data (only for DRAM method)"""
        if self.method_type != 'dram':
            return
            
        # Parse costmap from MarkerArray (same as your original code)
        self.dram_costmap = {}
        for marker in msg.markers:
            if marker.ns == "exploration_heatmap":
                for i, point in enumerate(marker.points):
                    x = point.x
                    y = point.y
                    
                    if i < len(marker.colors):
                        color = marker.colors[i]
                        # Check if this is a recovery point
                        is_recovery = (color.b > 0.5)
                        
                        if is_recovery:
                            if color.r > 0.4:  # Purple (3+ paths)
                                open_paths = 3
                            elif color.b > 0.7:  # Dark blue (2 paths)
                                open_paths = 2
                            else:  # Light blue (1 path)
                                open_paths = 1
                        else:
                            open_paths = 0
                        
                        self.dram_costmap[(x, y)] = {
                            'safety': 1.0 if color.g > 0.5 else 0.0,
                            'is_recovery': is_recovery,
                            'open_paths': open_paths
                        }
        
        if len(self.dram_costmap) > 0:
            self.get_logger().debug(f'🗺️ Updated DRAM costmap: {len(self.dram_costmap)} cells')

    def get_robot_pose(self):
        """Get current robot pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'odom', rclpy.time.Time(), 
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            
            # Extract yaw from quaternion
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            robot_theta = math.atan2(2.0 * (qw * qz + qx * qy), 
                                   1.0 - 2.0 * (qy * qy + qz * qz))
            return robot_x, robot_y, robot_theta, True
        except Exception as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return 0.0, 0.0, 0.0, False

    def generate_goal(self):
        """Generate a new goal based on method type"""
        
        # Get robot pose
        robot_x, robot_y, robot_theta, tf_success = self.get_robot_pose()
        if not tf_success:
            return
        
        # Check if current goal is still valid
        if self.current_goal is not None and self.goal_start_time is not None:
            goal_x = self.current_goal['x']
            goal_y = self.current_goal['y']
            distance_to_goal = math.hypot(robot_x - goal_x, robot_y - goal_y)
            goal_age = time.time() - self.goal_start_time
            
            if distance_to_goal > 1.0 and goal_age < 10.0:
                return  # Keep current goal
            elif distance_to_goal <= 1.0:
                self.get_logger().info(f'✅ Goal reached! Distance: {distance_to_goal:.2f}m')
        
        # Generate new goal based on method type
        if self.method_type == 'dram':
            new_goal_x, new_goal_y = self.find_dram_goal(robot_x, robot_y, robot_theta)
        else:
            new_goal_x, new_goal_y = self.find_baseline_goal(robot_x, robot_y, robot_theta)
        
        if new_goal_x is not None and new_goal_y is not None:
            # Create and publish goal
            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.header.frame_id = 'map'
            goal_msg.pose.position.x = new_goal_x
            goal_msg.pose.position.y = new_goal_y
            goal_msg.pose.position.z = 0.0
            
            # Set orientation towards goal
            goal_angle = math.atan2(new_goal_y - robot_y, new_goal_x - robot_x)
            goal_msg.pose.orientation.z = math.sin(goal_angle / 2.0)
            goal_msg.pose.orientation.w = math.cos(goal_angle / 2.0)
            
            self.goal_pub.publish(goal_msg)
            
            # Update internal state
            self.current_goal = {'x': new_goal_x, 'y': new_goal_y}
            self.goal_start_time = time.time()
            
            distance = math.hypot(new_goal_x - robot_x, new_goal_y - robot_y)
            method_symbol = "🧠" if self.method_type == 'dram' else "🔧"
            self.get_logger().info(f'{method_symbol} New goal: ({new_goal_x:.2f}, {new_goal_y:.2f}), '
                                 f'distance: {distance:.2f}m')
        else:
            self.get_logger().warn('⚠️ Could not find valid goal')

    def find_dram_goal(self, robot_x, robot_y, robot_theta):
        """Find goal using DRAM costmap intelligence"""
        
        # First, try to use DRAM costmap if available
        if self.dram_costmap:
            # Look for recovery points first (DRAM's advantage!)
            recovery_points = []
            safe_points = []
            
            for (x, y), data in self.dram_costmap.items():
                distance = math.hypot(x - robot_x, y - robot_y)
                if 1.0 <= distance <= self.goal_distance:
                    if data['is_recovery'] and data['open_paths'] > 0:
                        recovery_points.append({
                            'x': x, 'y': y, 'distance': distance,
                            'open_paths': data['open_paths'],
                            'score': data['open_paths'] / distance  # Prefer closer points with more paths
                        })
                    elif data['safety'] > 0.5:
                        safe_points.append({
                            'x': x, 'y': y, 'distance': distance,
                            'score': 1.0 / distance  # Prefer closer safe points
                        })
            
            # Prioritize recovery points (DRAM's key advantage)
            if recovery_points:
                best_point = max(recovery_points, key=lambda p: p['score'])
                self.get_logger().info(f'🎯 DRAM: Selected recovery point with {best_point["open_paths"]} paths')
                return best_point['x'], best_point['y']
            
            # Use safe exploration points
            if safe_points:
                best_point = max(safe_points, key=lambda p: p['score'])
                self.get_logger().debug('🗺️ DRAM: Selected safe exploration point')
                return best_point['x'], best_point['y']
        
        # Fallback to laser-based exploration (same as baselines)
        self.get_logger().debug('📡 DRAM: Fallback to laser-based goal generation')
        return self.find_baseline_goal(robot_x, robot_y, robot_theta)

    def find_baseline_goal(self, robot_x, robot_y, robot_theta):
        """Find goal using basic laser-based exploration (for baselines)"""
        
        # Use spiral exploration pattern (same as simple goal generator)
        self.exploration_angle += math.pi / 4  # 45 degree increments
        if self.exploration_angle > 2 * math.pi:
            self.exploration_angle -= 2 * math.pi
        
        # Try multiple angles around the exploration direction
        angles_to_try = [
            self.exploration_angle,
            self.exploration_angle + math.pi / 6,
            self.exploration_angle - math.pi / 6,
            self.exploration_angle + math.pi / 3,
            self.exploration_angle - math.pi / 3,
        ]
        
        for angle in angles_to_try:
            goal_x = robot_x + self.goal_distance * math.cos(angle)
            goal_y = robot_y + self.goal_distance * math.sin(angle)
            
            if self.is_goal_valid_laser(robot_x, robot_y, goal_x, goal_y):
                return goal_x, goal_y
        
        return None, None

    def is_goal_valid_laser(self, robot_x, robot_y, goal_x, goal_y):
        """Check if goal is valid using laser data (baseline validation)"""
        
        if self.laser_data is None:
            return True
        
        # Check distance
        distance = math.hypot(goal_x - robot_x, goal_y - robot_y)
        if distance < 1.0 or distance > self.goal_distance + 1.0:
            return False
        
        # Check if goal direction has obstacles using laser data
        goal_angle = math.atan2(goal_y - robot_y, goal_x - robot_x)
        robot_theta = 0.0  # Simplified - assume robot facing forward
        
        relative_angle = goal_angle - robot_theta
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi
        
        # Check laser data in that direction
        angle_min = self.laser_data.angle_min
        angle_max = self.laser_data.angle_max
        
        if angle_min <= relative_angle <= angle_max:
            angle_increment = self.laser_data.angle_increment
            index = int((relative_angle - angle_min) / angle_increment)
            
            if 0 <= index < len(self.laser_data.ranges):
                range_val = self.laser_data.ranges[index]
                if not math.isnan(range_val) and not math.isinf(range_val):
                    if range_val < self.min_obstacle_distance:
                        return False
        
        return True

def main(args=None):
    rclpy.init(args=args)
    node = DramAwareGoalGenerator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
