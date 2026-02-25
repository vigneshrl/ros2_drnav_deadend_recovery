#!/usr/bin/env python3

"""
Simple Goal Generator for Baseline Methods

This is a basic goal generator that provides simple waypoints for comparison studies.
It does NOT use:
- DRAM costmaps
- Recovery points  
- Dead-end detection
- Complex scoring

It simply generates goals in a basic exploration pattern or towards frontiers.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformListener, Buffer
import numpy as np
import math
import time
from typing import List, Tuple, Optional

class SimpleGoalGenerator(Node):
    def __init__(self):
        super().__init__('simple_goal_generator')
        
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
        
        # Transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Internal state
        self.laser_data = None
        self.current_goal = None
        self.goal_start_time = None
        self.exploration_angle = 0.0  # Current exploration direction
        self.exploration_pattern = 'spiral'  # 'spiral', 'random', 'frontier'
        
        # Timer for goal generation
        self.create_timer(1.0 / self.goal_generation_rate, self.generate_goal)
        
        self.get_logger().info('Simple Goal Generator initialized')
        self.get_logger().info(f'Goal distance: {self.goal_distance}m')
        self.get_logger().info(f'Generation rate: {self.goal_generation_rate}Hz')

    def laser_callback(self, msg):
        """Store laser scan data"""
        self.laser_data = msg

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
        """Generate a new goal based on simple exploration strategy"""
        
        # Get robot pose
        robot_x, robot_y, robot_theta, tf_success = self.get_robot_pose()
        if not tf_success:
            return
        
        # Check if current goal is still valid (not reached and not too old)
        if self.current_goal is not None and self.goal_start_time is not None:
            # Check if goal is reached
            goal_x = self.current_goal['x']
            goal_y = self.current_goal['y']
            distance_to_goal = math.hypot(robot_x - goal_x, robot_y - goal_y)
            
            # Check if goal is too old (timeout)
            goal_age = time.time() - self.goal_start_time
            
            if distance_to_goal > 1.0 and goal_age < 10.0:  # Goal not reached and not too old
                return  # Keep current goal
            elif distance_to_goal <= 1.0:
                self.get_logger().info(f'✅ Goal reached! Distance: {distance_to_goal:.2f}m')
        
        # Generate new goal
        new_goal_x, new_goal_y = self.find_new_goal(robot_x, robot_y, robot_theta)
        
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
            self.get_logger().info(f'🎯 New goal: ({new_goal_x:.2f}, {new_goal_y:.2f}), '
                                 f'distance: {distance:.2f}m')
        else:
            self.get_logger().warn('⚠️ Could not find valid goal')

    def find_new_goal(self, robot_x, robot_y, robot_theta):
        """Find a new goal using simple exploration strategies"""
        
        if self.exploration_pattern == 'spiral':
            return self.spiral_exploration(robot_x, robot_y, robot_theta)
        elif self.exploration_pattern == 'random':
            return self.random_exploration(robot_x, robot_y, robot_theta)
        elif self.exploration_pattern == 'frontier':
            return self.frontier_exploration(robot_x, robot_y, robot_theta)
        else:
            return self.spiral_exploration(robot_x, robot_y, robot_theta)

    def spiral_exploration(self, robot_x, robot_y, robot_theta):
        """Generate goals in a spiral pattern for systematic exploration"""
        
        # Increment exploration angle for spiral pattern
        self.exploration_angle += math.pi / 4  # 45 degree increments
        if self.exploration_angle > 2 * math.pi:
            self.exploration_angle -= 2 * math.pi
            # Increase distance slightly for expanding spiral
            self.goal_distance = min(self.goal_distance + 0.5, 5.0)
        
        # Try multiple angles around the spiral direction
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
            
            if self.is_goal_valid(robot_x, robot_y, goal_x, goal_y):
                return goal_x, goal_y
        
        return None, None

    def random_exploration(self, robot_x, robot_y, robot_theta):
        """Generate random goals for exploration"""
        
        # Try multiple random directions
        for _ in range(12):  # Try 12 random directions
            angle = np.random.uniform(0, 2 * math.pi)
            distance = np.random.uniform(2.0, self.goal_distance)
            
            goal_x = robot_x + distance * math.cos(angle)
            goal_y = robot_y + distance * math.sin(angle)
            
            if self.is_goal_valid(robot_x, robot_y, goal_x, goal_y):
                return goal_x, goal_y
        
        return None, None

    def frontier_exploration(self, robot_x, robot_y, robot_theta):
        """Generate goals towards frontiers (open areas) using laser data"""
        
        if self.laser_data is None:
            # Fallback to spiral if no laser data
            return self.spiral_exploration(robot_x, robot_y, robot_theta)
        
        # Find the direction with the most open space
        best_angle = None
        best_distance = 0.0
        
        # Sample directions based on laser data
        angle_increment = self.laser_data.angle_increment
        angle_min = self.laser_data.angle_min
        
        for i, range_val in enumerate(self.laser_data.ranges):
            if math.isnan(range_val) or math.isinf(range_val):
                continue
                
            # Convert to global angle
            laser_angle = angle_min + i * angle_increment
            global_angle = robot_theta + laser_angle
            
            # Look for directions with good clearance
            if range_val > self.min_obstacle_distance and range_val < self.laser_range_max:
                if range_val > best_distance:
                    best_distance = range_val
                    best_angle = global_angle
        
        if best_angle is not None:
            # Place goal partway towards the furthest clear direction
            goal_distance = min(self.goal_distance, best_distance * 0.7)
            goal_x = robot_x + goal_distance * math.cos(best_angle)
            goal_y = robot_y + goal_distance * math.sin(best_angle)
            
            if self.is_goal_valid(robot_x, robot_y, goal_x, goal_y):
                return goal_x, goal_y
        
        # Fallback to spiral exploration
        return self.spiral_exploration(robot_x, robot_y, robot_theta)

    def is_goal_valid(self, robot_x, robot_y, goal_x, goal_y):
        """Check if a goal is valid (not too close to obstacles)"""
        
        if self.laser_data is None:
            return True  # Assume valid if no laser data
        
        # Simple validation: check if path to goal is clear
        # This is a simplified check - in reality you'd do proper path planning
        
        # Check distance from robot
        distance = math.hypot(goal_x - robot_x, goal_y - robot_y)
        if distance < 1.0 or distance > self.goal_distance + 1.0:
            return False
        
        # Check if goal direction has obstacles using laser data
        goal_angle = math.atan2(goal_y - robot_y, goal_x - robot_x)
        robot_theta = math.atan2(goal_y - robot_y, goal_x - robot_x)  # Simplified
        
        # Find corresponding laser ray
        relative_angle = goal_angle - robot_theta
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi
        
        # Check if this direction is clear in laser data
        angle_min = self.laser_data.angle_min
        angle_max = self.laser_data.angle_max
        
        if angle_min <= relative_angle <= angle_max:
            angle_increment = self.laser_data.angle_increment
            index = int((relative_angle - angle_min) / angle_increment)
            
            if 0 <= index < len(self.laser_data.ranges):
                range_val = self.laser_data.ranges[index]
                if not math.isnan(range_val) and not math.isinf(range_val):
                    if range_val < self.min_obstacle_distance:
                        return False  # Too close to obstacle
        
        return True

def main(args=None):
    rclpy.init(args=args)
    node = SimpleGoalGenerator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()