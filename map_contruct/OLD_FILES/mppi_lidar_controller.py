#!/usr/bin/env python3

"""
MPPI (Model Predictive Path Integral) Controller with LiDAR-based Dead-end Detection

This controller uses:
1. LiDAR data to detect obstacles and dead-ends
2. MPPI optimization for path planning
3. Recovery point detection and navigation
4. Performance metrics collection
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Float32MultiArray, Bool
from tf2_ros import TransformListener, Buffer
import numpy as np
import math
import time
import json
import os
from collections import defaultdict

class MPPILidarController(Node):
    def __init__(self):
        super().__init__('mppi_lidar_controller')
        
        # Subscriptions
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
        self.path_status_pub = self.create_publisher(Float32MultiArray, '/mppi_lidar/path_status', 10)
        self.recovery_points_pub = self.create_publisher(Float32MultiArray, '/mppi_lidar/recovery_points', 10)
        self.dead_end_pub = self.create_publisher(Bool, '/mppi_lidar/is_dead_end', 10)
        self.planned_path_pub = self.create_publisher(Path, '/mppi_lidar/planned_path', 10)
        
        # Transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # MPPI Parameters
        self.num_samples = 1000  # Number of trajectory samples
        self.horizon = 20  # Planning horizon steps
        self.dt = 0.1  # Time step
        self.lambda_param = 0.01  # Temperature parameter
        self.sigma_v = 0.3  # Velocity noise std
        self.sigma_w = 0.5  # Angular velocity noise std
        
        # Control constraints
        self.max_v = 0.8
        self.min_v = 0.0
        self.max_w = 1.0
        self.min_w = -1.0
        
        # Dead-end detection parameters
        self.obstacle_threshold = 0.5  # Distance to consider as obstacle (m)
        self.dead_end_threshold = 3  # Consecutive dead-end detections to trigger recovery
        self.recovery_distance_threshold = 2.0  # Distance to recovery point to exit recovery mode
        
        # State variables
        self.current_scan = None
        self.robot_pose = None
        self.recovery_points = []
        self.dead_end_count = 0
        self.consecutive_dead_ends = 0
        self.is_in_recovery_mode = False
        self.target_recovery_point = None
        
        # Goal management
        self.current_goal = None
        self.goal_tolerance = 1.0  # meters
        
        # Enhanced performance metrics
        self.metrics = {
            'start_time': time.time(),
            'path_length': 0.0,
            'total_energy': 0.0,
            'dead_end_detections': 0,
            'false_positive_dead_ends': 0,
            'false_negatives': 0,
            'recovery_point_detections': 0,
            'recovery_activations': 0,
            'success': False,
            'time_to_goal': 0.0,
            'completion_time': 0.0,
            
            # Proactive metrics
            'detection_lead_distances': [],
            'detection_lead_times': [],
            'distance_to_first_recovery': [],
            'freezes': 0,
            'time_trapped': 0.0,
            'ede_integral': 0.0,
            
            # State tracking for proactive metrics
            'last_dead_end_position': None,
            'dead_end_detection_time': None,
            'freeze_start_time': None,
            'last_velocity_time': time.time(),
            'path_segments': [],
            'dead_end_probabilities': []
        }
        self.last_pose = None
        self.last_cmd = Twist()
        
        # Create output directory for results
        self.output_dir = self.create_output_directory()
        
        # Control timer
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('🚀 MPPI LiDAR Controller initialized')
        self.get_logger().info(f'📊 Parameters: samples={self.num_samples}, horizon={self.time_horizon}s, λ={self.lambda_param}')
        self.get_logger().info(f'🎯 Dead-end threshold: {self.dead_end_threshold} consecutive detections')
        self.get_logger().info('📡 Subscribing to /scan and /move_base_simple/goal')
        self.get_logger().info('✅ Ready to receive goals from goal_generator')
    
    def create_output_directory(self):
        """Create timestamped output directory"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"/home/mrvik/dram_ws/evaluation_results/mppi_lidar_{timestamp}"
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
    
    def scan_callback(self, msg):
        """Process LiDAR scan data"""
        self.current_scan = msg
        
        # Analyze scan for dead-end detection
        self.analyze_scan_for_dead_ends(msg)
    
    def analyze_scan_for_dead_ends(self, scan):
        """Analyze LiDAR scan to detect dead-ends and recovery points"""
        if not scan.ranges:
            return
        
        # Convert scan to 3-directional analysis (front, left, right)
        num_rays = len(scan.ranges)
        
        # Define angular sectors
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
        path_probs = [
            1.0 if front_clear else 0.0,
            1.0 if left_clear else 0.0, 
            1.0 if right_clear else 0.0
        ]
        
        # Determine dead-end status
        open_paths = sum(path_probs)
        is_dead_end = (open_paths == 0)
        is_recovery_point = (open_paths >= 2)
        
        # Update dead-end tracking with enhanced metrics
        if is_dead_end:
            self.consecutive_dead_ends += 1
            self.dead_end_count += 1
            self.metrics['dead_end_detections'] += 1
            
            # Track detection lead (for LiDAR baseline, this is reactive)
            self.track_detection_lead(True)
        else:
            self.consecutive_dead_ends = 0
            self.track_detection_lead(False)
        
        # Detect false negatives
        self.detect_false_negatives()
        
        # Store recovery point if detected
        if is_recovery_point and self.robot_pose:
            self.add_recovery_point(self.robot_pose[0], self.robot_pose[1], int(open_paths))
        
        # Publish path status
        path_msg = Float32MultiArray()
        path_msg.data = path_probs
        self.path_status_pub.publish(path_msg)
        
        # Publish dead-end status
        dead_end_msg = Bool()
        dead_end_msg.data = is_dead_end
        self.dead_end_pub.publish(dead_end_msg)
        
        self.get_logger().debug(f'LiDAR Analysis: F={front_clear}, L={left_clear}, R={right_clear} | '
                               f'Dead-end: {is_dead_end} | Recovery: {is_recovery_point}')
    
    def check_sector_clear(self, scan, start_idx, end_idx):
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
    
    def add_recovery_point(self, x, y, open_paths):
        """Add a recovery point if not already exists nearby"""
        # Check for duplicates within 1m
        for rp in self.recovery_points:
            if math.hypot(rp['x'] - x, rp['y'] - y) < 1.0:
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
                'map', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
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
    
    def mppi_planning(self, goal_x, goal_y):
        """MPPI trajectory optimization"""
        if not self.robot_pose or not self.current_scan:
            return Twist()
        
        robot_x, robot_y, robot_yaw = self.robot_pose
        
        # Generate random control sequences
        control_sequences = []
        costs = []
        
        for _ in range(self.num_samples):
            # Generate random control sequence
            controls = []
            for t in range(self.horizon):
                v = np.random.normal(0.3, self.sigma_v)  # Forward bias
                w = np.random.normal(0.0, self.sigma_w)
                
                # Apply constraints
                v = np.clip(v, self.min_v, self.max_v)
                w = np.clip(w, self.min_w, self.max_w)
                
                controls.append((v, w))
            
            control_sequences.append(controls)
            
            # Evaluate trajectory cost
            cost = self.evaluate_trajectory_cost(robot_x, robot_y, robot_yaw, controls, goal_x, goal_y)
            costs.append(cost)
        
        # Convert costs to weights using exponential weighting
        costs = np.array(costs)
        weights = np.exp(-costs / self.lambda_param)
        weights = weights / np.sum(weights)
        
        # Compute weighted average of first control action
        weighted_v = 0.0
        weighted_w = 0.0
        
        for i, (v, w) in enumerate([seq[0] for seq in control_sequences]):
            weighted_v += weights[i] * v
            weighted_w += weights[i] * w
        
        # Create command
        cmd = Twist()
        cmd.linear.x = weighted_v
        cmd.angular.z = weighted_w
        
        return cmd
    
    def evaluate_trajectory_cost(self, start_x, start_y, start_yaw, controls, goal_x, goal_y):
        """Evaluate cost of a trajectory"""
        x, y, yaw = start_x, start_y, start_yaw
        total_cost = 0.0
        
        for v, w in controls:
            # Simulate forward
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
            yaw += w * self.dt
            
            # Goal distance cost
            goal_dist = math.hypot(x - goal_x, y - goal_y)
            goal_cost = goal_dist * 1.0
            
            # Obstacle cost from LiDAR
            obstacle_cost = self.get_obstacle_cost(x, y)
            
            # Control effort cost
            control_cost = v * v * 0.1 + w * w * 0.1
            
            total_cost += goal_cost + obstacle_cost + control_cost
        
        return total_cost
    
    def get_obstacle_cost(self, x, y):
        """Get obstacle cost at position using current LiDAR scan"""
        if not self.current_scan or not self.robot_pose:
            return 0.0
        
        # Simple obstacle cost based on distance to robot
        # In a full implementation, this would transform LiDAR data to world coordinates
        robot_x, robot_y, _ = self.robot_pose
        dist_to_robot = math.hypot(x - robot_x, y - robot_y)
        
        # Higher cost if too close to current robot position (potential obstacles)
        if dist_to_robot < 0.5:
            return 100.0
        
        return 0.0
    
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
                
                # Track recovery distance
                self.track_recovery_distance()
                
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
        
        # Plan trajectory using MPPI
        cmd = self.mppi_planning(goal_x, goal_y)
        
        # Publish command
        self.cmd_pub.publish(cmd)
        self.last_cmd = cmd
        
        # Log status
        mode = "RECOVERY" if self.is_in_recovery_mode else "EXPLORATION"
        self.get_logger().debug(f'MPPI [{mode}]: v={cmd.linear.x:.2f}, w={cmd.angular.z:.2f} | '
                               f'Goal: ({goal_x:.1f}, {goal_y:.1f}) | Dead-ends: {self.consecutive_dead_ends}')
    
    def update_metrics(self):
        """Update enhanced performance metrics"""
        if not self.robot_pose or not self.last_pose:
            self.last_pose = self.robot_pose
            return
        
        current_time = time.time()
        
        # Calculate distance traveled (path length)
        dx = self.robot_pose[0] - self.last_pose[0]
        dy = self.robot_pose[1] - self.last_pose[1]
        distance = math.hypot(dx, dy)
        self.metrics['path_length'] += distance
        
        # Calculate energy consumption (simplified)
        v = abs(self.last_cmd.linear.x)
        w = abs(self.last_cmd.angular.z)
        energy = (v * v + w * w) * 0.1  # Simplified energy model
        self.metrics['total_energy'] += energy
        
        # Track freezes (robot stuck)
        if v < 0.01 and w < 0.01:  # Nearly stationary
            if self.metrics['freeze_start_time'] is None:
                self.metrics['freeze_start_time'] = current_time
        else:
            if self.metrics['freeze_start_time'] is not None:
                freeze_duration = current_time - self.metrics['freeze_start_time']
                if freeze_duration > 5.0:  # Stuck for more than 5 seconds
                    self.metrics['freezes'] += 1
                    self.metrics['time_trapped'] += freeze_duration
                self.metrics['freeze_start_time'] = None
            self.metrics['last_velocity_time'] = current_time
        
        # Update EDE integral (Exposure to Dead-End)
        if distance > 0:
            # For LiDAR baseline, dead-end probability is binary (0 or 1)
            dead_end_prob = 1.0 if self.consecutive_dead_ends > 0 else 0.0
            self.metrics['ede_integral'] += dead_end_prob * distance
            
            # Store for analysis
            self.metrics['path_segments'].append(distance)
            self.metrics['dead_end_probabilities'].append(dead_end_prob)
        
        # Update completion time
        self.metrics['completion_time'] = current_time - self.metrics['start_time']
        
        self.last_pose = self.robot_pose
    
    def track_detection_lead(self, is_dead_end: bool):
        """Track detection lead metrics"""
        if not self.robot_pose:
            return
        
        current_time = time.time()
        robot_x, robot_y, _ = self.robot_pose
        
        if is_dead_end and self.metrics['last_dead_end_position'] is None:
            # First dead-end detection at this location
            self.metrics['last_dead_end_position'] = (robot_x, robot_y)
            self.metrics['dead_end_detection_time'] = current_time
            
            # For LiDAR baseline, detection lead is typically 0 (reactive)
            # But we can measure distance to actual cul-de-sac if known
            detection_lead = 0.0  # Reactive detection
            self.metrics['detection_lead_distances'].append(detection_lead)
            self.metrics['detection_lead_times'].append(0.0)
            
        elif not is_dead_end and self.metrics['last_dead_end_position'] is not None:
            # Exited dead-end state
            self.metrics['last_dead_end_position'] = None
            self.metrics['dead_end_detection_time'] = None
    
    def track_recovery_distance(self):
        """Track distance to first recovery action"""
        if (self.is_in_recovery_mode and self.target_recovery_point and 
            self.metrics['last_dead_end_position'] is not None):
            
            # Calculate distance from dead-end detection to recovery start
            dead_end_x, dead_end_y = self.metrics['last_dead_end_position']
            recovery_x, recovery_y = self.target_recovery_point['x'], self.target_recovery_point['y']
            
            recovery_distance = math.hypot(recovery_x - dead_end_x, recovery_y - dead_end_y)
            self.metrics['distance_to_first_recovery'].append(recovery_distance)
    
    def detect_false_negatives(self):
        """Detect false negatives (missed dead-ends that robot drove into)"""
        # Simple heuristic: if robot gets stuck without detecting dead-end first
        if (self.metrics['freeze_start_time'] is not None and 
            time.time() - self.metrics['freeze_start_time'] > 10.0 and
            not self.is_dead_end):
            self.metrics['false_negatives'] += 1
    
    def save_metrics(self):
        """Save performance metrics to file"""
        metrics_file = os.path.join(self.output_dir, 'mppi_lidar_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.get_logger().info(f'📊 Metrics saved to {metrics_file}')

def main(args=None):
    rclpy.init(args=args)
    node = MPPILidarController()
    
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
