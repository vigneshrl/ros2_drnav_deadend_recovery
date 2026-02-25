#!/usr/bin/env python3

"""
Unified Goal Generator for All Methods

This node generates short-horizon waypoints (3-5m ahead) using a unified scoring framework:
Score(θ) = J_geom(θ) + λ·EDE(θ)

Where:
- J_geom(θ): Geometric costs (collision, feasibility, smoothness, range bias)
- EDE(θ): Exposure to Dead-End (semantic risk from DRaM)
- λ: Method-specific weight (>0 for DRaM methods, =0 for LiDAR baselines)

Methods:
- DRaM Multi/Single: λ > 0, uses semantic risk map
- DWA/MPPI LiDAR: λ = 0, pure geometric scoring
- Recovery mode: Uses recovery points when all rays blocked
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Float32MultiArray, Bool, ColorRGBA
from tf2_ros import TransformListener, Buffer
import numpy as np
import math
import time
from typing import List, Tuple, Optional, Dict
import tf2_geometry_msgs

class GoalGenerator(Node):
    def __init__(self):
        super().__init__('goal_generator')
        
        # Declare parameters for method configuration
        self.declare_parameter('method_type', 'multi_camera_dram')  # multi_camera_dram, single_camera_dram, dwa_lidar, mppi_lidar
        self.declare_parameter('lambda_ede', 1.0)  # EDE weight (0 for LiDAR methods)
        self.declare_parameter('goal_generation_rate', 7.0)  # Hz (5-10 Hz as specified)
        self.declare_parameter('horizon_distance', 4.0)  # meters (3-5m as specified)
        
        # Get parameters
        self.method_type = self.get_parameter('method_type').get_parameter_value().string_value
        self.lambda_ede = self.get_parameter('lambda_ede').get_parameter_value().double_value
        self.goal_generation_rate = self.get_parameter('goal_generation_rate').get_parameter_value().double_value
        self.horizon_distance = self.get_parameter('horizon_distance').get_parameter_value().double_value
        
        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
        self.waypoint_viz_pub = self.create_publisher(MarkerArray, '/goal_generator/waypoints', 10)
        self.rays_viz_pub = self.create_publisher(MarkerArray, '/goal_generator/rays', 10)
        
        # Subscribers (adapt based on method)
        if 'dram' in self.method_type:
            # DRaM methods need risk map and recovery points
            topic_prefix = '/single_camera' if 'single' in self.method_type else '/dead_end_detection'
            
            self.risk_map_sub = self.create_subscription(
                MarkerArray, '/dram_exploration_map', self.risk_map_callback, 10
            )
            self.recovery_points_sub = self.create_subscription(
                Float32MultiArray, f'{topic_prefix}/recovery_points', self.recovery_points_callback, 10
            )
            self.dead_end_sub = self.create_subscription(
                Bool, f'{topic_prefix}/is_dead_end', self.dead_end_callback, 10
            )
            # Subscribe to path status for proper recovery triggering
            self.path_status_sub = self.create_subscription(
                Float32MultiArray, f'{topic_prefix}/path_status', self.path_status_callback, 10
            )
        else:
            # LiDAR methods don't use risk map
            self.risk_map_sub = None
            self.recovery_points_sub = None
            self.dead_end_sub = None
            self.path_status_sub = None
        
        self.occupancy_map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.occupancy_map_callback, 10
        )
        
        # Transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Unified scoring parameters (same for all methods)
        self.num_rays = 36                    # number of heading samples [-π, π]
        self.ray_resolution = 0.15            # meters - discretization along ray (0.1-0.2m)
        self.min_horizon = 2.0                # minimum acceptable horizon
        self.score_threshold = 100.0          # threshold for "no good rays"
        
        # J_geom components (identical for all methods)
        self.collision_weight = 50.0          # penalty for collision/clearance cost
        self.feasibility_weight = 1000.0      # huge penalty if ray hits obstacle before R
        self.smoothness_weight = 0.5          # yaw change penalty to avoid twitch
        self.range_bias_weight = 0.1          # light penalty for very short rays
        self.inflation_radius = 0.3           # robot inflation radius
        
        # Recovery scoring parameters (same α,β,γ,δ for all)
        self.alpha_progress = 1.0             # goal progress weight
        self.beta_clearance = 2.0             # clearance weight  
        self.gamma_roughness = 0.5            # slope/roughness penalty
        self.delta_ede_recovery = 0.3         # EDE from recovery point (only for DRaM)
        
        # State tracking
        self.robot_pose = None
        self.last_robot_yaw = 0.0
        self.risk_grid = {}                   # (x, y) -> risk_value [0,1]
        self.occupancy_grid = None
        self.recovery_points = []
        self.consecutive_dead_ends = 0
        self.recovery_threshold = 4           # consecutive dead-ends before recovery
        self.stuck_counter = 0
        self.last_robot_pose = None
        
        # Path status tracking for proper recovery triggering
        self.current_path_status = None 
        
        
        #change the threshold for indoor as the detection is poor with the glasses 
        self.path_blocked_threshold = 0.56    # threshold for considering path blocked
        self.is_truly_blocked = False         # all 3 directions blocked
        
        # Create timer for goal generation (5-10 Hz)
        self.create_timer(1.0 / self.goal_generation_rate, self.generate_goal_callback)
        
        self.get_logger().info(f'🎯 Unified Goal Generator initialized')
        self.get_logger().info(f'📊 Method: {self.method_type}, λ_EDE: {self.lambda_ede:.2f}')
        self.get_logger().info(f'📊 Rate: {self.goal_generation_rate}Hz, Horizon: {self.horizon_distance}m, Rays: {self.num_rays}')

    def risk_map_callback(self, msg: MarkerArray):
        """Process risk map from DRaM heatmap visualization"""
        self.risk_grid.clear()
        
        for marker in msg.markers:
            if marker.ns == "exploration_heatmap" and marker.type == Marker.POINTS:
                # Process heatmap points
                for i, point in enumerate(marker.points):
                    if i < len(marker.colors):
                        color = marker.colors[i]
                        
                        # Convert color to risk value
                        # Green (safe) = low risk, Red (dangerous) = high risk
                        if color.g > 0.5:  # Green
                            risk_value = 0.1
                        elif color.r > 0.5:  # Red
                            risk_value = 0.9
                        else:  # Recovery points (blue/purple)
                            risk_value = 0.0  # Very safe
                        
                        # Store in grid
                        grid_x = round(point.x / 0.3) * 0.3  # Snap to grid
                        grid_y = round(point.y / 0.3) * 0.3
                        self.risk_grid[(grid_x, grid_y)] = risk_value

    def occupancy_map_callback(self, msg: OccupancyGrid):
        """Store occupancy grid for obstacle checking"""
        self.occupancy_grid = msg

    def recovery_points_callback(self, msg: Float32MultiArray):
        """Store recovery points"""
        self.recovery_points.clear()
        data = msg.data
        
        for i in range(0, len(data), 3):
            if i + 2 < len(data):
                point_type = int(data[i])
                x = data[i + 1]
                y = data[i + 2]
                open_paths = 2 if point_type == 1 else 1
                
                self.recovery_points.append({
                    'x': x, 'y': y, 'type': point_type, 'open_paths': open_paths
                })

    def path_status_callback(self, msg: Float32MultiArray):
        """Process path status for intelligent recovery triggering"""
        if len(msg.data) >= 3:
            self.current_path_status = [msg.data[0], msg.data[1], msg.data[2]]  # [front, left, right]
            
            # Check if ALL 3 directions are blocked (below threshold)
            front_blocked = msg.data[0] < self.path_blocked_threshold
            left_blocked = msg.data[1] < self.path_blocked_threshold
            right_blocked = msg.data[2] < self.path_blocked_threshold
            
            self.is_truly_blocked = front_blocked and left_blocked and right_blocked
            
            self.get_logger().debug(f'📊 Path Status: F={msg.data[0]:.3f}{"❌" if front_blocked else "✅"}, '
                                  f'L={msg.data[1]:.3f}{"❌" if left_blocked else "✅"}, '
                                  f'R={msg.data[2]:.3f}{"❌" if right_blocked else "✅"} → '
                                  f'{"🚨 ALL BLOCKED" if self.is_truly_blocked else "🟢 PATH AVAILABLE"}')

    def dead_end_callback(self, msg: Bool):
        """Track dead-end detections (kept for compatibility)"""
        if msg.data:
            self.consecutive_dead_ends += 1
        else:
            self.consecutive_dead_ends = 0

    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        """Get current robot pose in map frame"""
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

    def is_robot_stuck(self) -> bool:
        """Check if robot is stuck (hasn't moved much)"""
        if self.last_robot_pose is None:
            self.last_robot_pose = self.robot_pose
            return False
        
        if self.robot_pose is None:
            return False
        
        # Calculate distance moved
        dx = self.robot_pose[0] - self.last_robot_pose[0]
        dy = self.robot_pose[1] - self.last_robot_pose[1]
        distance_moved = math.hypot(dx, dy)
        
        # Update last pose
        self.last_robot_pose = self.robot_pose
        
        # Check if stuck (moved less than 0.1m in last second)
        if distance_moved < 0.1:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        return self.stuck_counter > 5  # Stuck for 5 seconds

    def bilinear_sample_risk(self, x: float, y: float) -> float:
        """Bilinear sample risk from grid (for DRaM methods only)"""
        if self.lambda_ede == 0.0 or not self.risk_grid:
            return 0.0  # No risk for LiDAR methods
        
        # Snap to grid for lookup
        grid_x = round(x / 0.3) * 0.3
        grid_y = round(y / 0.3) * 0.3
        
        # Return risk if available, cap to [0,1]
        risk = self.risk_grid.get((grid_x, grid_y), 0.0)  # Unknown = no risk
        return max(0.0, min(1.0, risk))

    def get_costmap_value(self, x: float, y: float) -> float:
        """Get inflated costmap value at point (for collision cost)"""
        if self.occupancy_grid is None:
            return 0.0
        
        # Convert world coordinates to grid coordinates
        grid_x = int((x - self.occupancy_grid.info.origin.position.x) / self.occupancy_grid.info.resolution)
        grid_y = int((y - self.occupancy_grid.info.origin.position.y) / self.occupancy_grid.info.resolution)
        
        # Check bounds
        if (grid_x < 0 or grid_x >= self.occupancy_grid.info.width or 
            grid_y < 0 or grid_y >= self.occupancy_grid.info.height):
            return 100.0  # Out-of-bounds = high cost
        
        # Get occupancy value
        idx = grid_y * self.occupancy_grid.info.width + grid_x
        if idx >= len(self.occupancy_grid.data):
            return 100.0
        
        occupancy = self.occupancy_grid.data[idx]
        if occupancy < 0:  # Unknown
            return 25.0  # Moderate cost for unknown
        
        return float(occupancy)  # [0-100]

    def is_point_occupied(self, x: float, y: float) -> bool:
        """Check if point is occupied (for feasibility check)"""
        return self.get_costmap_value(x, y) > 50.0

    def get_clearance_cost(self, x: float, y: float) -> float:
        """Get clearance cost (1/distance to nearest obstacle)"""
        if self.occupancy_grid is None:
            return 0.0
        
        min_dist = float('inf')
        resolution = self.occupancy_grid.info.resolution
        
        # Check in small radius around point
        search_radius = int(self.inflation_radius / resolution) + 1
        
        base_x = int((x - self.occupancy_grid.info.origin.position.x) / resolution)
        base_y = int((y - self.occupancy_grid.info.origin.position.y) / resolution)
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                grid_x = base_x + dx
                grid_y = base_y + dy
                
                if (grid_x >= 0 and grid_x < self.occupancy_grid.info.width and
                    grid_y >= 0 and grid_y < self.occupancy_grid.info.height):
                    
                    idx = grid_y * self.occupancy_grid.info.width + grid_x
                    if idx < len(self.occupancy_grid.data) and self.occupancy_grid.data[idx] > 80:
                        dist = math.hypot(dx * resolution, dy * resolution)
                        min_dist = min(min_dist, dist)
        
        if min_dist == float('inf'):
            return 0.0  # No obstacles nearby
        
        return 1.0 / max(min_dist, 0.1)  # Avoid division by zero

    # def compute_j_geom(self, start_x: float, start_y: float, heading: float, distance: float) -> Tuple[float, bool]:
    #     """
    #     Compute J_geom(θ) - geometric cost components (identical for all methods)
        
    #     Returns:
    #         (j_geom_score, is_feasible)
    #     """
    #     collision_cost = 0.0
    #     feasible = True
    #     actual_distance = distance
        
    #     # Sample points along the ray
    #     num_samples = int(distance / self.ray_resolution)
        
    #     for i in range(1, num_samples + 1):
    #         # Calculate point along ray
    #         sample_distance = i * self.ray_resolution
    #         sample_x = start_x + sample_distance * math.cos(heading)
    #         sample_y = start_y + sample_distance * math.sin(heading)
            
    #         # Feasibility: check if ray hits obstacle before R
    #         if self.is_point_occupied(sample_x, sample_y):
    #             feasible = False
    #             actual_distance = sample_distance
    #             break
            
    #         # Collision/clearance cost: integral of inflated costmap + clearance
    #         costmap_val = self.get_costmap_value(sample_x, sample_y)
    #         clearance_cost = self.get_clearance_cost(sample_x, sample_y)
    #         collision_cost += (costmap_val / 100.0 + clearance_cost) * self.ray_resolution
        
    #     # Feasibility penalty: huge penalty if blocked before R
    #     feasibility_cost = self.feasibility_weight if not feasible else 0.0
        
    #     # Smoothness: yaw change penalty to avoid twitch
    #     yaw_change = abs(heading - self.last_robot_yaw)
    #     yaw_change = min(yaw_change, 2 * math.pi - yaw_change)  # Wrap to [-π, π]
    #     smoothness_cost = self.smoothness_weight * yaw_change
        
    #     # Range bias: light penalty for very short rays
    #     range_bias_cost = self.range_bias_weight * max(0, self.horizon_distance - actual_distance)
        
    #     # Total J_geom
    #     j_geom = (self.collision_weight * collision_cost + 
    #               feasibility_cost + 
    #               smoothness_cost + 
    #               range_bias_cost)
        
    #     return j_geom, feasible

    # def compute_ede(self, start_x: float, start_y: float, heading: float, distance: float) -> float:
    #     """
    #     Compute EDE(θ) - Exposure to Dead-End (for DRaM methods only)
    #     EDE(θ) = Σᵢ pᵢ·Δs where pᵢ is dead-end probability, Δs is segment length
    #     """
    #     if self.lambda_ede == 0.0:
    #         return 0.0  # LiDAR methods don't use EDE
        
    #     ede_score = 0.0
    #     num_samples = int(distance / self.ray_resolution)
        
    #     for i in range(1, num_samples + 1):
    #         # Calculate point along ray
    #         sample_distance = i * self.ray_resolution
    #         sample_x = start_x + sample_distance * math.cos(heading)
    #         sample_y = start_y + sample_distance * math.sin(heading)
            
    #         # Stop at obstacles
    #         if self.is_point_occupied(sample_x, sample_y):
    #             break
            
    #         # Bilinear sample risk probability p_i ∈ [0,1]
    #         risk_prob = self.bilinear_sample_risk(sample_x, sample_y)
            
    #         # Accumulate EDE: p_i * Δs
    #         ede_score += risk_prob * self.ray_resolution
        
    #     return ede_score

    # def compute_unified_score(self, start_x: float, start_y: float, heading: float) -> Tuple[float, bool]:
    #     """
    #     Compute unified score: Score(θ) = J_geom(θ) + λ·EDE(θ)
        
    #     Returns:
    #         (total_score, is_feasible)
    #     """
    #     # Compute geometric cost (same for all methods)
    #     j_geom, feasible = self.compute_j_geom(start_x, start_y, heading, self.horizon_distance)
        
    #     # Compute EDE (only for DRaM methods when λ > 0)
    #     ede = self.compute_ede(start_x, start_y, heading, self.horizon_distance)
        
    #     # Unified score
    #     total_score = j_geom + self.lambda_ede * ede
        
    #     return total_score, feasible

    def sample_exploration_waypoint(self) -> Optional[Tuple[float, float, float]]:
        """
        Sample waypoints using unified scoring framework
        At 5-10 Hz: sample headings θ ∈ [-π,π], pick argmin Score(θ)
        """
        if self.robot_pose is None:
            return None
        
        robot_x, robot_y, robot_yaw = self.robot_pose
        self.last_robot_yaw = robot_yaw  # Update for smoothness cost
        
        best_score = float('inf')
        best_waypoint = None
        ray_scores = []  # For visualization
        feasible_rays = 0
        
        # Sample headings θ ∈ [-π, π]
        for i in range(self.num_rays):
            heading = (2 * math.pi * i) / self.num_rays - math.pi  # [-π, π]
            
            # Compute unified score: Score(θ) = J_geom(θ) + λ·EDE(θ)
            score, feasible = self.compute_unified_score(robot_x, robot_y, heading)
            ray_scores.append((heading, score, feasible))
            
            if feasible:
                feasible_rays += 1
            
            # Track best direction (argmin θ*)
            if score < best_score:
                best_score = score
                best_heading = heading
                
                # Calculate waypoint position at range R
                waypoint_x = robot_x + self.horizon_distance * math.cos(heading)
                waypoint_y = robot_y + self.horizon_distance * math.sin(heading)
                best_waypoint = (waypoint_x, waypoint_y, heading)
        
        # Visualize rays
        self.visualize_rays(robot_x, robot_y, ray_scores)
        
        # Check if "no good rays" (all blocked or risk too high)
        if best_score > self.score_threshold or feasible_rays == 0:
            self.get_logger().warn(f'🚫 No good rays: best_score={best_score:.1f}, feasible={feasible_rays}/{self.num_rays}')
            return None
        
        self.get_logger().debug(f'✅ Best ray: θ={best_heading:.2f}rad, score={best_score:.1f}, feasible={feasible_rays}/{self.num_rays}')
        return best_waypoint

    # def compute_recovery_score(self, candidate_x: float, candidate_y: float) -> float:
    #     """
    #     Compute recovery score using shared α,β,γ,δ parameters
    #     RecScore = α·goal-progress + β·clearance - γ·slope/roughness - δ·EDE_from_here
    #     """
    #     if self.robot_pose is None:
    #         return float('inf')
        
    #     robot_x, robot_y, robot_yaw = self.robot_pose
        
    #     # α·goal-progress (assume goal is forward exploration for now)
    #     forward_x = robot_x + math.cos(robot_yaw)
    #     forward_y = robot_y + math.sin(robot_yaw)
    #     progress = -math.hypot(candidate_x - forward_x, candidate_y - forward_y)  # Negative = closer is better
    #     goal_progress_term = self.alpha_progress * progress
        
    #     # β·clearance (distance to nearest obstacle)
    #     clearance = 1.0 / max(self.get_clearance_cost(candidate_x, candidate_y), 0.1)
    #     clearance_term = self.beta_clearance * clearance
        
    #     # γ·slope/roughness (simplified as distance penalty)
    #     distance_to_candidate = math.hypot(candidate_x - robot_x, candidate_y - robot_y)
    #     roughness_term = self.gamma_roughness * distance_to_candidate
        
    #     # δ·EDE_from_here (only for DRaM methods)
    #     if self.lambda_ede > 0.0:
    #         # Sample a few directions from recovery point to estimate EDE
    #         ede_from_recovery = 0.0
    #         test_directions = 8
    #         for i in range(test_directions):
    #             test_heading = (2 * math.pi * i) / test_directions
    #             ede_from_recovery += self.compute_ede(candidate_x, candidate_y, test_heading, 2.0)  # 2m horizon
    #         ede_from_recovery /= test_directions
    #         ede_term = self.delta_ede_recovery * ede_from_recovery
    #     else:
    #         ede_term = 0.0
        
    #     # Total recovery score (lower is better)
    #     rec_score = goal_progress_term + clearance_term - roughness_term - ede_term
    #     return -rec_score  # Negate so higher is better, then we take min

    # def select_recovery_waypoint(self) -> Optional[Tuple[float, float, float]]:
    #     """
    #     Select recovery waypoint from candidate set (behind & to the sides)
    #     If no recovery points available, generate candidates behind robot
    #     """
    #     if self.robot_pose is None:
    #         return None
        
    #     robot_x, robot_y, robot_yaw = self.robot_pose
    #     candidates = []
        
    #     # Use recovery points if available (DRaM methods)
    #     if self.recovery_points:
    #         for rp in self.recovery_points:
    #             candidates.append((rp['x'], rp['y']))
    #     else:
    #         # Generate recovery candidates behind & to the sides (LiDAR methods)
    #         recovery_distance = 2.0  # meters behind
    #         recovery_angles = [math.pi, 3*math.pi/4, 5*math.pi/4, math.pi/2, -math.pi/2]  # Behind, diagonals, sides
            
    #         for angle_offset in recovery_angles:
    #             candidate_heading = robot_yaw + angle_offset
    #             candidate_x = robot_x + recovery_distance * math.cos(candidate_heading)
    #             candidate_y = robot_y + recovery_distance * math.sin(candidate_heading)
                
    #             # Check if candidate is not occupied
    #             if not self.is_point_occupied(candidate_x, candidate_y):
    #                 candidates.append((candidate_x, candidate_y))
        
    #     if not candidates:
    #         return None
        
    #     # Score each candidate and pick best
    #     best_score = float('inf')
    #     best_candidate = None
        
    #     for candidate_x, candidate_y in candidates:
    #         score = self.compute_recovery_score(candidate_x, candidate_y)
    #         if score < best_score:
    #             best_score = score
    #             best_candidate = (candidate_x, candidate_y)
        
    #     if best_candidate:
    #         # Calculate heading to recovery point
    #         dx = best_candidate[0] - robot_x
    #         dy = best_candidate[1] - robot_y
    #         heading = math.atan2(dy, dx)
            
    #         return (best_candidate[0], best_candidate[1], heading)
        
    #     return None

    def generate_goal_callback(self):
        """
        Main goal generation callback (5-10 Hz)
        1. Check if ALL 3 camera directions are blocked
        2. If blocked, use recovery waypoint  
        3. Otherwise, sample headings θ ∈ [-π,π] and pick argmin Score(θ)
        """
        self.robot_pose = self.get_robot_pose()
        
        if self.robot_pose is None:
            return
        
        # Check if robot is stuck (physical movement)
        is_stuck = self.is_robot_stuck()
        
        # INTELLIGENT RECOVERY TRIGGERING: Only when ALL 3 directions are blocked
        should_recover = False
        if 'dram' in self.method_type and self.is_truly_blocked:
            should_recover = True
            self.get_logger().info('🚨 ALL 3 DIRECTIONS BLOCKED - Triggering recovery mode!')
        elif is_stuck:
            should_recover = True
            self.get_logger().info('🚨 ROBOT STUCK - Triggering recovery mode!')
        
        waypoint = None
        
        if should_recover:
            # Use recovery waypoint when truly blocked
            waypoint = self.select_recovery_waypoint()
            if waypoint:
                self.get_logger().info(f'🚨 Recovery: heading to ({waypoint[0]:.1f}, {waypoint[1]:.1f})')
        
        if waypoint is None:
            # Normal exploration: sample headings, pick argmin θ*
            waypoint = self.sample_exploration_waypoint()
        
        # Publish waypoint as goal
        if waypoint:
            self.publish_goal(waypoint[0], waypoint[1], waypoint[2])
            self.visualize_waypoint(waypoint[0], waypoint[1])
        else:
            self.get_logger().warn('❌ No valid waypoint found - robot may be trapped!')

    def publish_goal(self, x: float, y: float, heading: float):
        """Publish waypoint as navigation goal"""
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = 0.0
        
        # Convert heading to quaternion
        goal_msg.pose.orientation.x = 0.0
        goal_msg.pose.orientation.y = 0.0
        goal_msg.pose.orientation.z = math.sin(heading / 2.0)
        goal_msg.pose.orientation.w = math.cos(heading / 2.0)
        
        self.goal_pub.publish(goal_msg)

    def visualize_waypoint(self, x: float, y: float):
        """Visualize current waypoint"""
        marker_array = MarkerArray()
        
        # Waypoint marker
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "current_waypoint"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.8
        marker.scale.y = 0.8
        marker.scale.z = 0.8
        
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        marker.lifetime.sec = 2
        marker_array.markers.append(marker)
        
        self.waypoint_viz_pub.publish(marker_array)

    def visualize_rays(self, robot_x: float, robot_y: float, ray_scores: List[Tuple[float, float, bool]]):
        """Visualize sampled rays with color-coded unified scores"""
        marker_array = MarkerArray()
        
        # Normalize scores for color coding
        feasible_scores = [score for _, score, feasible in ray_scores if feasible]
        if not feasible_scores:
            return  # No feasible rays to visualize
        
        min_score = min(feasible_scores)
        max_score = max(feasible_scores)
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        for i, (heading, score, feasible) in enumerate(ray_scores):
            # Create line marker for ray
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "unified_rays"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            # Ray start and end points
            start_point = Point()
            start_point.x = robot_x
            start_point.y = robot_y
            start_point.z = 0.1
            
            end_point = Point()
            end_point.x = robot_x + self.horizon_distance * math.cos(heading)
            end_point.y = robot_y + self.horizon_distance * math.sin(heading)
            end_point.z = 0.1
            
            marker.points = [start_point, end_point]
            
            marker.scale.x = 0.03  # Thin line width
            
            # Color based on feasibility and score
            if not feasible:
                # Red for infeasible rays
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.3
            else:
                # Green-to-red gradient for feasible rays (green = low score/good)
                normalized_score = (score - min_score) / score_range if score_range > 0 else 0.0
                marker.color.r = normalized_score
                marker.color.g = 1.0 - normalized_score
                marker.color.b = 0.0
                marker.color.a = 0.8
            
            marker.lifetime.sec = 0  # Persist until next update
            marker_array.markers.append(marker)
        
        # Add method info text
        info_marker = Marker()
        info_marker.header.frame_id = "map"
        info_marker.header.stamp = self.get_clock().now().to_msg()
        info_marker.ns = "method_info"
        info_marker.id = 9999
        info_marker.type = Marker.TEXT_VIEW_FACING
        info_marker.action = Marker.ADD
        
        info_marker.pose.position.x = robot_x
        info_marker.pose.position.y = robot_y + 2.0
        info_marker.pose.position.z = 1.0
        info_marker.pose.orientation.w = 1.0
        
        info_marker.scale.z = 0.5
        info_marker.color.r = 1.0
        info_marker.color.g = 1.0
        info_marker.color.b = 1.0
        info_marker.color.a = 1.0
        
        info_marker.text = f"{self.method_type}\nλ={self.lambda_ede:.1f}"
        info_marker.lifetime.sec = 0
        marker_array.markers.append(info_marker)
        
        self.rays_viz_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    
    node = GoalGenerator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
