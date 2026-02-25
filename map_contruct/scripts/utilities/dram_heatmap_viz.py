#!/usr/bin/env python3

"""
DRaM Exploration Risk Map

A risk-first visualization approach:
1. Start with entire map as RED (unknown/dangerous)
2. As robot explores and makes predictions, update regions to GREEN (safe)
3. Hold predictions until robot revisits that area
4. Mark recovery points with long persistence
5. No legends - simple red/green risk visualization

Red = Unknown/Dangerous, Green = Explored/Safe
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformListener, Buffer
import numpy as np
import math
import time
from collections import defaultdict

class DRaMExplorationMap(Node):
    def __init__(self):
        super().__init__('dram_exploration_map')

        # Subscribe to path status (dead-end probabilities)
        self.path_subscription = self.create_subscription(
            Float32MultiArray,
            '/dead_end_detection/path_status',
            self.path_status_callback,
            10
        )


        #single camera abalation study
        # self.path_subscription = self.create_subscription(
        #     Float32MultiArray,
        #     '/single_camera/path_status',
        #     self.path_status_callback,
        #     10
        # )

        # Subscribe to recovery points
        self.recovery_subscription = self.create_subscription(
            Float32MultiArray,
            '/dead_end_detection/recovery_points',
            self.recovery_points_callback,
            10
        )

        # Subscribe to SLAM map for map bounds
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # Publishers
        self.risk_map_pub = self.create_publisher(
            MarkerArray,
            '/dram_exploration_map',
            10
        )

        # Transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Exploration data storage
        self.explored_grid = {}  # grid_pos -> {'safety': float, 'timestamp': float, 'explored': bool}
        self.recovery_points = []
        self.current_map = None
        self.map_bounds = None  # Will store map boundaries
        
        # Visualization parameters
        self.grid_resolution = 0.3  # 0.3m cells for smoother heatmap
        self.recovery_persistence = 300.0  # Keep recovery points for 5 minutes
        self.exploration_radius = 3.0  # How far around robot to update when exploring
        
        # Initialize red background map when we get map data
        self.background_initialized = False
        self.background_update_timer = None
        
        self.get_logger().info('DRaM Exploration Map initialized - GREEN for safe paths, RED only when dead ends detected')

    def get_safety_color(self, safety_level):
        """Get RGB color for safety level: only show green for safe areas, red only for actual dead ends"""
        # Only show colors when there's a clear prediction
        if safety_level >= 0.5:  # Safe/open path detected
            return (0.0, 1.0, 0.0)  # Pure bright green
        else:  # Dead end detected
            return (1.0, 0.0, 0.0)  # Pure bright red (only when dead end is predicted)

    def get_recovery_point_color(self, open_paths):
        """Get color for recovery points based on number of open paths"""
        if open_paths >= 3:  # 3 sides open
            return (0.5, 0.0, 1.0)  # Purple
        elif open_paths >= 2:  # 2 sides open
            return (0.0, 0.0, 0.8)  # Dark blue
        else:  # 1 side open
            return (0.3, 0.6, 1.0)  # Light blue

    def get_robot_position(self):
        """Get current robot position"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'body', rclpy.time.Time(), 
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            return (
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.rotation.z,
                'map'
            )
        except Exception:
            return (0.0, 0.0, 0.0, 'body')

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int(math.floor(x / self.grid_resolution))
        grid_y = int(math.floor(y / self.grid_resolution))
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates (cell center)"""
        world_x = (grid_x + 0.5) * self.grid_resolution
        world_y = (grid_y + 0.5) * self.grid_resolution
        return (world_x, world_y)

    def is_navigable_cell(self, world_x, world_y):
        """Check if a world coordinate is in a navigable (free) map cell"""
        if not self.current_map:
            return False
            
        # Convert world coordinates to map pixel coordinates
        map_x = int((world_x - self.current_map.info.origin.position.x) / self.current_map.info.resolution)
        map_y = int((world_y - self.current_map.info.origin.position.y) / self.current_map.info.resolution)
        
        # Check bounds
        if (map_x < 0 or map_x >= self.current_map.info.width or 
            map_y < 0 or map_y >= self.current_map.info.height):
            return False
        
        # Check a small area around the point to ensure it's truly free
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_x = map_x + dx
                check_y = map_y + dy
                
                # Check bounds for the expanded area
                if (check_x < 0 or check_x >= self.current_map.info.width or 
                    check_y < 0 or check_y >= self.current_map.info.height):
                    continue
                
                # Get occupancy value
                idx = check_y * self.current_map.info.width + check_x
                if idx >= len(self.current_map.data):
                    continue
                    
                occupancy = self.current_map.data[idx]
                
                # If any surrounding cell is occupied (walls/obstacles), reject this location
                # Only accept cells that are definitely free (occupancy == 0)
                if occupancy != 0:
                    return False
        
        return True  # All surrounding cells are free

    def initialize_map_bounds(self):
        """Initialize map bounds - no background coloring, only show predictions when made"""
        if not self.current_map or self.background_initialized:
            return
            
        # Get map bounds
        width = self.current_map.info.width
        height = self.current_map.info.height
        resolution = self.current_map.info.resolution
        origin_x = self.current_map.info.origin.position.x
        origin_y = self.current_map.info.origin.position.y
        
        # Calculate world bounds
        min_x = origin_x
        max_x = origin_x + width * resolution
        min_y = origin_y
        max_y = origin_y + height * resolution
        
        self.map_bounds = {
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y
        }
        
        self.background_initialized = True
        self.get_logger().info(f'🗺️ Map bounds initialized: {min_x:.1f} to {max_x:.1f} x {min_y:.1f} to {max_y:.1f}')
        self.get_logger().info('🎯 Clean visualization: GREEN for safe paths, RED only for detected dead ends')

    def path_status_callback(self, msg):
        """Process path status and update exploration map"""
        if len(msg.data) != 3:
            return

        robot_pos = self.get_robot_position()
        if robot_pos is None:
            return

        robot_x, robot_y, robot_yaw, frame_id = robot_pos
        current_time = time.time()

        # Calculate safety from the 3 path probabilities
        # Higher probability of open paths = higher safety
        front_prob, left_prob, right_prob = msg.data
        
        # Only mark as dead end (red) if ALL paths are clearly blocked
        # Only mark as safe (green) if ANY path is clearly open
        threshold = 0.54
        
        # Check if this is a clear dead end prediction (all paths blocked)
        all_paths_blocked = all(prob <= threshold for prob in [front_prob, left_prob, right_prob])
        any_path_open = any(prob > threshold for prob in [front_prob, left_prob, right_prob])
        
        if any_path_open:
            safety_level = 1.0  # Green - safe to proceed
        elif all_paths_blocked:
            safety_level = 0.0  # Red - dead end detected
        else:
            # Uncertain - don't visualize (skip this update)
            return
        
        # Determine what action we're taking
        if safety_level >= 0.5:
            action_type = "🟢 MARKING GREEN (safe path detected)"
        else:
            action_type = "🔴 MARKING RED (dead end detected)"
        
        # Debug logging to see what values we're getting
        open_paths = [f"F={front_prob:.3f}{'✓' if front_prob > threshold else '✗'}", 
                     f"L={left_prob:.3f}{'✓' if left_prob > threshold else '✗'}", 
                     f"R={right_prob:.3f}{'✓' if right_prob > threshold else '✗'}"]
        self.get_logger().info(f'🔍 Path probs: {", ".join(open_paths)} → {action_type}')
        
        # Update exploration area around robot position
        self.update_exploration_area(robot_x, robot_y, safety_level, current_time)
        
        # Publish updated exploration map
        self.publish_exploration_map()
        
        # Count areas (only show when there are clear predictions)
        total_explored = sum(1 for cell in self.explored_grid.values() if cell["explored"])
        green_areas = sum(1 for cell in self.explored_grid.values() if cell["explored"] and cell["safety"] >= 0.5)
        red_areas = sum(1 for cell in self.explored_grid.values() if cell["explored"] and cell["safety"] < 0.5)
        
        self.get_logger().info(f'📍 At ({robot_x:.1f}, {robot_y:.1f}) | '
                              f'Map: 🟢{green_areas} safe areas, 🔴{red_areas} dead ends (Total: {total_explored})')

    def update_exploration_area(self, robot_x, robot_y, safety_level, timestamp):
        """Update the exploration area around the robot with safety predictions"""
        # Update cells within exploration radius
        for dx in np.arange(-self.exploration_radius, self.exploration_radius + self.grid_resolution, self.grid_resolution):
            for dy in np.arange(-self.exploration_radius, self.exploration_radius + self.grid_resolution, self.grid_resolution):
                # Calculate distance from robot
                distance = math.sqrt(dx*dx + dy*dy)
                if distance <= self.exploration_radius:
                    # Calculate world position
                    world_x = robot_x + dx
                    world_y = robot_y + dy
                    
                    # Only update if this is a navigable cell
                    if not self.is_navigable_cell(world_x, world_y):
                        continue
                    
                    grid_pos = self.world_to_grid(world_x, world_y)
                    
                    # Weight safety by distance (closer = more influence)
                    distance_weight = max(0.1, 1.0 - (distance / self.exploration_radius))
                    weighted_safety = safety_level * distance_weight
                    
                    # Update or create grid cell with PERSISTENT GREEN logic
                    if grid_pos in self.explored_grid:
                        # If cell exists, check if it's already green (safe)
                        old_safety = self.explored_grid[grid_pos]['safety']
                        
                        # PERSISTENCE RULE: Once marked, keep the marking until robot revisits
                        if distance <= self.grid_resolution:
                            # Robot is at this location - update with current observation
                            new_safety = weighted_safety
                            if old_safety >= 0.5 and weighted_safety < 0.5:
                                self.get_logger().info(f'🔴 Dead end detected at previously safe location ({world_x:.1f}, {world_y:.1f})')
                            elif old_safety < 0.5 and weighted_safety >= 0.5:
                                self.get_logger().info(f'🟢 Safe path found at previously blocked location ({world_x:.1f}, {world_y:.1f})')
                        else:
                            # Robot is far - keep existing status (no change)
                            new_safety = old_safety
                    else:
                        # New cell - initialize with current prediction
                        new_safety = weighted_safety
                        if weighted_safety >= 0.5:
                            self.get_logger().info(f'🟢 Safe path detected at ({world_x:.1f}, {world_y:.1f})')
                        else:
                            self.get_logger().info(f'🔴 Dead end detected at ({world_x:.1f}, {world_y:.1f})')
                    
                    self.explored_grid[grid_pos] = {
                        'safety': new_safety,
                        'timestamp': timestamp,
                        'explored': True  # Mark as explored
                    }

    def recovery_points_callback(self, msg):
        """Store all recovery points with open paths information"""
        self.get_logger().info(f'🔍 Received recovery points message with {len(msg.data)} elements')
        
        if len(msg.data) % 3 != 0:
            self.get_logger().warn(f'⚠️ Recovery points data length {len(msg.data)} is not divisible by 3')
            return

        current_time = time.time()

        # Clear existing recovery points to start fresh
        self.recovery_points = []

        # Add all recovery points from the message
        for i in range(0, len(msg.data), 3):
            point_type = int(msg.data[i])
            x = msg.data[i + 1]
            y = msg.data[i + 2]

            # Determine number of open paths from point type
            # Type 1 = 2+ openings (preferred), Type 2 = 1 opening (backup)
            open_paths = 2 if point_type == 1 else 1

            self.recovery_points.append({
                'type': point_type,
                'x': x,
                'y': y,
                'open_paths': open_paths,
                'timestamp': current_time
            })
            
            self.get_logger().info(f'📍 Added recovery point: Type {point_type}, ({x:.2f}, {y:.2f}), {open_paths} open paths')

        self.get_logger().info(f'📍 Total {len(self.recovery_points)} recovery points stored')

    def map_callback(self, msg):
        """Store map and initialize red background"""
        self.current_map = msg
        
        # Initialize map bounds when we first get map data
        if not self.background_initialized:
            self.initialize_map_bounds()

    def publish_exploration_map(self):
        """Publish the exploration map - only show areas with clear predictions"""
        marker_array = MarkerArray()
        marker_id = 0
        
        robot_pos = self.get_robot_position()
        if robot_pos is None:
            return
            
        _, _, _, frame_id = robot_pos

        # 1. Create heatmap showing only areas with clear predictions
        if self.explored_grid:
            # Create a single POINTS marker for the heatmap
            heatmap_marker = Marker()
            heatmap_marker.header.frame_id = frame_id
            heatmap_marker.header.stamp = self.get_clock().now().to_msg()
            heatmap_marker.ns = "exploration_heatmap"
            heatmap_marker.id = marker_id
            heatmap_marker.type = Marker.POINTS
            heatmap_marker.action = Marker.ADD
            
            # Set point size for clear visibility
            heatmap_marker.scale.x = self.grid_resolution * 1.5  # Point width - larger for better visibility
            heatmap_marker.scale.y = self.grid_resolution * 1.5  # Point height
            heatmap_marker.scale.z = 0.03  # Point thickness - slightly thicker
            
            heatmap_marker.pose.orientation.w = 1.0
            
            # Add only explored areas with clear predictions
            for (grid_x, grid_y), data in self.explored_grid.items():
                # Only show explored areas with definitive predictions
                if not data.get('explored', False):
                    continue
                    
                world_x, world_y = self.grid_to_world(grid_x, grid_y)
                safety_level = data['safety']

                # Create point
                point = Point()
                point.x = world_x
                point.y = world_y
                point.z = 0.03  # Slightly above ground
                heatmap_marker.points.append(point)

                # Create color for this point
                color = ColorRGBA()
                r, g, b = self.get_safety_color(safety_level)
                color.r = r
                color.g = g
                color.b = b
                color.a = 0.9  # High visibility for clear predictions
                heatmap_marker.colors.append(color)
            
            # Only add marker if we have points to show
            if heatmap_marker.points:
                heatmap_marker.lifetime.sec = 0  # Persistent until updated
                marker_array.markers.append(heatmap_marker)
            marker_id += 1

        # 2. Add recovery points as pins with text labels
        for i, rp in enumerate(self.recovery_points):
            # Create pin marker (CYLINDER)
            pin_marker = Marker()
            pin_marker.header.frame_id = frame_id
            pin_marker.header.stamp = self.get_clock().now().to_msg()
            pin_marker.ns = "recovery_pins"
            pin_marker.id = marker_id
            pin_marker.type = Marker.CYLINDER
            pin_marker.action = Marker.ADD
            
            # Position the pin
            pin_marker.pose.position.x = rp['x']
            pin_marker.pose.position.y = rp['y']
            pin_marker.pose.position.z = 0.1  # Height of pin
            pin_marker.pose.orientation.w = 1.0
            
            # Size of pin
            pin_marker.scale.x = 0.3  # Width
            pin_marker.scale.y = 0.3  # Depth
            pin_marker.scale.z = 0.2  # Height
            
            # Color based on open paths
            r, g, b = self.get_recovery_point_color(rp['open_paths'])
            pin_marker.color.r = r
            pin_marker.color.g = g
            pin_marker.color.b = b
            pin_marker.color.a = 0.9  # Solid color
            
            pin_marker.lifetime.sec = 0  # Persistent
            marker_array.markers.append(pin_marker)
            marker_id += 1
            
            # Create text label showing number of open paths
            text_marker = Marker()
            text_marker.header.frame_id = frame_id
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "recovery_labels"
            text_marker.id = marker_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # Position text above the pin
            text_marker.pose.position.x = rp['x']
            text_marker.pose.position.y = rp['y']
            text_marker.pose.position.z = 0.3  # Above the pin
            text_marker.pose.orientation.w = 1.0
            
            # Text content
            open_paths = rp['open_paths']
            text_marker.text = f"{open_paths} open"
            
            # Text size
            text_marker.scale.z = 0.15  # Text height
            
            # Text color (white for visibility)
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.lifetime.sec = 0  # Persistent
            marker_array.markers.append(text_marker)
            marker_id += 1

        # Publish exploration map
        self.risk_map_pub.publish(marker_array)



def main(args=None):
    rclpy.init(args=args)
    
    node = DRaMExplorationMap()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
