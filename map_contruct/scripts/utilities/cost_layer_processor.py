#!/usr/bin/env python3

"""
Real-time Binary Cost Layer Processor with Recovery Points

This node processes 3-directional path status predictions from the inference node
and creates a binary costmap with recovery point detection. It:

1. Receives 3 path probabilities (front, left, right) from inference
2. Converts probabilities to binary values using 0.56 threshold
3. Creates colored markers: Green=Open(1), Red=Blocked(0), Yellow=Recovery
4. Detects recovery points when ≥2 paths are open
5. Maintains historical costmap data over time
6. Visualizes current status and accumulated path history
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped, Point
import numpy as np
import math
import time

class BayesianCostLayerProcessor(Node):
    def __init__(self):
        super().__init__('bayesian_cost_layer_processor')

        # Subscribe to the model's path status output
        self.path_subscription = self.create_subscription(
            Float32MultiArray,
            '/dead_end_detection/path_status',
            self.path_status_callback,
            10
        )
        # self.path_subscription = self.create_subscription(
        #     Float32MultiArray, 
        #     '/single_camera/path_status',
        #     self.path_status_callback,
        #     10
        # )

        # Subscribe to the SLAM map for positioning
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # Publisher for the cost layer visualization
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/cost_layer',
            10
        )
        
        # Publisher for recovery points with type information
        self.recovery_points_pub = self.create_publisher(
            Float32MultiArray,
            '/dead_end_detection/recovery_points',
            10
        )

        # Transform listener for robot position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Store robot position and map info
        self.robot_position = None
        self.map_info = None
        
        # Angular sector parameters for costmap
        self.cost_history = {}  # sector_key -> {'cost_value': int, 'timestamp': float, ...}
        self.sector_radius = 3.0  # How far sectors extend (meters)
        self.sector_angle = np.pi / 3  # 60 degrees per sector (adjustable)
        self.max_history_age = 30.0  # Keep cost data for 30 seconds
        
        # Recovery points storage with type information
        self.recovery_points = []  # List of {'x': float, 'y': float, 'type': int, 'timestamp': float}
        self.max_recovery_age = 60.0  # Keep recovery points for 60 seconds
        
        # Current detection results (3 directional outputs)
        self.current_path_status = None  # Raw probabilities
        self.current_path_binary = None  # Binary values (0/1)
        
        self.get_logger().info('Bayesian Cost Layer Processor initialized')

    def bayesian_update(self, logodds, p):
        """Update log-odds with new observation probability"""
        if p <= 0.0:
            p = 1e-6
        if p >= 1.0:
            p = 1.0 - 1e-6
        l_obs = math.log(p / (1 - p))
        return logodds + l_obs

    def get_robot_position(self):
        """Get current robot position - use body frame if map frame not available"""
        try:
            # First try to get position in map frame
            transform = self.tf_buffer.lookup_transform(
                'map', 'body', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
            )
            return (
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.rotation.z,
                'map'  # Return frame used
            )
        except Exception:
            # If map frame not available, use body frame (origin at robot)
            self.get_logger().debug('Map frame not available, using body frame origin')
            return (0.0, 0.0, 0.0, 'body')  # Robot at origin in body frame

    def map_callback(self, msg):
        """Store map information for coordinate conversion"""
        self.map_info = msg.info
        
    def path_status_callback(self, msg):
        """Process 3-directional path status from inference node"""
        if len(msg.data) != 3:
            self.get_logger().warning(f'Expected 3 path status values, got {len(msg.data)}')
            return
            
        # Get current robot position
        robot_pos = self.get_robot_position()
        if robot_pos is None:
            self.get_logger().debug('No robot position available')
            return
            
        robot_x, robot_y, robot_yaw, frame_id = robot_pos
        current_time = time.time()
        
        # Convert probabilities to binary using 0.56 threshold (consistent with visualization)
        threshold = 0.56
        path_binary = [1 if prob > threshold else 0 for prob in msg.data]
        front_open, left_open, right_open = path_binary
        
        # Determine overall status
        open_paths_count = sum(path_binary)
        is_dead_end = (open_paths_count == 0)
        is_recovery_point = (open_paths_count >= 2)
        
        # Store the current detection results (both probabilities and binary)
        self.current_path_status = msg.data
        self.current_path_binary = path_binary
        
        self.get_logger().info(f'🎯 Path probs: F={msg.data[0]:.3f}, L={msg.data[1]:.3f}, R={msg.data[2]:.3f} | '
                              f'Binary: F={front_open}, L={left_open}, R={right_open} | '
                              f'Open: {open_paths_count}/3 | '
                              f'Status: {"🟡 RECOVERY" if is_recovery_point else "🔴 DEAD END" if is_dead_end else "🟢 OPEN"}')
        
        # Create directional cost markers relative to robot position
        directions = ['front', 'left', 'right']
        direction_angles = [0.0, np.pi/2, -np.pi/2]  # Front, left, right relative to robot
        detection_range = 2.0  # How far ahead to place the cost markers
        
        marker_array = MarkerArray()
        marker_id = 0
        
        # Create angular sector markers for each direction
        # Use class parameters for consistency
        
        for i, (direction, angle_offset, binary_value) in enumerate(zip(directions, direction_angles, path_binary)):
            # Calculate the central angle for this direction
            central_angle = robot_yaw + angle_offset
            
            # Store sector information for history
            sector_key = f"{direction}_{robot_x:.1f}_{robot_y:.1f}_{current_time}"
            self.cost_history[sector_key] = {
                'cost_value': binary_value,
                'timestamp': current_time,
                'direction': direction,
                'robot_x': robot_x,
                'robot_y': robot_y,
                'robot_yaw': robot_yaw,
                'angle_offset': angle_offset
            }
            
            # Create angular sector using TRIANGLE_LIST
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"sector_{direction}"
            marker.id = marker_id
            marker.type = Marker.TRIANGLE_LIST
            marker.action = Marker.ADD
            
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            
            # Color based on binary value: Green=Open(1), Red=Blocked(0)
            if binary_value == 1:  # Open path
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.6
            else:  # Blocked path
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.6
            
            # Create sector geometry (pie slice)
            num_triangles = 10  # More triangles = smoother arc
            angle_start = central_angle - self.sector_angle / 2
            angle_end = central_angle + self.sector_angle / 2
            
            # Center point (robot position)
            center = Point()
            center.x = robot_x
            center.y = robot_y
            center.z = 0.1
            
            # Create triangular fan from center to arc
            for j in range(num_triangles):
                # Calculate points on the arc
                angle1 = angle_start + (angle_end - angle_start) * j / num_triangles
                angle2 = angle_start + (angle_end - angle_start) * (j + 1) / num_triangles
                
                point1 = Point()
                point1.x = robot_x + self.sector_radius * math.cos(angle1)
                point1.y = robot_y + self.sector_radius * math.sin(angle1)
                point1.z = 0.1
                
                point2 = Point()
                point2.x = robot_x + self.sector_radius * math.cos(angle2)
                point2.y = robot_y + self.sector_radius * math.sin(angle2)
                point2.z = 0.1
                
                # Add triangle (center, point1, point2)
                marker.points.extend([center, point1, point2])
            
            marker.lifetime.sec = 5
            marker_array.markers.append(marker)
            marker_id += 1
        
        # Create recovery point marker and store recovery points
        # Type 1: 2+ paths open (preferred), Type 2: 1 path open (backup)
        if is_recovery_point or open_paths_count == 1:
            # Place recovery marker at robot position (center)
            recovery_marker = Marker()
            recovery_marker.header.frame_id = frame_id
            recovery_marker.header.stamp = self.get_clock().now().to_msg()
            recovery_marker.ns = "recovery_point"
            recovery_marker.id = marker_id
            recovery_marker.type = Marker.CYLINDER
            recovery_marker.action = Marker.ADD
            
            recovery_marker.pose.position.x = robot_x
            recovery_marker.pose.position.y = robot_y
            recovery_marker.pose.position.z = 0.3
            recovery_marker.pose.orientation.w = 1.0
            
            recovery_marker.scale.x = 1.0  # 1 meter diameter cylinder
            recovery_marker.scale.y = 1.0  # 1 meter diameter cylinder
            recovery_marker.scale.z = 0.4
            
            # Yellow for recovery point
            recovery_marker.color.r = 1.0
            recovery_marker.color.g = 1.0
            recovery_marker.color.b = 0.0
            recovery_marker.color.a = 0.9
            
            recovery_marker.lifetime.sec = 10  # Recovery points last longer
            marker_array.markers.append(recovery_marker)
            marker_id += 1
            
            # Store recovery point with type information
            # Type 1: 2+ openings (preferred), Type 2: 1 opening (backup)
            recovery_type = 1 if open_paths_count >= 2 else 2
            recovery_point = {
                'x': robot_x,
                'y': robot_y,
                'type': recovery_type,
                'timestamp': current_time,
                'open_paths': open_paths_count
            }
            
            # Add to recovery points list (avoid duplicates within 1.0 meter)
            is_duplicate = False
            for existing in self.recovery_points:
                if (abs(existing['x'] - robot_x) < 1.0 and 
                    abs(existing['y'] - robot_y) < 1.0):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                self.recovery_points.append(recovery_point)
                self.get_logger().info(f'🎯 Saved recovery point Type {recovery_type} at ({robot_x:.2f}, {robot_y:.2f}) with {open_paths_count} openings')
        
        # Add center status marker at robot position
        status_marker = Marker()
        status_marker.header.frame_id = frame_id
        status_marker.header.stamp = self.get_clock().now().to_msg()
        status_marker.ns = "status_center"
        status_marker.id = marker_id
        status_marker.type = Marker.SPHERE
        status_marker.action = Marker.ADD
        
        status_marker.pose.position.x = robot_x
        status_marker.pose.position.y = robot_y
        status_marker.pose.position.z = 0.05
        status_marker.pose.orientation.w = 1.0
        
        status_marker.scale.x = 0.3
        status_marker.scale.y = 0.3
        status_marker.scale.z = 0.1
        
        # Color based on overall status
        if is_recovery_point:  # 2+ paths open - Yellow
            status_marker.color.r = 1.0
            status_marker.color.g = 1.0
            status_marker.color.b = 0.0
        elif is_dead_end:  # All paths blocked - Red
            status_marker.color.r = 1.0
            status_marker.color.g = 0.0
            status_marker.color.b = 0.0
        else:  # 1 path open - Green
            status_marker.color.r = 0.0
            status_marker.color.g = 1.0
            status_marker.color.b = 0.0
        
        status_marker.color.a = 1.0
        status_marker.lifetime.sec = 3
        marker_array.markers.append(status_marker)
        marker_id += 1
        
        # Add persistent historical sector markers (costmap)
        historical_count = 0
        for sector_key, data in self.cost_history.items():
            # Skip if too old
            if current_time - data['timestamp'] > self.max_history_age:
                continue
                
            # Skip current frame data (avoid duplicates)
            if abs(data['timestamp'] - current_time) < 1.0:
                continue
                
            # Get stored sector information
            cost_value = data.get('cost_value', 0)
            direction = data.get('direction', 'unknown')
            hist_robot_x = data.get('robot_x', robot_x)
            hist_robot_y = data.get('robot_y', robot_y)
            hist_robot_yaw = data.get('robot_yaw', robot_yaw)
            hist_angle_offset = data.get('angle_offset', 0)
            
            # Create historical sector marker (faded)
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"history_{direction}"
            marker.id = marker_id
            marker.type = Marker.TRIANGLE_LIST
            marker.action = Marker.ADD
            
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            
            # Faded colors for historical data
            if cost_value == 1:  # Historical open path
                marker.color.r = 0.0
                marker.color.g = 0.7
                marker.color.b = 0.0
                marker.color.a = 0.3
            else:  # Historical blocked path
                marker.color.r = 0.7
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.3
            
            # Recreate historical sector geometry
            hist_central_angle = hist_robot_yaw + hist_angle_offset
            hist_sector_radius = self.sector_radius * 0.8  # Slightly smaller for history
            hist_sector_angle = self.sector_angle
            
            angle_start = hist_central_angle - hist_sector_angle / 2
            angle_end = hist_central_angle + hist_sector_angle / 2
            
            # Center point (historical robot position)
            center = Point()
            center.x = hist_robot_x
            center.y = hist_robot_y
            center.z = 0.05
            
            # Create triangular fan for historical sector
            num_triangles = 8  # Fewer triangles for historical data
            for j in range(num_triangles):
                angle1 = angle_start + (angle_end - angle_start) * j / num_triangles
                angle2 = angle_start + (angle_end - angle_start) * (j + 1) / num_triangles
                
                point1 = Point()
                point1.x = hist_robot_x + hist_sector_radius * math.cos(angle1)
                point1.y = hist_robot_y + hist_sector_radius * math.sin(angle1)
                point1.z = 0.05
                
                point2 = Point()
                point2.x = hist_robot_x + hist_sector_radius * math.cos(angle2)
                point2.y = hist_robot_y + hist_sector_radius * math.sin(angle2)
                point2.z = 0.05
                
                # Add triangle (center, point1, point2)
                marker.points.extend([center, point1, point2])
            
            marker.lifetime.sec = 20  # Historical sectors last longer
            marker_array.markers.append(marker)
            marker_id += 1
            historical_count += 1
        
        # Clean up old history
        cutoff_time = current_time - self.max_history_age
        self.cost_history = {k: v for k, v in self.cost_history.items() 
                           if v['timestamp'] > cutoff_time}
        
        # Clean up old recovery points
        recovery_cutoff_time = current_time - self.max_recovery_age
        self.recovery_points = [rp for rp in self.recovery_points 
                               if rp['timestamp'] > recovery_cutoff_time]
        
        # Publish recovery points for the DWA planner
        self.publish_recovery_points()
        
        # Publish the marker array
        self.marker_pub.publish(marker_array)
        
        # Enhanced logging
        status_text = "🟡 RECOVERY" if is_recovery_point else "🔴 DEAD END" if is_dead_end else "🟢 OPEN"
        self.get_logger().info(f'📍 Published {marker_id} markers | Current: 3 paths + 1 status + {"1 recovery" if is_recovery_point else "0 recovery"} | Historical: {historical_count} | Status: {status_text}')

    def publish_recovery_points(self):
        """Publish recovery points with type information for DWA planner"""
        if not self.recovery_points:
            self.get_logger().debug('📍 No recovery points to publish')
            return
            
        # Format: [type1, x1, y1, type2, x2, y2, ...] 
        # Type 1 = 2+ openings (preferred), Type 2 = 1 opening (backup)
        recovery_data = []
        
        for rp in self.recovery_points:
            recovery_data.extend([
                float(rp['type']),  # Recovery point type
                float(rp['x']),     # X coordinate
                float(rp['y'])      # Y coordinate
            ])
            self.get_logger().info(f'📍 Publishing recovery point: Type {rp["type"]}, ({rp["x"]:.2f}, {rp["y"]:.2f})')
        
        # Create and publish message
        msg = Float32MultiArray()
        msg.data = recovery_data
        self.recovery_points_pub.publish(msg)
        self.get_logger().info(f'📍 Published {len(self.recovery_points)} recovery points to /dead_end_detection/recovery_points')
        
        # Log recovery points summary
        type1_count = len([rp for rp in self.recovery_points if rp['type'] == 1])
        type2_count = len([rp for rp in self.recovery_points if rp['type'] == 2])
        self.get_logger().debug(f'🎯 Published {len(self.recovery_points)} recovery points: '
                               f'{type1_count} Type-1 (2+ paths), {type2_count} Type-2 (1 path)')

def main(args=None):
    rclpy.init(args=args)
    node = BayesianCostLayerProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()