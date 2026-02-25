#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np

class CostLayerVisualizer(Node):
    def __init__(self):
        super().__init__('cost_layer_visualizer')
        
        # Subscribe to the map topic
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',  # Subscribe to /map topic
            self.map_callback,
            10)
        
        # Publisher for the cost layer visualization
        self.cost_pub = self.create_publisher(
            MarkerArray,
            '/cost_layer',
            10)
        self.last_marker = None
        self.timer = self.create_timer(1.0, self.publish_marker)
        self.get_logger().info('Cost Layer Visualizer initialized')

    def map_callback(self, msg):
        self.get_logger().info(f'Received map data: height={msg.info.height}, width={msg.info.width}, resolution={msg.info.resolution}')
        
        # Create markers for high and low cost areas
        high_cost_marker = Marker()
        low_cost_marker = Marker()
        
        # Common settings for both markers
        for marker in [high_cost_marker, low_cost_marker]:
            marker.header.frame_id = 'body'  # Use body frame consistently
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE_LIST  # Changed from POINTS to SPHERE_LIST
            marker.action = Marker.ADD
            
            # Make spheres larger and more visible
            marker.scale.x = msg.info.resolution * 4.0  # Sphere diameter
            marker.scale.y = msg.info.resolution * 4.0
            marker.scale.z = msg.info.resolution * 4.0
            # marker.color.a = 1.0
            
        # Set unique IDs
        high_cost_marker.id = 0
        low_cost_marker.id = 1
        
        # Set colors with higher opacity
        high_cost_marker.color.r = 1.0
        high_cost_marker.color.g = 0.0
        high_cost_marker.color.b = 0.0
        high_cost_marker.color.a = 0.8
        
        low_cost_marker.color.r = 0.0
        low_cost_marker.color.g = 1.0
        low_cost_marker.color.b = 0.0
        low_cost_marker.color.a = 0.8
        
        # Convert map data to numpy array
        map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        
        # Process each cell, but skip some cells to reduce density
        step = 2  # Process every 2nd cell to reduce density
        for i in range(0, msg.info.height, step):
            for j in range(0, msg.info.width, step):
                if map_data[i, j] != -1:  # Only process known cells
                    point = Point()
                    point.x = msg.info.origin.position.x + j * msg.info.resolution
                    point.y = msg.info.origin.position.y + i * msg.info.resolution
                    point.z = 0.5  # Lower height but still visible
                    
                    if map_data[i, j] > 50:  # High cost
                        high_cost_marker.points.append(point)
                    else:  # Low cost
                        low_cost_marker.points.append(point)
        
        # Create marker array and add markers
        marker_array = MarkerArray()
        if high_cost_marker.points:
            marker_array.markers.append(high_cost_marker)
        if low_cost_marker.points:
            marker_array.markers.append(low_cost_marker)
        
        # Publish markers
        self.cost_pub.publish(marker_array)
        self.last_marker = marker_array
        self.get_logger().info(f'Published markers - High cost: {len(high_cost_marker.points)}, Low cost: {len(low_cost_marker.points)}')

    def publish_marker(self):
        if self.last_marker:
            self.cost_pub.publish(self.last_marker)

def main(args=None):
    rclpy.init(args=args)
    node = CostLayerVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 


