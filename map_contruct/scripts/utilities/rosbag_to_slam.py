#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2
import numpy as np
from tf2_ros import TransformException, TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time
from rclpy.clock import ClockType
from rclpy.duration import Duration


class RosbagToSLAM(Node):
    def __init__(self):
        super().__init__('rosbag_to_slam')
        
        # Publisher for the converted laser scan
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        
        # # Transform broadcaster
        # self.tf_broadcaster = TransformBroadcaster(self)
        
        # Parameters
        self.declare_parameter('point_cloud_topic', '/os_cloud_node/points')
        # self.declare_parameter('base_frame', 'map')
        self.declare_parameter('laser_frame', 'os_sensor')
        # self.declare_parameter('body_frame', 'base_link')
        
        self.point_cloud_topic = self.get_parameter('point_cloud_topic').value
        # self.base_frame = self.get_parameter('base_frame').value
        self.laser_frame = self.get_parameter('laser_frame').value
        # self.body_frame = self.get_parameter('body_frame').value
        
        # Timer to broadcast transform
        # self.timer = self.create_timer(0.1, self.broadcast_transform)
        
        # Subscribe to point cloud topic
        self.point_cloud_sub = self.create_subscription(
            PointCloud2,
            self.point_cloud_topic,
            self.point_cloud_callback,
            10
        )
        
        # Laser scan parameters - adjusted for better coverage
        self.scan = LaserScan()
        self.scan.header.frame_id = self.laser_frame
        self.scan.angle_min = -3.14159  # Full 360 degrees
        self.scan.angle_max = 3.14159
        self.scan.angle_increment = 0.0349  # 2 degree in radians
        self.scan.time_increment = 0.0
        self.scan.scan_time = 0.1
        self.scan.range_min = 0.3  # Adjusted min range
        self.scan.range_max = 100.0  # Increased max range
        
        # Initialize ranges array
        self.num_readings = int((self.scan.angle_max - self.scan.angle_min) / self.scan.angle_increment)
        self.scan.ranges = [float('inf')] * self.num_readings
        
        # self.get_logger().info(f'Initialized with parameters:')
        # self.get_logger().info(f'  point_cloud_topic: {self.point_cloud_topic}')
        # self.get_logger().info(f'  base_frame: {self.base_frame}')
        # self.get_logger().info(f'  laser_frame: {self.laser_frame}')
        # # self.get_logger().info(f'  body_frame: {self.body_frame}')
        # self.get_logger().info(f'  num_readings: {self.num_readings}')
        self.get_logger().info('Waiting for point cloud messages...')

    # def broadcast_transform(self):
    #     # First broadcast transform from map (global frame) to base_link (robot's local frame)
    #     t_map_to_base = TransformStamped()
    #     t_map_to_base.header.stamp = self.get_clock().now().to_msg()
    #     t_map_to_base.header.frame_id = "map"  # Global frame
    #     t_map_to_base.child_frame_id = "body"  # Robot's local frame

    #     # Set the transform from map to base_link
    #     # These values position the robot within the global map frame
    #     t_map_to_base.transform.translation.x = 0.0
    #     t_map_to_base.transform.translation.y = 0.0
    #     t_map_to_base.transform.translation.z = 0.0
    #     t_map_to_base.transform.rotation.x = 0.0
    #     t_map_to_base.transform.rotation.y = 0.0
    #     t_map_to_base.transform.rotation.z = 0.0
    #     t_map_to_base.transform.rotation.w = 1.0

        # # Now broadcast transform from base_link to laser_frame
        # t_utm_to_local = TransformStamped()
        # t_utm_to_local.header.stamp = self.get_clock().now().to_msg()
        # t_utm_to_local.header.frame_id = "utm_frame"
        # t_utm_to_local.child_frame_id = "utm_local_frame"

        # # Set the transform from base_link to laser
        # # Adjust these values based on your robot's configuration
        # t_utm_to_local.transform.translation.x = 0.0
        # t_utm_to_local.transform.translation.y = 0.0
        # t_utm_to_local.transform.translation.z = 0.0  # Assuming the lidar is 0.2m above the base
        # t_utm_to_local.transform.rotation.x = 0.0
        # t_utm_to_local.transform.rotation.y = 0.0
        # t_utm_to_local.transform.rotation.z = 0.0
        # t_utm_to_local.transform.rotation.w = 1.0

        #         # Now broadcast transform from base_link to laser_frame
        # t_utm_local_to_gps = TransformStamped()
        # t_utm_local_to_gps.header.stamp = self.get_clock().now().to_msg()
        # t_utm_local_to_gps.header.frame_id = "utm_local_frame"
        # t_utm_local_to_gps.child_frame_id = "gps_tracking_link"

        # # Set the transform from base_link to laser
        # # Adjust these values based on your robot's configuration
        # t_utm_local_to_gps.transform.translation.x = 0.0
        # t_utm_local_to_gps.transform.translation.y = 0.0
        # t_utm_local_to_gps.transform.translation.z = 0.0 # Assuming the lidar is 0.2m above the base
        # t_utm_local_to_gps.transform.rotation.x = 0.0
        # t_utm_local_to_gps.transform.rotation.y = 0.0
        # t_utm_local_to_gps.transform.rotation.z = 0.0
        # t_utm_local_to_gps.transform.rotation.w = 1.0

        #                 # Now broadcast transform from base_link to laser_frame
        # t_gps_to_base = TransformStamped()
        # t_gps_to_base.header.stamp = self.get_clock().now().to_msg()
        # t_gps_to_base.header.frame_id = "gps_tracking_link"
        # t_gps_to_base.child_frame_id = "body_gps"

        # # Set the transform from base_link to laser
        # # Adjust these values based on your robot's configuration
        # t_gps_to_base.transform.translation.x = 0.0
        # t_gps_to_base.transform.translation.y = 0.0
        # t_gps_to_base.transform.translation.z = 0.0 # Assuming the lidar is 0.2m above the base
        # t_gps_to_base.transform.rotation.x = 0.0
        # t_gps_to_base.transform.rotation.y = 0.0
        # t_gps_to_base.transform.rotation.z = 0.0
        # t_gps_to_base.transform.rotation.w = 1.0

        #                         # Now broadcast transform from base_link to laser_frame
        # t_base_to_gps = TransformStamped()
        # t_base_to_gps.header.stamp = self.get_clock().now().to_msg()
        # t_base_to_gps.header.frame_id = "body_gps"
        # t_base_to_gps.child_frame_id = "body"

        # # Set the transform from base_link to laser
        # # Adjust these values based on your robot's configuration
        # t_base_to_gps.transform.translation.x = 0.0
        # t_base_to_gps.transform.translation.y = 0.0
        # t_base_to_gps.transform.translation.z = 0.0  # Assuming the lidar is 0.2m above the base
        # t_base_to_gps.transform.rotation.x = 0.0
        # t_base_to_gps.transform.rotation.y = 0.0
        # t_base_to_gps.transform.rotation.z = 0.0
        # t_base_to_gps.transform.rotation.w = 1.0


        # Broadcast the transforms
        # self.tf_broadcaster.sendTransform([t_map_to_base])
        self.get_logger().info('Publishing scan')
    def point_cloud_callback(self, cloud_msg):
        
        try:
            # Log point cloud info
            # self.get_logger().info(f'Received point cloud with {cloud_msg.width * cloud_msg.height} points')
            # self.get_logger().info(f'Point cloud frame: {cloud_msg.header.frame_id}')

            # Set the scan message timestamp
            self.scan.header.stamp = cloud_msg.header.stamp
            
            # Ensure scan is published in the laser frame
            self.scan.header.frame_id = self.laser_frame
            
            # Convert point cloud to numpy array
            points_list = list(point_cloud2.read_points(cloud_msg, 
                                                      field_names=("x", "y", "z"), 
                                                      skip_nans=True))
            
            if not points_list:
                self.get_logger().warn('No valid points in point cloud')
                return
            
            # Convert to numpy array and ensure proper shape
            points = np.array(points_list)
            
            # Check if we have valid 3D points
            if points.ndim != 2 or points.shape[1] != 3:
                self.get_logger().error(f'Invalid point cloud shape: {points.shape}. Expected (N, 3)')
                return
                
            if len(points) == 0:
                self.get_logger().warn('No valid points in point cloud after conversion')
                return
                
            # Filter points by height - keep a larger vertical slice
            mask = (points[:, 2] > -0.1) & (points[:, 2] < 0.5)
            points = points[mask]
            
            if len(points) == 0:
                self.get_logger().warn('No points remaining after height filter')
                return
            
            # Convert 3D points to 2D polar coordinates
            ranges = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            angles = np.arctan2(points[:, 1], points[:, 0])
            
            # Reset ranges array
            self.scan.ranges = [float('inf')] * self.num_readings
            
            # Fill in ranges
            points_used = 0
            for i in range(len(ranges)):
                if ranges[i] < self.scan.range_min or ranges[i] > self.scan.range_max:
                    continue
                
                # Normalize angle to [-pi, pi]
                angle = angles[i]
                while angle > np.pi:
                    angle -= 2 * np.pi
                while angle < -np.pi:
                    angle += 2 * np.pi
                    
                angle_idx = int((angle - self.scan.angle_min) / self.scan.angle_increment)
                if 0 <= angle_idx < len(self.scan.ranges):
                    if ranges[i] < self.scan.ranges[angle_idx]:
                        self.scan.ranges[angle_idx] = float(ranges[i])
                        points_used += 1
            
            # Log detailed scan info
            valid_ranges = sum(1 for r in self.scan.ranges if r < float('inf'))
            # self.get_logger().info(f'Points processed: {len(points)}, Points used: {points_used}, Valid ranges: {valid_ranges}')
            
            if valid_ranges > 0:
                # self.get_logger().info('Publishing scan')
                self.scan_pub.publish(self.scan)
            else:
                self.get_logger().warn('No valid ranges in scan, skipping publication')
                    
        except Exception as e:
            self.get_logger().error(f'Error in point_cloud_callback: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    node = RosbagToSLAM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 





#run command
#ros2 launch map_contruct rosbag_slam.launch.py bag_path:=/home/vicky/IROS2025/rosbags/deadend_recovery_bag_6_0-001.db3