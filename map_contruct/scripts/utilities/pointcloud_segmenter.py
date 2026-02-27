#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from std_msgs.msg import Header

class PointCloudSegmenter(Node):
    def __init__(self):
        super().__init__('pointcloud_segmenter')

        # BEST_EFFORT matches Isaac Sim and real Ouster driver QoS
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribe to main point cloud
        self.subscription = self.create_subscription(
            PointCloud2,
            '/os_cloud_node/points',
            self.pointcloud_callback,
            qos
        )
        
        # Publishers for segmented point clouds
        self.front_pub = self.create_publisher(PointCloud2, '/lidar/front/points', 10)
        self.left_pub = self.create_publisher(PointCloud2, '/lidar/left/points', 10)
        self.right_pub = self.create_publisher(PointCloud2, '/lidar/right/points', 10)
        
        self.get_logger().info('PointCloud Segmenter initialized')
    
    def pointcloud_callback(self, msg: PointCloud2):
        """Segment point cloud into front, left, right regions"""
        try:
            # Parse field offsets from message header — works with any LiDAR format
            field_offsets = {f.name: f.offset for f in msg.fields if f.name in ('x', 'y', 'z')}
            if not all(k in field_offsets for k in ('x', 'y', 'z')):
                self.get_logger().warn('PointCloud2 missing x/y/z fields')
                return

            n_pts = msg.width * msg.height
            if n_pts == 0:
                return

            raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n_pts, msg.point_step)
            x = raw[:, field_offsets['x']:field_offsets['x'] + 4].copy().view(np.float32).flatten()
            y = raw[:, field_offsets['y']:field_offsets['y'] + 4].copy().view(np.float32).flatten()
            z = raw[:, field_offsets['z']:field_offsets['z'] + 4].copy().view(np.float32).flatten()
            points = np.column_stack([x, y, z])

            # Remove NaN/Inf points
            valid = np.isfinite(points).all(axis=1)
            points = points[valid]
            if len(points) == 0:
                return
            
            # Calculate angles for each point
            angles = np.arctan2(points[:, 1], points[:, 0])  # atan2(y, x)
            
            # Define angular sectors (in radians)
            # Front: -30° to +30° (-0.524 to +0.524 rad)
            front_mask = (angles >= -0.524) & (angles <= 0.524)
            
            # Right: -120° to -60° (-2.094 to -1.047 rad) 
            right_mask = (angles >= -2.094) & (angles <= -1.047)
            
            # Left: +60° to +120° (+1.047 to +2.094 rad)
            left_mask = (angles >= 1.047) & (angles <= 2.094)
            
            # Create segmented point clouds
            front_points = points[front_mask]
            left_points = points[left_mask]
            right_points = points[right_mask]
            
            # Publish segmented clouds
            if len(front_points) > 0:
                front_msg = self.create_pointcloud2_msg(front_points, msg.header)
                self.front_pub.publish(front_msg)
                
            if len(left_points) > 0:
                left_msg = self.create_pointcloud2_msg(left_points, msg.header)
                self.left_pub.publish(left_msg)
                
            if len(right_points) > 0:
                right_msg = self.create_pointcloud2_msg(right_points, msg.header)
                self.right_pub.publish(right_msg)
                
            self.get_logger().info(f'Segmented: Front={len(front_points)}, Left={len(left_points)}, Right={len(right_points)}')
            
        except Exception as e:
            self.get_logger().error(f'Error in point cloud segmentation: {e}')
    
    def create_pointcloud2_msg(self, points: np.ndarray, header: Header) -> PointCloud2:
        """Create PointCloud2 message from numpy array"""
        # Create point cloud message
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.is_dense = True
        cloud_msg.is_bigendian = False
        
        # Define fields (x, y, z)
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.point_step = 12  # 3 floats * 4 bytes each
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        
        # Convert points to bytes
        cloud_msg.data = points.astype(np.float32).tobytes()
        
        return cloud_msg

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSegmenter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
