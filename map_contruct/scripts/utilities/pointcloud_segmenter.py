#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from std_msgs.msg import Header

class PointCloudSegmenter(Node):
    def __init__(self):
        super().__init__('pointcloud_segmenter')
        
        # Subscribe to main point cloud
        self.subscription = self.create_subscription(
            PointCloud2,
            '/os_cloud_node/points',
            self.pointcloud_callback,
            10
        )
        
        # Publishers for segmented point clouds
        self.front_pub = self.create_publisher(PointCloud2, '/lidar/front/points', 10)
        self.left_pub = self.create_publisher(PointCloud2, '/lidar/left/points', 10)
        self.right_pub = self.create_publisher(PointCloud2, '/lidar/right/points', 10)
        
        self.get_logger().info('PointCloud Segmenter initialized')
    
    def pointcloud_callback(self, msg: PointCloud2):
        """Segment point cloud into front, left, right regions"""
        try:
            # Convert PointCloud2 to numpy array
            points_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            if not points_list:
                return
            
            # Convert structured array to regular float array
            if len(points_list) > 0:
                # Handle both structured and regular arrays
                if isinstance(points_list[0], (list, tuple)):
                    # Regular list/tuple format
                    points = np.array(points_list, dtype=np.float32)
                else:
                    # Structured array format - extract x, y, z
                    structured_array = np.array(points_list)
                    points = np.column_stack([
                        structured_array['x'],
                        structured_array['y'], 
                        structured_array['z']
                    ]).astype(np.float32)
            else:
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
                
            self.get_logger().debug(f'Segmented: Front={len(front_points)}, Left={len(left_points)}, Right={len(right_points)}')
            
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
            pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
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
