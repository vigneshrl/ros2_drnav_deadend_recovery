#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from std_msgs.msg import Header

class PointCloudSegmenter(Node):
    def __init__(self):
        super().__init__('pointcloud_segmenter')

        # BEST_EFFORT matches Isaac Sim and real Ouster driver QoS
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribe to main point cloud
        self.subscription = self.create_subscription(
            PointCloud2,
            '/j100_0893/sensors/lidar3d_0/points', #change the topic as per your robtos namespace
            self.pointcloud_callback,
            qos
        )
        
        # Publishers for segmented point clouds
        self.front_pub = self.create_publisher(PointCloud2, '/lidar/front/points', qos)
        self.left_pub = self.create_publisher(PointCloud2, '/lidar/left/points', 10)
        self.right_pub = self.create_publisher(PointCloud2, '/lidar/right/points', 10)
        
        self.get_logger().info('PointCloud Segmenter initialized')
    
    def pointcloud_callback(self, msg: PointCloud2):
        try:
            data_struct = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.column_stack([
                data_struct['x'],
                data_struct['y'],
                data_struct['z']
            ]).astype(np.float32)
            
            if points.shape[0] == 0:
                return
            angles = np.arctan2(points[:, 1], points[:, 0])
            front_mask = (angles >= -0.524) & (angles <= 0.524)
            right_mask = (angles >= -2.094) & (angles <= -1.047)
            left_mask = (angles >= 1.047) & (angles <= 2.094)
            self.publish_cloud(self.front_pub, points[front_mask], msg.header)
            self.publish_cloud(self.left_pub, points[left_mask], msg.header)
            self.publish_cloud(self.right_pub, points[right_mask], msg.header)
            
            self.get_logger().debug(f'Segmented: Front={np.sum(front_mask)}, Left={np.sum(left_mask)}, Right={np.sum(right_mask)}')
            
        except Exception as e:
            self.get_logger().error(f'Error in point cloud segmentation: {e}')

    def publish_cloud(self, publisher, points, header):
        if points.shape[0] > 0:
            # Using create_cloud is much faster and cleaner than manual message assembly
            fields = [
                pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
            ]
            msg = pc2.create_cloud(header, fields, points)
            publisher.publish(msg)

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
