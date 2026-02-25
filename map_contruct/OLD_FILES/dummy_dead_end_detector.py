#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray
import time

class DummyDeadEndDetector(Node):
    def __init__(self):
        super().__init__('dummy_dead_end_detector')
        
        # Publishers
        self.dead_end_pub = self.create_publisher(Bool, '/dead_end_detection/is_dead_end', 10)
        self.path_status_pub = self.create_publisher(Float32MultiArray, '/dead_end_detection/path_status', 10)
        self.recovery_point_pub = self.create_publisher(Float32MultiArray, '/dead_end_detection/recovery_points', 10)
        
        # Timer to publish dummy data
        self.create_timer(0.5, self.publish_dummy_data)  # 2 Hz
        
        self.get_logger().info('Dummy Dead End Detector initialized - all paths open, no dead ends')

    def publish_dummy_data(self):
        """Publish dummy data indicating no dead ends"""
        
        # Publish dead end status (always false - no dead ends)
        dead_end_msg = Bool()
        dead_end_msg.data = False
        self.dead_end_pub.publish(dead_end_msg)
        
        # Publish path status (all paths open with high confidence)
        path_status_msg = Float32MultiArray()
        path_status_msg.data = [0.8, 0.8, 0.8]  # Front, left, right all open
        self.path_status_pub.publish(path_status_msg)
        
        # Publish some dummy recovery points
        recovery_msg = Float32MultiArray()
        recovery_msg.data = [
            1.0, 10.0, 0.0,   # Type 1, x=10, y=0
            1.0, 0.0, 10.0,   # Type 1, x=0, y=10
            2.0, -10.0, 0.0   # Type 2, x=-10, y=0
        ]
        self.recovery_point_pub.publish(recovery_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DummyDeadEndDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



