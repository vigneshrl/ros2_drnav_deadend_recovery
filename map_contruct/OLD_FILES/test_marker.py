#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

class TestMarker(Node):
    def __init__(self):
        super().__init__('test_marker')
        self.marker_pub = self.create_publisher(MarkerArray, 'test_marker', 10)
        self.timer = self.create_timer(1.0, self.publish_marker)
        self.marker_array = MarkerArray()

        for i in range(5):
            m = Marker()
            m.header.frame_id = "map"
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(i)
            m.pose.position.y = 0.0
            m.pose.position.z = 0.0
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = 0.2
            m.color.a = 1.0
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.id = i  # Important: Each marker needs a unique ID
            self.marker_array.markers.append(m)

    def publish_marker(self):
        self.marker_pub.publish(self.marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = TestMarker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()