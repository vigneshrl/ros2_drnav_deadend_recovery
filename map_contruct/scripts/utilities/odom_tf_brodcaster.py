# import rclpy
# from tf2_ros import TransformBroadcaster
# from geometry_msgs.msg import TransformStamped
# from nav_msgs.msg import Odometry

# def odom_callback(msg, broadcaster):
#     t = TransformStamped()
#     t.header.stamp = msg.header.stamp
#     t.header.frame_id = msg.header.frame_id  # lidar_origin
#     t.child_frame_id = msg.child_frame_id    # body
#     t.transform.translation.x = msg.pose.pose.position.x
#     t.transform.translation.y = msg.pose.pose.position.y
#     t.transform.translation.z = msg.pose.pose.position.z
#     t.transform.rotation = msg.pose.pose.orientation
#     broadcaster.sendTransform(t)

# def main(args=None):
#     rclpy.init(args=args)
#     node = rclpy.create_node('odom_tf_broadcaster')
#     broadcaster = TransformBroadcaster(node)
#     subscription = node.create_subscription(Odometry, '/odom_lidar', lambda msg: odom_callback(msg, broadcaster), 10)
#     rclpy.spin(node)
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()



# import rclpy
# from rclpy.node import Node
# from rclpy.qos import qos_profile_sensor_data  # Import this
# from tf2_ros import TransformBroadcaster
# from geometry_msgs.msg import TransformStamped
# from nav_msgs.msg import Odometry

# def odom_callback(msg, broadcaster):
#     t = TransformStamped()
#     t.header.stamp = msg.header.stamp
#     t.header.frame_id = 'msg.header.frame_id'  # lidar_origin
#     t.child_frame_id = msg.child_frame_id    # body
#     t.transform.translation.x = msg.pose.pose.position.x
#     t.transform.translation.y = msg.pose.pose.position.y
#     t.transform.translation.z = msg.pose.pose.position.z
#     t.transform.rotation = msg.pose.pose.orientation
#     broadcaster.sendTransform(t)

# def main(args=None):
#     rclpy.init(args=args)
#     node = Node('odom_tf_broadcaster')
#     broadcaster = TransformBroadcaster(node)
#     subscription = node.create_subscription(
#         Odometry,
#         '/odom_lidar',
#         lambda msg: odom_callback(msg, broadcaster),
#         qos_profile=qos_profile_sensor_data  # Add this for compatibility
#     )
#     rclpy.spin(node)
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()





#dynamic_transform_publisher between odom and base_link
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

class OdomTFBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')

        # Parameters (so you can override via CLI or a launch file)
        self.declare_parameter('odom_topic', '/odom_lidar')
        self.declare_parameter('parent_frame', 'odom')   # <- publish from here
        self.declare_parameter('child_frame', 'body')    # <- to here

        odom_topic   = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.parent  = self.get_parameter('parent_frame').get_parameter_value().string_value
        self.child   = self.get_parameter('child_frame').get_parameter_value().string_value

        self.tf_broadcaster = TransformBroadcaster(self)

        self.subscription = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            qos_profile_sensor_data
        )
        self.get_logger().info(
            f'Publishing dynamic TF {self.parent} -> {self.child} from odometry: {odom_topic}'
        )

    def odom_callback(self, msg: Odometry):
        t = TransformStamped()
        # Use the odometry timestamp if provided; otherwise, use node clock
        if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
            t.header.stamp = self.get_clock().now().to_msg()
        else:
            t.header.stamp = msg.header.stamp

        t.header.frame_id = self.parent       # force parent frame = "odom"
        t.child_frame_id  = self.child        # force child  frame = "body"

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation = msg.pose.pose.orientation  # dynamic orientation

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = OdomTFBroadcaster()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
