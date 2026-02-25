# from launch import LaunchDescription
# from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
# from launch.substitutions import LaunchConfiguration
# from launch_ros.actions import Node
# from ament_index_python.packages import get_package_share_directory
# import os

# def generate_launch_description():
#     # Declare arguments
#     # bag_path_arg = DeclareLaunchArgument(
#     #     'bag_path',
#     #     default_value='',
#     #     description='Path to the ROS2 bag file'
#     # )

#     # Get the package share directory
#     pkg_share = get_package_share_directory('map_contruct')

#     # Static transform publisher for map->odom
#     # odom_to_body_publisher = Node(
#     #     package='tf2_ros',
#     #     executable='static_transform_publisher',
#     #     name='odom_to_body_publisher',
#     #     arguments=['0', '0', '0', '0', '0', '0', 'odom_lidar', 'body'],
#     #     parameters=[{'use_sim_time': True}]
#     # )

#     # Static transform publisher for utm_frame->body
#     # utm_to_body_publisher = Node(
#     #     package='tf2_ros',
#     #     executable='static_transform_publisher',
#     #     name='utm_to_body_publisher',
#     #     arguments=['0', '0', '0', '0', '0', '0', 'utm_frame', 'body'],
#     #     parameters=[{'use_sim_time': True}]
#     # )

#     # Static transform publisher for body->os_lidar
#     # body_to_lidar_publisher = Node(
#         # package='tf2_ros',
#         # executable='static_transform_publisher',
#         # name='body_to_lidar_publisher',
#         # arguments=['0', '0', '0.2', '0', '0', '0', 'map', 'os_lidar'],
#         # parameters=[{'use_sim_time': True}]
#     # )
#     map_to_body = Node(
#         package='tf2_ros',
#         executable='static_transform_publisher',
#         name='map_to_body_publisher',
#         arguments=['0', '0', '0', '0', '0', '0', 'map', 'utm_frame'],
#         parameters=[{'use_sim_time': True}]
#     )
#     ros2_slam_node = Node(
#         package='map_contruct',
#         executable='rosbag_to_slam',
#         name='point_cloud_convertor',
#         output='screen',
#         arguments=['--ros-args', '--log-level', 'DEBUG'],
#     )

#     # SLAM Toolbox node
#     slam_toolbox_node = Node(
#         package='slam_toolbox',
#         executable='sync_slam_toolbox_node',
#         name='slam_toolbox',
#         output='screen',
#         parameters=[{
#             'base_frame': 'body',
#             'odom_frame': 'lidar_origin',
#             'map_frame': 'map',
#             'use_sim_time': True,
#             'resolution': 0.05,
#             'max_laser_range': 20.0,
#             'minimum_travel_distance': 0.1,
#             'minimum_travel_heading': 0.1,
#             'map_update_interval': 1.0,
#             'transform_timeout': 2.0,
#             'update_factor': 3.0,
#             'laser_max_beams': 360,
#             'laser_min_range': 0.1,
#             'laser_max_range': 30.0,
#             'max_update_rate': 10.0,
#             'transform_publish_period': 0.05,
#             'map_start_pose': [0.0, 0.0, 0.0],
#             'mode': 'mapping',
#             'odom_topic': '/odom_lidar',
#             'scan_topic': '/scan',
#             'debug_logging': True,
#             'throttle_scans': 1,
#             'publish_frame_transforms': True,
#             'use_odom': True,
#             'use_pose_graph_backend': True,
#             'use_scan_matching': True,
#             'use_scan_barycenter': True,
#             'use_optimized_poses': True,
#             'scan_matcher_type': 'CSM',
#             'ceres_loss_function': 'HuberLoss',
#             'ceres_trust_region_strategy': 'LEVENBERG_MARQUARDT',
#             'enable_interactive_mode': True,
#             'queue_size': 100,
#             'tf_buffer_duration': 20.0,
#             'tf_tolerance': 0.1,
#             'scan_queue_size': 100,
#             'scan_buffer_duration': 10.0,
#             'scan_tolerance': 0.1,
#             'debug': True,
#             'qos_overrides./scan.subscriber.reliability': 'best_effort',
#             'qos_overrides./odom_lidar.subscriber.reliability': 'best_effort',
#             'qos_overrides./map.publisher.durability': 'transient_local'
#         }]
#     )

#     # ROS Bag Processing Node
#     rosbag_to_slam_node = Node(
#         package='map_contruct',
#         executable='rosbag_to_slam',
#         name='rosbag_to_slam',
#         output='screen',
#         parameters=[{
#             'point_cloud_topic': '/os_cloud_node/points',
#             'base_frame': 'body',
#             'laser_frame': 'os_sensor',
#             'use_sim_time': True
#         }],
#         arguments=['--ros-args', '--log-level', 'debug']
#     )

#     # # TF2 Buffer Server - helps with transform lookups
#     # tf2_buffer_server = Node(
#     #     package='tf2_ros',
#     #     executable='buffer_server',
#     #     name='tf2_buffer_server',
#     #     parameters=[{'use_sim_time': True}]
#     # )

#     # RViz node with custom config
#     # rviz_node = Node(
#     #     package='rviz2',
#     #     executable='rviz2',
#     #     name='rviz2',
#     #     arguments=['-d', os.path.join(pkg_share, 'config', 'slam.rviz')],
#     #     parameters=[{'use_sim_time': True}]
#     # )

#     # ROS Bag Play Node with delay and slower speed
#     # bag_play_node = TimerAction(
#     #     period=3.0,
#     #     actions=[
#     #         ExecuteProcess(
#     #             cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_path'), '--clock', '-r', '0.2'],
#     #             output='screen'
#     #         )
#     #     ]
#     # )

#     # Add a node to monitor TF
#     tf_monitor = Node(
#         package='tf2_ros',
#         executable='tf2_monitor',
#         name='tf_monitor',
#         output='screen'
#     )

#     # # Cost Layer Processor Node
#     # cost_layer_processor_node = Node(
#     #     package='map_contruct',
#     #     executable='cost_layer_processor',
#     #     name='cost_layer_processor',
#     #     output='screen',
#     #     parameters=[{'use_sim_time': True}]
#     # )

#     return LaunchDescription([
#         # bag_path_arg,
#         # odom_to_body_publisher,  # odom_lidar -> body
#         # body_publisher,  # utm_frame -> body
#         # body_to_lidar_publisher,  # body -> os_lidar
#         # tf2_buffer_server,
#         map_to_body,
#         ros2_slam_node,
#         slam_toolbox_node,
#         rosbag_to_slam_node,
#         tf_monitor,
#         # rviz_node,
#         # cost_layer_processor_node,
#         # bag_play_node
#     ])



#version 2

# ... existing code ...
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('map_contruct')
    # json_path_arg = DeclareLaunchArgument(
    #     'json_path',
    #     description='Path to the json file',
    #     default_value=os.path.join(pkg_share, 'data', 'merged_data.json')
    # )

    # Static transform: odom_lidar -> body
    # odom_to_body_publisher = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='odom_to_body_publisher',
    #     arguments=['0', '0', '0', '0', '0', '0', 'lidar_origin', 'body'],
    #     parameters=[{'use_sim_time': True}]
    # )

    # Static transform: body -> os_sensor
    # body_to_lidar_publisher = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',  
    #     name='body_to_lidar_publisher',
    #     arguments=['0', '0', '0', '0', '0', '0', 'body', 'map'],
    #     parameters=[{'use_sim_time': True}]
    # )

    # SLAM Toolbox node
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[{
            'resolution': 0.1,
            'base_frame': 'body',
            'odom_frame': 'lidar_origin',
            'map_frame': 'map',
            'use_sim_time': True,
            'qos_overrides./map.publisher.durability': 'transient_local',
            'qos_overrides./map.publisher.reliability': 'reliable',
            'publish_period_sec': 0.05,  # Increased to reduce processing load
            'publish_frame_transforms': True,
            'map_update_interval': 5.0,
            'qos_overrides./map.publisher.history_depth': 1,
            'map_start_pose': [0.0, 0.0, 0.0],
            'mode': 'mapping',
            'use_pose_graph_backend': True,
            'use_scan_matching': True,
            'queue_size': 2000,  # Further increased queue size
            'transform_publish_period': 0.05,  # Increased transform publish period
            'tf_buffer_duration': 30.0,  # Increased buffer duration
            'scan_queue_size': 2000,  # Increased scan queue size
            'scan_buffer_duration': 20.0,  # Increased scan buffer duration
            'scan_tolerance': 0.2,  # Increased scan tolerance
            'qos_overrides./scan.subscriber.reliability': 'best_effort',
            'qos_overrides./scan.subscriber.history_depth': 2000,
            'qos_overrides./scan.subscriber.durability': 'volatile',
            'minimum_travel_distance': 0.1,  # Reduced minimum travel distance
            'minimum_travel_heading': 0.1,  # Reduced minimum travel heading
            'update_factor': 1.0,  # Reduced update factor
            'transform_timeout': 0.5  # Increased transform timeout
        }]
    )

    # Use official pointcloud_to_laserscan package
    pointcloud_to_laserscan_node = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        output='screen',
        remappings=[
            ('cloud_in', '/os_cloud_node/points'),
            ('scan', '/scan')
        ],
        parameters=[{
            'target_frame': '',  # Use your laser frame
            'transform_tolerance': 0.01,
            'min_height': -0.1,
            'max_height': 0.5,
            'angle_min': -3.14159,  # Full 360 degrees
            'angle_max': 3.14159,
            'angle_increment': 0.0349,  # 2 degrees
            'scan_time': 0.1,
            'range_min': 0.3,
            'range_max': 100.0,
            'use_inf': True,
            'use_sim_time': True
        }]
    )

    # Add a static transform publisher for os_sensor to utm_frame
    os_sensor_to_utm = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='os_sensor_to_utm_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'utm_frame'],
        parameters=[{'use_sim_time': True}]
    )

    # cost_layer_processor_node = Node(
    #     package='map_contruct',
    #     executable='cost_layer_processor',
    #     name='cost_layer_processor',
    #     output='screen',
    #     parameters=[{'use_sim_time': True}],
    #     arguments=[LaunchConfiguration('json_path')]
    # )
#     body_gps_to_body_publisher = Node(
#     package='tf2_ros',
#     executable='static_transform_publisher',
#     name='body_gps_to_body_publisher',
#     arguments=['0', '0', '0', '0', '0', '0', 'odom', 'utm_frame'],
#     parameters=[{'use_sim_time': True}]
# )

    # (Optional) RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        parameters=[{
            'use_sim_time': True,
            'ogre_glsl': False  # Disable advanced shading
        }]
    )


    return LaunchDescription([
        # odom_to_body_publisher,
        # body_gps_to_body_publisher,gett
        # body_to_lidar_publisher,
        # body_gps_to_body_publisher,
        # json_path_arg,
        slam_toolbox_node,
        rosbag_to_slam_node,
        # cost_layer_processor_node,
        os_sensor_to_utm,  # Add the static transform publisher
        # rviz_node,
        # ... any other nodes ...
    ])