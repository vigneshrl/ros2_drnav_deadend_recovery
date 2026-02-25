#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """
    Launch file for capturing predictive dead-end scenarios for paper figures
    """
    
    # Declare arguments
    # bag_path_arg = DeclareLaunchArgument(
    #     'bag_path',
    #     default_value='',
    #     description='Path to the ROS2 bag file to replay'
    # )
    
    # bag_rate_arg = DeclareLaunchArgument(
    #     'bag_rate',
    #     default_value='0.5',  # Slower for better capture
    #     description='Playback rate for the bag file'
    # )
    
    robot_mode_arg = DeclareLaunchArgument(
        'robot_mode',
        default_value='false',
        description='Set to true for robot mode, false for rosbag mode'
    )
    
    save_visualizations_arg = DeclareLaunchArgument(
        'save_visualizations',
        default_value='false',
        description='Set to true to save DRAM visualization images'
    )
    
    auto_capture_arg = DeclareLaunchArgument(
        'auto_capture',
        default_value='true',
        description='Automatically capture interesting scenarios'
    )
    
    sequence_mode_arg = DeclareLaunchArgument(
        'sequence_mode',
        default_value='true',
        description='Generate sequence visualizations (3-frame progressions)'
    )

    # Static transforms for coordinate frames
    map_to_odom_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        parameters=[{'use_sim_time': True}]
    )
    
    odom_to_base_link_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='odom_to_base_link_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link'],
        parameters=[{'use_sim_time': True}]
    )

    # Your DRAM model inference (with PyTorch)
    # dead_end_detection_node = Node(
    #     package='map_contruct',
    #     executable='infer_vis',
    #     name='dead_end_detection_visual_node',
    #     output='screen',
    #     parameters=[{
    #         'robot_mode': LaunchConfiguration('robot_mode'),
    #         'save_visualizations': LaunchConfiguration('save_visualizations'),
    #         'use_sim_time': True
    #     }],
    #     arguments=['--ros-args', '--log-level', 'info']
    # )

    # Predictive Dead-End Visualizer (the new node)
    predictive_visualizer_node = Node(
        package='map_contruct',
        executable='predictive_deadend_visualizer',
        name='predictive_deadend_visualizer',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'save_interval': 3.0,  # Capture every 3 seconds
            'auto_capture': LaunchConfiguration('auto_capture'),
            'sequence_mode': LaunchConfiguration('sequence_mode'),
            'output_dir': '/home/mrvik/dram_ws/predictive_figures'
        }],
        arguments=['--ros-args', '--log-level', 'info']
    )

    # SLAM Toolbox for mapping/localization
    # slam_toolbox_node = Node(
    #     package='slam_toolbox',
    #     executable='sync_slam_toolbox_node',
    #     name='slam_toolbox',
    #     output='screen',
    #     parameters=[{
    #         'resolution': 0.05,  # Higher resolution for better figures
    #         'base_frame': 'base_link',
    #         'odom_frame': 'odom',
    #         'map_frame': 'map',
    #         'use_sim_time': True,
    #         'qos_overrides./map.publisher.durability': 'transient_local',
    #         'qos_overrides./map.publisher.reliability': 'reliable',
    #         'publish_period_sec': 0.1,
    #         'publish_frame_transforms': True,
    #         'map_update_interval': 1.0,  # Faster updates for real-time capture
    #         'queue_size': 2000,
    #         'transform_publish_period': 0.05,
    #         'tf_buffer_duration': 30.0,
    #         'scan_queue_size': 2000,
    #         'scan_buffer_duration': 10.0,
    #         'scan_tolerance': 0.2,
    #         'qos_overrides./scan.subscriber.reliability': 'best_effort',
    #         'minimum_travel_distance': 0.05,  # More sensitive
    #         'minimum_travel_heading': 0.05,
    #         'update_factor': 1.0,
    #         'transform_timeout': 0.5,
    #         'mode': 'mapping'  # Use mapping mode for better costmap
    #     }]
    # )

    # Convert pointcloud to laserscan
    # pointcloud_to_laserscan_node = Node(
    #     package='pointcloud_to_laserscan',
    #     executable='pointcloud_to_laserscan_node',
    #     name='pointcloud_to_laserscan',
    #     output='screen',
    #     remappings=[
    #         ('cloud_in', '/os_cloud_node/points'),
    #         ('scan', '/scan')
    #     ],
    #     parameters=[{
    #         'target_frame': '',
    #         'transform_tolerance': 0.01,
    #         'min_height': -0.5,
    #         'max_height': 2.0,
    #         'angle_min': -3.14159,
    #         'angle_max': 3.14159,
    #         'angle_increment': 0.0175,  # 1 degree for higher resolution
    #         'scan_time': 0.1,
    #         'range_min': 0.3,
    #         'range_max': 100.0,
    #         'use_inf': True,
    #         'use_sim_time': True
    #     }]
    # )

    # RViz for real-time monitoring
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{
            'use_sim_time': True
        }]
    )

    # # ROS Bag Play Node with delay
    # bag_play_node = TimerAction(
    #     period=5.0,  # Wait for nodes to initialize
    #     actions=[
    #         ExecuteProcess(
    #             cmd=[
    #                 'ros2', 'bag', 'play', 
    #                 LaunchConfiguration('bag_path'),
    #                 '--clock',
    #                 '-r', LaunchConfiguration('bag_rate'),
    #                 '--qos-profile-overrides-path', '/dev/null'
    #             ],
    #             output='screen'
    #         )
    #     ]
    # )

    return LaunchDescription([
        # Arguments
        # bag_path_arg,
        # bag_rate_arg,
        robot_mode_arg,
        save_visualizations_arg,
        auto_capture_arg,
        sequence_mode_arg,
        
        # Static transforms
        map_to_odom_publisher,
        odom_to_base_link_publisher,
        
        # Core nodes
        dead_end_detection_node,
        predictive_visualizer_node,
        
        # SLAM and perception
        pointcloud_to_laserscan_node,
        slam_toolbox_node,
        
        # Visualization
        rviz_node,
        
        # Bag playback (delayed)
        bag_play_node
    ])
