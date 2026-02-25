#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """
    Launch file for running DRAM model with DWA planner on ROS2 bag data
    """
    
    # Declare arguments
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value='',
        description='Path to the ROS2 bag file to replay'
    )
    
    bag_rate_arg = DeclareLaunchArgument(
        'bag_rate',
        default_value='1.0',
        description='Playback rate for the bag file (e.g., 0.5 for half speed)'
    )
    
    robot_mode_arg = DeclareLaunchArgument(
        'robot_mode',
        default_value='false',
        description='Set to true for robot mode, false for rosbag mode'
    )
    
    save_visualizations_arg = DeclareLaunchArgument(
        'save_visualizations',
        default_value='false',
        description='Set to true to save visualization images'
    )

    # Get the package share directory
    pkg_share = get_package_share_directory('map_contruct')

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

    # Dead End Detection Node with Visualization (your model)
    # dead_end_detection_node = Node(
    #     package='map_contruct',
    #     executable='infer_vis',  # Your inference node
    #     name='dead_end_detection_visual_node',
    #     output='screen',
    #     parameters=[{
    #         'robot_mode': LaunchConfiguration('robot_mode'),
    #         'save_visualizations': LaunchConfiguration('save_visualizations'),
    #         'use_sim_time': True
    #     }],
    #     arguments=['--ros-args', '--log-level', 'info']
    # )

    # DWA Planner Node with Action Logging
    dwa_planner_node = Node(
        package='map_contruct',
        executable='dwa_rosbag_planner',  # Your new DWA planner
        name='dwa_rosbag_planner_node',
        output='screen',
        parameters=[{
            'use_sim_time': True
        }],
        arguments=['--ros-args', '--log-level', 'info']
    )

    # SLAM Toolbox node for mapping (optional but useful for localization)
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[{
            'resolution': 0.1,
            'base_frame': 'base_link',
            'odom_frame': 'odom',
            'map_frame': 'map',
            'use_sim_time': True,
            'qos_overrides./map.publisher.durability': 'transient_local',
            'qos_overrides./map.publisher.reliability': 'reliable',
            'publish_period_sec': 0.1,
            'publish_frame_transforms': True,
            'map_update_interval': 2.0,
            'queue_size': 1000,
            'transform_publish_period': 0.05,
            'tf_buffer_duration': 30.0,
            'scan_queue_size': 1000,
            'scan_buffer_duration': 10.0,
            'scan_tolerance': 0.2,
            'qos_overrides./scan.subscriber.reliability': 'best_effort',
            'minimum_travel_distance': 0.1,
            'minimum_travel_heading': 0.1,
            'update_factor': 1.0,
            'transform_timeout': 0.5,
            'mode': 'localization'  # Use localization mode for rosbag playback
        }]
    )

    # Convert pointcloud to laserscan for SLAM
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
            'target_frame': '',
            'transform_tolerance': 0.01,
            'min_height': -0.5,
            'max_height': 2.0,
            'angle_min': -3.14159,
            'angle_max': 3.14159,
            'angle_increment': 0.0349,  # 2 degrees
            'scan_time': 0.1,
            'range_min': 0.3,
            'range_max': 100.0,
            'use_inf': True,
            'use_sim_time': True
        }]
    )

    # Goal Generator Node (simple waypoint publisher)
    goal_generator_node = Node(
        package='map_contruct',
        executable='simple_goal_generator',
        name='simple_goal_generator',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'goal_distance': 10.0,  # Distance to generate goals ahead
            'goal_update_rate': 0.5  # Hz
        }]
    )

    # RViz for visualization (optional)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{
            'use_sim_time': True
        }]
    )

    # ROS Bag Play Node with delay to allow nodes to initialize
    bag_play_node = TimerAction(
        period=5.0,  # Wait 5 seconds for nodes to initialize
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play', 
                    LaunchConfiguration('bag_path'),
                    '--clock',  # Publish simulation time
                    '-r', LaunchConfiguration('bag_rate'),
                    '--qos-profile-overrides-path', '/dev/null'  # Ignore QoS mismatches
                ],
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        # Arguments
        bag_path_arg,
        bag_rate_arg,
        robot_mode_arg,
        save_visualizations_arg,
        
        # Static transforms
        map_to_odom_publisher,
        odom_to_base_link_publisher,
        
        # Core nodes
        # dead_end_detection_node,
        dwa_planner_node,
        
        # SLAM and perception
        pointcloud_to_laserscan_node,
        slam_toolbox_node,
        
        # Goal generation
        goal_generator_node,
        
        # Visualization
        rviz_node,
        
        # Bag playback (delayed)
        bag_play_node
    ])

