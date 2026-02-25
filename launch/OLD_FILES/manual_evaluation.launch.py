#!/usr/bin/env python3

"""
Manual Evaluation Launch File

This launch file helps you run evaluation with manual localization and goal setting.
Choose between:
1. Global map + AMCL localization (use 2D Pose Estimate)
2. SLAM mode (real-time mapping, no localization needed)
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, GroupAction, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'mode',
            default_value='global_map',
            description='Mode: global_map or slam'
        ),
        DeclareLaunchArgument(
            'method',
            default_value='multi_camera_dram', 
            description='Method to evaluate: multi_camera_dram, single_camera_dram, dwa_lidar, mppi_lidar'
        ),
        DeclareLaunchArgument(
            'map_file',
            default_value='/path/to/your/map.yaml',
            description='Path to global map YAML file (for global_map mode)'
        ),
        DeclareLaunchArgument(
            'use_goal_generator',
            default_value='true',
            description='Use autonomous goal generation instead of manual goal setting'
        ),
        
        # Global map mode: Map server + AMCL
        GroupAction(
            condition=IfCondition("'global_map' in LaunchConfiguration('mode')"),
            actions=[
                Node(
                    package='nav2_map_server',
                    executable='map_server',
                    name='map_server',
                    parameters=[{'yaml_filename': LaunchConfiguration('map_file')}],
                    output='screen'
                ),
                Node(
                    package='nav2_amcl',
                    executable='amcl',
                    name='amcl',
                    output='screen'
                ),
            ]
        ),
        
        # SLAM mode: Real-time mapping
        GroupAction(
            condition=IfCondition("'slam' in LaunchConfiguration('mode')"),
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'launch', 'slam_toolbox', 'online_async_launch.py'],
                    output='screen'
                ),
            ]
        ),
        
        # Multi-camera DRaM method
        GroupAction(
            condition=IfCondition("'multi_camera_dram' in LaunchConfiguration('method')"),
            actions=[
                Node(
                    package='map_contruct',
                    executable = 'odom_tf_broadcaster',
                    name='odom_tf_broadcaster_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable = 'pointcloud_segmenter',
                    name='pointcloud_segmenter_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable='infer_vis',
                    name='infer_vis_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable='cost_layer_processor',
                    name='cost_layer_processor_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable='dram_heatmap_viz',
                    name='dram_heatmap_viz_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable='dwa_planner',
                    name='dwa_planner_node',
                    output='screen'
                ),
            ]
        ),
        
        # Single-camera DRaM method
        GroupAction(
            condition=IfCondition("'single_camera_dram' in LaunchConfiguration('method')"),
            actions=[
                Node(
                    package='map_contruct',
                    executable = 'odom_tf_broadcaster',
                    name='odom_tf_broadcaster_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable = 'pointcloud_segmenter',
                    name='pointcloud_segmenter_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable='single_camera_inference',
                    name='single_camera_inference_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable='cost_layer_processor',
                    name='cost_layer_processor_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable='dram_heatmap_viz',
                    name='dram_heatmap_viz_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable='dwa_planner',
                    name='dwa_planner_node',
                    output='screen'
                ),
            ]
        ),
        
        # DWA LiDAR baseline
        GroupAction(
            condition=IfCondition("'dwa_lidar' in LaunchConfiguration('method')"),
            actions=[
                                Node(
                    package='map_contruct',
                    executable = 'odom_tf_broadcaster',
                    name='odom_tf_broadcaster_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable = 'pointcloud_segmenter',
                    name='pointcloud_segmenter_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable='dwa_lidar_controller',
                    name='dwa_lidar_controller_node',
                    output='screen'
                ),
            ]
        ),
        
        # MPPI LiDAR baseline
        GroupAction(
            condition=IfCondition("'mppi_lidar' in LaunchConfiguration('method')"),
            actions=[
                Node(
                    package='map_contruct',
                    executable = 'odom_tf_broadcaster',
                    name='odom_tf_broadcaster_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable = 'pointcloud_segmenter',
                    name='pointcloud_segmenter_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable='mppi_lidar_controller',
                    name='mppi_lidar_controller_node',
                    output='screen'
                ),
            ]
        ),
        
        # Goal Generator for autonomous exploration
        GroupAction(
            condition=IfCondition("'true' in LaunchConfiguration('use_goal_generator')"),
            actions=[
                Node(
                    package='map_contruct',
                    executable='goal_generator',
                    name='goal_generator_node',
                    output='screen',
                    parameters=[{
                        'method_type': LaunchConfiguration('method'),
                        'lambda_ede': 1.0,  # Will be overridden based on method
                        'goal_generation_rate': 7.0,  # 5-10 Hz
                        'horizon_distance': 4.0  # 3-5m
                    }]
                ),
            ]
        ),
        
        # Always start RViz for visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        ),
    ])
