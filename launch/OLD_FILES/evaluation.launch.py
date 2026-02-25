#!/usr/bin/env python3

"""
Launch file for running evaluation of different dead-end detection methods
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'method',
            default_value='multi_camera_dram',
            description='Method to evaluate: multi_camera_dram, single_camera_dram, dwa_lidar, mppi_lidar, or evaluation_framework'
        ),
        
        # Multi-camera DRaM method (your method)
        GroupAction(
            condition=IfCondition("'multi_camera_dram' in LaunchConfiguration('method')"),
            actions=[
                Node(
                    package='map_contruct',
                    executable='inference',
                    name='inference_node',
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
            ]
        ),
        
        # Single-camera DRaM method (ablation)
        GroupAction(
            condition=IfCondition("'single_camera_dram' in LaunchConfiguration('method')"),
            actions=[
                Node(
                    package='map_contruct',
                    executable='single_camera_inference',
                    name='single_camera_inference_node',
                    output='screen'
                ),
                Node(
                    package='map_contruct',
                    executable='dram_heatmap_viz',
                    name='dram_heatmap_viz_node',
                    output='screen'
                ),
            ]
        ),
        
        # DWA with LiDAR method (comparison)
        GroupAction(
            condition=IfCondition("'dwa_lidar' in LaunchConfiguration('method')"),
            actions=[
                Node(
                    package='map_contruct',
                    executable='dwa_lidar_controller',
                    name='dwa_lidar_controller_node',
                    output='screen'
                ),
            ]
        ),
        
        # MPPI with LiDAR method (comparison)
        GroupAction(
            condition=IfCondition("'mppi_lidar' in LaunchConfiguration('method')"),
            actions=[
                Node(
                    package='map_contruct',
                    executable='mppi_lidar_controller',
                    name='mppi_lidar_controller_node',
                    output='screen'
                ),
            ]
        ),
        
        # Evaluation framework (runs all methods)
        GroupAction(
            condition=IfCondition("'evaluation_framework' in LaunchConfiguration('method')"),
            actions=[
                Node(
                    package='map_contruct',
                    executable='evaluation_framework',
                    name='evaluation_framework_node',
                    output='screen'
                ),
            ]
        ),
    ])
