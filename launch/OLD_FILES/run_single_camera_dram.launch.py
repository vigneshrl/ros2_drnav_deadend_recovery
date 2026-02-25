#!/usr/bin/env python3

"""
Complete Single-Camera DRaM Method Launch
- Inference with single camera (front only)
- Cost layer processor 
- DRaM heatmap visualization
- DWA planner
- Goal generator (λ=1.0)
- Metrics collection
"""

from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        # Single-camera inference (ROBOT MODE - optimized for real robot)
        Node(
            package='map_contruct',
            executable='single_camera_inference',
            name='single_camera_inference',
            output='screen',
            parameters=[{
                'robot_mode': True,           # Enable robot optimizations
                'save_visualizations': False  # Disable heavy visualization
            }],
            env={
                'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
                'VIRTUAL_ENV': os.environ.get('VIRTUAL_ENV', ''),
                'PATH': os.environ.get('PATH', '')
            }
        ),
        
        # Cost layer processor (subscribes to /single_camera topics)
        Node(
            package='map_contruct',
            executable='cost_layer_processor',
            name='cost_layer_processor',
            output='screen',
            remappings=[
                ('/dead_end_detection/path_status', '/single_camera/path_status'),
                ('/dead_end_detection/is_dead_end', '/single_camera/is_dead_end'),
                ('/dead_end_detection/recovery_points', '/single_camera/recovery_points')
            ]
        ),
        
        # DRaM heatmap visualization (subscribes to /single_camera topics)
        Node(
            package='map_contruct',
            executable='dram_heatmap_viz',
            name='dram_heatmap_viz',
            output='screen',
            remappings=[
                ('/dead_end_detection/path_status', '/single_camera/path_status'),
                ('/dead_end_detection/recovery_points', '/single_camera/recovery_points')
            ]
        ),
        
        # DWA planner (subscribes to /single_camera topics)
        Node(
            package='map_contruct',
            executable='dwa_planner',
            name='dwa_planner',
            output='screen',
            remappings=[
                ('/dead_end_detection/recovery_points', '/single_camera/recovery_points'),
                ('/dead_end_detection/is_dead_end', '/single_camera/is_dead_end')
            ]
        ),
        
        # Goal generator (λ=1.0 for DRaM)
        Node(
            package='map_contruct',
            executable='goal_generator',
            name='goal_generator',
            output='screen',
            parameters=[{
                'method_type': 'single_camera_dram',
                'lambda_ede': 1.0,
                'goal_generation_rate': 7.0,
                'horizon_distance': 4.0
            }]
        ),
    ])
