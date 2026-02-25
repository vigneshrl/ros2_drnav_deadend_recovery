#!/usr/bin/env python3

"""
Multi-Camera DRaM Method Launch for ROSBAG Testing
- Full visualization and logging enabled
- Higher processing rates
- File saving enabled
"""

from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        # Multi-camera inference (ROSBAG MODE - full visualization)
        Node(
            package='map_contruct',
            executable='infer_vis',
            name='multi_camera_inference',
            output='screen',
            parameters=[{
                'robot_mode': False,          # Disable robot optimizations
                'save_visualizations': True   # Enable full visualization
            }],
            env={
                'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
                'VIRTUAL_ENV': os.environ.get('VIRTUAL_ENV', ''),
                'PATH': os.environ.get('PATH', '')
            }
        ),
        
        # Cost layer processor
        Node(
            package='map_contruct',
            executable='cost_layer_processor',
            name='cost_layer_processor',
            output='screen'
        ),
        
        # DRaM heatmap visualization
        Node(
            package='map_contruct',
            executable='dram_heatmap_viz',
            name='dram_heatmap_viz',
            output='screen'
        ),
        
        # DWA planner
        Node(
            package='map_contruct',
            executable='dwa_planner',
            name='dwa_planner',
            output='screen'
        ),
        
        # Goal generator (λ=1.0 for DRaM)
        Node(
            package='map_contruct',
            executable='goal_generator',
            name='goal_generator',
            output='screen',
            parameters=[{
                'method_type': 'multi_camera_dram',
                'lambda_ede': 1.0,
                'goal_generation_rate': 7.0,
                'horizon_distance': 4.0
            }]
        ),
    ])

