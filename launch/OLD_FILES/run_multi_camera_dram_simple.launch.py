#!/usr/bin/env python3

"""
Simplified Multi-Camera DRaM Launch File
Uses wrapper scripts to handle environment setup
"""

from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        # Multi-camera inference using wrapper script
        Node(
            package='map_contruct',
            executable='run_infer_vis.sh',
            name='multi_camera_inference',
            output='screen',
            cwd='/home/mrvik/dram_ws'
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

