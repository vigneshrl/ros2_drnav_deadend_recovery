#!/usr/bin/env python3

"""
Complete Multi-Camera DRaM Method Launch
- Inference with multi-camera
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
        # Multi-camera inference (ROBOT MODE - optimized for real robot)
        # Node(
        #     package='map_contruct',
        #     executable='infer_vis',
        #     name='multi_camera_inference',
        #     output='screen',
        #     parameters=[{
        #         'robot_mode': True,           # Enable robot optimizations
        #         'save_visualizations': False  # Disable heavy visualization
        #     }],
        #     env={
        #         'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
        #         'VIRTUAL_ENV': os.environ.get('VIRTUAL_ENV', ''),
        #         'PATH': os.environ.get('PATH', '')
        #     }
        # ),
        
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
            executable='dwa_rosbag_planner',
            name='dwa_rosbag_planner',
            output='screen'
        ),
        
        # Goal generator (λ=1.0 for DRaM)
        # Node(
        #     package='map_contruct',
        #     executable='goal_generator',
        #     name='goal_generator',
        #     output='screen',
        #     parameters=[{
        #         'method_type': 'multi_camera_dram',
        #         'lambda_ede': 1.0,
        #         'goal_generation_rate': 7.0,
        #         'horizon_distance': 4.0
        #     }]
        # ),
    ])
