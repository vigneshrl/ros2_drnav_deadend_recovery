#!/usr/bin/env python3

"""
Inference + DWA Launch File

Launches inference node with DWA planner for DRaM-based navigation.
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Inference Node (Dead End Detection)
        Node(
            package='map_contruct',
            executable='infer_vis',
            name='dead_end_detection_visual_node',
            output='screen',
            parameters=[{
                'robot_mode': True,
                'save_visualizations': True
            }]
        ),
        
        # DWA Planner
        Node(
            package='map_contruct',
            executable='dwa_planner',
            name='dwa_planner_node',
            output='screen'
        ),
    ])

