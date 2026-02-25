#!/usr/bin/env python3

"""
DWA + Goal Generator Launch File

Launches DWA planner with Goal Generator for navigation.
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # DWA Planner
        Node(
            package='map_contruct',
            executable='dwa_planner',
            name='dwa_planner_node',
            output='screen'
        ),
        
        # Goal Generator
        Node(
            package='map_contruct',
            executable='goal_generator',
            name='goal_generator',
            output='screen',
            parameters=[{
                'method_type': 'dwa_lidar',
                'lambda_ede': 0.0,  # No semantic risk for DWA
                'goal_generation_rate': 7.0,
                'horizon_distance': 4.0
            }]
        ),
    ])

