#!/usr/bin/env python3

"""
MPPI + Goal Generator Launch File

Launches MPPI planner with Goal Generator for navigation.
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # MPPI Planner
        Node(
            package='map_contruct',
            executable='mppi_planner',
            name='mppi_planner_node',
            output='screen'
        ),
        
        # Goal Generator
        Node(
            package='map_contruct',
            executable='goal_generator',
            name='goal_generator',
            output='screen',
            parameters=[{
                'method_type': 'mppi_lidar',
                'lambda_ede': 0.0,  # No semantic risk for MPPI
                'goal_generation_rate': 7.0,
                'horizon_distance': 4.0
            }]
        ),
    ])

