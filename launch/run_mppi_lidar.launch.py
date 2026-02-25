#!/usr/bin/env python3

"""
Complete MPPI LiDAR Method Launch
- MPPI LiDAR controller (with metrics)
- Goal generator (λ=0.0)
- Pure LiDAR-based navigation
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # MPPI LiDAR controller (includes dead-end detection and metrics)
        Node(
            package='map_contruct',
            executable='mppi_lidar_controller',
            name='mppi_lidar_controller',
            output='screen'
        ),
        
        # Goal generator (λ=0.0 for LiDAR - no semantic risk)
        Node(
            package='map_contruct',
            executable='goal_generator',
            name='goal_generator',
            output='screen',
            parameters=[{
                'method_type': 'mppi_lidar',
                'lambda_ede': 0.0,  # No semantic risk
                'goal_generation_rate': 7.0,
                'horizon_distance': 4.0
            }],
            remappings=[
                ('/dead_end_detection/recovery_points', '/mppi_lidar/recovery_points'),
                ('/dead_end_detection/is_dead_end', '/mppi_lidar/is_dead_end')
            ]
        ),
    ])
