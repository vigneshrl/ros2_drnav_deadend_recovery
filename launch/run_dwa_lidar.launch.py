#!/usr/bin/env python3

"""
Complete DWA LiDAR Method Launch
- DWA LiDAR controller (with metrics)
- Goal generator (λ=0.0)
- Pure LiDAR-based navigation
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # DWA LiDAR controller (includes dead-end detection and metrics)
        Node(
            package='map_contruct',
            executable='dwa_lidar_controller',
            name='dwa_lidar_controller',
            output='screen'
        ),
        
        # Goal generator (λ=0.0 for LiDAR - no semantic risk)
        Node(
            package='map_contruct',
            executable='goal_generator',
            name='goal_generator',
            output='screen',
            parameters=[{
                'method_type': 'dwa_lidar',
                'lambda_ede': 0.0,  # No semantic risk
                'goal_generation_rate': 7.0,
                'horizon_distance': 4.0
            }],
            remappings=[
                ('/dead_end_detection/recovery_points', '/dwa_lidar/recovery_points'),
                ('/dead_end_detection/is_dead_end', '/dwa_lidar/is_dead_end')
            ]
        ),
    ])
