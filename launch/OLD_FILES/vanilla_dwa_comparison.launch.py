#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'rosbag_path',
            default_value='/path/to/your/rosbag',
            description='Path to the rosbag file'
        ),
        
        # DRAM-Aware Goal Generator (baseline mode - laser only)
        Node(
            package='map_contruct',
            executable='dram_aware_goal_generator.py',
            name='dram_aware_goal_generator',
            output='screen',
            parameters=[{
                'method_type': 'vanilla_dwa',
                'goal_distance': 3.0,
                'goal_generation_rate': 5.0
            }]
        ),
        
        # Vanilla DWA Planner
        Node(
            package='map_contruct',
            executable='vanilla_dwa_planner.py',
            name='vanilla_dwa_planner',
            output='screen'
        ),
        
        # RViz for visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', '/home/mrvik/dram_ws/src/map_contruct/rviz/comparison_config.rviz'],
            output='screen'
        )
    ])
