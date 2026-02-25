#!/usr/bin/env python3

"""
Goal Generator Launch File

Launches the unified goal generator with method-specific parameters:
- DRaM Multi/Single: λ > 0 (semantic risk enabled)
- DWA/MPPI LiDAR: λ = 0 (pure geometric scoring)

Usage:
ros2 launch map_contruct goal_generator.launch.py method:=multi_camera_dram
ros2 launch map_contruct goal_generator.launch.py method:=dwa_lidar
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    """Setup launch based on method type"""
    method = LaunchConfiguration('method').perform(context)
    
    # Configure λ_EDE based on method
    lambda_configs = {
        'multi_camera_dram': 1.0,    # Use semantic risk
        'single_camera_dram': 1.0,   # Use semantic risk  
        'dwa_lidar': 0.0,           # Pure geometric
        'mppi_lidar': 0.0,          # Pure geometric
    }
    
    lambda_ede = lambda_configs.get(method, 0.0)
    
    return [
        Node(
            package='map_contruct',
            executable='goal_generator',
            name='goal_generator_node',
            output='screen',
            parameters=[{
                'method_type': method,
                'lambda_ede': lambda_ede,
                'goal_generation_rate': LaunchConfiguration('rate'),
                'horizon_distance': LaunchConfiguration('horizon')
            }]
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'method',
            default_value='multi_camera_dram',
            description='Method: multi_camera_dram, single_camera_dram, dwa_lidar, mppi_lidar'
        ),
        DeclareLaunchArgument(
            'rate',
            default_value='7.0',
            description='Goal generation rate in Hz (5-10 Hz recommended)'
        ),
        DeclareLaunchArgument(
            'horizon',
            default_value='4.0',
            description='Horizon distance in meters (3-5m recommended)'
        ),
        
        OpaqueFunction(function=launch_setup)
    ])
