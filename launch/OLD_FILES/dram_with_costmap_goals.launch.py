#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # DRAM-Aware Goal Generator (uses costmap intelligence)
        Node(
            package='map_contruct',
            executable='dram_aware_goal_generator.py',
            name='dram_aware_goal_generator',
            output='screen',
            parameters=[{
                'method_type': 'dram',
                'goal_distance': 3.0,
                'goal_generation_rate': 5.0
            }]
        ),
        
        # Your DRAM Planner
        Node(
            package='map_contruct',
            executable='dwa_rosbag_planner.py',
            name='dwa_rosbag_planner',
            output='screen'
        ),
        
        # Multi-camera dead-end detection
        # Node(
        #     package='map_contruct',
        #     executable='inference.py',
        #     name='front_left_inference',
        #     output='screen',
        #     parameters=[{
        #         'camera_topic': '/argus/ar0234_front_left/image_raw',
        #         'model_path': '/home/mrvik/dram_ws/src/map_contruct/models/model_front_left.pth',
        #         'camera_position': 'front_left'
        #     }]
        # ),
        
        # Node(
        #     package='map_contruct',
        #     executable='inference.py',
        #     name='front_right_inference',
        #     output='screen',
        #     parameters=[{
        #         'camera_topic': '/argus/ar0234_front_right/image_raw',
        #         'model_path': '/home/mrvik/dram_ws/src/map_contruct/models/model_front_right.pth',
        #         'camera_position': 'front_right'
        #     }]
        # ),
        
        # Dead-end prediction and visualization
        # Node(
        #     package='map_contruct',
        #     executable='comprehensive_deadend_visualizer.py',
        #     name='comprehensive_deadend_visualizer',
        #     output='screen'
        # ),
        
        # DRAM heatmap visualization (provides the costmap!)
        Node(
            package='map_contruct',
            executable='dram_heatmap_viz.py',
            name='dram_heatmap_viz',
            output='screen'
        ),
        Node(
            package='map_contruct',
            executable='cost_layer_processor',
            name='cost_layer_processor',
            output='screen'
        ),
    ])
