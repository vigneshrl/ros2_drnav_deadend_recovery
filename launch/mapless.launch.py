#!/usr/bin/env python3
"""
Mapless Navigation Launch File  —  v1.0.0
==========================================
Clean architecture: each method is self-contained.
Goal is given via RViz2 2D Nav Goal button (publishes to /move_base_simple/goal).
SLAM must be running separately to provide /map and the map→odom TF.

Usage
-----
  ros2 launch map_contruct mapless.launch.py method:=dwa
  ros2 launch map_contruct mapless.launch.py method:=mppi
  ros2 launch map_contruct mapless.launch.py method:=nav2_dwb
  ros2 launch map_contruct mapless.launch.py method:=dram model_path:=/path/to/model_best.pth

  # With bag recording:
  ros2 launch map_contruct mapless.launch.py method:=dram \\
      model_path:=/path/to/model_best.pth record:=true run_id:=1

Methods
-------
  dwa       — DWA local planner  (subscribes to /move_base_simple/goal + /map)
  mppi      — MPPI local planner (subscribes to /move_base_simple/goal + /map)
  nav2_dwb  — Nav2 DWB planner   (subscribes to /move_base_simple/goal + /map)
  dram      — DR.Nav: infer_vis + dram_risk_map + direct_vel_controller
                      (goal from RViz2, model drives velocity directly)

All methods share:
  - odom_tf_broadcaster : publishes odom→base_link TF from /odom_lidar

NOTE: SLAM (e.g. slam_toolbox) must be launched separately.
NOTE: Give the robot a goal in RViz2 using the 2D Nav Goal button.
"""

import datetime
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    method     = LaunchConfiguration('method').perform(context)
    use_rviz   = LaunchConfiguration('use_rviz').perform(context).lower() == 'true'
    record     = LaunchConfiguration('record').perform(context).lower() == 'true'
    run_id     = LaunchConfiguration('run_id').perform(context)
    model_path = LaunchConfiguration('model_path').perform(context)

    valid = ('dwa', 'mppi', 'nav2_dwb', 'dram')
    if method not in valid:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(valid)}")

    nodes = []

    # ── Always: odom → base_link TF ─────────────────────────────────────────
    nodes.append(Node(
        package='map_contruct',
        executable='odom_tf_broadcaster',
        name='odom_tf_broadcaster',
        output='screen',
    ))

    # ── Optional: bag recording ──────────────────────────────────────────────
    if record:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        bag_name  = f"run_{run_id}_{timestamp}" if run_id else f"run_{timestamp}"
        bag_dir   = os.path.join('bags', method, bag_name)
        os.makedirs(os.path.join('bags', method), exist_ok=True)

        topics = [
            '/odom',
            '/odom_lidar',
            '/move_base_simple/goal',
            '/dead_end_detection/is_dead_end',
            '/dead_end_detection/path_status',
            '/cmd_vel',
            '/tf',
        ]
        nodes.append(ExecuteProcess(
            cmd=['ros2', 'bag', 'record'] + topics + ['-o', bag_dir],
            output='screen',
            name='bag_recorder',
        ))
        print(f'\n[bag_recorder] Recording to: {bag_dir}')
        print(f'[bag_recorder] Topics: {" ".join(topics)}\n')

    # ── Method: baselines ────────────────────────────────────────────────────
    # Each baseline subscribes to /move_base_simple/goal (set via RViz2)
    # and /map (from SLAM). No goal_generator needed.

    if method == 'dwa':
        nodes.append(Node(
            package='map_contruct',
            executable='dwa_planner',
            name='dwa_planner',
            output='screen',
        ))

    elif method == 'mppi':
        nodes.append(Node(
            package='map_contruct',
            executable='mppi_planner',
            name='mppi_planner',
            output='screen',
        ))

    elif method == 'nav2_dwb':
        nodes.append(Node(
            package='map_contruct',
            executable='nav2_dwb_planner',
            name='nav2_dwb_planner',
            output='screen',
        ))

    # ── Method: DR.Nav ───────────────────────────────────────────────────────
    # Perception stack: pointcloud_segmenter → infer_vis → dram_risk_map
    # Control:          direct_vel_controller (F/L/R scores → cmd_vel directly)
    # Goal:             from RViz2 /move_base_simple/goal

    elif method == 'dram':
        nodes.append(Node(
            package='map_contruct',
            executable='pointcloud_segmenter',
            name='pointcloud_segmenter',
            output='screen',
        ))
        nodes.append(Node(
            package='map_contruct',
            executable='infer_vis',
            name='infer_vis',
            output='screen',
            parameters=[{
                'robot_mode':          True,
                'save_visualizations': False,
                'model_path':          model_path,
            }],
        ))
        nodes.append(Node(
            package='map_contruct',
            executable='dram_risk_map',
            name='dram_risk_map',
            output='screen',
        ))
        nodes.append(Node(
            package='map_contruct',
            executable='direct_vel_controller',
            name='direct_vel_controller',
            output='screen',
        ))

    # ── Optional: RViz ──────────────────────────────────────────────────────
    if use_rviz:
        nodes.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
        ))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'method',
            default_value='dwa',
            description='Navigation method: dwa | mppi | nav2_dwb | dram',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='Launch RViz for visualization: true | false',
        ),
        DeclareLaunchArgument(
            'record',
            default_value='false',
            description='Auto-record a bag: true | false',
        ),
        DeclareLaunchArgument(
            'run_id',
            default_value='',
            description='Run label appended to bag name (e.g. 1, 2, 3)',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value='',
            description='Absolute path to model weights .pth file (required for dram)',
        ),
        OpaqueFunction(function=launch_setup),
    ])
