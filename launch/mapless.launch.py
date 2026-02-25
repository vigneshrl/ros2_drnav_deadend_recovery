#!/usr/bin/env python3
"""
Mapless Navigation Launch File
===============================
Runs all methods in a mapless environment (no pre-built map).
SLAM must be running separately to provide /map (e.g. slam_toolbox).

Usage:
  ros2 launch map_contruct mapless.launch.py method:=dwa
  ros2 launch map_contruct mapless.launch.py method:=mppi
  ros2 launch map_contruct mapless.launch.py method:=nav2_dwb
  ros2 launch map_contruct mapless.launch.py method:=dram

Methods:
  dwa       — DWA baseline    (goal_generator λ=0 + dwa_planner)
  mppi      — MPPI baseline   (goal_generator λ=0 + mppi_planner)
  nav2_dwb  — Nav2 DWB baseline (goal_generator λ=0 + nav2_dwb_planner)
  dram      — DR.Nav full method (infer_vis + dram_risk_map + goal_generator λ=1 + dwa_planner)

All methods share:
  - odom_tf_broadcaster  : publishes odom -> body TF from /odom_lidar
  - goal_generator       : selects waypoints using unified scoring

NOTE: SLAM (e.g. slam_toolbox) must be launched separately to provide /map.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# ─── method → goal_generator config ────────────────────────────────────────
METHOD_CONFIG = {
    'dwa':      {'method_type': 'dwa_lidar',          'lambda_ede': 0.0},
    'mppi':     {'method_type': 'mppi_lidar',          'lambda_ede': 0.0},
    'nav2_dwb': {'method_type': 'nav2_dwb_lidar',      'lambda_ede': 0.0},
    'dram':     {'method_type': 'multi_camera_dram',   'lambda_ede': 1.0},
}


def launch_setup(context, *args, **kwargs):
    method   = LaunchConfiguration('method').perform(context)
    use_rviz = LaunchConfiguration('use_rviz').perform(context).lower() == 'true'

    if method not in METHOD_CONFIG:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: {list(METHOD_CONFIG.keys())}"
        )

    cfg = METHOD_CONFIG[method]
    nodes = []

    # ── Always: odom → body TF broadcaster ──────────────────────────────────
    nodes.append(Node(
        package='map_contruct',
        executable='odom_tf_broadcaster',
        name='odom_tf_broadcaster',
        output='screen',
        parameters=[{
            'odom_topic':   '/odom_lidar',
            'parent_frame': 'odom',
            'child_frame':  'body',
        }]
    ))

    # ── Always: goal generator ───────────────────────────────────────────────
    nodes.append(Node(
        package='map_contruct',
        executable='goal_generator',
        name='goal_generator',
        output='screen',
        parameters=[{
            'method_type':           cfg['method_type'],
            'lambda_ede':            cfg['lambda_ede'],
            'goal_generation_rate':  7.0,
            'horizon_distance':      4.0,
        }]
    ))

    # ── Method-specific: perception stack (DRAM only) ────────────────────────
    if method == 'dram':
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
                'robot_mode':          True,   # optimised for real-time
                'save_visualizations': False,  # disable heavy disk writes
            }]
        ))
        nodes.append(Node(
            package='map_contruct',
            executable='dram_risk_map',
            name='dram_risk_map',
            output='screen',
        ))

    # ── Method-specific: local planner ───────────────────────────────────────
    if method in ('dwa', 'dram'):
        # DRAM uses DWA as its local controller
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

    # ── Optional: RViz ───────────────────────────────────────────────────────
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
            description='Navigation method: dwa | mppi | nav2_dwb | dram'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='Launch RViz for visualization: true | false'
        ),
        OpaqueFunction(function=launch_setup),
    ])
