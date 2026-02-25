#!/usr/bin/env python3
"""
Map-Based Navigation Launch File
==================================
Run this in Terminal 2 AFTER nav2_bringup is up in Terminal 1.

Nav2 handles the full planning pipeline for all methods:
  nav2_bringup → map_server + amcl + planner_server + controller_server + bt_navigator

For BASELINE methods (dwa, mppi, nav2_dwb):
  - No second launch file needed.
  - Just pass a different nav2_params.yaml to nav2_bringup that configures
    the appropriate controller plugin:
      DWB  → nav2_params_dwb.yaml   (default in most nav2_bringup setups)
      DWA  → nav2_params_dwa.yaml   (set controller plugin to DWA)
      MPPI → nav2_params_mppi.yaml  (set controller plugin to nav2_mppi_controller)
  - Send goals via RViz "Nav2 Goal" button — Nav2 handles everything.

For DRAM (your method):
  - Run nav2_bringup in Terminal 1 (navigation, localization, map).
  - Run this file in Terminal 2: adds the perception stack on top.
  - Send goals via RViz "Nav2 Goal" button — Nav2 drives, DRAM monitors risk.
  - dram_risk_map publishes /dram_exploration_map (EDE heatmap visible in RViz).

Usage:
  # Terminal 1
  ros2 launch nav2_bringup <your_bringup>.launch.py

  # Terminal 2 (DRAM only — baselines don't need this)
  ros2 launch map_contruct map_based.launch.py method:=dram
  ros2 launch map_contruct map_based.launch.py method:=dram use_rviz:=true

Methods:
  dram  — infer_vis + dram_risk_map (perception only, Nav2 drives)
  dwa / mppi / nav2_dwb — handled entirely by nav2_bringup, no launch needed here
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    method   = LaunchConfiguration('method').perform(context)
    use_rviz = LaunchConfiguration('use_rviz').perform(context).lower() == 'true'

    valid_methods = ['dwa', 'mppi', 'nav2_dwb', 'dram']
    if method not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {valid_methods}"
        )

    nodes = []

    if method in ('dwa', 'mppi', 'nav2_dwb'):
        # Baselines are fully handled by nav2_bringup.
        # This launch file is a no-op for them — just print a reminder.
        print(
            f'\n[map_based] method={method}: '
            f'No additional nodes needed. '
            f'Ensure nav2_bringup is running with the correct nav2_params.yaml '
            f'for the {method} controller plugin.\n'
        )
        # Nothing to launch — return empty list.

    elif method == 'dram':
        # DRAM perception stack on top of nav2_bringup.
        # Nav2 drives the robot; our nodes provide the EDE risk heatmap.

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
            }]
        ))
        nodes.append(Node(
            package='map_contruct',
            executable='dram_risk_map',
            name='dram_risk_map',
            output='screen',
        ))

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
            default_value='dram',
            description=(
                'Navigation method: dram | dwa | mppi | nav2_dwb  '
                '(dwa/mppi/nav2_dwb are no-ops — handled by nav2_bringup)'
            )
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='Launch RViz: true | false'
        ),
        OpaqueFunction(function=launch_setup),
    ])
