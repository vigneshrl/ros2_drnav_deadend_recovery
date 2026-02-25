#!/usr/bin/env python3

"""
Simple script to run the unified goal generator for different methods

Usage:
python3 run_goal_generator.py multi_camera_dram
python3 run_goal_generator.py dwa_lidar
"""

import sys
import subprocess

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 run_goal_generator.py <method>")
        print("Methods: multi_camera_dram, single_camera_dram, dwa_lidar, mppi_lidar")
        sys.exit(1)
    
    method = sys.argv[1]
    
    # Validate method
    valid_methods = ['multi_camera_dram', 'single_camera_dram', 'dwa_lidar', 'mppi_lidar']
    if method not in valid_methods:
        print(f"Error: Invalid method '{method}'")
        print(f"Valid methods: {', '.join(valid_methods)}")
        sys.exit(1)
    
    print(f"🎯 Launching unified goal generator for: {method}")
    
    # Run the launch file
    cmd = [
        'ros2', 'launch', 'map_contruct', 'goal_generator.launch.py',
        f'method:={method}'
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Goal generator stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running goal generator: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

