#!/usr/bin/env python3

"""
Setup script for Enhanced Evaluation Framework

This script helps you configure:
1. Global map path
2. Goal points in your environment  
3. Evaluation parameters
"""

import os
import yaml
import sys

def setup_evaluation():
    """Interactive setup for evaluation framework"""
    print("🧪 Enhanced Evaluation Framework Setup")
    print("=" * 50)
    
    config_file = "/home/mrvik/dram_ws/src/map_contruct/config/evaluation_config.yaml"
    
    # Load existing config
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    print("\n📋 STEP 1: Global Map Configuration")
    print("-" * 30)
    
    # Global map setup
    use_global = input("Do you have a pre-built global map? (y/n): ").lower().startswith('y')
    config['use_global_map'] = use_global
    
    if use_global:
        default_path = config.get('global_map_file', '/path/to/your/map.yaml')
        map_path = input(f"Enter path to your map YAML file [{default_path}]: ").strip()
        if map_path:
            config['global_map_file'] = map_path
        else:
            config['global_map_file'] = default_path
            
        # Verify map exists
        if not os.path.exists(config['global_map_file']):
            print(f"⚠️  Warning: Map file not found at {config['global_map_file']}")
            print("   Please make sure the path is correct before running evaluation")
    else:
        print("📍 You'll need to run SLAM during evaluation")
        config['global_map_file'] = ""
    
    print("\n🎯 STEP 2: Goal Points Configuration")
    print("-" * 30)
    
    setup_goals = input("Do you want to configure goal points? (y/n): ").lower().startswith('y')
    
    if setup_goals:
        goals = []
        goal_count = 1
        
        while True:
            print(f"\nGoal {goal_count}:")
            name = input(f"  Goal name [Goal_{chr(64+goal_count)}]: ").strip()
            if not name:
                name = f"Goal_{chr(64+goal_count)}"
            
            try:
                x = float(input("  X coordinate: "))
                y = float(input("  Y coordinate: "))
                description = input("  Description (optional): ").strip()
                
                goal = {
                    'name': name,
                    'x': x,
                    'y': y,
                    'description': description if description else f"Target location {goal_count}"
                }
                goals.append(goal)
                goal_count += 1
                
                if not input("Add another goal? (y/n): ").lower().startswith('y'):
                    break
                    
            except ValueError:
                print("❌ Invalid coordinates. Please enter numbers.")
                continue
        
        config['goal_points'] = goals
        print(f"✅ Configured {len(goals)} goal points")
    
    print("\n⚙️  STEP 3: Evaluation Parameters")
    print("-" * 30)
    
    # Trial parameters
    current_duration = config.get('trial_duration', 300)
    duration_input = input(f"Trial duration in seconds [{current_duration}]: ").strip()
    if duration_input:
        try:
            config['trial_duration'] = int(duration_input)
        except ValueError:
            pass
    
    current_trials = config.get('num_trials', 10)
    trials_input = input(f"Number of trials per method [{current_trials}]: ").strip()
    if trials_input:
        try:
            config['num_trials'] = int(trials_input)
        except ValueError:
            pass
    
    # Success criteria
    current_tolerance = config.get('goal_tolerance', 2.0)
    tolerance_input = input(f"Goal tolerance in meters [{current_tolerance}]: ").strip()
    if tolerance_input:
        try:
            config['goal_tolerance'] = float(tolerance_input)
        except ValueError:
            pass
    
    print("\n🔬 STEP 4: Methods to Evaluate")
    print("-" * 30)
    
    methods = config.get('methods', {})
    method_names = [
        ('multi_camera_dram', 'Multi-Camera DRaM (Your Method)'),
        ('single_camera_dram', 'Single-Camera DRaM (Ablation)'),
        ('dwa_lidar', 'DWA with LiDAR (Baseline)'),
        ('mppi_lidar', 'MPPI with LiDAR (Baseline)')
    ]
    
    for method_id, method_name in method_names:
        if method_id not in methods:
            methods[method_id] = {'enabled': True}
        
        current_enabled = methods[method_id].get('enabled', True)
        enable_input = input(f"Enable {method_name}? (y/n) [{'y' if current_enabled else 'n'}]: ").strip()
        
        if enable_input:
            methods[method_id]['enabled'] = enable_input.lower().startswith('y')
        else:
            methods[method_id]['enabled'] = current_enabled
    
    config['methods'] = methods
    
    # Save configuration
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("\n✅ CONFIGURATION COMPLETE")
    print("=" * 50)
    print(f"📄 Config saved to: {config_file}")
    
    # Summary
    print("\n📋 SUMMARY:")
    print(f"  Global Map: {'Yes' if config['use_global_map'] else 'No (SLAM mode)'}")
    if config['use_global_map']:
        print(f"  Map File: {config['global_map_file']}")
    
    if 'goal_points' in config:
        print(f"  Goal Points: {len(config['goal_points'])}")
        for goal in config['goal_points']:
            print(f"    - {goal['name']}: ({goal['x']:.1f}, {goal['y']:.1f})")
    
    print(f"  Trial Duration: {config['trial_duration']} seconds")
    print(f"  Trials per Method: {config['num_trials']}")
    
    enabled_methods = [k for k, v in config['methods'].items() if v.get('enabled', True)]
    print(f"  Enabled Methods: {len(enabled_methods)}")
    for method in enabled_methods:
        print(f"    - {method}")
    
    total_time = config['trial_duration'] * config['num_trials'] * len(enabled_methods)
    print(f"  Estimated Total Time: {total_time//60} minutes")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Make sure your global map file exists (if using global map)")
    print("2. Start your robot/simulation")
    print("3. Run: ros2 run map_contruct enhanced_evaluation_framework")
    
    return config

if __name__ == '__main__':
    try:
        setup_evaluation()
    except KeyboardInterrupt:
        print("\n🛑 Setup cancelled")
        sys.exit(1)
