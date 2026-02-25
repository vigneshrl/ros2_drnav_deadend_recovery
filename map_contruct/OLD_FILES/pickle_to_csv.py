#!/usr/bin/env python3

import pickle
import csv
import json
import math
from datetime import datetime
import os

def pickle_to_csv(pickle_path, output_dir=None):
    """Convert pickle file to CSV format"""
    
    if output_dir is None:
        output_dir = os.path.dirname(pickle_path)
    
    print(f"📂 Loading: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'camera_timestamp_to_action' not in data:
        print("❌ No camera-action data found in pickle file")
        return
    
    camera_actions = data['camera_timestamp_to_action']
    
    # Prepare CSV data
    csv_data = []
    for timestamp, action_data in camera_actions.items():
        x, y, theta = action_data['robot_pose']
        
        row = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'v': action_data['v'],
            'omega': action_data['omega'],
            'robot_x': x,
            'robot_y': y,
            'robot_theta': theta,
            'ros_timestamp_sec': action_data['ros_timestamp_sec'],
            'ros_timestamp_nanosec': action_data['ros_timestamp_nanosec']
        }
        csv_data.append(row)
    
    # Sort by timestamp
    csv_data.sort(key=lambda x: x['timestamp'])
    
    # Save as CSV
    csv_file = os.path.join(output_dir, 'camera_actions.csv')
    with open(csv_file, 'w', newline='') as f:
        if csv_data:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
    
    print(f"💾 CSV saved: {csv_file}")
    
    # Save as JSON for easy reading
    json_file = os.path.join(output_dir, 'camera_actions.json')
    with open(json_file, 'w') as f:
        json.dump({
            'metadata': {
                'total_frames': len(csv_data),
                'duration_seconds': csv_data[-1]['timestamp'] - csv_data[0]['timestamp'] if csv_data else 0,
                'description': 'Camera timestamps synchronized with robot actions'
            },
            'data': csv_data
        }, f, indent=2)
    
    print(f"📝 JSON saved: {json_file}")
    
    # Create summary statistics
    if csv_data:
        velocities = [row['v'] for row in csv_data]
        omegas = [row['omega'] for row in csv_data]
        
        stats = {
            'total_frames': len(csv_data),
            'duration_seconds': csv_data[-1]['timestamp'] - csv_data[0]['timestamp'],
            'average_fps': len(csv_data) / (csv_data[-1]['timestamp'] - csv_data[0]['timestamp']),
            'velocity_stats': {
                'min': min(velocities),
                'max': max(velocities),
                'mean': sum(velocities) / len(velocities),
                'std': math.sqrt(sum((v - sum(velocities)/len(velocities))**2 for v in velocities) / len(velocities))
            },
            'omega_stats': {
                'min': min(omegas),
                'max': max(omegas),
                'mean': sum(omegas) / len(omegas),
                'std': math.sqrt(sum((o - sum(omegas)/len(omegas))**2 for o in omegas) / len(omegas))
            }
        }
        
        stats_file = os.path.join(output_dir, 'camera_actions_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Statistics saved: {stats_file}")
        
        # Print summary
        print("\n📊 SUMMARY STATISTICS:")
        print(f"🎬 Total frames: {stats['total_frames']}")
        print(f"⏱️  Duration: {stats['duration_seconds']:.2f} seconds")
        print(f"🎯 Average FPS: {stats['average_fps']:.2f}")
        print(f"🚗 Velocity range: {stats['velocity_stats']['min']:.3f} to {stats['velocity_stats']['max']:.3f} m/s")
        print(f"🔄 Omega range: {stats['omega_stats']['min']:.3f} to {stats['omega_stats']['max']:.3f} rad/s")

def find_and_convert_latest():
    """Find and convert the most recent camera_actions.pkl file"""
    base_dir = "/home/mrvik/dram_ws"
    
    # Find camera_actions.pkl files
    camera_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'camera_actions.pkl':
                full_path = os.path.join(root, file)
                mtime = os.path.getmtime(full_path)
                camera_files.append((full_path, mtime))
    
    if not camera_files:
        print("❌ No camera_actions.pkl files found!")
        return
    
    # Sort by modification time (newest first)
    camera_files.sort(key=lambda x: x[1], reverse=True)
    
    latest_file = camera_files[0][0]
    print(f"🔍 Converting latest file: {latest_file}")
    
    pickle_to_csv(latest_file)

if __name__ == "__main__":
    print("🥒➡️📊 PICKLE TO CSV CONVERTER")
    print("=" * 50)
    
    find_and_convert_latest()
