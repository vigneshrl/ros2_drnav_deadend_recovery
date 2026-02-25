#!/usr/bin/env python3

"""
Simple script to run individual methods for testing
Usage: python3 run_method.py <method_name>
"""

import sys
import subprocess
import time
import signal
import os

def run_method(method_name):
    """Run a specific evaluation method"""
    
    methods = {
        'multi_camera': {
            'nodes': ['inference', 'cost_layer_processor', 'dram_heatmap_viz'],
            'description': 'Multi-Camera DRaM (Your Method)'
        },
        'single_camera': {
            'nodes': ['single_camera_inference', 'dram_heatmap_viz'],
            'description': 'Single-Camera DRaM (Ablation)'
        },
        'dwa_lidar': {
            'nodes': ['dwa_lidar_controller'],
            'description': 'DWA with LiDAR (Comparison)'
        },
        'mppi_lidar': {
            'nodes': ['mppi_lidar_controller'],
            'description': 'MPPI with LiDAR (Comparison)'
        }
    }
    
    if method_name not in methods:
        print(f"❌ Unknown method: {method_name}")
        print(f"Available methods: {', '.join(methods.keys())}")
        return
    
    method = methods[method_name]
    print(f"🚀 Starting: {method['description']}")
    print(f"📋 Nodes: {', '.join(method['nodes'])}")
    
    processes = []
    
    try:
        # Start all nodes
        for node_name in method['nodes']:
            print(f"▶️  Starting {node_name}...")
            cmd = ['ros2', 'run', 'map_contruct', node_name]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            processes.append(process)
            time.sleep(3)  # Stagger startup
        
        print(f"✅ All nodes started successfully!")
        print(f"🔄 Running... (Press Ctrl+C to stop)")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
            # Check if any process died
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    print(f"⚠️  Node {method['nodes'][i]} exited unexpectedly")
    
    except KeyboardInterrupt:
        print(f"\n🛑 Stopping {method['description']}...")
    
    finally:
        # Stop all processes
        for i, process in enumerate(processes):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
                print(f"🔸 Stopped {method['nodes'][i]}")
            except:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass
        
        print("✅ All nodes stopped")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 run_method.py <method_name>")
        print("Methods: multi_camera, single_camera, dwa_lidar, mppi_lidar")
        sys.exit(1)
    
    method_name = sys.argv[1]
    run_method(method_name)

if __name__ == '__main__':
    main()

