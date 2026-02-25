#!/usr/bin/env python3

"""
Simple Method Comparison Tool

Quick and easy comparison of your DRAM method vs baseline planners.
Shows key metrics that matter most for demonstrating superiority.
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import math

def find_latest_results():
    """Find the most recent result directories for each method"""
    base_dir = "/home/mrvik/dram_ws/rosbag_iribe_indorr"
    
    # Define all possible method patterns
    method_patterns = {
        'DRAM': 'dram_rosbag_results_*',
        'Vanilla_DWA': 'vanilla_dwa_results_*',
        'MPPI': 'mppi_results_*',
        'Nav2_DWB': 'nav2_dwb_results_*'
    }
    
    found_dirs = {}
    
    print("🔍 Searching for result directories...")
    
    # Only include methods that actually have results
    for method, pattern in method_patterns.items():
        matching_dirs = list(Path(base_dir).glob(pattern))
        if matching_dirs:
            latest_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
            found_dirs[method] = str(latest_dir)
            print(f"✅ Found {method}: {latest_dir.name}")
    
    if not found_dirs:
        print("❌ No result directories found!")
    else:
        print(f"\n📊 Will analyze {len(found_dirs)} methods: {', '.join(found_dirs.keys())}")
    
    return found_dirs

def load_camera_actions(result_dir):
    """Load camera actions from pickle file"""
    camera_pickle = Path(result_dir) / 'camera_actions.pkl'
    
    if not camera_pickle.exists():
        print(f"❌ No camera_actions.pkl found in {result_dir}")
        return None
    
    try:
        with open(camera_pickle, 'rb') as f:
            data = pickle.load(f)
            
            # Handle both old and new formats
            if isinstance(data, dict):
                # New format: {timestamp: {'v': float, 'omega': float}}
                if 'camera_timestamp_to_action' in data:
                    # Old format
                    camera_actions = data.get('camera_timestamp_to_action', {})
                else:
                    # New format - data is directly the timestamp->action mapping
                    camera_actions = data
            else:
                camera_actions = data
                
            return camera_actions
    except Exception as e:
        print(f"❌ Error loading {camera_pickle}: {e}")
        return None

def analyze_method_performance(method_name, camera_actions):
    """Analyze performance of a single method"""
    if not camera_actions:
        return None
    
    timestamps = sorted(camera_actions.keys())
    velocities = [camera_actions[t]['v'] for t in timestamps]
    omegas = [camera_actions[t]['omega'] for t in timestamps]
    
    # Handle robot_pose data if available (for backward compatibility)
    poses = []
    for t in timestamps:
        if 'robot_pose' in camera_actions[t]:
            poses.append(camera_actions[t]['robot_pose'])
        else:
            # If no pose data, create dummy poses
            poses.append((0.0, 0.0, 0.0))
    
    # Key Performance Metrics
    
    # 1. Total Time
    total_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    
    # 2. Distance Traveled
    total_distance = 0.0
    for i in range(1, len(poses)):
        dx = poses[i][0] - poses[i-1][0]
        dy = poses[i][1] - poses[i-1][1]
        total_distance += math.hypot(dx, dy)
    
    # 3. Average Speed
    avg_speed = np.mean(velocities) if velocities else 0
    
    # 4. Time Stuck (velocity < 0.01 m/s)
    stuck_frames = sum(1 for v in velocities if abs(v) < 0.01)
    stuck_percentage = (stuck_frames / len(velocities)) * 100 if velocities else 0
    
    # 5. Energy Consumption (simplified)
    energy = sum(v**2 + w**2 for v, w in zip(velocities, omegas)) * 0.1
    
    # 6. Path Efficiency (straight line vs actual path)
    if len(poses) > 1:
        start_pos = poses[0]
        end_pos = poses[-1]
        straight_distance = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
        path_efficiency = straight_distance / max(total_distance, 0.001)
    else:
        path_efficiency = 0
    
    # 7. Exploration Area
    if poses:
        x_coords = [p[0] for p in poses]
        y_coords = [p[1] for p in poses]
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        exploration_area = x_range * y_range
    else:
        exploration_area = 0
    
    metrics = {
        'total_time': total_time,
        'total_distance': total_distance,
        'avg_speed': avg_speed,
        'stuck_percentage': stuck_percentage,
        'energy_consumption': energy,
        'path_efficiency': path_efficiency,
        'exploration_area': exploration_area,
        'total_frames': len(timestamps)
    }
    
    print(f"\n📊 {method_name} Performance:")
    print(f"   ⏱️  Total Time: {total_time:.1f} seconds")
    print(f"   📏 Distance: {total_distance:.2f} meters")
    print(f"   🚗 Avg Speed: {avg_speed:.3f} m/s")
    print(f"   🛑 Stuck: {stuck_percentage:.1f}% of time")
    print(f"   ⚡ Energy: {energy:.2f} units")
    print(f"   📐 Path Efficiency: {path_efficiency:.3f}")
    print(f"   🗺️  Exploration: {exploration_area:.2f} m²")
    
    return metrics

def create_comparison_chart(all_metrics):
    """Create a simple comparison chart"""
    if not all_metrics:
        print("❌ No data to plot")
        return
    
    methods = list(all_metrics.keys())
    
    # Define colors for each method
    colors = {
        'DRAM': '#2E8B57',      # Sea Green
        'Vanilla_DWA': '#FF6347',  # Tomato
        'MPPI': '#4169E1',         # Royal Blue
        'Nav2_DWB': '#FFD700'      # Gold
    }
    
    # Create subplots for key metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    methods_list = ', '.join(methods)
    fig.suptitle(f'Method Comparison: {methods_list}', fontsize=14, fontweight='bold')
    
    # 1. Completion Time (lower is better)
    ax1 = axes[0, 0]
    times = [all_metrics[m]['total_time'] for m in methods]
    method_colors = [colors.get(m, '#808080') for m in methods]
    bars1 = ax1.bar(methods, times, color=method_colors, alpha=0.7)
    ax1.set_title('Completion Time\n(Lower = Better)', fontweight='bold')
    ax1.set_ylabel('Seconds')
    
    # Add value labels
    for bar, time in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 2. Stuck Percentage (lower is better)
    ax2 = axes[0, 1]
    stuck_pcts = [all_metrics[m]['stuck_percentage'] for m in methods]
    bars2 = ax2.bar(methods, stuck_pcts, color=method_colors, alpha=0.7)
    ax2.set_title('Time Stuck (Dead-ends)\n(Lower = Better)', fontweight='bold')
    ax2.set_ylabel('Percentage (%)')
    
    for bar, pct in zip(bars2, stuck_pcts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stuck_pcts)*0.01,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Average Speed (higher is better)
    ax3 = axes[1, 0]
    speeds = [all_metrics[m]['avg_speed'] for m in methods]
    bars3 = ax3.bar(methods, speeds, color=method_colors, alpha=0.7)
    ax3.set_title('Average Speed\n(Higher = Better)', fontweight='bold')
    ax3.set_ylabel('m/s')
    
    for bar, speed in zip(bars3, speeds):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speeds)*0.01,
                f'{speed:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Path Efficiency (higher is better)
    ax4 = axes[1, 1]
    efficiencies = [all_metrics[m]['path_efficiency'] for m in methods]
    bars4 = ax4.bar(methods, efficiencies, color=method_colors, alpha=0.7)
    ax4.set_title('Path Efficiency\n(Higher = Better)', fontweight='bold')
    ax4.set_ylabel('Straight-line / Actual')
    
    for bar, eff in zip(bars4, efficiencies):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiencies)*0.01,
                f'{eff:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/home/mrvik/dram_ws/simple_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison chart saved: {filename}")
    
    plt.show()
    return filename

def print_summary_report(all_metrics):
    """Print a summary report"""
    methods = list(all_metrics.keys())
    
    if 'DRAM' not in all_metrics:
        print(f"\n📊 COMPARISON SUMMARY")
        print("="*50)
        print(f"Analyzed methods: {', '.join(methods)}")
        print("Note: No DRAM results found for detailed comparison")
        return
    
    dram_metrics = all_metrics['DRAM']
    baseline_methods = [m for m in all_metrics.keys() if m != 'DRAM']
    
    print("\n" + "="*60)
    print("🏆 DRAM METHOD SUPERIORITY ANALYSIS")
    print("="*60)
    print(f"Comparing DRAM vs: {', '.join(baseline_methods)}")
    
    improvements = []
    
    for baseline in baseline_methods:
        baseline_metrics = all_metrics[baseline]
        
        print(f"\n📊 DRAM vs {baseline}:")
        
        # Compare key metrics
        metrics_to_compare = [
            ('total_time', 'Completion Time', 'seconds', 'lower'),
            ('stuck_percentage', 'Time Stuck', '%', 'lower'),
            ('avg_speed', 'Average Speed', 'm/s', 'higher'),
            ('path_efficiency', 'Path Efficiency', 'ratio', 'higher'),
            ('exploration_area', 'Exploration Area', 'm²', 'higher')
        ]
        
        for metric_key, metric_name, unit, direction in metrics_to_compare:
            dram_val = dram_metrics.get(metric_key, 0)
            baseline_val = baseline_metrics.get(metric_key, 0)
            
            if baseline_val > 0:
                if direction == 'lower':
                    improvement = ((baseline_val - dram_val) / baseline_val) * 100
                    comparison = "better" if dram_val < baseline_val else "worse"
                else:  # higher
                    improvement = ((dram_val - baseline_val) / baseline_val) * 100
                    comparison = "better" if dram_val > baseline_val else "worse"
                
                symbol = "✅" if comparison == "better" else "❌"
                print(f"   {symbol} {metric_name}: {abs(improvement):.1f}% {comparison}")
                
                if comparison == "better":
                    improvements.append((metric_name, improvement))
    
    # Overall summary
    print(f"\n🎯 KEY ADVANTAGES OF DRAM METHOD:")
    print("-" * 40)
    
    # Group improvements by metric
    metric_improvements = {}
    for metric, improvement in improvements:
        if metric not in metric_improvements:
            metric_improvements[metric] = []
        metric_improvements[metric].append(improvement)
    
    for metric, improvements_list in metric_improvements.items():
        avg_improvement = np.mean(improvements_list)
        print(f"✅ {metric}: Average {avg_improvement:.1f}% better than baselines")
    
    # Count wins
    total_comparisons = len(baseline_methods) * 5  # 5 metrics per comparison
    wins = len(improvements)
    win_percentage = (wins / total_comparisons) * 100
    
    print(f"\n🏆 OVERALL PERFORMANCE:")
    print(f"   DRAM wins in {wins}/{total_comparisons} comparisons ({win_percentage:.1f}%)")
    
    if win_percentage > 60:
        print("🎉 DRAM method is SIGNIFICANTLY BETTER than baseline methods!")
    elif win_percentage > 40:
        print("👍 DRAM method shows good improvements over baselines.")
    else:
        print("🤔 Results are mixed - may need more analysis.")

def main():
    print("🔍 Simple Method Comparison Tool")
    print("=" * 50)
    
    # Find result directories
    result_dirs = find_latest_results()
    
    if not result_dirs:
        print("❌ No result directories found!")
        print("   Make sure you have run the planners and generated pickle files.")
        return
    
    # Analyze each method
    all_metrics = {}
    
    for method_name, result_dir in result_dirs.items():
        print(f"\n🔍 Analyzing {method_name}...")
        camera_actions = load_camera_actions(result_dir)
        
        if camera_actions:
            metrics = analyze_method_performance(method_name, camera_actions)
            if metrics:
                all_metrics[method_name] = metrics
        else:
            print(f"⚠️ Skipping {method_name} - no valid data")
    
    if not all_metrics:
        print("❌ No valid data found for any method!")
        return
    
    # Create comparison visualization
    chart_file = create_comparison_chart(all_metrics)
    
    # Print summary report
    print_summary_report(all_metrics)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/home/mrvik/dram_ws/simple_comparison_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'methods_analyzed': list(all_metrics.keys()),
            'performance_metrics': all_metrics,
            'chart_file': chart_file
        }, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"📄 Results saved: {results_file}")
    if 'chart_file' in locals():
        print(f"📊 Chart saved: {chart_file}")

if __name__ == "__main__":
    main()
