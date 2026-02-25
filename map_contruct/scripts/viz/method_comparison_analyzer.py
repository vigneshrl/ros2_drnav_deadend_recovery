#!/usr/bin/env python3

"""
Method Comparison Analyzer

This script analyzes pickle files from different planning methods and generates
comprehensive comparison reports showing why DRAM method is superior.

Metrics analyzed:
1. Completion Time
2. Total Distance Traveled  
3. Energy Consumption
4. Path Efficiency
5. Velocity Smoothness
6. Dead-end Recovery Performance
7. Exploration Coverage
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import os
from datetime import datetime
import math
from typing import Dict, List, Tuple, Any

class MethodComparisonAnalyzer:
    def __init__(self):
        self.methods_data = {}  # Store data for each method
        self.comparison_results = {}
        self.base_dir = "/home/mrvik/dram_ws"
        
        # Define method names and their expected result directories
        self.method_configs = {
            'DRAM': {
                'pattern': 'dram_rosbag_results_*',
                'description': 'Your DRAM Method (with recovery points, dead-end detection)',
                'color': '#2E8B57'  # Sea Green
            },
            'Vanilla_DWA': {
                'pattern': 'vanilla_dwa_results_*',
                'description': 'Vanilla DWA (basic goal-following)',
                'color': '#FF6347'  # Tomato Red
            },
            'MPPI': {
                'pattern': 'mppi_results_*',
                'description': 'MPPI (sampling-based optimization)',
                'color': '#4169E1'  # Royal Blue
            },
            'Nav2_DWB': {
                'pattern': 'nav2_dwb_results_*',
                'description': 'Nav2 DWB (acceleration-limited planning)',
                'color': '#FFD700'  # Gold
            }
        }
        
        print("🔍 Method Comparison Analyzer initialized")
        print(f"📂 Base directory: {self.base_dir}")

    def find_result_directories(self):
        """Find all result directories for each method"""
        found_methods = {}
        
        for method_name, config in self.method_configs.items():
            pattern = config['pattern']
            matching_dirs = list(Path(self.base_dir).glob(pattern))
            
            if matching_dirs:
                # Get the most recent directory
                latest_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
                found_methods[method_name] = {
                    'directory': str(latest_dir),
                    'config': config
                }
                print(f"✅ Found {method_name}: {latest_dir}")
            else:
                print(f"❌ No results found for {method_name} (pattern: {pattern})")
        
        return found_methods

    def load_method_data(self, method_name: str, directory: str):
        """Load pickle and JSON data for a method"""
        data = {}
        
        # Load camera actions pickle
        camera_pickle_path = Path(directory) / 'camera_actions.pkl'
        if camera_pickle_path.exists():
            with open(camera_pickle_path, 'rb') as f:
                camera_data = pickle.load(f)
                data['camera_actions'] = camera_data.get('camera_timestamp_to_action', {})
                data['total_frames'] = camera_data.get('total_frames', 0)
        
        # Load main action log
        action_pickle_path = Path(directory) / 'action_log.pkl'
        if action_pickle_path.exists():
            with open(action_pickle_path, 'rb') as f:
                action_data = pickle.load(f)
                data['actions'] = action_data.get('actions', [])
                data['metrics'] = action_data.get('metrics', {})
                data['session_info'] = action_data.get('session_info', {})
        
        # Load metrics JSON
        metrics_files = list(Path(directory).glob('*_metrics.json'))
        if metrics_files:
            with open(metrics_files[0], 'r') as f:
                data['json_metrics'] = json.load(f)
        
        print(f"📊 Loaded {method_name}: {len(data.get('camera_actions', {}))} camera frames, "
              f"{len(data.get('actions', []))} actions")
        
        return data

    def calculate_performance_metrics(self, method_name: str, data: Dict):
        """Calculate comprehensive performance metrics"""
        camera_actions = data.get('camera_actions', {})
        actions = data.get('actions', [])
        metrics = data.get('metrics', {})
        
        if not camera_actions:
            print(f"⚠️ No camera actions found for {method_name}")
            return {}
        
        # Extract velocities and poses
        timestamps = sorted(camera_actions.keys())
        velocities = [camera_actions[t]['v'] for t in timestamps]
        omegas = [camera_actions[t]['omega'] for t in timestamps]
        poses = [camera_actions[t]['robot_pose'] for t in timestamps]
        
        # 1. Basic Performance Metrics
        total_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        avg_velocity = np.mean(velocities)
        max_velocity = np.max(velocities)
        
        # 2. Distance and Path Metrics
        total_distance = 0.0
        path_points = []
        for i, pose in enumerate(poses):
            x, y, theta = pose
            path_points.append((x, y))
            if i > 0:
                dx = x - poses[i-1][0]
                dy = y - poses[i-1][1]
                total_distance += math.hypot(dx, dy)
        
        # 3. Energy Consumption
        energy_consumption = sum(v**2 + w**2 for v, w in zip(velocities, omegas)) * 0.1
        
        # 4. Path Efficiency (straight-line distance vs actual distance)
        if len(path_points) > 1:
            start_point = path_points[0]
            end_point = path_points[-1]
            straight_line_distance = math.hypot(end_point[0] - start_point[0], 
                                              end_point[1] - start_point[1])
            path_efficiency = straight_line_distance / max(total_distance, 0.001)
        else:
            path_efficiency = 0.0
        
        # 5. Velocity Smoothness (standard deviation of velocity changes)
        velocity_changes = [abs(velocities[i] - velocities[i-1]) 
                          for i in range(1, len(velocities))]
        velocity_smoothness = 1.0 / (1.0 + np.std(velocity_changes)) if velocity_changes else 1.0
        
        # 6. Exploration Coverage (area of bounding box)
        if path_points:
            x_coords = [p[0] for p in path_points]
            y_coords = [p[1] for p in path_points]
            exploration_area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        else:
            exploration_area = 0.0
        
        # 7. Dead-end Recovery (count zero velocities - proxy for being stuck)
        stuck_count = sum(1 for v in velocities if abs(v) < 0.01)
        stuck_percentage = (stuck_count / len(velocities)) * 100 if velocities else 0
        
        # 8. Angular Velocity Analysis (turning behavior)
        avg_angular_velocity = np.mean([abs(w) for w in omegas])
        max_angular_velocity = np.max([abs(w) for w in omegas])
        
        performance_metrics = {
            'total_time': total_time,
            'total_distance': total_distance,
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'energy_consumption': energy_consumption,
            'path_efficiency': path_efficiency,
            'velocity_smoothness': velocity_smoothness,
            'exploration_area': exploration_area,
            'stuck_percentage': stuck_percentage,
            'avg_angular_velocity': avg_angular_velocity,
            'max_angular_velocity': max_angular_velocity,
            'total_frames': len(timestamps),
            'avg_fps': len(timestamps) / total_time if total_time > 0 else 0,
            
            # Derived metrics
            'distance_per_time': total_distance / max(total_time, 0.001),
            'energy_efficiency': total_distance / max(energy_consumption, 0.001),
            'exploration_efficiency': exploration_area / max(total_time, 0.001),
        }
        
        return performance_metrics

    def generate_comparison_report(self, methods_data: Dict):
        """Generate comprehensive comparison report"""
        print("\n" + "="*80)
        print("📊 METHOD PERFORMANCE COMPARISON REPORT")
        print("="*80)
        
        # Calculate metrics for all methods
        all_metrics = {}
        for method_name, data in methods_data.items():
            metrics = self.calculate_performance_metrics(method_name, data)
            all_metrics[method_name] = metrics
            
        # Create comparison table
        metrics_to_compare = [
            ('total_time', 'Total Time (s)', 'lower_is_better'),
            ('total_distance', 'Distance Traveled (m)', 'context_dependent'),
            ('avg_velocity', 'Average Velocity (m/s)', 'higher_is_better'),
            ('energy_consumption', 'Energy Consumption', 'lower_is_better'),
            ('path_efficiency', 'Path Efficiency', 'higher_is_better'),
            ('velocity_smoothness', 'Velocity Smoothness', 'higher_is_better'),
            ('exploration_area', 'Exploration Area (m²)', 'higher_is_better'),
            ('stuck_percentage', 'Stuck Percentage (%)', 'lower_is_better'),
            ('distance_per_time', 'Speed (m/s)', 'higher_is_better'),
            ('energy_efficiency', 'Energy Efficiency (m/J)', 'higher_is_better'),
            ('exploration_efficiency', 'Exploration Rate (m²/s)', 'higher_is_better'),
        ]
        
        print(f"\n{'Metric':<25} {'DRAM':<12} {'Vanilla_DWA':<12} {'MPPI':<12} {'Nav2_DWB':<12} {'Winner':<10}")
        print("-" * 85)
        
        winners = {}
        for metric_key, metric_name, direction in metrics_to_compare:
            row_data = {}
            for method in all_metrics:
                value = all_metrics[method].get(metric_key, 0)
                row_data[method] = value
            
            # Determine winner
            if direction == 'higher_is_better':
                winner = max(row_data.keys(), key=lambda k: row_data[k])
            elif direction == 'lower_is_better':
                winner = min(row_data.keys(), key=lambda k: row_data[k])
            else:  # context_dependent
                winner = "Context"
            
            winners[metric_key] = winner
            
            # Format and print row
            dram_val = f"{row_data.get('DRAM', 0):.2f}"
            vanilla_val = f"{row_data.get('Vanilla_DWA', 0):.2f}"
            mppi_val = f"{row_data.get('MPPI', 0):.2f}"
            nav2_val = f"{row_data.get('Nav2_DWB', 0):.2f}"
            
            print(f"{metric_name:<25} {dram_val:<12} {vanilla_val:<12} {mppi_val:<12} {nav2_val:<12} {winner:<10}")
        
        # Count wins
        win_counts = {}
        for method in all_metrics:
            win_counts[method] = sum(1 for winner in winners.values() if winner == method)
        
        print("\n" + "="*50)
        print("🏆 OVERALL PERFORMANCE SUMMARY")
        print("="*50)
        
        for method, wins in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
            config = self.method_configs.get(method, {})
            description = config.get('description', method)
            print(f"{method:<15} {wins:>2} wins - {description}")
        
        # Detailed DRAM advantages
        print(f"\n🎯 WHY DRAM METHOD IS SUPERIOR:")
        print("-" * 40)
        
        dram_metrics = all_metrics.get('DRAM', {})
        baseline_avg = {}
        
        # Calculate baseline averages
        baseline_methods = ['Vanilla_DWA', 'MPPI', 'Nav2_DWB']
        for metric_key, _, _ in metrics_to_compare:
            baseline_values = [all_metrics[method].get(metric_key, 0) 
                             for method in baseline_methods if method in all_metrics]
            baseline_avg[metric_key] = np.mean(baseline_values) if baseline_values else 0
        
        improvements = []
        if 'DRAM' in all_metrics:
            for metric_key, metric_name, direction in metrics_to_compare:
                dram_val = dram_metrics.get(metric_key, 0)
                baseline_val = baseline_avg.get(metric_key, 0)
                
                if baseline_val > 0:
                    if direction == 'higher_is_better':
                        improvement = ((dram_val - baseline_val) / baseline_val) * 100
                        if improvement > 0:
                            improvements.append(f"✅ {metric_name}: {improvement:.1f}% better")
                    elif direction == 'lower_is_better':
                        improvement = ((baseline_val - dram_val) / baseline_val) * 100
                        if improvement > 0:
                            improvements.append(f"✅ {metric_name}: {improvement:.1f}% better")
        
        for improvement in improvements[:8]:  # Show top 8 improvements
            print(f"   {improvement}")
        
        return all_metrics, winners

    def create_visualizations(self, methods_data: Dict, all_metrics: Dict):
        """Create comprehensive visualizations"""
        print(f"\n📈 Creating performance visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DRAM Method vs Baseline Planners - Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Completion Time Comparison
        ax1 = axes[0, 0]
        methods = list(all_metrics.keys())
        times = [all_metrics[m].get('total_time', 0) for m in methods]
        colors = [self.method_configs.get(m, {}).get('color', '#808080') for m in methods]
        
        bars1 = ax1.bar(methods, times, color=colors, alpha=0.7)
        ax1.set_title('Completion Time (Lower is Better)', fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time in zip(bars1, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 2. Energy Efficiency Comparison
        ax2 = axes[0, 1]
        energy_eff = [all_metrics[m].get('energy_efficiency', 0) for m in methods]
        bars2 = ax2.bar(methods, energy_eff, color=colors, alpha=0.7)
        ax2.set_title('Energy Efficiency (Higher is Better)', fontweight='bold')
        ax2.set_ylabel('Distance per Energy (m/J)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, eff in zip(bars2, energy_eff):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energy_eff)*0.01,
                    f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Path Efficiency Comparison
        ax3 = axes[0, 2]
        path_eff = [all_metrics[m].get('path_efficiency', 0) for m in methods]
        bars3 = ax3.bar(methods, path_eff, color=colors, alpha=0.7)
        ax3.set_title('Path Efficiency (Higher is Better)', fontweight='bold')
        ax3.set_ylabel('Straight-line / Actual Distance')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, eff in zip(bars3, path_eff):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(path_eff)*0.01,
                    f'{eff:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Stuck Percentage (Dead-end Performance)
        ax4 = axes[1, 0]
        stuck_pct = [all_metrics[m].get('stuck_percentage', 0) for m in methods]
        bars4 = ax4.bar(methods, stuck_pct, color=colors, alpha=0.7)
        ax4.set_title('Stuck Percentage (Lower is Better)', fontweight='bold')
        ax4.set_ylabel('Percentage of Time Stuck (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, pct in zip(bars4, stuck_pct):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stuck_pct)*0.01,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 5. Exploration Efficiency
        ax5 = axes[1, 1]
        explore_eff = [all_metrics[m].get('exploration_efficiency', 0) for m in methods]
        bars5 = ax5.bar(methods, explore_eff, color=colors, alpha=0.7)
        ax5.set_title('Exploration Efficiency (Higher is Better)', fontweight='bold')
        ax5.set_ylabel('Area Explored per Time (m²/s)')
        ax5.tick_params(axis='x', rotation=45)
        
        for bar, eff in zip(bars5, explore_eff):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(explore_eff)*0.01,
                    f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Overall Performance Radar Chart
        ax6 = axes[1, 2]
        
        # Normalize metrics for radar chart (0-1 scale)
        radar_metrics = ['path_efficiency', 'velocity_smoothness', 'energy_efficiency']
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for method in methods:
            values = []
            for metric in radar_metrics:
                # Normalize to 0-1 scale
                all_values = [all_metrics[m].get(metric, 0) for m in methods]
                max_val = max(all_values) if max(all_values) > 0 else 1
                normalized = all_metrics[method].get(metric, 0) / max_val
                values.append(normalized)
            
            values += values[:1]  # Complete the circle
            
            color = self.method_configs.get(method, {}).get('color', '#808080')
            ax6.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
            ax6.fill(angles, values, alpha=0.25, color=color)
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels([m.replace('_', ' ').title() for m in radar_metrics])
        ax6.set_title('Normalized Performance Radar', fontweight='bold')
        ax6.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax6.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"/home/mrvik/dram_ws/method_comparison_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved: {plot_filename}")
        
        plt.show()
        
        return plot_filename

    def create_trajectory_comparison(self, methods_data: Dict):
        """Create trajectory visualization comparing all methods"""
        print(f"\n🗺️ Creating trajectory comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Robot Trajectories - Method Comparison', fontsize=16, fontweight='bold')
        
        method_names = list(methods_data.keys())
        
        for idx, (method_name, data) in enumerate(methods_data.items()):
            if idx >= 4:  # Only show first 4 methods
                break
                
            ax = axes[idx // 2, idx % 2]
            
            camera_actions = data.get('camera_actions', {})
            if not camera_actions:
                ax.text(0.5, 0.5, 'No trajectory data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method_name} - No Data')
                continue
            
            # Extract trajectory points
            timestamps = sorted(camera_actions.keys())
            x_coords = [camera_actions[t]['robot_pose'][0] for t in timestamps]
            y_coords = [camera_actions[t]['robot_pose'][1] for t in timestamps]
            velocities = [camera_actions[t]['v'] for t in timestamps]
            
            # Create trajectory plot with velocity color coding
            if len(x_coords) > 1:
                scatter = ax.scatter(x_coords, y_coords, c=velocities, cmap='viridis', 
                                   s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
                
                # Add trajectory line
                ax.plot(x_coords, y_coords, 'k--', alpha=0.3, linewidth=1)
                
                # Mark start and end points
                ax.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
                ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Velocity (m/s)')
            
            config = self.method_configs.get(method_name, {})
            description = config.get('description', method_name)
            ax.set_title(f'{method_name}\n{description}', fontsize=10)
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.axis('equal')
        
        plt.tight_layout()
        
        # Save trajectory comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        traj_filename = f"/home/mrvik/dram_ws/trajectory_comparison_{timestamp}.png"
        plt.savefig(traj_filename, dpi=300, bbox_inches='tight')
        print(f"🗺️ Trajectory comparison saved: {traj_filename}")
        
        plt.show()
        
        return traj_filename

    def run_full_analysis(self):
        """Run complete method comparison analysis"""
        print("🚀 Starting comprehensive method comparison analysis...")
        
        # Find all result directories
        methods_found = self.find_result_directories()
        
        if not methods_found:
            print("❌ No method result directories found!")
            return
        
        # Load data for each method
        methods_data = {}
        for method_name, info in methods_found.items():
            directory = info['directory']
            data = self.load_method_data(method_name, directory)
            if data:
                methods_data[method_name] = data
        
        if not methods_data:
            print("❌ No valid method data loaded!")
            return
        
        # Generate comparison report
        all_metrics, winners = self.generate_comparison_report(methods_data)
        
        # Create visualizations
        plot_filename = self.create_visualizations(methods_data, all_metrics)
        traj_filename = self.create_trajectory_comparison(methods_data)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"/home/mrvik/dram_ws/comparison_results_{timestamp}.json"
        
        results_data = {
            'timestamp': timestamp,
            'methods_analyzed': list(methods_data.keys()),
            'performance_metrics': all_metrics,
            'winners_by_metric': winners,
            'plot_file': plot_filename,
            'trajectory_file': traj_filename,
            'summary': {
                'total_methods': len(methods_data),
                'dram_wins': sum(1 for w in winners.values() if w == 'DRAM'),
                'total_metrics': len(winners)
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n✅ Complete analysis finished!")
        print(f"📊 Results saved to: {results_file}")
        print(f"📈 Plots saved to: {plot_filename}")
        print(f"🗺️ Trajectories saved to: {traj_filename}")
        
        return results_data

def main():
    analyzer = MethodComparisonAnalyzer()
    results = analyzer.run_full_analysis()
    
    if results:
        print(f"\n🎉 Analysis Complete!")
        print(f"   Methods analyzed: {results['summary']['total_methods']}")
        print(f"   DRAM wins: {results['summary']['dram_wins']}/{results['summary']['total_metrics']} metrics")
        
        if results['summary']['dram_wins'] > results['summary']['total_metrics'] // 2:
            print(f"🏆 DRAM method is SUPERIOR to baseline methods!")
        else:
            print(f"🤔 Results are mixed - check the detailed analysis.")

if __name__ == "__main__":
    main()







