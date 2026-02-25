#!/usr/bin/env python3

"""
Enhanced Evaluation Framework for Dead-End Detection Methods

Enhanced metrics with normalization, bootstrap confidence intervals, and proactive metrics:
1. Normalized metrics: Energy/100m, False-positives/100m
2. Bootstrap 95% confidence intervals
3. Proactive metrics: Detection lead, Distance to first recovery, False negatives, EDE integral
4. Minimal ablations support

Metrics collected:
- Success rate (%)
- Time to goal (s) 
- Path length (m)
- Energy (Wh/100m) or proxy/100m
- False dead-ends (/100m)
- False negatives (#/run)
- Detection lead (m)
- Distance to first recovery (m)
- Freezes (#/run), Time trapped (s/run)
- EDE integral (↓)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
import subprocess
import time
import json
import os
import signal
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

class EnhancedEvaluationFramework(Node):
    def __init__(self):
        super().__init__('enhanced_evaluation_framework')
        
        # Evaluation configuration
        self.manual_localization = True   # Use RViz 2D Pose Estimate
        self.manual_goal_setting = True   # Use RViz 2D Nav Goal
        self.use_global_map = True        # Set to True for pre-built global map, False for SLAM
        self.global_map_file = "/path/to/your/global_map.yaml"  # Update this path
        self.use_slam = False             # Set to True to use SLAM instead of global map
        
        # Evaluation configuration with ablations
        self.methods = {
            # Main method
            'multi_camera_dram': {
                'name': 'Multi-Camera DRaM (Your Method)',
                'nodes': ['inference', 'cost_layer_processor', 'dram_heatmap_viz'],
                'metrics_file': 'multi_camera_metrics.json',
                'description': 'Full method: 3 cameras + DRaM model + semantic cost layer',
                'category': 'main'
            },
            
            # Ablation studies
            'single_camera_dram': {
                'name': 'Single-Camera DRaM (Ablation)',
                'nodes': ['single_camera_inference', 'dram_heatmap_viz'],
                'metrics_file': 'single_camera_metrics.json', 
                'description': 'Ablation: Only front camera + DRaM model',
                'category': 'ablation'
            },
            'dram_no_semantic_cost': {
                'name': 'DRaM w/o Semantic Cost (Ablation)',
                'nodes': ['inference', 'dram_heatmap_viz'],  # Skip cost_layer_processor
                'metrics_file': 'dram_no_semantic_metrics.json',
                'description': 'Ablation: DRaM without semantic cost layer',
                'category': 'ablation'
            },
            'dram_single_frame': {
                'name': 'DRaM Single-Frame (Ablation)',
                'nodes': ['inference_single_frame', 'cost_layer_processor', 'dram_heatmap_viz'],
                'metrics_file': 'dram_single_frame_metrics.json',
                'description': 'Ablation: No Bayesian update, single-frame semantics only',
                'category': 'ablation'
            },
            
            # Comparison baselines
            'dwa_lidar': {
                'name': 'DWA with LiDAR (Baseline)',
                'nodes': ['dwa_lidar_controller'],
                'metrics_file': 'dwa_lidar_metrics.json',
                'description': 'Reactive baseline: DWA planner using only LiDAR',
                'category': 'baseline'
            },
            'mppi_lidar': {
                'name': 'MPPI with LiDAR (Baseline)',
                'nodes': ['mppi_lidar_controller'],
                'metrics_file': 'mppi_lidar_metrics.json',
                'description': 'Reactive baseline: MPPI planner using only LiDAR',
                'category': 'baseline'
            }
        }
        
        # Evaluation parameters
        self.trial_duration = 300  # 5 minutes per trial
        self.num_trials = 10  # Increased for better statistics
        self.bootstrap_samples = 1000  # For confidence intervals
        self.results_dir = self.create_results_directory()
        
        # Publishers for goal and map
        self.goal_pub = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        
        # Results storage
        self.all_results = {}
        self.map_server_process = None
        
        self.get_logger().info('🧪 Enhanced Evaluation Framework initialized')
        self.get_logger().info(f'📊 Will evaluate {len(self.methods)} methods with {self.num_trials} trials each')
        self.get_logger().info(f'🔬 Bootstrap CI samples: {self.bootstrap_samples}')

    def create_results_directory(self):
        """Create timestamped results directory"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = f"/home/mrvik/dram_ws/evaluation_results/enhanced_comparison_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        return results_dir
    
    def start_mapping_system(self):
        """Start mapping/localization system"""
        if self.use_slam:
            # Start SLAM for real-time mapping
            try:
                cmd = ['ros2', 'launch', 'slam_toolbox', 'online_async_launch.py']
                self.map_server_process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
                time.sleep(5)  # Wait for SLAM to start
                self.get_logger().info('🗺️  Started SLAM for real-time mapping')
                
            except Exception as e:
                self.get_logger().error(f'❌ Failed to start SLAM: {str(e)}')
                self.map_server_process = None
                
        elif self.use_global_map and os.path.exists(self.global_map_file):
            # Start map server + AMCL for localization
            try:
                # Start map server
                map_cmd = ['ros2', 'run', 'nav2_map_server', 'map_server', 
                          '--ros-args', '-p', f'yaml_filename:={self.global_map_file}']
                self.map_server_process = subprocess.Popen(
                    map_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
                time.sleep(3)
                
                # Start AMCL for localization
                amcl_cmd = ['ros2', 'run', 'nav2_amcl', 'amcl']
                self.amcl_process = subprocess.Popen(
                    amcl_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
                time.sleep(3)
                
                self.get_logger().info(f'🗺️  Started map server + AMCL with: {self.global_map_file}')
                
            except Exception as e:
                self.get_logger().error(f'❌ Failed to start mapping system: {str(e)}')
                self.map_server_process = None
        else:
            self.get_logger().warn('⚠️  No mapping system configured')
    
    def stop_mapping_system(self):
        """Stop mapping/localization system"""
        # Stop map server
        if hasattr(self, 'map_server_process') and self.map_server_process:
            try:
                os.killpg(os.getpgid(self.map_server_process.pid), signal.SIGTERM)
                self.map_server_process.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(self.map_server_process.pid), signal.SIGKILL)
                except:
                    pass
            self.map_server_process = None
        
        # Stop AMCL
        if hasattr(self, 'amcl_process') and self.amcl_process:
            try:
                os.killpg(os.getpgid(self.amcl_process.pid), signal.SIGTERM)
                self.amcl_process.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(self.amcl_process.pid), signal.SIGKILL)
                except:
                    pass
            self.amcl_process = None
    
    def publish_goal(self, goal_point: Dict):
        """Publish navigation goal"""
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        
        goal_msg.pose.position.x = float(goal_point['x'])
        goal_msg.pose.position.y = float(goal_point['y'])
        goal_msg.pose.position.z = 0.0
        
        goal_msg.pose.orientation.x = 0.0
        goal_msg.pose.orientation.y = 0.0
        goal_msg.pose.orientation.z = 0.0
        goal_msg.pose.orientation.w = 1.0
        
        # Publish goal multiple times to ensure it's received
        for _ in range(5):
            self.goal_pub.publish(goal_msg)
            time.sleep(0.1)
        
        self.get_logger().info(f'🎯 Published goal: {goal_point["name"]} at ({goal_point["x"]:.1f}, {goal_point["y"]:.1f})')
    
    def get_next_goal(self) -> Dict:
        """Get next goal point in sequence"""
        goal = self.goal_points[self.current_goal_idx]
        self.current_goal_idx = (self.current_goal_idx + 1) % len(self.goal_points)
        return goal

    def collect_trial_metrics(self, method_id: str, trial: int, start_time: float) -> Dict:
        """Collect enhanced metrics from a completed trial"""
        base_dir = "/home/mrvik/dram_ws/evaluation_results"
        
        # Enhanced metrics structure
        metrics = {
            'method': method_id,
            'trial': trial,
            'start_time': start_time,
            'duration': time.time() - start_time,
            
            # Core metrics
            'success_rate': 0.0,
            'time_to_goal': 0.0,
            'path_length': 0.0,
            'total_energy': 0.0,
            
            # Raw counts
            'dead_end_detections': 0,
            'false_positive_dead_ends': 0,
            'false_negatives': 0,
            'recovery_point_detections': 0,
            'recovery_activations': 0,
            'freezes': 0,
            
            # Proactive metrics
            'detection_lead_distances': [],  # List of lead distances
            'detection_lead_times': [],     # List of lead times
            'distance_to_first_recovery': [],  # List of recovery distances
            'time_trapped': 0.0,
            'ede_integral': 0.0,
            
            # Normalized metrics (calculated later)
            'energy_per_100m': 0.0,
            'false_positives_per_100m': 0.0,
            'detection_lead_mean': 0.0,
            'distance_to_recovery_mean': 0.0
        }
        
        try:
            # Find most recent metrics file for this method
            method_dirs = [d for d in os.listdir(base_dir) if method_id in d]
            if method_dirs:
                latest_dir = max(method_dirs, key=lambda x: os.path.getctime(os.path.join(base_dir, x)))
                metrics_file = os.path.join(base_dir, latest_dir, f'{method_id}_metrics.json')
                
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        file_metrics = json.load(f)
                    
                    # Update metrics with file data
                    metrics.update(file_metrics)
                    
                    # Calculate normalized metrics
                    if metrics['path_length'] > 0:
                        metrics['energy_per_100m'] = (metrics['total_energy'] / metrics['path_length']) * 100
                        metrics['false_positives_per_100m'] = (metrics['false_positive_dead_ends'] / metrics['path_length']) * 100
                    
                    # Calculate proactive metric means
                    if metrics['detection_lead_distances']:
                        metrics['detection_lead_mean'] = np.mean(metrics['detection_lead_distances'])
                    if metrics['distance_to_first_recovery']:
                        metrics['distance_to_recovery_mean'] = np.mean(metrics['distance_to_first_recovery'])
                    
                    # Success determination
                    if metrics['time_to_goal'] > 0 and metrics['time_to_goal'] < self.trial_duration:
                        metrics['success_rate'] = 1.0
                    
        except Exception as e:
            self.get_logger().warn(f'⚠️  Could not load metrics for {method_id}: {str(e)}')
        
        return metrics

    def bootstrap_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
        """Calculate bootstrap confidence interval"""
        if not data or len(data) < 2:
            return 0.0, 0.0, 0.0
        
        data = np.array(data)
        mean = np.mean(data)
        
        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(self.bootstrap_samples):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, (alpha/2) * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return mean, lower, upper

    def calculate_summary_stats(self, trial_results: List[Dict]) -> Dict:
        """Calculate enhanced summary statistics with bootstrap CIs"""
        if not trial_results:
            return {}
        
        summary = {}
        
        # Core metrics for bootstrap CI
        metrics_for_ci = [
            'success_rate', 'time_to_goal', 'path_length', 'energy_per_100m',
            'false_positives_per_100m', 'false_negatives', 'detection_lead_mean',
            'distance_to_recovery_mean', 'freezes', 'time_trapped', 'ede_integral'
        ]
        
        for metric in metrics_for_ci:
            values = [trial.get(metric, 0) for trial in trial_results]
            mean, ci_lower, ci_upper = self.bootstrap_confidence_interval(values)
            
            summary[metric] = {
                'mean': mean,
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_trials': len(values)
            }
        
        # Special handling for list metrics (detection leads, recovery distances)
        all_detection_leads = []
        all_recovery_distances = []
        
        for trial in trial_results:
            all_detection_leads.extend(trial.get('detection_lead_distances', []))
            all_recovery_distances.extend(trial.get('distance_to_first_recovery', []))
        
        if all_detection_leads:
            mean, ci_lower, ci_upper = self.bootstrap_confidence_interval(all_detection_leads)
            summary['detection_lead_aggregate'] = {
                'mean': mean, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
                'n_detections': len(all_detection_leads)
            }
        
        if all_recovery_distances:
            mean, ci_lower, ci_upper = self.bootstrap_confidence_interval(all_recovery_distances)
            summary['recovery_distance_aggregate'] = {
                'mean': mean, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
                'n_recoveries': len(all_recovery_distances)
            }
        
        return summary

    def generate_enhanced_comparison_table(self):
        """Generate enhanced comparison table with all requested metrics"""
        table_data = []
        
        for method_id, results in self.all_results.items():
            if 'summary' not in results or not results['summary']:
                continue
                
            summary = results['summary']
            
            row = {
                'Method': results['config']['name'],
                'Category': results['config']['category'].title(),
                
                # Core performance metrics
                'Success Rate (%)': self.format_ci_metric(summary.get('success_rate', {}), multiplier=100, decimals=1),
                'Time to Goal (s)': self.format_ci_metric(summary.get('time_to_goal', {}), decimals=1),
                'Path Length (m)': self.format_ci_metric(summary.get('path_length', {}), decimals=1),
                
                # Normalized efficiency metrics
                'Energy (units/100m)': self.format_ci_metric(summary.get('energy_per_100m', {}), decimals=2),
                'False Dead-Ends (/100m)': self.format_ci_metric(summary.get('false_positives_per_100m', {}), decimals=2),
                
                # Proactive advantage metrics
                'False Negatives (#/run)': self.format_ci_metric(summary.get('false_negatives', {}), decimals=1),
                'Detection Lead (m)': self.format_ci_metric(summary.get('detection_lead_aggregate', {}), decimals=1),
                'Distance to Recovery (m)': self.format_ci_metric(summary.get('recovery_distance_aggregate', {}), decimals=1),
                
                # Robustness metrics
                'Freezes (#/run)': self.format_ci_metric(summary.get('freezes', {}), decimals=1),
                'Time Trapped (s/run)': self.format_ci_metric(summary.get('time_trapped', {}), decimals=1),
                'EDE Integral (↓)': self.format_ci_metric(summary.get('ede_integral', {}), decimals=3),
            }
            table_data.append(row)
        
        # Sort by category (main, ablation, baseline) and then by performance
        df = pd.DataFrame(table_data)
        category_order = {'Main': 0, 'Ablation': 1, 'Baseline': 2}
        df['sort_key'] = df['Category'].map(category_order)
        df = df.sort_values('sort_key').drop('sort_key', axis=1)
        
        # Save as CSV
        csv_file = os.path.join(self.results_dir, 'enhanced_comparison_table.csv')
        df.to_csv(csv_file, index=False)
        
        # Save as formatted text with proper paper table format
        txt_file = os.path.join(self.results_dir, 'paper_table.txt')
        with open(txt_file, 'w') as f:
            f.write("ENHANCED DEAD-END DETECTION METHODS COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            f.write("Table format ready for paper submission:\n")
            f.write("Values shown as: Mean [95% CI Lower, 95% CI Upper]\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            f.write("Key findings:\n")
            f.write("- Detection Lead > 0 indicates proactive behavior\n")
            f.write("- Lower Distance to Recovery = faster recovery\n")
            f.write("- EDE Integral measures exposure to dead-end risk\n")
            f.write("- Normalized metrics account for path length differences\n")

    def format_ci_metric(self, metric_dict: Dict, multiplier: float = 1.0, decimals: int = 2) -> str:
        """Format metric with confidence interval for table display"""
        if not metric_dict or 'mean' not in metric_dict:
            return "N/A"
        
        mean = metric_dict['mean'] * multiplier
        ci_lower = metric_dict.get('ci_lower', mean) * multiplier
        ci_upper = metric_dict.get('ci_upper', mean) * multiplier
        
        if decimals == 0:
            return f"{mean:.0f} [{ci_lower:.0f}, {ci_upper:.0f}]"
        elif decimals == 1:
            return f"{mean:.1f} [{ci_lower:.1f}, {ci_upper:.1f}]"
        elif decimals == 2:
            return f"{mean:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"
        else:
            return f"{mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"

    def generate_enhanced_plots(self):
        """Generate enhanced plots with confidence intervals"""
        plt.style.use('seaborn-v0_8')
        
        # Key metrics for visualization
        key_metrics = [
            ('success_rate', 'Success Rate (%)', 100, 1),
            ('energy_per_100m', 'Energy Consumption (units/100m)', 1, 2),
            ('false_positives_per_100m', 'False Positives (/100m)', 1, 2),
            ('detection_lead_aggregate', 'Detection Lead (m)', 1, 1),
            ('distance_to_recovery_aggregate', 'Distance to Recovery (m)', 1, 1),
            ('ede_integral', 'EDE Integral (↓)', 1, 3)
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = {'main': '#1f77b4', 'ablation': '#ff7f0e', 'baseline': '#2ca02c'}
        
        for idx, (metric_key, title, multiplier, decimals) in enumerate(key_metrics):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            methods = []
            means = []
            ci_lowers = []
            ci_uppers = []
            method_colors = []
            
            for method_id, results in self.all_results.items():
                if 'summary' not in results or not results['summary']:
                    continue
                
                metric_data = results['summary'].get(metric_key, {})
                if metric_data and 'mean' in metric_data:
                    methods.append(results['config']['name'].replace(' ', '\n'))
                    means.append(metric_data['mean'] * multiplier)
                    ci_lowers.append(metric_data.get('ci_lower', metric_data['mean']) * multiplier)
                    ci_uppers.append(metric_data.get('ci_upper', metric_data['mean']) * multiplier)
                    method_colors.append(colors.get(results['config']['category'], '#gray'))
            
            if methods:
                x_pos = np.arange(len(methods))
                
                # Plot bars with error bars (confidence intervals)
                bars = ax.bar(x_pos, means, color=method_colors, alpha=0.7, capsize=5)
                
                # Add confidence interval error bars
                yerr_lower = [mean - ci_lower for mean, ci_lower in zip(means, ci_lowers)]
                yerr_upper = [ci_upper - mean for mean, ci_upper in zip(means, ci_uppers)]
                ax.errorbar(x_pos, means, yerr=[yerr_lower, yerr_upper], 
                           fmt='none', color='black', capsize=5, capthick=2)
                
                ax.set_title(f'{title}', fontsize=12, fontweight='bold')
                ax.set_ylabel(f'{title}')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(methods, rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, mean, ci_lower, ci_upper in zip(bars, means, ci_lowers, ci_uppers):
                    height = bar.get_height()
                    if decimals == 0:
                        label = f'{mean:.0f}'
                    elif decimals == 1:
                        label = f'{mean:.1f}'
                    elif decimals == 2:
                        label = f'{mean:.2f}'
                    else:
                        label = f'{mean:.3f}'
                    
                    ax.text(bar.get_x() + bar.get_width()/2., height + (ci_upper - mean) + 0.05 * max(means),
                           label, ha='center', va='bottom', fontweight='bold')
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, color=colors[cat], alpha=0.7, label=cat.title()) 
                          for cat in colors.keys()]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, 'enhanced_comparison_plots.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_proactive_analysis(self):
        """Generate specific analysis of proactive vs reactive behavior"""
        analysis_file = os.path.join(self.results_dir, 'proactive_analysis.txt')
        
        with open(analysis_file, 'w') as f:
            f.write("PROACTIVE vs REACTIVE ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            # Analyze detection lead times
            f.write("DETECTION LEAD ANALYSIS:\n")
            f.write("(Positive values indicate proactive detection)\n\n")
            
            for method_id, results in self.all_results.items():
                if 'summary' not in results:
                    continue
                    
                method_name = results['config']['name']
                category = results['config']['category']
                
                detection_lead = results['summary'].get('detection_lead_aggregate', {})
                if detection_lead:
                    mean = detection_lead.get('mean', 0)
                    ci_lower = detection_lead.get('ci_lower', 0)
                    ci_upper = detection_lead.get('ci_upper', 0)
                    n_detections = detection_lead.get('n_detections', 0)
                    
                    behavior = "PROACTIVE" if mean > 0.5 else "REACTIVE"
                    f.write(f"{method_name} ({category}):\n")
                    f.write(f"  Detection Lead: {mean:.2f}m [{ci_lower:.2f}, {ci_upper:.2f}]\n")
                    f.write(f"  Behavior: {behavior}\n")
                    f.write(f"  Detections: {n_detections}\n\n")
            
            # Analyze recovery efficiency
            f.write("\nRECOVERY EFFICIENCY ANALYSIS:\n")
            f.write("(Lower values indicate faster recovery)\n\n")
            
            for method_id, results in self.all_results.items():
                if 'summary' not in results:
                    continue
                    
                method_name = results['config']['name']
                recovery_dist = results['summary'].get('recovery_distance_aggregate', {})
                false_negs = results['summary'].get('false_negatives', {})
                
                if recovery_dist:
                    mean_dist = recovery_dist.get('mean', 0)
                    ci_lower = recovery_dist.get('ci_lower', 0)
                    ci_upper = recovery_dist.get('ci_upper', 0)
                    
                    f.write(f"{method_name}:\n")
                    f.write(f"  Distance to Recovery: {mean_dist:.2f}m [{ci_lower:.2f}, {ci_upper:.2f}]\n")
                    
                    if false_negs:
                        fn_mean = false_negs.get('mean', 0)
                        f.write(f"  False Negatives: {fn_mean:.1f}/run\n")
                    
                    f.write("\n")

    def run_evaluation(self):
        """Run complete enhanced evaluation"""
        self.get_logger().info('🚀 Starting enhanced evaluation with proactive metrics...')
        
        for method_id, method_config in self.methods.items():
            self.get_logger().info(f'\n📋 Evaluating: {method_config["name"]}')
            self.get_logger().info(f'📝 Category: {method_config["category"].title()}')
            self.get_logger().info(f'📄 Description: {method_config["description"]}')
            
            method_results = []
            
            for trial in range(self.num_trials):
                self.get_logger().info(f'🔄 Trial {trial + 1}/{self.num_trials}')
                
                trial_result = self.run_single_trial(method_id, method_config, trial)
                if trial_result:
                    method_results.append(trial_result)
                
                time.sleep(5)
            
            # Store results with enhanced statistics
            self.all_results[method_id] = {
                'config': method_config,
                'trials': method_results,
                'summary': self.calculate_summary_stats(method_results)
            }
            
            success_rate = len(method_results) / self.num_trials * 100
            self.get_logger().info(f'✅ Completed {method_config["name"]}: {success_rate:.1f}% trial success rate')
        
        # Generate enhanced reports
        self.generate_final_enhanced_report()
        
        self.get_logger().info('🎉 Enhanced evaluation completed!')

    def generate_final_enhanced_report(self):
        """Generate comprehensive enhanced evaluation report"""
        # Save raw results
        results_file = os.path.join(self.results_dir, 'enhanced_evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        
        # Generate enhanced comparison table
        self.generate_enhanced_comparison_table()
        
        # Generate enhanced plots with confidence intervals
        self.generate_enhanced_plots()
        
        # Generate proactive analysis
        self.generate_proactive_analysis()
        
        # Generate detailed text report
        self.generate_enhanced_text_report()
        
        self.get_logger().info(f'📊 Enhanced evaluation report saved to: {self.results_dir}')

    def generate_enhanced_text_report(self):
        """Generate detailed enhanced text report"""
        report_file = os.path.join(self.results_dir, 'enhanced_evaluation_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("ENHANCED DEAD-END DETECTION METHODS EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Trial Duration: {self.trial_duration} seconds\n")
            f.write(f"Trials per Method: {self.num_trials}\n")
            f.write(f"Bootstrap Samples: {self.bootstrap_samples}\n")
            f.write(f"Confidence Level: 95%\n\n")
            
            f.write("METHODOLOGY:\n")
            f.write("- Normalized metrics per 100m path length\n")
            f.write("- Bootstrap confidence intervals (95%)\n")
            f.write("- Proactive metrics: Detection lead, Recovery distance\n")
            f.write("- EDE integral: Exposure to Dead-End risk\n\n")
            
            # Results by category
            categories = ['main', 'ablation', 'baseline']
            for category in categories:
                f.write(f"\n{category.upper()} METHODS:\n")
                f.write("-" * 30 + "\n")
                
                for method_id, results in self.all_results.items():
                    if results['config']['category'] != category:
                        continue
                        
                    f.write(f"\n{results['config']['name']}:\n")
                    f.write(f"Description: {results['config']['description']}\n")
                    f.write(f"Successful Trials: {len(results['trials'])}/{self.num_trials}\n")
                    
                    if 'summary' in results and results['summary']:
                        f.write("\nPerformance Metrics (Mean [95% CI]):\n")
                        summary = results['summary']
                        
                        # Core metrics
                        if 'success_rate' in summary:
                            sr = summary['success_rate']
                            f.write(f"  Success Rate: {sr.get('mean', 0)*100:.1f}% [{sr.get('ci_lower', 0)*100:.1f}, {sr.get('ci_upper', 0)*100:.1f}]\n")
                        
                        # Normalized efficiency metrics
                        if 'energy_per_100m' in summary:
                            ep = summary['energy_per_100m']
                            f.write(f"  Energy/100m: {ep.get('mean', 0):.2f} [{ep.get('ci_lower', 0):.2f}, {ep.get('ci_upper', 0):.2f}]\n")
                        
                        if 'false_positives_per_100m' in summary:
                            fp = summary['false_positives_per_100m']
                            f.write(f"  False Positives/100m: {fp.get('mean', 0):.2f} [{fp.get('ci_lower', 0):.2f}, {fp.get('ci_upper', 0):.2f}]\n")
                        
                        # Proactive metrics
                        if 'detection_lead_aggregate' in summary:
                            dl = summary['detection_lead_aggregate']
                            f.write(f"  Detection Lead: {dl.get('mean', 0):.2f}m [{dl.get('ci_lower', 0):.2f}, {dl.get('ci_upper', 0):.2f}]\n")
                        
                        if 'recovery_distance_aggregate' in summary:
                            rd = summary['recovery_distance_aggregate']
                            f.write(f"  Recovery Distance: {rd.get('mean', 0):.2f}m [{rd.get('ci_lower', 0):.2f}, {rd.get('ci_upper', 0):.2f}]\n")
                    
                    f.write("\n")

    # Include the original methods from the base class
    def run_single_trial(self, method_id: str, method_config: Dict, trial: int) -> Optional[Dict]:
        """Run a single trial with manual localization and goal setting"""
        try:
            # Start mapping/localization system
            self.start_mapping_system()
            time.sleep(5)  # Wait for system to start
            
            # Start method nodes
            processes = self.start_method_nodes(method_config['nodes'])
            time.sleep(10)  # Wait for nodes to initialize
            
            # Instructions for manual operation
            if self.manual_localization and self.use_global_map:
                self.get_logger().info('🎯 MANUAL LOCALIZATION REQUIRED:')
                self.get_logger().info('   1. Open RViz: rviz2')
                self.get_logger().info('   2. Add Map display to see the map')
                self.get_logger().info('   3. Use "2D Pose Estimate" to localize the robot')
                self.get_logger().info('   4. Use "2D Nav Goal" to set navigation goal')
                input('Press ENTER when robot is localized and goal is set...')
            elif self.manual_goal_setting and self.use_slam:
                self.get_logger().info('🎯 MANUAL GOAL SETTING REQUIRED:')
                self.get_logger().info('   1. Open RViz: rviz2') 
                self.get_logger().info('   2. Add Map display to see SLAM map')
                self.get_logger().info('   3. Use "2D Nav Goal" to set navigation goal')
                self.get_logger().info('   4. Robot is auto-localized by SLAM')
                input('Press ENTER when goal is set...')
            
            # Record start time
            start_time = time.time()
            
            self.get_logger().info(f'⏱️  Running trial for {self.trial_duration} seconds...')
            self.get_logger().info('🔄 Trial in progress - robot should be navigating to goal')
            
            # Monitor trial
            time.sleep(self.trial_duration)
            
            # Stop everything
            self.stop_processes(processes)
            self.stop_mapping_system()
            
            # Collect metrics
            metrics = self.collect_trial_metrics(method_id, trial, start_time)
            
            return metrics
            
        except Exception as e:
            self.get_logger().error(f'❌ Trial failed: {str(e)}')
            # Cleanup on failure
            if hasattr(self, 'map_server_process') and self.map_server_process:
                self.stop_mapping_system()
            return None

    def start_method_nodes(self, node_names: List[str]) -> List[subprocess.Popen]:
        """Start ROS nodes for a method"""
        processes = []
        
        for node_name in node_names:
            try:
                cmd = ['ros2', 'run', 'map_contruct', node_name]
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
                processes.append(process)
                self.get_logger().info(f'▶️  Started node: {node_name}')
                time.sleep(2)
                
            except Exception as e:
                self.get_logger().error(f'❌ Failed to start {node_name}: {str(e)}')
        
        return processes

    def stop_processes(self, processes: List[subprocess.Popen]):
        """Stop all processes gracefully"""
        for process in processes:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass

def main():
    """Main enhanced evaluation function"""
    rclpy.init()
    
    framework = EnhancedEvaluationFramework()
    
    try:
        framework.run_evaluation()
    except KeyboardInterrupt:
        framework.get_logger().info('🛑 Enhanced evaluation interrupted by user')
    except Exception as e:
        framework.get_logger().error(f'❌ Enhanced evaluation failed: {str(e)}')
    finally:
        framework.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
