#!/usr/bin/env python3

"""
Evaluation Framework for Dead-End Detection Methods

This framework runs different methods and collects performance metrics:
1. Multi-camera DRaM model (your method)
2. Single-camera DRaM model (ablation)
3. DWA with LiDAR (comparison)
4. MPPI with LiDAR (comparison)

Metrics collected:
- Success rate
- Energy consumption  
- False positive dead-end detection
- Recovery point detection accuracy
- Completion time
- Path efficiency
"""

import rclpy
from rclpy.node import Node
import subprocess
import time
import json
import os
import signal
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class EvaluationFramework(Node):
    def __init__(self):
        super().__init__('evaluation_framework')
        
        # Evaluation configuration
        self.methods = {
            'multi_camera_dram': {
                'name': 'Multi-Camera DRaM (Your Method)',
                'nodes': ['inference', 'cost_layer_processor', 'dram_heatmap_viz'],
                'metrics_file': 'multi_camera_metrics.json',
                'description': 'Your method using 3 cameras and DRaM model'
            },
            'single_camera_dram': {
                'name': 'Single-Camera DRaM (Ablation)',
                'nodes': ['single_camera_inference', 'dram_heatmap_viz'],
                'metrics_file': 'single_camera_metrics.json', 
                'description': 'Ablation study with only front camera'
            },
            'dwa_lidar': {
                'name': 'DWA with LiDAR (Comparison)',
                'nodes': ['dwa_lidar_controller'],
                'metrics_file': 'dwa_lidar_metrics.json',
                'description': 'DWA planner using only LiDAR data'
            },
            'mppi_lidar': {
                'name': 'MPPI with LiDAR (Comparison)',
                'nodes': ['mppi_lidar_controller'],
                'metrics_file': 'mppi_lidar_metrics.json',
                'description': 'MPPI planner using only LiDAR data'
            }
        }
        
        # Evaluation parameters
        self.trial_duration = 300  # 5 minutes per trial
        self.num_trials = 5  # Number of trials per method
        self.results_dir = self.create_results_directory()
        
        # Results storage
        self.all_results = {}
        
        self.get_logger().info('🧪 Evaluation Framework initialized')
        self.get_logger().info(f'📊 Will evaluate {len(self.methods)} methods with {self.num_trials} trials each')

    def create_results_directory(self):
        """Create timestamped results directory"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = f"/home/mrvik/dram_ws/evaluation_results/comparison_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

    def run_evaluation(self):
        """Run complete evaluation of all methods"""
        self.get_logger().info('🚀 Starting evaluation...')
        
        for method_id, method_config in self.methods.items():
            self.get_logger().info(f'\n📋 Evaluating: {method_config["name"]}')
            self.get_logger().info(f'📝 Description: {method_config["description"]}')
            
            method_results = []
            
            for trial in range(self.num_trials):
                self.get_logger().info(f'🔄 Trial {trial + 1}/{self.num_trials}')
                
                # Run single trial
                trial_result = self.run_single_trial(method_id, method_config, trial)
                if trial_result:
                    method_results.append(trial_result)
                
                # Wait between trials
                time.sleep(5)
            
            # Store results for this method
            self.all_results[method_id] = {
                'config': method_config,
                'trials': method_results,
                'summary': self.calculate_summary_stats(method_results)
            }
            
            self.get_logger().info(f'✅ Completed {method_config["name"]}: {len(method_results)} successful trials')
        
        # Generate final report
        self.generate_final_report()
        
        self.get_logger().info('🎉 Evaluation completed!')

    def run_single_trial(self, method_id: str, method_config: Dict, trial: int) -> Optional[Dict]:
        """Run a single trial of a method"""
        try:
            # Start method nodes
            processes = self.start_method_nodes(method_config['nodes'])
            
            # Wait for nodes to initialize
            time.sleep(10)
            
            # Record start time
            start_time = time.time()
            
            # Monitor trial
            self.get_logger().info(f'⏱️  Running trial for {self.trial_duration} seconds...')
            time.sleep(self.trial_duration)
            
            # Stop nodes
            self.stop_processes(processes)
            
            # Collect metrics
            metrics = self.collect_trial_metrics(method_id, trial, start_time)
            
            return metrics
            
        except Exception as e:
            self.get_logger().error(f'❌ Trial failed: {str(e)}')
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
                    preexec_fn=os.setsid  # Create process group
                )
                processes.append(process)
                self.get_logger().info(f'▶️  Started node: {node_name}')
                time.sleep(2)  # Stagger startup
                
            except Exception as e:
                self.get_logger().error(f'❌ Failed to start {node_name}: {str(e)}')
        
        return processes

    def stop_processes(self, processes: List[subprocess.Popen]):
        """Stop all processes gracefully"""
        for process in processes:
            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except:
                # Force kill if graceful shutdown fails
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass

    def collect_trial_metrics(self, method_id: str, trial: int, start_time: float) -> Dict:
        """Collect metrics from a completed trial"""
        # Look for metrics files in evaluation_results directories
        base_dir = "/home/mrvik/dram_ws/evaluation_results"
        
        metrics = {
            'method': method_id,
            'trial': trial,
            'start_time': start_time,
            'duration': time.time() - start_time,
            'success_rate': 0.0,
            'total_distance': 0.0,
            'total_energy': 0.0,
            'dead_end_detections': 0,
            'false_positive_dead_ends': 0,
            'recovery_point_detections': 0,
            'recovery_activations': 0,
            'completion_time': 0.0,
            'path_efficiency': 0.0
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
                    
                    # Calculate derived metrics
                    if metrics['total_distance'] > 0:
                        metrics['path_efficiency'] = 1.0 / (1.0 + metrics['total_energy'] / metrics['total_distance'])
                    
                    if metrics['completion_time'] > 0:
                        metrics['success_rate'] = 1.0  # Completed successfully
                    
        except Exception as e:
            self.get_logger().warn(f'⚠️  Could not load metrics for {method_id}: {str(e)}')
        
        return metrics

    def calculate_summary_stats(self, trial_results: List[Dict]) -> Dict:
        """Calculate summary statistics for a method"""
        if not trial_results:
            return {}
        
        # Extract metrics
        metrics = {}
        for key in ['success_rate', 'total_distance', 'total_energy', 'dead_end_detections', 
                   'false_positive_dead_ends', 'recovery_point_detections', 'recovery_activations',
                   'completion_time', 'path_efficiency']:
            values = [trial.get(key, 0) for trial in trial_results]
            metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return metrics

    def generate_final_report(self):
        """Generate comprehensive evaluation report"""
        # Save raw results
        results_file = os.path.join(self.results_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        # Generate comparison table
        self.generate_comparison_table()
        
        # Generate plots
        self.generate_plots()
        
        # Generate text report
        self.generate_text_report()
        
        self.get_logger().info(f'📊 Final report saved to: {self.results_dir}')

    def generate_comparison_table(self):
        """Generate comparison table of all methods"""
        table_data = []
        
        for method_id, results in self.all_results.items():
            if 'summary' not in results or not results['summary']:
                continue
                
            row = {
                'Method': results['config']['name'],
                'Success Rate (%)': f"{results['summary'].get('success_rate', {}).get('mean', 0) * 100:.1f}",
                'Avg Distance (m)': f"{results['summary'].get('total_distance', {}).get('mean', 0):.2f}",
                'Avg Energy': f"{results['summary'].get('total_energy', {}).get('mean', 0):.2f}",
                'Dead-End Detections': f"{results['summary'].get('dead_end_detections', {}).get('mean', 0):.1f}",
                'False Positives': f"{results['summary'].get('false_positive_dead_ends', {}).get('mean', 0):.1f}",
                'Recovery Points': f"{results['summary'].get('recovery_point_detections', {}).get('mean', 0):.1f}",
                'Recovery Activations': f"{results['summary'].get('recovery_activations', {}).get('mean', 0):.1f}",
                'Avg Time (s)': f"{results['summary'].get('completion_time', {}).get('mean', 0):.1f}",
                'Path Efficiency': f"{results['summary'].get('path_efficiency', {}).get('mean', 0):.3f}"
            }
            table_data.append(row)
        
        # Save as CSV
        df = pd.DataFrame(table_data)
        csv_file = os.path.join(self.results_dir, 'comparison_table.csv')
        df.to_csv(csv_file, index=False)
        
        # Save as formatted text
        txt_file = os.path.join(self.results_dir, 'comparison_table.txt')
        with open(txt_file, 'w') as f:
            f.write("DEAD-END DETECTION METHODS COMPARISON\n")
            f.write("=" * 50 + "\n\n")
            f.write(df.to_string(index=False))

    def generate_plots(self):
        """Generate comparison plots"""
        plt.style.use('default')
        
        # Metrics to plot
        metrics_to_plot = [
            ('success_rate', 'Success Rate', '%'),
            ('total_energy', 'Energy Consumption', 'Units'),
            ('dead_end_detections', 'Dead-End Detections', 'Count'),
            ('recovery_point_detections', 'Recovery Points Detected', 'Count'),
            ('path_efficiency', 'Path Efficiency', 'Score')
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (metric_key, title, unit) in enumerate(metrics_to_plot):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            methods = []
            means = []
            stds = []
            
            for method_id, results in self.all_results.items():
                if 'summary' not in results or not results['summary']:
                    continue
                    
                metric_data = results['summary'].get(metric_key, {})
                if metric_data:
                    methods.append(results['config']['name'].replace(' ', '\n'))
                    mean_val = metric_data.get('mean', 0)
                    if metric_key == 'success_rate':
                        mean_val *= 100  # Convert to percentage
                    means.append(mean_val)
                    stds.append(metric_data.get('std', 0))
            
            if methods:
                bars = ax.bar(methods, means, yerr=stds, capsize=5, 
                             color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)])
                ax.set_title(f'{title}')
                ax.set_ylabel(f'{title} ({unit})')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, mean in zip(bars, means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(stds)*0.1,
                           f'{mean:.2f}', ha='center', va='bottom')
        
        # Remove empty subplots
        for idx in range(len(metrics_to_plot), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, 'comparison_plots.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_text_report(self):
        """Generate detailed text report"""
        report_file = os.path.join(self.results_dir, 'evaluation_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("DEAD-END DETECTION METHODS EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Trial Duration: {self.trial_duration} seconds\n")
            f.write(f"Trials per Method: {self.num_trials}\n\n")
            
            for method_id, results in self.all_results.items():
                f.write(f"\n{results['config']['name'].upper()}\n")
                f.write("-" * len(results['config']['name']) + "\n")
                f.write(f"Description: {results['config']['description']}\n")
                f.write(f"Successful Trials: {len(results['trials'])}/{self.num_trials}\n\n")
                
                if 'summary' in results and results['summary']:
                    f.write("Performance Metrics (Mean ± Std):\n")
                    summary = results['summary']
                    
                    if 'success_rate' in summary:
                        sr = summary['success_rate']
                        f.write(f"  Success Rate: {sr.get('mean', 0)*100:.1f}% ± {sr.get('std', 0)*100:.1f}%\n")
                    
                    if 'total_distance' in summary:
                        td = summary['total_distance']
                        f.write(f"  Total Distance: {td.get('mean', 0):.2f} ± {td.get('std', 0):.2f} m\n")
                    
                    if 'total_energy' in summary:
                        te = summary['total_energy']
                        f.write(f"  Energy Consumption: {te.get('mean', 0):.2f} ± {te.get('std', 0):.2f}\n")
                    
                    if 'dead_end_detections' in summary:
                        ded = summary['dead_end_detections']
                        f.write(f"  Dead-End Detections: {ded.get('mean', 0):.1f} ± {ded.get('std', 0):.1f}\n")
                    
                    if 'false_positive_dead_ends' in summary:
                        fp = summary['false_positive_dead_ends']
                        f.write(f"  False Positives: {fp.get('mean', 0):.1f} ± {fp.get('std', 0):.1f}\n")
                    
                    if 'recovery_point_detections' in summary:
                        rpd = summary['recovery_point_detections']
                        f.write(f"  Recovery Points: {rpd.get('mean', 0):.1f} ± {rpd.get('std', 0):.1f}\n")
                
                f.write("\n")

def main():
    """Main evaluation function"""
    rclpy.init()
    
    framework = EvaluationFramework()
    
    try:
        framework.run_evaluation()
    except KeyboardInterrupt:
        framework.get_logger().info('🛑 Evaluation interrupted by user')
    except Exception as e:
        framework.get_logger().error(f'❌ Evaluation failed: {str(e)}')
    finally:
        framework.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
