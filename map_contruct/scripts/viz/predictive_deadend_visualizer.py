#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray, Marker
from tf2_ros import TransformListener, Buffer
import numpy as np
import cv2
import time
import os
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyBboxPatch
import math

class PredictiveDeadEndVisualizer(Node):
    def __init__(self):
        super().__init__('predictive_deadend_visualizer')
        
        # Parameters
        self.save_interval = self.declare_parameter('save_interval', 2.0).get_parameter_value().double_value
        self.output_dir = self.declare_parameter('output_dir', '/home/mrvik/dram_ws/predictive_figures').get_parameter_value().string_value
        self.auto_capture = self.declare_parameter('auto_capture', True).get_parameter_value().bool_value
        self.sequence_mode = self.declare_parameter('sequence_mode', True).get_parameter_value().bool_value
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subscriptions
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.costmap_callback,
            10
        )
        
        self.dead_end_sub = self.create_subscription(
            Bool,
            '/dead_end_detection/is_dead_end',
            self.dead_end_callback,
            10
        )
        
        self.path_status_sub = self.create_subscription(
            Float32MultiArray,
            '/dead_end_detection/path_status',
            self.path_status_callback,
            10
        )
        
        self.recovery_points_sub = self.create_subscription(
            Float32MultiArray,
            '/dead_end_detection/recovery_points',
            self.recovery_points_callback,
            10
        )
        
        # Publishers for annotations
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/predictive_annotations',
            10
        )
        
        # Transform listener for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Data storage
        self.costmap_data = None
        self.robot_pose = None
        self.is_dead_end = False
        self.path_probabilities = None  # [front, left, right]
        self.recovery_points = []
        
        # Capture state
        self.capture_count = 0
        self.sequence_frames = []
        self.last_save_time = 0
        
        # Timer for periodic capture
        if self.auto_capture:
            self.create_timer(self.save_interval, self.auto_capture_callback)
        
        # Timer for publishing annotations
        self.create_timer(0.5, self.publish_annotations)
        
        self.get_logger().info(f'Predictive Dead End Visualizer initialized')
        self.get_logger().info(f'Output directory: {self.output_dir}')
        self.get_logger().info(f'Auto capture: {self.auto_capture}, Sequence mode: {self.sequence_mode}')

    def costmap_callback(self, msg):
        """Store the latest costmap data"""
        self.costmap_data = {
            'data': np.array(msg.data).reshape((msg.info.height, msg.info.width)),
            'info': msg.info,
            'header': msg.header
        }

    def dead_end_callback(self, msg):
        """Store dead end detection status"""
        self.is_dead_end = msg.data

    def path_status_callback(self, msg):
        """Store path probability data"""
        if len(msg.data) >= 3:
            self.path_probabilities = list(msg.data[:3])  # Convert to list [front, left, right]

    def recovery_points_callback(self, msg):
        """Store recovery points data"""
        data = list(msg.data)  # Convert to list first
        self.recovery_points = []
        if len(data) % 3 == 0:
            for i in range(0, len(data), 3):
                self.recovery_points.append({
                    'type': int(data[i]),
                    'x': float(data[i+1]),
                    'y': float(data[i+2])
                })

    def get_robot_pose(self):
        """Get current robot pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            # Extract yaw from quaternion
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            
            self.robot_pose = {'x': x, 'y': y, 'yaw': yaw}
            return True
        except Exception as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return False

    def auto_capture_callback(self):
        """Automatically capture interesting scenarios"""
        current_time = time.time()
        
        # Only capture if we have all necessary data
        if not self.has_complete_data():
            return
            
        # Capture conditions:
        # 1. Robot is moving towards a potential dead end (path prob < 0.7)
        # 2. Not yet at the dead end (is_dead_end = False)
        # 3. At least one path has low probability
        
        if (self.path_probabilities and 
            not self.is_dead_end and 
            min(self.path_probabilities) < 0.7 and
            current_time - self.last_save_time > self.save_interval):
            
            self.capture_scenario("auto_capture")
            self.last_save_time = current_time

    def has_complete_data(self):
        """Check if we have all necessary data for visualization"""
        return (self.costmap_data is not None and 
                self.get_robot_pose() and
                self.path_probabilities is not None)

    def capture_scenario(self, scenario_name="manual"):
        """Capture the current scenario for visualization"""
        if not self.has_complete_data():
            self.get_logger().warn("Cannot capture - missing data")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create scenario data
        scenario_data = {
            'timestamp': timestamp,
            'scenario_name': scenario_name,
            'robot_pose': self.robot_pose.copy(),
            'costmap': {
                'data': self.costmap_data['data'].copy(),
                'info': {
                    'resolution': self.costmap_data['info'].resolution,
                    'width': self.costmap_data['info'].width,
                    'height': self.costmap_data['info'].height,
                    'origin': {
                        'x': self.costmap_data['info'].origin.position.x,
                        'y': self.costmap_data['info'].origin.position.y
                    }
                }
            },
            'is_dead_end': self.is_dead_end,
            'path_probabilities': list(self.path_probabilities) if self.path_probabilities else None,
            'recovery_points': list(self.recovery_points) if self.recovery_points else []
        }
        
        if self.sequence_mode:
            # Add to sequence
            self.sequence_frames.append(scenario_data)
            
            # If we have 3 frames, create sequence visualization
            if len(self.sequence_frames) >= 3:
                self.create_sequence_visualization(self.sequence_frames[-3:])
                
        else:
            # Create single frame visualization
            self.create_single_visualization(scenario_data)
            
        self.capture_count += 1
        self.get_logger().info(f"Captured scenario: {scenario_name} (#{self.capture_count})")

    def create_single_visualization(self, data):
        """Create a single annotated costmap visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Draw costmap
        costmap_img = self.prepare_costmap_image(data)
        extent = self.get_costmap_extent(data['costmap']['info'])
        
        im = ax.imshow(costmap_img, extent=extent, origin='lower', interpolation='nearest')
        
        # Add robot pose
        self.draw_robot_pose(ax, data['robot_pose'])
        
        # Add predictive annotations
        self.draw_predictive_annotations(ax, data)
        
        # Add title and labels
        dead_end_status = "DEAD END DETECTED" if data['is_dead_end'] else "PREDICTING AHEAD"
        prob_text = ""
        if data['path_probabilities']:
            prob_text = f"Path Probs: F={data['path_probabilities'][0]:.2f}, L={data['path_probabilities'][1]:.2f}, R={data['path_probabilities'][2]:.2f}"
        
        ax.set_title(f'Predictive Dead-End Detection\n{dead_end_status}\n{prob_text}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        
        # Add legend
        self.add_legend(ax)
        
        # Save figure
        filename = f"predictive_deadend_{data['timestamp']}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metadata
        metadata_file = os.path.join(self.output_dir, f"metadata_{data['timestamp']}.json")
        with open(metadata_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            save_data = data.copy()
            save_data['costmap']['data'] = save_data['costmap']['data'].tolist()
            json.dump(save_data, f, indent=2)
        
        self.get_logger().info(f"Saved visualization: {filepath}")

    def create_sequence_visualization(self, sequence_data):
        """Create a sequence of 3 timesteps showing prediction progression"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        titles = [
            "Frame 1: Entering Corridor\nPredicting Dead End Ahead",
            "Frame 2: Approaching\nPrediction Holds",
            "Frame 3: At Decision Point\nValidating Prediction"
        ]
        
        for i, (ax, data) in enumerate(zip(axes, sequence_data)):
            # Draw costmap
            costmap_img = self.prepare_costmap_image(data)
            extent = self.get_costmap_extent(data['costmap']['info'])
            
            im = ax.imshow(costmap_img, extent=extent, origin='lower', interpolation='nearest')
            
            # Add robot pose
            self.draw_robot_pose(ax, data['robot_pose'])
            
            # Add predictive annotations
            self.draw_predictive_annotations(ax, data)
            
            # Add title
            ax.set_title(titles[i], fontsize=12, fontweight='bold')
            ax.set_xlabel('X (meters)', fontsize=10)
            if i == 0:
                ax.set_ylabel('Y (meters)', fontsize=10)
        
        # Add overall title
        fig.suptitle('Predictive Dead-End Detection Sequence', fontsize=16, fontweight='bold')
        
        # Add legend to the last subplot
        self.add_legend(axes[-1])
        
        plt.tight_layout()
        
        # Save sequence
        timestamp = sequence_data[-1]['timestamp']
        filename = f"predictive_sequence_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.get_logger().info(f"Saved sequence visualization: {filepath}")

    def prepare_costmap_image(self, data):
        """Prepare costmap image with semantic coloring"""
        costmap = data['costmap']['data']
        height, width = costmap.shape
        
        # Create RGB image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Standard costmap coloring
        # Free space (0) = white
        # Unknown (-1) = gray  
        # Occupied (100) = black
        
        free_mask = (costmap == 0)
        unknown_mask = (costmap == -1)
        occupied_mask = (costmap >= 50)
        
        # Base colors
        img[free_mask] = [255, 255, 255]  # White for free
        img[unknown_mask] = [128, 128, 128]  # Gray for unknown
        img[occupied_mask] = [0, 0, 0]  # Black for obstacles
        
        # Add semantic dead-end predictions
        if data['path_probabilities']:
            # Create predictive overlay based on path probabilities
            robot_pose = data['robot_pose']
            
            # Find areas ahead of robot that might be dead ends
            self.add_semantic_predictions(img, costmap, robot_pose, data['path_probabilities'])
        
        return img

    def add_semantic_predictions(self, img, costmap, robot_pose, path_probs):
        """Add semantic dead-end predictions to the image"""
        height, width = costmap.shape
        info = self.costmap_data['info']
        
        # Convert robot pose to grid coordinates
        robot_grid_x = int((robot_pose['x'] - info.origin.position.x) / info.resolution)
        robot_grid_y = int((robot_pose['y'] - info.origin.position.y) / info.resolution)
        
        # Create prediction overlay
        # Areas with low path probability get colored as predicted dead ends
        min_prob = min(path_probs)
        
        if min_prob < 0.7:  # If any path has low probability
            # Color areas ahead of robot based on prediction confidence
            yaw = robot_pose['yaw']
            
            # Create a cone ahead of the robot
            for dy in range(-20, 21):
                for dx in range(0, 40):  # Look ahead
                    gx = robot_grid_x + int(dx * np.cos(yaw) - dy * np.sin(yaw))
                    gy = robot_grid_y + int(dx * np.sin(yaw) + dy * np.cos(yaw))
                    
                    if 0 <= gx < width and 0 <= gy < height:
                        # Only color free space
                        if costmap[gy, gx] == 0:
                            # Distance-based intensity
                            distance = np.sqrt(dx*dx + dy*dy)
                            if distance < 30:  # Within prediction range
                                intensity = max(0, 1.0 - min_prob - distance/50.0)
                                if intensity > 0:
                                    # Orange/yellow for predicted dead end
                                    img[gy, gx] = [
                                        255,  # Red
                                        int(255 * (1 - intensity * 0.5)),  # Green (less for more red)
                                        0  # Blue
                                    ]

    def get_costmap_extent(self, info):
        """Get extent for imshow"""
        origin_x = info['origin']['x']
        origin_y = info['origin']['y']
        resolution = info['resolution']
        width = info['width']
        height = info['height']
        
        return [
            origin_x,
            origin_x + width * resolution,
            origin_y,
            origin_y + height * resolution
        ]

    def draw_robot_pose(self, ax, robot_pose):
        """Draw robot pose as an arrow"""
        x, y, yaw = robot_pose['x'], robot_pose['y'], robot_pose['yaw']
        
        # Draw robot as a blue arrow
        arrow_length = 1.0
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        
        ax.arrow(x, y, dx, dy, 
                head_width=0.3, head_length=0.2, 
                fc='blue', ec='blue', linewidth=3,
                label='Robot Pose')
        
        # Add a circle at robot position
        circle = Circle((x, y), 0.2, color='blue', alpha=0.7)
        ax.add_patch(circle)

    def draw_predictive_annotations(self, ax, data):
        """Draw annotations showing predictions"""
        robot_pose = data['robot_pose']
        
        # If predicting dead end ahead, add annotation
        if data['path_probabilities'] and min(data['path_probabilities']) < 0.7:
            # Draw prediction area ahead
            x, y, yaw = robot_pose['x'], robot_pose['y'], robot_pose['yaw']
            
            # Prediction zone ahead of robot
            pred_distance = 5.0
            pred_x = x + pred_distance * np.cos(yaw)
            pred_y = y + pred_distance * np.sin(yaw)
            
            # Draw prediction circle
            pred_circle = Circle((pred_x, pred_y), 1.5, 
                               fill=False, edgecolor='orange', 
                               linewidth=3, linestyle='--',
                               alpha=0.8)
            ax.add_patch(pred_circle)
            
            # Add text annotation
            ax.annotate('Predicted\nDead End\nAhead', 
                       xy=(pred_x, pred_y), 
                       xytext=(pred_x + 2, pred_y + 2),
                       fontsize=12, fontweight='bold',
                       color='darkorange',
                       arrowprops=dict(arrowstyle='->', 
                                     color='darkorange', 
                                     linewidth=2),
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor='white', 
                               edgecolor='darkorange',
                               alpha=0.9))
        
        # Draw recovery points if available
        for rp in data['recovery_points']:
            recovery_circle = Circle((rp['x'], rp['y']), 0.5,
                                   fill=True, facecolor='green', 
                                   edgecolor='darkgreen',
                                   alpha=0.6)
            ax.add_patch(recovery_circle)

    def add_legend(self, ax):
        """Add legend to the plot"""
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Free Space'),
            Patch(facecolor='black', edgecolor='black', label='Obstacles'),
            Patch(facecolor='orange', edgecolor='black', label='Predicted Dead End'),
            Patch(facecolor='green', edgecolor='black', label='Recovery Points'),
            Patch(facecolor='blue', edgecolor='black', label='Robot Position')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    def publish_annotations(self):
        """Publish visualization markers to RViz"""
        if not self.has_complete_data():
            return
            
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # Add prediction markers
        if self.path_probabilities and min(self.path_probabilities) < 0.7:
            robot_pose = self.robot_pose
            
            # Prediction zone marker
            pred_marker = Marker()
            pred_marker.header.frame_id = "map"
            pred_marker.header.stamp = self.get_clock().now().to_msg()
            pred_marker.id = 1
            pred_marker.type = Marker.CYLINDER
            pred_marker.action = Marker.ADD
            
            pred_distance = 5.0
            pred_marker.pose.position.x = robot_pose['x'] + pred_distance * np.cos(robot_pose['yaw'])
            pred_marker.pose.position.y = robot_pose['y'] + pred_distance * np.sin(robot_pose['yaw'])
            pred_marker.pose.position.z = 0.1
            pred_marker.pose.orientation.w = 1.0
            
            pred_marker.scale.x = 3.0
            pred_marker.scale.y = 3.0
            pred_marker.scale.z = 0.2
            
            pred_marker.color.r = 1.0
            pred_marker.color.g = 0.5
            pred_marker.color.b = 0.0
            pred_marker.color.a = 0.5
            
            marker_array.markers.append(pred_marker)
            
            # Text marker
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.id = 2
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = pred_marker.pose.position.x
            text_marker.pose.position.y = pred_marker.pose.position.y
            text_marker.pose.position.z = 1.0
            text_marker.pose.orientation.w = 1.0
            
            text_marker.scale.z = 0.5
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.text = "Predicted Dead End"
            
            marker_array.markers.append(text_marker)
        
        self.marker_pub.publish(marker_array)

    def manual_capture(self):
        """Service call for manual capture"""
        self.capture_scenario("manual_capture")

def main(args=None):
    rclpy.init(args=args)
    node = PredictiveDeadEndVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
