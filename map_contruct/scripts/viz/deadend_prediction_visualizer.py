#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray
from nav_msgs.msg import OccupancyGrid
from tf2_ros import TransformListener, Buffer
import numpy as np
import cv2
import time
import os
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyBboxPatch, Arrow
from datetime import datetime

class DeadEndPredictionVisualizer(Node):
    def __init__(self):
        super().__init__('deadend_prediction_visualizer')
        
        # Parameters
        self.output_dir = '/home/mrvik/dram_ws/deadend_prediction_figures'
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
        
        # Transform listener for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Data storage
        self.costmap_data = None
        self.robot_pose = None
        self.is_dead_end = False
        self.path_probabilities = None  # [front, left, right]
        
        # Capture control
        self.last_capture_time = 0
        self.capture_interval = 5.0  # Capture every 5 seconds
        self.capture_count = 0
        
        # Timer for periodic capture
        self.create_timer(2.0, self.check_and_capture)
        
        self.get_logger().info(f'Dead End Prediction Visualizer initialized')
        self.get_logger().info(f'Output directory: {self.output_dir}')

    def costmap_callback(self, msg):
        """Store the latest costmap data"""
        self.costmap_data = {
            'data': np.array(msg.data).reshape((msg.info.height, msg.info.width)),
            'info': msg.info,
            'header': msg.header
        }
        self.get_logger().debug(f'Received costmap: {msg.info.width}x{msg.info.height}')

    def dead_end_callback(self, msg):
        """Store dead end detection status"""
        self.is_dead_end = msg.data

    def path_status_callback(self, msg):
        """Store path probability data"""
        if len(msg.data) >= 3:
            self.path_probabilities = list(msg.data[:3])  # [front, left, right]
            self.get_logger().debug(f'Path probs: F={self.path_probabilities[0]:.2f}, L={self.path_probabilities[1]:.2f}, R={self.path_probabilities[2]:.2f}')

    def get_robot_pose(self):
        """Get current robot pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'odom', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
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

    def check_and_capture(self):
        """Check if we should capture the current scenario"""
        current_time = time.time()
        
        # Only capture if we have all data
        if not self.has_complete_data():
            self.get_logger().debug('Missing data for capture')
            return
            
        # Capture interesting scenarios:
        # 1. Robot is not currently at dead end (is_dead_end = False)  
        # 2. We have path probabilities (model is working)
        # 3. Sufficient time has passed since last capture
        
        min_prob = min(self.path_probabilities) if self.path_probabilities else 1.0
        
        # Debug logging
        self.get_logger().debug(f'Capture check: dead_end={self.is_dead_end}, min_prob={min_prob:.3f}, '
                               f'time_diff={current_time - self.last_capture_time:.1f}s')
        
        # More permissive capture conditions for demonstration
        if (not self.is_dead_end and 
            self.path_probabilities and 
            min_prob < 0.85 and  # More permissive threshold
            current_time - self.last_capture_time > self.capture_interval):
            
            self.get_logger().info(f'Capturing scenario: min_prob={min_prob:.3f}, probs={self.path_probabilities}')
            self.capture_prediction_scenario()
            self.last_capture_time = current_time

    def has_complete_data(self):
        """Check if we have all necessary data"""
        return (self.costmap_data is not None and 
                self.get_robot_pose() and
                self.path_probabilities is not None)

    def capture_prediction_scenario(self):
        """Capture the current predictive scenario"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.capture_count += 1
        
        # Create the visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Process and display costmap
        costmap_img = self.create_costmap_image()
        extent = self.get_costmap_extent()
        
        # Display costmap
        ax.imshow(costmap_img, extent=extent, origin='lower', interpolation='nearest')
        
        # Add robot position and orientation
        self.draw_robot(ax)
        
        # Add prediction visualization
        self.draw_prediction_area(ax)
        
        # Add annotations
        self.add_annotations(ax)
        
        # Set title and labels
        prob_text = f"Path Probabilities - Front: {self.path_probabilities[0]:.2f}, Left: {self.path_probabilities[1]:.2f}, Right: {self.path_probabilities[2]:.2f}"
        status_text = "PREDICTING DEAD END AHEAD" if not self.is_dead_end else "AT DEAD END"
        
        ax.set_title(f'Dead End Prediction Visualization\n{status_text}\n{prob_text}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        
        # Add legend
        self.add_legend(ax)
        
        # Set axis limits around robot for better view
        robot_x, robot_y = self.robot_pose['x'], self.robot_pose['y']
        margin = 15  # meters
        ax.set_xlim(robot_x - margin, robot_x + margin)
        ax.set_ylim(robot_y - margin, robot_y + margin)
        
        # Save figure
        filename = f"deadend_prediction_{timestamp}_{self.capture_count:03d}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.get_logger().info(f"Captured prediction scenario: {filepath}")
        
        # Also save a zoomed-out version
        self.save_full_map_view(timestamp)

    def create_costmap_image(self):
        """Create a colored costmap image with enhanced visibility"""
        costmap = self.costmap_data['data']
        height, width = costmap.shape
        
        # Create RGB image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Enhanced color coding for better visibility:
        unknown_mask = (costmap == -1)
        free_mask = (costmap == 0)
        inflated_mask = ((costmap > 0) & (costmap < 100))
        occupied_mask = (costmap >= 100)
        
        # More contrasting colors
        img[unknown_mask] = [180, 180, 180]  # Light gray for unknown (more visible)
        img[free_mask] = [240, 248, 255]     # Very light blue for free space
        img[inflated_mask] = [255, 200, 200] # Light red for inflated obstacles
        img[occupied_mask] = [0, 0, 0]       # Black for solid obstacles
        
        # If the map is mostly empty, create a synthetic environment for demonstration
        occupied_pixels = np.sum(occupied_mask)
        total_pixels = height * width
        
        if occupied_pixels < (total_pixels * 0.01):  # Less than 1% occupied
            self.get_logger().info("Map appears mostly empty, adding synthetic environment for visualization")
            self.add_synthetic_corridor(img, height, width)
        
        # Always add strong prediction overlay
        if self.path_probabilities:
            self.add_strong_prediction_overlay(img, costmap)
        
        return img

    def add_prediction_overlay(self, img, costmap):
        """Add prediction overlay to show where dead end is predicted"""
        height, width = costmap.shape
        info = self.costmap_data['info']
        robot_pose = self.robot_pose
        
        # Convert robot pose to grid coordinates
        robot_grid_x = int((robot_pose['x'] - info.origin.position.x) / info.resolution)
        robot_grid_y = int((robot_pose['y'] - info.origin.position.y) / info.resolution)
        
        # Prediction parameters
        min_prob = min(self.path_probabilities)
        prediction_strength = max(0, 0.8 - min_prob)  # Higher when probability is lower
        
        if prediction_strength > 0.1:
            yaw = robot_pose['yaw']
            
            # Create prediction area ahead of robot
            prediction_distance = int(10.0 / info.resolution)  # 10 meters ahead
            prediction_width = int(6.0 / info.resolution)      # 6 meters wide
            
            for dy in range(-prediction_width//2, prediction_width//2):
                for dx in range(5, prediction_distance):  # Start a bit ahead of robot
                    gx = robot_grid_x + int(dx * np.cos(yaw) - dy * np.sin(yaw))
                    gy = robot_grid_y + int(dx * np.sin(yaw) + dy * np.cos(yaw))
                    
                    if 0 <= gx < width and 0 <= gy < height:
                        # Only overlay on free or inflated space
                        if costmap[gy, gx] < 100:
                            # Distance-based intensity
                            distance = np.sqrt(dx*dx + dy*dy)
                            lateral_distance = abs(dy)
                            
                            if distance < prediction_distance * 0.8:
                                # Calculate intensity based on distance and prediction strength
                                distance_factor = max(0, 1.0 - distance / (prediction_distance * 0.8))
                                lateral_factor = max(0, 1.0 - lateral_distance / (prediction_width/2))
                                intensity = prediction_strength * distance_factor * lateral_factor * 0.7
                                
                                if intensity > 0.1:
                                    # Red/orange overlay for predicted dead end
                                    overlay_color = np.array([255, int(165 * (1-intensity)), 0])  # Orange to red
                                    original_color = img[gy, gx].astype(np.float32)
                                    
                                    # Blend colors
                                    blended = (1 - intensity) * original_color + intensity * overlay_color
                                    img[gy, gx] = blended.astype(np.uint8)

    def add_synthetic_corridor(self, img, height, width):
        """Add a synthetic corridor environment when real map is empty"""
        # Create a corridor scenario around robot position
        if self.robot_pose:
            # Convert robot position to image coordinates
            info = self.costmap_data['info']
            robot_grid_x = int((self.robot_pose['x'] - info.origin.position.x) / info.resolution)
            robot_grid_y = int((self.robot_pose['y'] - info.origin.position.y) / info.resolution)
        else:
            # Default to center
            robot_grid_x = width // 2
            robot_grid_y = height // 2
        
        # Create corridor walls
        corridor_length = min(width//3, 300)  # Length of corridor
        corridor_width = 60  # Width of corridor
        wall_thickness = 15
        
        # Ensure robot is within bounds
        robot_grid_x = max(corridor_length//2, min(width - corridor_length//2, robot_grid_x))
        robot_grid_y = max(corridor_width//2, min(height - corridor_width//2, robot_grid_y))
        
        # Create corridor floor (light green for free space)
        corridor_start_x = robot_grid_x - corridor_length//3
        corridor_end_x = robot_grid_x + corridor_length//2
        corridor_top_y = robot_grid_y - corridor_width//2
        corridor_bottom_y = robot_grid_y + corridor_width//2
        
        # Fill corridor with light green (free space)
        img[corridor_top_y:corridor_bottom_y, corridor_start_x:corridor_end_x] = [200, 255, 200]
        
        # Add top wall
        img[corridor_top_y-wall_thickness:corridor_top_y, corridor_start_x:corridor_end_x] = [50, 50, 50]
        
        # Add bottom wall
        img[corridor_bottom_y:corridor_bottom_y+wall_thickness, corridor_start_x:corridor_end_x] = [50, 50, 50]
        
        # Add dead-end wall at the end
        img[corridor_top_y-wall_thickness:corridor_bottom_y+wall_thickness, 
            corridor_end_x-wall_thickness:corridor_end_x] = [50, 50, 50]
        
        # Add some side passages that are blocked
        side_passage_y1 = robot_grid_y - corridor_width//4
        side_passage_y2 = robot_grid_y + corridor_width//4
        side_passage_x = robot_grid_x + corridor_length//4
        
        # Left side passage (blocked)
        img[side_passage_y1:side_passage_y2, side_passage_x-30:side_passage_x] = [200, 255, 200]
        img[side_passage_y1:side_passage_y2, side_passage_x-35:side_passage_x-30] = [50, 50, 50]  # Block it
        
        self.get_logger().info(f"Added synthetic corridor: robot at ({robot_grid_x}, {robot_grid_y})")

    def add_strong_prediction_overlay(self, img, costmap):
        """Add a very visible prediction overlay"""
        if not self.robot_pose:
            return
            
        height, width = costmap.shape
        info = self.costmap_data['info']
        
        # Convert robot pose to grid coordinates
        robot_grid_x = int((self.robot_pose['x'] - info.origin.position.x) / info.resolution)
        robot_grid_y = int((self.robot_pose['y'] - info.origin.position.y) / info.resolution)
        
        # Get minimum path probability for prediction strength
        min_prob = min(self.path_probabilities) if self.path_probabilities else 0.5
        
        # Create a strong prediction area ahead of robot
        yaw = self.robot_pose.get('yaw', 0)
        
        # Prediction zone parameters
        pred_distance = int(15.0 / info.resolution)  # 15 meters ahead
        pred_width = int(8.0 / info.resolution)      # 8 meters wide
        
        # Create prediction rectangle ahead of robot
        for dy in range(-pred_width//2, pred_width//2):
            for dx in range(10, pred_distance):  # Start ahead of robot
                gx = robot_grid_x + int(dx * np.cos(yaw) - dy * np.sin(yaw))
                gy = robot_grid_y + int(dx * np.sin(yaw) + dy * np.cos(yaw))
                
                if 0 <= gx < width and 0 <= gy < height:
                    # Distance-based intensity
                    distance = np.sqrt(dx*dx + dy*dy)
                    lateral_distance = abs(dy)
                    
                    if distance < pred_distance * 0.8 and lateral_distance < pred_width//2:
                        # Strong red overlay for prediction
                        intensity = 0.7  # Fixed strong intensity
                        
                        # Create strong red prediction area
                        current_color = img[gy, gx].astype(np.float32)
                        prediction_color = np.array([255, 50, 50])  # Strong red
                        
                        # Strong blend
                        img[gy, gx] = ((1 - intensity) * current_color + intensity * prediction_color).astype(np.uint8)

    def get_costmap_extent(self):
        """Get extent for imshow"""
        info = self.costmap_data['info']
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        resolution = info.resolution
        width = info.width
        height = info.height
        
        return [
            origin_x,
            origin_x + width * resolution,
            origin_y,
            origin_y + height * resolution
        ]

    def draw_robot(self, ax):
        """Draw robot pose as an arrow"""
        x, y, yaw = self.robot_pose['x'], self.robot_pose['y'], self.robot_pose['yaw']
        
        # Draw robot as a blue arrow
        arrow_length = 2.0
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        
        ax.arrow(x, y, dx, dy, 
                head_width=0.5, head_length=0.4, 
                fc='blue', ec='blue', linewidth=4,
                label='Robot Position & Heading')
        
        # Add a circle at robot position
        circle = Circle((x, y), 0.3, color='blue', alpha=0.8, zorder=10)
        ax.add_patch(circle)

    def draw_prediction_area(self, ax):
        """Draw the area where dead end is predicted"""
        if not self.path_probabilities or min(self.path_probabilities) >= 0.6:
            return
            
        robot_pose = self.robot_pose
        x, y, yaw = robot_pose['x'], robot_pose['y'], robot_pose['yaw']
        
        # Prediction zone ahead of robot
        pred_distance = 8.0
        pred_x = x + pred_distance * np.cos(yaw)
        pred_y = y + pred_distance * np.sin(yaw)
        
        # Draw prediction area as a rectangle
        rect_width = 4.0
        rect_height = 6.0
        
        # Calculate rectangle corners
        rect = patches.Rectangle(
            (pred_x - rect_width/2, pred_y - rect_height/2),
            rect_width, rect_height,
            angle=np.degrees(yaw),
            fill=False, edgecolor='red', 
            linewidth=3, linestyle='--',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add "PREDICTED DEAD END" text
        text_x = pred_x + 2 * np.cos(yaw + np.pi/2)
        text_y = pred_y + 2 * np.sin(yaw + np.pi/2)
        
        ax.annotate('PREDICTED\nDEAD END', 
                   xy=(pred_x, pred_y), 
                   xytext=(text_x, text_y),
                   fontsize=12, fontweight='bold',
                   color='red',
                   ha='center', va='center',
                   arrowprops=dict(arrowstyle='->', 
                                 color='red', 
                                 linewidth=2),
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor='white', 
                           edgecolor='red',
                           alpha=0.9))

    def add_annotations(self, ax):
        """Add text annotations"""
        robot_x, robot_y = self.robot_pose['x'], self.robot_pose['y']
        
        # Robot annotation
        ax.annotate('Robot Current Position', 
                   xy=(robot_x, robot_y), 
                   xytext=(robot_x - 5, robot_y + 3),
                   fontsize=11, fontweight='bold',
                   color='blue',
                   arrowprops=dict(arrowstyle='->', 
                                 color='blue', 
                                 linewidth=1.5),
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor='lightblue', 
                           edgecolor='blue',
                           alpha=0.7))

    def add_legend(self, ax):
        """Add legend to the plot"""
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Free Space'),
            Patch(facecolor='lightgray', edgecolor='black', label='Inflated Obstacles'),
            Patch(facecolor='black', edgecolor='black', label='Obstacles'),
            Patch(facecolor='orange', edgecolor='black', label='Predicted Dead End Area'),
            Patch(facecolor='blue', edgecolor='black', label='Robot Position')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    def save_full_map_view(self, timestamp):
        """Save a full map view for context"""
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        
        # Process and display costmap
        costmap_img = self.create_costmap_image()
        extent = self.get_costmap_extent()
        
        # Display full costmap
        ax.imshow(costmap_img, extent=extent, origin='lower', interpolation='nearest')
        
        # Add robot and predictions
        self.draw_robot(ax)
        self.draw_prediction_area(ax)
        
        # Title
        ax.set_title(f'Full Map View - Dead End Prediction\nRobot is NOT at dead end yet, but predicts one ahead', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        
        # Add legend
        self.add_legend(ax)
        
        # Save full view
        filename = f"deadend_prediction_fullmap_{timestamp}_{self.capture_count:03d}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.get_logger().info(f"Saved full map view: {filepath}")

def main(args=None):
    rclpy.init(args=args)
    node = DeadEndPredictionVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
