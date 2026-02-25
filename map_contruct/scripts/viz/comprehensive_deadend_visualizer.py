#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray
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
from matplotlib.patches import Circle, Rectangle
from datetime import datetime
from PIL import Image as PILImage
import cv_bridge

class ComprehensiveDeadEndVisualizer(Node):
    def __init__(self):
        super().__init__('comprehensive_deadend_visualizer')
        
        # Parameters
        self.output_dir = '/home/mrvik/dram_ws/comprehensive_figures'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # CV Bridge for image conversion
        self.bridge = cv_bridge.CvBridge()
        
        # Subscriptions - Use DRAM exploration map instead of SLAM map
        self.costmap_sub = self.create_subscription(
            MarkerArray,
            '/dram_exploration_map',
            self.dram_costmap_callback,
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
        
        # No camera subscriptions needed for this version
        
        # Transform listener for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Data storage
        self.dram_costmap_data = None  # DRAM exploration map data
        self.robot_pose = None
        self.is_dead_end = False
        self.path_probabilities = None  # [front, left, right]
        
        # Capture control
        self.last_capture_time = 0
        self.capture_interval = 3.0  # Capture every 3 seconds
        self.capture_count = 0
        
        # Timer for periodic capture
        self.create_timer(2.0, self.check_and_capture)
        
        self.get_logger().info(f'Comprehensive Dead End Visualizer initialized')
        self.get_logger().info(f'Output directory: {self.output_dir}')

    def dram_costmap_callback(self, msg):
        """Store DRAM exploration map data"""
        # Parse DRAM exploration map from MarkerArray
        self.dram_costmap_data = {
            'markers': msg.markers,
            'timestamp': msg.markers[0].header.stamp if msg.markers else None
        }
        self.get_logger().debug(f'Received DRAM exploration map with {len(msg.markers)} markers')

    def dead_end_callback(self, msg):
        """Store dead end detection status"""
        self.is_dead_end = msg.data

    def path_status_callback(self, msg):
        """Store path probability data"""
        if len(msg.data) >= 3:
            self.path_probabilities = list(msg.data[:3])  # [front, left, right]

    # Camera callbacks removed - using single DRAM costmap visualization

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
        
        # Only capture if we have all necessary data
        if not self.has_complete_data():
            return
            
        min_prob = min(self.path_probabilities) if self.path_probabilities else 1.0
        
        # Capture when we have interesting predictions
        if (not self.is_dead_end and 
            self.path_probabilities and 
            min_prob < 0.85 and  # Prediction threshold
            current_time - self.last_capture_time > self.capture_interval):
            
            self.get_logger().info(f'Capturing comprehensive scenario: min_prob={min_prob:.3f}')
            self.capture_comprehensive_scenario()
            self.last_capture_time = current_time

    def has_complete_data(self):
        """Check if we have all necessary data"""
        return (self.dram_costmap_data is not None and 
                self.get_robot_pose() and
                self.path_probabilities is not None)

    def capture_comprehensive_scenario(self):
        """Create DRAM costmap visualization with robot pose and predictions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.capture_count += 1
        
        # Create single large figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Create DRAM costmap visualization
        self.create_dram_costmap_plot(ax)
        
        # Add robot position and prediction boxes
        self.add_robot_and_predictions(ax)
        
        # Set plot properties with OPEN/BLOCKED labels
        path_labels = []
        path_labels.append(f"Front: {'BLOCKED' if self.path_probabilities[0] < 0.7 else 'OPEN'}")
        path_labels.append(f"Left: {'BLOCKED' if self.path_probabilities[1] < 0.7 else 'OPEN'}")  
        path_labels.append(f"Right: {'BLOCKED' if self.path_probabilities[2] < 0.7 else 'OPEN'}")
        
        status_text = "PREDICTING DEAD END AHEAD" if not self.is_dead_end else "AT DEAD END"
        path_text = " | ".join(path_labels)
        
        ax.set_title(f'DRAM Exploration Map - Dead-End Prediction\n{status_text}\n{path_text}', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('X (meters)', fontsize=14)
        ax.set_ylabel('Y (meters)', fontsize=14)
        
        # Add legend
        self.add_dram_legend(ax)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filename = f"dram_costmap_{timestamp}_{self.capture_count:03d}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.get_logger().info(f"Saved DRAM costmap visualization: {filepath}")

    def create_dram_costmap_plot(self, ax):
        """Create DRAM exploration costmap plot"""
        if not self.dram_costmap_data or not self.dram_costmap_data['markers']:
            self.get_logger().warn("No DRAM costmap data available")
            return
            
        # Process DRAM exploration map markers
        for marker in self.dram_costmap_data['markers']:
            if marker.ns == "exploration_heatmap":
                # Process heatmap points
                for i, point in enumerate(marker.points):
                    x = point.x
                    y = point.y
                    
                    # Get color information
                    if i < len(marker.colors):
                        color = marker.colors[i]
                        
                        # Determine point type based on color
                        if color.b > 0.5:  # Recovery points (blue component)
                            if color.r > 0.4:  # Purple (3+ paths)
                                ax.scatter(x, y, c='purple', s=100, marker='s', alpha=0.8, label='Recovery Point (3+ paths)')
                            elif color.b > 0.7:  # Dark blue (2 paths)  
                                ax.scatter(x, y, c='blue', s=80, marker='s', alpha=0.8, label='Recovery Point (2 paths)')
                            else:  # Light blue (1 path)
                                ax.scatter(x, y, c='lightblue', s=60, marker='s', alpha=0.8, label='Recovery Point (1 path)')
                        else:
                            # Safety heatmap points
                            if color.g > 0.5:  # Green = safe
                                ax.scatter(x, y, c='green', s=30, marker='o', alpha=0.6, label='Safe Area')
                            else:  # Red = unsafe
                                ax.scatter(x, y, c='red', s=30, marker='o', alpha=0.6, label='Unsafe Area')
        
        # Set axis properties
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    def create_comprehensive_map(self):
        """Create a comprehensive map combining occupancy grid and costmap"""
        costmap = self.costmap_data['data']
        height, width = costmap.shape
        
        # Create RGB image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Base occupancy grid coloring
        unknown_mask = (costmap == -1)
        free_mask = (costmap == 0)
        inflated_mask = ((costmap > 0) & (costmap < 100))
        occupied_mask = (costmap >= 100)
        
        # Enhanced colors for better visibility
        img[unknown_mask] = [169, 169, 169]  # Dark gray for unknown
        img[free_mask] = [245, 245, 245]     # Light gray for free space
        img[inflated_mask] = [255, 165, 0]   # Orange for inflated obstacles
        img[occupied_mask] = [0, 0, 0]       # Black for obstacles
        
        # Add synthetic environment if map is mostly empty
        occupied_pixels = np.sum(occupied_mask)
        total_pixels = height * width
        
        if occupied_pixels < (total_pixels * 0.01):  # Less than 1% occupied
            self.get_logger().info("Creating synthetic environment for comprehensive visualization")
            self.add_comprehensive_synthetic_environment(img, height, width)
        
        # Add prediction overlay
        if self.path_probabilities:
            self.add_prediction_boxes(img, costmap)
        
        return img

    def add_comprehensive_synthetic_environment(self, img, height, width):
        """Add a realistic synthetic environment"""
        # Get robot position in grid coordinates
        if self.robot_pose:
            info = self.costmap_data['info']
            robot_grid_x = int((self.robot_pose['x'] - info.origin.position.x) / info.resolution)
            robot_grid_y = int((self.robot_pose['y'] - info.origin.position.y) / info.resolution)
        else:
            robot_grid_x = width // 2
            robot_grid_y = height // 2
        
        # Ensure robot is within bounds
        robot_grid_x = max(100, min(width - 100, robot_grid_x))
        robot_grid_y = max(100, min(height - 100, robot_grid_y))
        
        # Create a more complex environment
        # Main corridor
        corridor_length = 200
        corridor_width = 80
        wall_thickness = 20
        
        # Main corridor (horizontal)
        start_x = robot_grid_x - 60
        end_x = robot_grid_x + corridor_length
        top_y = robot_grid_y - corridor_width//2
        bottom_y = robot_grid_y + corridor_width//2
        
        # Fill main corridor with light blue (navigable space)
        img[top_y:bottom_y, start_x:end_x] = [200, 230, 255]
        
        # Add walls
        img[top_y-wall_thickness:top_y, start_x:end_x] = [50, 50, 50]  # Top wall
        img[bottom_y:bottom_y+wall_thickness, start_x:end_x] = [50, 50, 50]  # Bottom wall
        
        # Add dead-end wall
        dead_end_x = end_x - 10
        img[top_y-wall_thickness:bottom_y+wall_thickness, dead_end_x:end_x] = [50, 50, 50]
        
        # Add side corridors (some blocked, some open)
        side_corridor_1_x = robot_grid_x + 50
        side_corridor_1_top = top_y - 60
        side_corridor_1_bottom = top_y
        
        # Right side corridor (blocked)
        img[side_corridor_1_top:side_corridor_1_bottom, side_corridor_1_x:side_corridor_1_x+40] = [200, 230, 255]
        img[side_corridor_1_top:side_corridor_1_bottom, side_corridor_1_x+40:side_corridor_1_x+50] = [50, 50, 50]  # Block it
        
        # Left side corridor (open but leads to dead end)
        side_corridor_2_x = robot_grid_x + 80
        side_corridor_2_top = bottom_y
        side_corridor_2_bottom = bottom_y + 60
        
        img[side_corridor_2_top:side_corridor_2_bottom, side_corridor_2_x:side_corridor_2_x+40] = [200, 230, 255]
        img[side_corridor_2_bottom-10:side_corridor_2_bottom, side_corridor_2_x:side_corridor_2_x+40] = [50, 50, 50]  # Dead end
        
        # Add some furniture/obstacles
        furniture_x = robot_grid_x + 20
        furniture_y = robot_grid_y - 10
        img[furniture_y:furniture_y+20, furniture_x:furniture_x+15] = [139, 69, 19]  # Brown furniture

    def add_prediction_boxes(self, img, costmap):
        """Add prediction boxes/areas to the map"""
        if not self.robot_pose:
            return
            
        height, width = costmap.shape
        info = self.costmap_data['info']
        
        # Convert robot pose to grid coordinates
        robot_grid_x = int((self.robot_pose['x'] - info.origin.position.x) / info.resolution)
        robot_grid_y = int((self.robot_pose['y'] - info.origin.position.y) / info.resolution)
        
        yaw = self.robot_pose.get('yaw', 0)
        min_prob = min(self.path_probabilities) if self.path_probabilities else 0.5
        
        # Create prediction areas for each direction
        self.add_directional_prediction_boxes(img, robot_grid_x, robot_grid_y, yaw, width, height)

    def add_directional_prediction_boxes(self, img, robot_x, robot_y, yaw, width, height):
        """Add prediction boxes for front, left, right directions"""
        box_size = 40  # Size of prediction box
        distance_ahead = 80  # Distance ahead to place boxes
        
        # Front prediction box
        front_x = robot_x + int(distance_ahead * np.cos(yaw))
        front_y = robot_y + int(distance_ahead * np.sin(yaw))
        
        if self.path_probabilities[0] < 0.7:  # Front path blocked
            self.draw_prediction_box(img, front_x, front_y, box_size, [255, 0, 0], width, height)  # Red
        
        # Left prediction box  
        left_yaw = yaw + np.pi/2
        left_x = robot_x + int(distance_ahead * 0.7 * np.cos(left_yaw))
        left_y = robot_y + int(distance_ahead * 0.7 * np.sin(left_yaw))
        
        if self.path_probabilities[1] < 0.7:  # Left path blocked
            self.draw_prediction_box(img, left_x, left_y, box_size, [255, 100, 0], width, height)  # Orange
        
        # Right prediction box
        right_yaw = yaw - np.pi/2  
        right_x = robot_x + int(distance_ahead * 0.7 * np.cos(right_yaw))
        right_y = robot_y + int(distance_ahead * 0.7 * np.sin(right_yaw))
        
        if self.path_probabilities[2] < 0.7:  # Right path blocked
            self.draw_prediction_box(img, right_x, right_y, box_size, [255, 150, 0], width, height)  # Light orange

    def draw_prediction_box(self, img, center_x, center_y, size, color, width, height):
        """Draw a prediction box at specified location"""
        half_size = size // 2
        
        # Ensure box is within image bounds
        x1 = max(0, center_x - half_size)
        x2 = min(width, center_x + half_size)
        y1 = max(0, center_y - half_size)
        y2 = min(height, center_y + half_size)
        
        if x2 > x1 and y2 > y1:
            # Fill box with semi-transparent color
            img[y1:y2, x1:x2] = color
            
            # Add border
            border_thickness = 3
            img[y1:y1+border_thickness, x1:x2] = [255, 255, 255]  # Top border
            img[y2-border_thickness:y2, x1:x2] = [255, 255, 255]  # Bottom border
            img[y1:y2, x1:x1+border_thickness] = [255, 255, 255]  # Left border
            img[y1:y2, x2-border_thickness:x2] = [255, 255, 255]  # Right border

    def add_robot_and_predictions(self, ax):
        """Add robot position and prediction annotations to the plot"""
        if not self.robot_pose:
            return
            
        x, y, yaw = self.robot_pose['x'], self.robot_pose['y'], self.robot_pose['yaw']
        
        # Draw robot as a large blue arrow
        arrow_length = 3.0
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        
        ax.arrow(x, y, dx, dy, 
                head_width=0.8, head_length=0.6, 
                fc='blue', ec='blue', linewidth=5,
                label='Robot Position')
        
        # Add robot circle
        robot_circle = Circle((x, y), 0.5, color='blue', alpha=0.8, zorder=10)
        ax.add_patch(robot_circle)
        
        # Add prediction boxes as rectangles on the plot
        if self.path_probabilities:
            distance_ahead = 8.0
            box_size = 2.0
            
            # Front prediction
            if self.path_probabilities[0] < 0.7:
                front_x = x + distance_ahead * np.cos(yaw)
                front_y = y + distance_ahead * np.sin(yaw)
                
                front_rect = Rectangle((front_x - box_size/2, front_y - box_size/2), 
                                     box_size, box_size,
                                     angle=np.degrees(yaw),
                                     facecolor='red', alpha=0.7, 
                                     edgecolor='white', linewidth=2)
                ax.add_patch(front_rect)
                
                # Add label
                ax.annotate('FRONT\nBLOCKED', xy=(front_x, front_y), 
                           xytext=(front_x + 2, front_y + 2),
                           fontsize=10, fontweight='bold', color='red',
                           ha='center', va='center',
                           arrowprops=dict(arrowstyle='->', color='red'),
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Similar for left and right predictions
            self.add_side_predictions(ax, x, y, yaw, distance_ahead, box_size)

    def add_side_predictions(self, ax, x, y, yaw, distance_ahead, box_size):
        """Add left and right prediction boxes"""
        # Left prediction
        if self.path_probabilities[1] < 0.7:
            left_yaw = yaw + np.pi/2
            left_x = x + distance_ahead * 0.7 * np.cos(left_yaw)
            left_y = y + distance_ahead * 0.7 * np.sin(left_yaw)
            
            left_rect = Rectangle((left_x - box_size/2, left_y - box_size/2), 
                                box_size, box_size,
                                facecolor='orange', alpha=0.7, 
                                edgecolor='white', linewidth=2)
            ax.add_patch(left_rect)
        
        # Right prediction
        if self.path_probabilities[2] < 0.7:
            right_yaw = yaw - np.pi/2
            right_x = x + distance_ahead * 0.7 * np.cos(right_yaw)
            right_y = y + distance_ahead * 0.7 * np.sin(right_yaw)
            
            right_rect = Rectangle((right_x - box_size/2, right_y - box_size/2), 
                                 box_size, box_size,
                                 facecolor='darkorange', alpha=0.7, 
                                 edgecolor='white', linewidth=2)
            ax.add_patch(right_rect)

    def add_camera_images(self, ax_front, ax_left, ax_right):
        """Add camera images to the subplots"""
        # Front camera
        if self.front_image is not None:
            ax_front.imshow(self.front_image)
            ax_front.set_title('Front Camera', fontsize=10, fontweight='bold')
            prob_text = f'P: {self.path_probabilities[0]:.2f}' if self.path_probabilities else 'P: N/A'
            ax_front.text(0.02, 0.98, prob_text, transform=ax_front.transAxes, 
                         fontsize=12, fontweight='bold', color='white',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7),
                         verticalalignment='top')
        ax_front.axis('off')
        
        # Left camera
        if self.left_image is not None:
            ax_left.imshow(self.left_image)
            ax_left.set_title('Left Camera', fontsize=10, fontweight='bold')
            prob_text = f'P: {self.path_probabilities[1]:.2f}' if self.path_probabilities else 'P: N/A'
            ax_left.text(0.02, 0.98, prob_text, transform=ax_left.transAxes, 
                        fontsize=12, fontweight='bold', color='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7),
                        verticalalignment='top')
        ax_left.axis('off')
        
        # Right camera
        if self.right_image is not None:
            ax_right.imshow(self.right_image)
            ax_right.set_title('Right Camera', fontsize=10, fontweight='bold')
            prob_text = f'P: {self.path_probabilities[2]:.2f}' if self.path_probabilities else 'P: N/A'
            ax_right.text(0.02, 0.98, prob_text, transform=ax_right.transAxes, 
                         fontsize=12, fontweight='bold', color='white',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                         verticalalignment='top')
        ax_right.axis('off')

    def add_dram_legend(self, ax):
        """Add DRAM costmap legend"""
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', label='Safe Area'),
            Patch(facecolor='red', edgecolor='black', label='Unsafe Area'),
            Patch(facecolor='purple', edgecolor='black', label='Recovery Point (3+ paths)'),
            Patch(facecolor='blue', edgecolor='black', label='Recovery Point (2 paths)'),
            Patch(facecolor='lightblue', edgecolor='black', label='Recovery Point (1 path)'),
            Patch(facecolor='red', edgecolor='white', label='Predicted BLOCKED'),
            Patch(facecolor='blue', edgecolor='black', label='Robot Position')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

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

def main(args=None):
    rclpy.init(args=args)
    node = ComprehensiveDeadEndVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
