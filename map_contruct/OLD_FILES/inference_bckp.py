#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Point
import numpy as np
import torch
from torchvision import transforms
from PIL import Image as PILImage
import cv2
import time
from typing import Dict, List, Optional

# Import your model and recovery points
from model_CA import DeadEndDetectionModel
from recovery_points import RecoveryPointManager

class DeadEndDetectionNode(Node):
    def __init__(self):
        super().__init__('dead_end_detection_node')
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeadEndDetectionModel()
        
        # Load model weights
        model_path = '/media/mrvik/ROGZ/Ubuntu_files/DRaM/RAL2025/saved_models_latest/model_best.pth'  # Update this path
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize recovery point manager
        self.recovery_manager = RecoveryPointManager(confidence_threshold=0.5)
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize subscribers
        self.front_cam_sub = self.create_subscription(
            Image, 
            '/argus/ar0234_front_left/image_raw', 
            self.front_cam_callback, 
            10
        )
        self.left_cam_sub = self.create_subscription(
            Image, 
            '/argus/ar0234_side_left/image_raw', 
            self.left_cam_callback, 
            10
        )
        self.right_cam_sub = self.create_subscription(
            Image, 
            '/argus/ar0234_side_right/image_raw', 
            self.right_cam_callback, 
            10
        )
        self.front_lidar_sub = self.create_subscription(
            PointCloud2, 
            '/lidar/front/points', 
            self.front_lidar_callback, 
            10
        )
        self.left_lidar_sub = self.create_subscription(
            PointCloud2, 
            '/lidar/left/points', 
            self.left_lidar_callback, 
            10
        )
        self.right_lidar_sub = self.create_subscription(
            PointCloud2, 
            '/lidar/right/points', 
            self.right_lidar_callback, 
            10
        )
        
        # Initialize publishers
        self.dead_end_pub = self.create_publisher(Bool, '/dead_end_detection/is_dead_end', 10)
        self.path_status_pub = self.create_publisher(Float32MultiArray, '/dead_end_detection/path_status', 10)
        self.recovery_point_pub = self.create_publisher(Float32MultiArray, '/dead_end_detection/recovery_points', 10)
        
        # Initialize message storage
        self.front_img = None
        self.left_img = None
        self.right_img = None
        self.front_lidar = None
        self.left_lidar = None
        self.right_lidar = None
        
        # Initialize synchronization
        self.last_processed_time = 0
        self.processing_interval = 0.1  # Process every 100ms
        
        self.get_logger().info('Dead End Detection Node initialized')

    def front_cam_callback(self, msg: Image):
        """Process front camera image"""
        try:
            # Convert ROS Image to PIL Image
            img = self.ros_image_to_pil(msg)
            self.front_img = self.transform(img)
        except Exception as e:
            self.get_logger().error(f'Error processing front camera image: {e}')

    def left_cam_callback(self, msg: Image):
        """Process left camera image"""
        try:
            img = self.ros_image_to_pil(msg)
            self.left_img = self.transform(img)
        except Exception as e:
            self.get_logger().error(f'Error processing left camera image: {e}')

    def right_cam_callback(self, msg: Image):
        """Process right camera image"""
        try:
            img = self.ros_image_to_pil(msg)
            self.right_img = self.transform(img)
        except Exception as e:
            self.get_logger().error(f'Error processing right camera image: {e}')

    def front_lidar_callback(self, msg: PointCloud2):
        """Process front LiDAR data"""
        try:
            points = self.ros_pointcloud_to_numpy(msg)  # Shape: (3, N)
            self.front_lidar = torch.from_numpy(points).float()
            self.get_logger().debug(f'Received front LiDAR: {points.shape[1]} points, shape: {points.shape}')
        except Exception as e:
            self.get_logger().error(f'Error processing front LiDAR data: {e}')

    def left_lidar_callback(self, msg: PointCloud2):
        """Process left LiDAR data"""
        try:
            points = self.ros_pointcloud_to_numpy(msg)  # Shape: (3, N)
            self.left_lidar = torch.from_numpy(points).float()
            self.get_logger().debug(f'Received left LiDAR: {points.shape[1]} points, shape: {points.shape}')
        except Exception as e:
            self.get_logger().error(f'Error processing left LiDAR data: {e}')

    def right_lidar_callback(self, msg: PointCloud2):
        """Process right LiDAR data"""
        try:
            points = self.ros_pointcloud_to_numpy(msg)  # Shape: (3, N)
            self.right_lidar = torch.from_numpy(points).float()
            self.get_logger().debug(f'Received right LiDAR: {points.shape[1]} points, shape: {points.shape}')
        except Exception as e:
            self.get_logger().error(f'Error processing right LiDAR data: {e}')

    def ros_image_to_pil(self, msg: Image) -> PILImage.Image:
        """Convert ROS Image message to PIL Image"""
        # Convert ROS Image to OpenCV format
        cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        # Convert BGR to RGB
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        return PILImage.fromarray(cv_image)

    def ros_pointcloud_to_numpy(self, msg: PointCloud2) -> np.ndarray:
        """Convert ROS PointCloud2 message to numpy array for PointNet"""
        import sensor_msgs_py.point_cloud2 as pc2
        
        # Convert PointCloud2 to list of points
        points_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        
        if not points_list:
            # Return empty point cloud with correct shape
            return np.zeros((3, 0), dtype=np.float32)
        
        # Convert structured array to regular float array
        if len(points_list) > 0:
            # Handle both structured and regular arrays
            if isinstance(points_list[0], (list, tuple)):
                # Regular list/tuple format
                points = np.array(points_list, dtype=np.float32)
            else:
                # Structured array format - extract x, y, z
                structured_array = np.array(points_list)
                points = np.column_stack([
                    structured_array['x'],
                    structured_array['y'], 
                    structured_array['z']
                ]).astype(np.float32)
        else:
            return np.zeros((3, 0), dtype=np.float32)
        
        # Convert to shape (3, N) for PointNet
        points = points.T  # Shape: (3, N)
        
        return points

    def process_data(self):
        """Process all sensor data and run model inference"""
        current_time = time.time()
        
        # Check if enough time has passed since last processing
        if current_time - self.last_processed_time < self.processing_interval:
            return
        
        # Check if we have all required data
        if not all([self.front_img is not None, self.left_img is not None, 
                   self.right_img is not None, self.front_lidar is not None,
                   self.left_lidar is not None, self.right_lidar is not None]):
            return
        
        try:
            # Prepare input tensors
            front_img = self.front_img.unsqueeze(0).to(self.device)
            left_img = self.left_img.unsqueeze(0).to(self.device)
            right_img = self.right_img.unsqueeze(0).to(self.device)
            
            # Prepare LiDAR tensors - PointNet expects [batch_size, 3, num_points]
            front_lidar = self.front_lidar.unsqueeze(0).to(self.device)  # [1, 3, N]
            left_lidar = self.left_lidar.unsqueeze(0).to(self.device)   # [1, 3, N]
            right_lidar = self.right_lidar.unsqueeze(0).to(self.device) # [1, 3, N]
            
            self.get_logger().debug(f'LiDAR tensor shapes: front={front_lidar.shape}, left={left_lidar.shape}, right={right_lidar.shape}')
            
            # Run model inference
            with torch.no_grad():
                outputs = self.model(
                    front_img, right_img, left_img,
                    front_lidar, right_lidar, left_lidar
                )
            
            # Process recovery points
            point = self.recovery_manager.update(outputs, current_time)
            
            # Get recovery strategy
            strategy = self.recovery_manager.get_recovery_strategy(outputs)
            
            # Publish results
            self.publish_results(outputs, strategy)
            
            # Clear processed data
            self.clear_data()
            
            # Update last processed time
            self.last_processed_time = current_time
            
        except Exception as e:
            self.get_logger().error(f'Error in process_data: {e}')

    def publish_results(self, outputs: Dict[str, torch.Tensor], strategy: dict):
        """Publish model outputs and recovery points"""
        # Publish dead end status
        dead_end_msg = Bool()
        dead_end_msg.data = strategy['is_dead_end']
        self.dead_end_pub.publish(dead_end_msg)
        
        # Publish path status
        path_status = torch.sigmoid(outputs['path_status']).cpu().numpy()
        path_msg = Float32MultiArray()
        path_msg.data = path_status.flatten().tolist()
        self.path_status_pub.publish(path_msg)
        
        # Publish recovery points
        if strategy['recovery_points']:
            recovery_msg = Float32MultiArray()
            recovery_data = []
            
            # Add last recovery point data
            if strategy['recovery_points']['last']:
                last_point = strategy['recovery_points']['last']
                recovery_data.extend([
                    last_point.index,
                    last_point.rank,
                    *last_point.open_directions,
                    *last_point.confidence
                ])
            
            # Add best recovery point data
            if strategy['recovery_points']['best']:
                best_point = strategy['recovery_points']['best']
                recovery_data.extend([
                    best_point.index,
                    best_point.rank,
                    *best_point.open_directions,
                    *best_point.confidence
                ])
            
            recovery_msg.data = recovery_data
            self.recovery_point_pub.publish(recovery_msg)

    def clear_data(self):
        """Clear processed sensor data"""
        self.front_img = None
        self.left_img = None
        self.right_img = None
        self.front_lidar = None
        self.left_lidar = None
        self.right_lidar = None

def main(args=None):
    rclpy.init(args=args)
    node = DeadEndDetectionNode()
    
    try:
        # Create timer for processing data
        node.create_timer(0.1, node.process_data)  # 10Hz
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 