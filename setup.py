from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'map_contruct'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        ('share/' + package_name, ['package.xml']),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vicky',
    maintainer_email='your.email@example.com',
    description='ROS2 package',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            # Baselines
            'dwa_planner = map_contruct.baselines.dwa.dwa_planner:main',
            'mppi_planner = map_contruct.baselines.mppi.mppi_planner:main',
            'nav2_dwb_planner = map_contruct.baselines.nav2_dwb.nav2_dwb_planner:main',
            
            # Goal Generator
            'goal_generator = map_contruct.goal_generator.goal_generator:main',
            
            # Scripts - Inference
            'infer_vis = map_contruct.scripts.inference.infer_vis:main',
            
            # Scripts - Utilities
            'pointcloud_segmenter = map_contruct.scripts.utilities.pointcloud_segmenter:main',
            'cost_layer_processor = map_contruct.scripts.utilities.cost_layer_processor:main',
            'odom_tf_broadcaster = map_contruct.scripts.utilities.odom_tf_brodcaster:main',
            'dram_heatmap_viz = map_contruct.scripts.utilities.dram_heatmap_viz:main',
            'dram_risk_map = map_contruct.scripts.utilities.dram_risk_map:main',
            'slam = map_contruct.scripts.utilities.slam:main',
            
            # Scripts - Visualization
            'comprehensive_deadend_visualizer = map_contruct.scripts.viz.comprehensive_deadend_visualizer:main',
            'predictive_deadend_visualizer = map_contruct.scripts.viz.predictive_deadend_visualizer:main',
            'deadend_prediction_visualizer = map_contruct.scripts.viz.deadend_prediction_visualizer:main',
            'method_comparison_analyzer = map_contruct.scripts.viz.method_comparison_analyzer:main',
            'simple_method_comparison = map_contruct.scripts.viz.simple_method_comparison:main',
            'bag_metrics = map_contruct.scripts.viz.bag_metrics:main',
            'evaluation_framework = map_contruct.scripts.viz.evaluation_framework:main',
            'enhanced_evaluation_framework = map_contruct.scripts.viz.enhanced_evaluation_framework:main',
            
            # Old files (kept for backward compatibility)
            'inference = map_contruct.OLD_FILES.inference:main',
            'single_camera_inference = map_contruct.OLD_FILES.single_camera_inference:main',
            'dwa_lidar_controller = map_contruct.OLD_FILES.dwa_lidar_controller:main',
            'mppi_lidar_controller = map_contruct.OLD_FILES.mppi_lidar_controller:main',
            'dwa_rosbag_planner = map_contruct.OLD_FILES.dwa_rosbag_planner:main',
            'dummy_dead_end_detector = map_contruct.OLD_FILES.dummy_dead_end_detector:main',
            'vanilla_dwa_planner = map_contruct.OLD_FILES.vanilla_dwa_planner:main',
            'simple_goal_generator = map_contruct.OLD_FILES.simple_goal_generator:main',
            'dram_aware_goal_generator = map_contruct.OLD_FILES.dram_aware_goal_generator:main',
        ], 
    },
)
