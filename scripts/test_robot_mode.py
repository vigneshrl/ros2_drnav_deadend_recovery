#!/usr/bin/env python3
"""
Test script to verify robot mode optimizations work correctly
"""

import rclpy
from rclpy.node import Node
import time
import psutil
import os

class RobotModeTest(Node):
    def __init__(self):
        super().__init__('robot_mode_test')
        
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        self.get_logger().info('🧪 Robot Mode Test Started')
        self.get_logger().info(f'📊 Initial Memory: {self.initial_memory:.1f} MB')
        
        # Create timer to monitor system resources
        self.create_timer(5.0, self.monitor_resources)
        
    def monitor_resources(self):
        current_time = time.time()
        runtime = current_time - self.start_time
        
        # Memory usage
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - self.initial_memory
        
        # CPU usage
        cpu_percent = self.process.cpu_percent()
        
        self.get_logger().info(f'''
╔═══════════ SYSTEM MONITOR ═══════════╗
║ ⏱️  Runtime: {runtime:.1f}s
║ 🧠 Memory: {current_memory:.1f} MB (+{memory_increase:+.1f} MB)
║ ⚡ CPU: {cpu_percent:.1f}%
╚═════════════════════════════════════╝''')
        
        # Alert if memory is growing too fast
        if memory_increase > 100:  # More than 100MB increase
            self.get_logger().warn('⚠️  High memory usage detected!')

def main():
    rclpy.init()
    node = RobotModeTest()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Test completed')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
