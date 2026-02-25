#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BagMetricsNode — run alongside rosbag2 playback.
Computes per-run metrics and appends a CSV row.

Metrics:
- success (0/1): reached goal within goal_tol_m before timeout_s
- travel_m: path length from /odom (fallback: integrate |v| from /cmd_vel)
- duration_s: (last_msg_time - first_msg_time)
- inf_ms_mean, inf_ms_std: from /dram/inference_ms (optional)
- planner_ms_mean, planner_ms_std: from /planner/step_ms (optional; Nav2/DWA/MPPI loop time)

Params (declare or pass via CLI):
- method: "DRaM+DWA" | "MPPI" | "DWA" | "Nav2-DWB"
- scenario: e.g., "S1"
- goal_x, goal_y (float)
- timeout_s (float, default 30.0)
- goal_tol_m (float, default 0.75)
- output_csv (str, default "results.csv")
- odom_topic (str, default "/odom")
- cmd_vel_topic (str, default "/cmd_vel")
- inf_topic (str, default "/dram/inference_ms")
- planner_ms_topic (str, default "/planner/step_ms")
- stop_on_success (bool, default True)

Notes:
- Works with sim time; uses message timestamps (header.stamp when available).
- Terminates itself when timeout or success (if stop_on_success).
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import math, os, csv, statistics as stats
from typing import Optional

def ensure_header(path, fieldnames):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    if not exists:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

class BagMetricsNode(Node):
    def __init__(self):
        super().__init__("bag_metrics_node")

        # ---- Parameters ----
        self.declare_parameters("", [
            ("method", "UNKNOWN"),
            ("scenario", "S?"),
            ("goal_x", 0.0),
            ("goal_y", 0.0),
            ("timeout_s", 30.0),
            ("goal_tol_m", 0.75),
            ("output_csv", "results.csv"),
            ("odom_topic", "/odom_lidar"),
            ("cmd_vel_topic", "/cmd_vel"),
            ("inf_topic", ""),
            ("planner_ms_topic", ""),
            ("stop_on_success", True),
        ])

        # Get parameter values with defaults
        self.method   = self.get_parameter("method").value
        self.scenario = self.get_parameter("scenario").value
        self.goal_x   = self.get_parameter("goal_x").value
        self.goal_y   = self.get_parameter("goal_y").value
        self.timeout_s= self.get_parameter("timeout_s").value
        self.goal_tol = self.get_parameter("goal_tol_m").value
        self.output_csv = self.get_parameter("output_csv").value
        self.odom_topic   = self.get_parameter("odom_topic").value
        self.cmd_vel_topic= self.get_parameter("cmd_vel_topic").value
        self.inf_topic    = self.get_parameter("inf_topic").value
        self.planner_ms_topic = self.get_parameter("planner_ms_topic").value
        self.stop_on_success = self.get_parameter("stop_on_success").value

        # ---- Buffers ----
        self.odom_buf = []      # (t, x, y)
        self.cmd_buf  = []      # (t, |v|)
        self.inf_buf  = []      # ms
        self.plan_buf = []      # ms

        self.start_t: Optional[float] = None
        self.last_t: Optional[float] = None
        self.success = 0
        self.success_t: Optional[float] = None

        # ---- Subscribers ----
        # Best effort QoS for odometry (more reliable with bag playback)
        # BEST_EFFORT is better for bag data as it doesn't require reliable delivery
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.cb_odom, odom_qos)
        self.sub_cmd  = self.create_subscription(Twist, self.cmd_vel_topic, self.cb_cmd, 50)
        
        # Optional subscribers (only if topics exist)
        self.sub_inf = None
        self.sub_plnr = None
        
        if self.inf_topic:
            self.sub_inf = self.create_subscription(Float32, self.inf_topic, self.cb_inf, 10)
        if self.planner_ms_topic:
            self.sub_plnr = self.create_subscription(Float32, self.planner_ms_topic, self.cb_plnr, 10)
        
        # Track subscription status
        self.odom_received = False
        self.cmd_received = False

        # Periodic watchdog to enforce timeout and finalize
        self.timer = self.create_timer(0.2, self.watchdog)

        # Prepare CSV header
        self.fields = [
            "method","scenario","success","travel_m","duration_s",
            "inf_ms_mean","inf_ms_std","planner_ms_mean","planner_ms_std"
        ]
        ensure_header(self.output_csv, self.fields)

        self.get_logger().info(
            f"[BagMetricsNode] method={self.method}, scenario={self.scenario}, "
            f"goal=({self.goal_x:.2f},{self.goal_y:.2f}), timeout={self.timeout_s}s, "
            f"odom={self.odom_topic}, cmd={self.cmd_vel_topic}"
        )
        if self.inf_topic:
            self.get_logger().info(f"  inf_topic={self.inf_topic}")
        if self.planner_ms_topic:
            self.get_logger().info(f"  planner_ms_topic={self.planner_ms_topic}")
        
        # Log available topics for debugging
        self.get_logger().info("Available topics: /odom_lidar, /cmd_vel, /scan, /map, /move_base_simple/goal")
        
        # Log current ROS2 time source
        current_time = self.get_clock().now()
        self.get_logger().info(f"🕐 ROS2 time source: {current_time.nanoseconds * 1e-9:.3f}s")
        
        # Check if this is simulation time
        try:
            from rclpy.clock import ClockType
            if self.get_clock().clock_type == ClockType.ROS_TIME:
                self.get_logger().info("✅ Using ROS simulation time")
            else:
                self.get_logger().warn("⚠️ Using system wall clock time")
        except:
            self.get_logger().info("🔍 Clock type detection failed")

    # -------- Callbacks --------
    def msg_time(self, header_stamp: Optional[Time]=None) -> float:
        if header_stamp is not None and hasattr(header_stamp, 'sec'):
            time_sec = float(header_stamp.sec) + float(header_stamp.nanosec) * 1e-9
            time_source = "header_stamp"
        else:
            # Fallback to node time (sim or wall)
            time_sec = self.get_clock().now().nanoseconds * 1e-9
            time_source = "node_clock"
        
        # Debug logging for first few messages
        if not hasattr(self, 'debug_msg_count'):
            self.debug_msg_count = 0
        if self.debug_msg_count < 3:
            self.get_logger().info(f"🔍 Debug msg #{self.debug_msg_count}: time={time_sec:.3f}s, source={time_source}")
            self.debug_msg_count += 1
        
        # ALWAYS use relative time for bag data to avoid timestamp mismatches
        if not hasattr(self, 'first_msg_time'):
            self.first_msg_time = time_sec
            self.get_logger().info(f"🕐 Starting relative time from: {time_sec:.3f}s")
        
        # Return relative time (always start from 0)
        relative_time = time_sec - self.first_msg_time
        
        # Handle negative relative times (shouldn't happen with bags)
        if relative_time < 0:
            self.get_logger().warn(f"⚠️ Negative relative time: {relative_time:.3f}s - using absolute time")
            return time_sec
        
        return relative_time

    def cb_odom(self, msg: Odometry):
        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info(f"✅ Receiving odometry data from {self.odom_topic}")
            
        t = self.msg_time(msg.header.stamp)
        if self.start_t is None:
            self.start_t = t
            self.get_logger().info(f"Started data collection at t={t:.2f}s")
        self.last_t = t

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.odom_buf.append((t, x, y))

        # Success check
        if self.success == 0:
            distance_to_goal = math.hypot(x - self.goal_x, y - self.goal_y)
            if distance_to_goal <= self.goal_tol:
                self.success = 1
                self.success_t = t
                self.get_logger().info(f"SUCCESS reached goal at t={t - self.start_t:.2f}s (distance: {distance_to_goal:.2f}m)")
                if self.stop_on_success:
                    self.finalize_and_exit()
            
            # Log progress every 10 seconds
            if len(self.odom_buf) % 50 == 0:  # Assuming ~5Hz odom
                self.get_logger().info(f"Progress: t={t - self.start_t:.1f}s, pos=({x:.2f},{y:.2f}), dist_to_goal={distance_to_goal:.2f}m")

    def cb_cmd(self, msg: Twist):
        if not self.cmd_received:
            self.cmd_received = True
            self.get_logger().info(f"✅ Receiving command data from {self.cmd_vel_topic}")
            
        t = self.msg_time()
        if self.start_t is None:
            self.start_t = t
            self.get_logger().info(f"Started data collection from cmd_vel at t={t:.2f}s")
        self.last_t = t
        self.cmd_buf.append((t, abs(msg.linear.x)))
        
        # Log command activity
        if len(self.cmd_buf) % 100 == 0:  # Every 100 commands
            v = abs(msg.linear.x)
            omega = abs(msg.angular.z)
            self.get_logger().info(f"Command activity: v={v:.2f} m/s, ω={omega:.2f} rad/s")

    def cb_inf(self, msg: Float32):
        if msg:
            self.inf_buf.append(float(msg.data))

    def cb_plnr(self, msg: Float32):
        if msg:
            self.plan_buf.append(float(msg.data))

    # -------- Watchdog / Finalize --------
    def watchdog(self):
        if self.start_t is None:
            # Check if we're receiving any data
            if not self.odom_received and not self.cmd_received:
                self.get_logger().warn("⚠️  No data received yet - waiting for topics...")
            return
            
        now = self.get_clock().now().nanoseconds * 1e-9
        # use last seen message time to be robust to paused playback
        t_ref = self.last_t if self.last_t is not None else now
        elapsed = t_ref - self.start_t
        
        # With relative time, elapsed should always be reasonable
        
        # Log progress every 10 seconds
        if int(elapsed) % 10 == 0 and elapsed > 0:
            self.get_logger().info(f"Elapsed: {elapsed:.1f}s / {self.timeout_s}s, odom_msgs: {len(self.odom_buf)}, cmd_msgs: {len(self.cmd_buf)}")
        
        if elapsed >= self.timeout_s and self.success == 0:
            self.get_logger().warn(f"TIMEOUT reached after {elapsed:.1f}s; finalizing metrics.")
            self.finalize_and_exit()

    def path_length(self) -> float:
        # Prefer /odom, fallback to integrating /cmd_vel
        if len(self.odom_buf) >= 2:
            d = 0.0
            for i in range(1, len(self.odom_buf)):
                _, x0, y0 = self.odom_buf[i-1]
                _, x1, y1 = self.odom_buf[i]
                d += math.hypot(x1 - x0, y1 - y0)
            return d
        # Fallback: integrate |v| with trapezoid rule
        if len(self.cmd_buf) >= 2:
            d = 0.0
            for i in range(1, len(self.cmd_buf)):
                t0, v0 = self.cmd_buf[i-1]
                t1, v1 = self.cmd_buf[i]
                dt = max(0.0, t1 - t0)
                d += 0.5 * (v0 + v1) * dt
            return d
        return 0.0

    def duration(self) -> float:
        if self.start_t is None or self.last_t is None:
            return 0.0
        return max(0.0, self.last_t - self.start_t)

    def finalize_and_exit(self):
        travel = self.path_length()
        duration = self.duration()

        # Inference stats
        inf_mean = f"{stats.mean(self.inf_buf):.1f}" if len(self.inf_buf) else ""
        inf_std  = f"{(stats.pstdev(self.inf_buf) if len(self.inf_buf)>1 else 0.0):.1f}" if len(self.inf_buf) else ""

        # Planner step stats (optional)
        pl_mean = f"{stats.mean(self.plan_buf):.1f}" if len(self.plan_buf) else ""
        pl_std  = f"{(stats.pstdev(self.plan_buf) if len(self.plan_buf)>1 else 0.0):.1f}" if len(self.plan_buf) else ""

        row = {
            "method": self.method,
            "scenario": self.scenario,
            "success": self.success,
            "travel_m": f"{travel:.2f}",
            "duration_s": f"{duration:.2f}",
            "inf_ms_mean": inf_mean,
            "inf_ms_std": inf_std,
            "planner_ms_mean": pl_mean,
            "planner_ms_std": pl_std,
        }

        # Append to CSV
        with open(self.output_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)

        self.get_logger().info(
            f"[Saved] method={self.method}, scenario={self.scenario}, success={self.success}, "
            f"travel={travel:.2f}m, duration={duration:.2f}s -> {os.path.abspath(self.output_csv)}"
        )
        # Graceful shutdown
        rclpy.shutdown()

def main():
    rclpy.init()
    node = BagMetricsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted; finalizing.")
        node.finalize_and_exit()

if __name__ == "__main__":
    main()
