#!/usr/bin/env python3
import csv, bisect, math, numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


def yaw_from_quat(q):
    # assuming flat ground (no significant roll/pitch)
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class VisionCostmapNode(Node):
    def __init__(self):
        super().__init__('vision_costmap_node')

        # ---------------- Params ----------------
        self.declare_parameter('csv_path', '/home/mrvik/dram_ws/src/map_contruct/map_contruct/labels.csv')
        self.declare_parameter('resolution', 0.10)       # m/cell
        self.declare_parameter('size_x', 80.0)           # meters
        self.declare_parameter('size_y', 40.0)           # meters
        self.declare_parameter('max_range', 6.0)         # label=1 paint length
        self.declare_parameter('dead_end_range', 2.5)    # label=0 patch center dist
        self.declare_parameter('fov_deg', 40.0)          # +/- half-cone total = 40°
        self.declare_parameter('time_tolerance', 1.0)    # sec; start generous
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_hz', 2.0)        # periodic publish to RViz
        self.declare_parameter('dynamic_origin', True)   # center on first odom
        self.declare_parameter('odom_topic', '/odom')    # or '/odom_lidar'
        # Allow static origin override if dynamic_origin=false
        self.declare_parameter('origin_x', -40.0)
        self.declare_parameter('origin_y', -20.0)
        # Optional CSV time shift if needed
        self.declare_parameter('csv_time_offset', 0.0)

        # --------------- Grid setup ---------------
        self.res = float(self.get_parameter('resolution').value)
        size_x = float(self.get_parameter('size_x').value)
        size_y = float(self.get_parameter('size_y').value)
        self.w = int(round(size_x / self.res))
        self.h = int(round(size_y / self.res))

        self.grid = np.full((self.h, self.w), -1, dtype=np.int16)  # -1 unknown, 0 free, 100 lethal

        dyn_origin = bool(self.get_parameter('dynamic_origin').value)
        if dyn_origin:
            # will set origin on first odom
            self.origin_locked = False
            self.origin_x = -size_x / 2.0
            self.origin_y = -size_y / 2.0
        else:
            self.origin_locked = True
            self.origin_x = float(self.get_parameter('origin_x').value)
            self.origin_y = float(self.get_parameter('origin_y').value)

        # --------------- Load CSV ---------------
        csv_path = str(self.get_parameter('csv_path').value)
        offset = float(self.get_parameter('csv_time_offset').value)
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                t = float(r['timestamp']) + offset
                l = int(r['label'])
                rows.append((t, l))
        rows.sort(key=lambda x: x[0])
        self.stamps = [t for t, _ in rows]
        self.labels = [l for _, l in rows]
        if self.stamps:
            self.get_logger().info(
                f'Loaded {len(self.labels)} labels from {csv_path} (offset {offset:+.3f}s), '
                f't=[{self.stamps[0]:.3f}, {self.stamps[-1]:.3f}]'
            )
        else:
            self.get_logger().warn(f'CSV {csv_path} contained 0 labels.')

        # --------------- Pub/Sub ---------------
        self.pub = self.create_publisher(OccupancyGrid, 'vision_costmap', 1)

        hz = max(0.1, float(self.get_parameter('publish_hz').value))
        self.create_timer(1.0 / hz, self.publish_grid)

        odom_topic = str(self.get_parameter('odom_topic').value)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.sub = self.create_subscription(Odometry, odom_topic, self.odom_cb, qos)
        self.get_logger().info(f"Subscribed to {odom_topic} (BEST_EFFORT)")

        self.paint_events = 0  # debug counter

    # --------------- Helpers ---------------
    def maybe_lock_origin(self, x, y):
        if not self.origin_locked:
            size_x = self.w * self.res
            size_y = self.h * self.res
            self.origin_x = x - size_x / 2.0
            self.origin_y = y - size_y / 2.0
            self.origin_locked = True
            self.get_logger().info(
                f'Origin set around first odom: ({self.origin_x:.2f}, {self.origin_y:.2f})'
            )

    def world_to_idx(self, x, y):
        ix = int((x - self.origin_x) / self.res)
        iy = int((y - self.origin_y) / self.res)
        return ix, iy

    # --------------- Painting ---------------
    def paint_cone(self, x, y, yaw, length):
        import numpy as _np
        fov = math.radians(float(self.get_parameter('fov_deg').value))
        steps = max(1, int(length / self.res))
        drawn = 0
        for s in range(1, steps + 1):
            r = s * self.res
            for a in _np.linspace(-fov / 2, fov / 2, 9):
                xx = x + r * math.cos(yaw + a)
                yy = y + r * math.sin(yaw + a)
                ix, iy = self.world_to_idx(xx, yy)
                if 0 <= ix < self.w and 0 <= iy < self.h:
                    self.grid[iy, ix] = 0  # free
                    drawn += 1
        self.paint_events += drawn

    def paint_patch_ahead(self, x, y, yaw, r):
        # lethal patch centered r meters ahead
        half_w = 0.6
        half_h = 0.6
        cx = x + r * math.cos(yaw)
        cy = y + r * math.sin(yaw)
        xs = np.arange(cx - half_w, cx + half_w, self.res)
        ys = np.arange(cy - half_h, cy + half_h, self.res)
        drawn = 0
        for xx in xs:
            for yy in ys:
                ix, iy = self.world_to_idx(xx, yy)
                if 0 <= ix < self.w and 0 <= iy < self.h:
                    self.grid[iy, ix] = 100  # lethal
                    drawn += 1
        self.paint_events += drawn

    # --------------- Callbacks ---------------
    def odom_cb(self, msg: Odometry):
        if not self.stamps:
            return

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # nearest CSV time
        i = bisect.bisect_left(self.stamps, t)
        cand = []
        if i < len(self.stamps):
            cand.append(i)
        if i > 0:
            cand.append(i - 1)
        j = min(cand, key=lambda k: abs(self.stamps[k] - t))
        dt = abs(self.stamps[j] - t)
        tol = float(self.get_parameter('time_tolerance').value)
        if dt > tol:
            return

        label = self.labels[j]

        # pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quat(msg.pose.pose.orientation)

        # ensure grid covers the robot
        self.maybe_lock_origin(x, y)

        if label == 1:
            self.paint_cone(x, y, yaw, float(self.get_parameter('max_range').value))
        else:
            self.paint_patch_ahead(x, y, yaw, float(self.get_parameter('dead_end_range').value))

    def publish_grid(self):
        # Build OccupancyGrid
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.get_parameter('frame_id').value)

        info = MapMetaData()
        info.resolution = float(self.res)
        info.width = int(self.w)
        info.height = int(self.h)
        info.origin.position.x = float(self.origin_x)
        info.origin.position.y = float(self.origin_y)
        info.origin.orientation.w = 1.0
        msg.info = info

        msg.data = [int(np.clip(v, -1, 100)) for v in self.grid.flatten()]
        # occasional debug
        known = sum(1 for v in msg.data if v != -1)
        self.get_logger().throttle(self.get_clock(), 2000, f'Known cells: {known}/{len(msg.data)}; painted={self.paint_events}')
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = VisionCostmapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
