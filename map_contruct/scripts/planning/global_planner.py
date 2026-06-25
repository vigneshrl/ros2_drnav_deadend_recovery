#!/usr/bin/env python3
"""
Global Planner (A*)
===================
Shared by DWA, MPPI, and DR.Nav methods.

Computes a collision-free global path from the robot's current position
to the user-given goal using A* on the SLAM occupancy grid.

Replanning triggers
-------------------
  - New goal received
  - Robot deviates > 1.0 m from the nearest path waypoint (checked every 2 s)

Publishes
---------
  /global_path       nav_msgs/Path          A* path in map frame
  /global_costmap    nav_msgs/OccupancyGrid inflated occupancy grid

Subscribes
----------
  /map               nav_msgs/OccupancyGrid from SLAM
  /goal_pose         PoseStamped            from RViz2 Nav2 Goal
"""

import heapq
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs  # noqa: F401
from scipy.ndimage import binary_dilation


class GlobalPlanner(Node):
    def __init__(self):
        super().__init__('global_planner')

        # ── Parameters ──────────────────────────────────────────────────────
        self.inflation_radius  = 0.40   # metres to inflate obstacles
        self.replan_dev_dist   = 1.0    # replan when robot deviates this far
        self.path_spacing      = 0.3    # metres between published waypoints
        self.replan_check_dt   = 2.0    # seconds between deviation checks

        # ── State ────────────────────────────────────────────────────────────
        self.occupancy_grid    = None
        self.inflated_mask     = None   # bool array (height, width) True = no-go
        self.current_goal      = None   # PoseStamped
        self.global_path       = []     # list of (wx, wy) in map frame

        # TF
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(OccupancyGrid, '/map',       self._map_cb,  10)
        self.create_subscription(PoseStamped,   '/goal_pose', self._goal_cb, 10)

        # Publishers
        self.path_pub    = self.create_publisher(Path,          '/global_path',    10)
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/global_costmap', 10)

        # Timers
        self.create_timer(self.replan_check_dt, self._replan_check)
        self.create_timer(1.0,                  self._publish_costmap)

        self.get_logger().info('Global Planner (A*) initialized — waiting for /map and /goal_pose')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _map_cb(self, msg: OccupancyGrid):
        self.occupancy_grid = msg
        self._rebuild_inflated()
        if self.current_goal is not None:
            self._replan()

    def _goal_cb(self, msg: PoseStamped):
        if msg.header.frame_id != 'map':
            try:
                msg = self.tf_buffer.transform(
                    msg, 'map', timeout=rclpy.duration.Duration(seconds=0.5))
            except Exception as e:
                self.get_logger().error(f'Goal TF to map failed: {e}')
                return
        self.current_goal = msg
        self.global_path  = []
        self.get_logger().info(
            f'New goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) '
            f'[map]')
        self._replan()

    # ── Map helpers ───────────────────────────────────────────────────────────

    def _rebuild_inflated(self):
        """Dilate occupied cells by inflation_radius — done once per map update."""
        info  = self.occupancy_grid.info
        raw   = np.array(self.occupancy_grid.data, dtype=np.int8).reshape(
                    (info.height, info.width))
        cells = max(1, int(math.ceil(self.inflation_radius / info.resolution)))
        self.inflated_mask = binary_dilation(raw > 50, iterations=cells)

    def _world_to_cell(self, wx, wy):
        info = self.occupancy_grid.info
        gx = int((wx - info.origin.position.x) / info.resolution)
        gy = int((wy - info.origin.position.y) / info.resolution)
        return gx, gy

    def _cell_to_world(self, gx, gy):
        info = self.occupancy_grid.info
        wx = gx * info.resolution + info.origin.position.x + info.resolution * 0.5
        wy = gy * info.resolution + info.origin.position.y + info.resolution * 0.5
        return wx, wy

    def _is_free(self, gx, gy):
        info = self.occupancy_grid.info
        if gx < 0 or gx >= info.width or gy < 0 or gy >= info.height:
            return False
        return not self.inflated_mask[gy, gx]

    def _nearest_free(self, gx, gy, search_r=25):
        """Snap an occupied cell to the nearest free cell within search_r."""
        for r in range(1, search_r):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if self._is_free(nx, ny):
                        return nx, ny
        return None, None

    # ── A* ────────────────────────────────────────────────────────────────────

    def _astar(self, start_cell, goal_cell):
        """
        A* on the inflated occupancy grid.
        Returns list of (gx, gy) cells from start to goal, or None if no path.
        """
        sx, sy = start_cell
        gx, gy = goal_cell

        # Snap if occupied
        if not self._is_free(sx, sy):
            sx, sy = self._nearest_free(sx, sy)
            if sx is None:
                self.get_logger().warn('A*: start cell occupied and no free neighbour found')
                return None

        if not self._is_free(gx, gy):
            gx, gy = self._nearest_free(gx, gy)
            if gx is None:
                self.get_logger().warn('A*: goal cell occupied and no free neighbour found')
                return None

        def h(x, y):
            return math.hypot(x - gx, y - gy)

        open_set  = [(h(sx, sy), sx, sy)]
        came_from = {}
        g_score   = {(sx, sy): 0.0}

        while open_set:
            _, cx, cy = heapq.heappop(open_set)

            if (cx, cy) == (gx, gy):
                # Reconstruct path
                path, node = [], (gx, gy)
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append((sx, sy))
                path.reverse()
                return path

            # 8-connected neighbours
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                nx, ny   = cx + dx, cy + dy
                if not self._is_free(nx, ny):
                    continue
                step = 1.0 if dx == 0 or dy == 0 else 1.414
                ng   = g_score[(cx, cy)] + step
                if ng < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[(nx, ny)]   = ng
                    heapq.heappush(open_set, (ng + h(nx, ny), nx, ny))

        self.get_logger().warn('A*: no path found to goal')
        return None

    # ── Replanning ────────────────────────────────────────────────────────────

    def _replan(self):
        if self.occupancy_grid is None or self.inflated_mask is None:
            self.get_logger().warn('Cannot plan: no map yet')
            return
        if self.current_goal is None:
            return

        rx, ry, ok = self._get_robot_pos()
        if not ok:
            self.get_logger().warn('Cannot plan: robot TF not available')
            return

        start_cell = self._world_to_cell(rx, ry)
        goal_cell  = self._world_to_cell(
            self.current_goal.pose.position.x,
            self.current_goal.pose.position.y)

        cells = self._astar(start_cell, goal_cell)
        if cells is None:
            return

        # Downsample to path_spacing resolution
        info = self.occupancy_grid.info
        skip = max(1, int(self.path_spacing / info.resolution))
        cells = cells[::skip]
        if cells[-1] != (goal_cell[0], goal_cell[1]):
            cells.append(goal_cell)

        self.global_path = [self._cell_to_world(gx, gy) for gx, gy in cells]
        self._publish_path()
        self.get_logger().info(f'A* path: {len(self.global_path)} waypoints')

    def _replan_check(self):
        """Replan if robot has deviated too far from the current path."""
        if not self.global_path or self.current_goal is None:
            return
        rx, ry, ok = self._get_robot_pos()
        if not ok:
            return
        min_d = min(math.hypot(rx - wx, ry - wy) for wx, wy in self.global_path)
        if min_d > self.replan_dev_dist:
            self.get_logger().info(f'Deviation {min_d:.2f} m — replanning')
            self._replan()

    # ── Publishing ────────────────────────────────────────────────────────────

    def _publish_path(self):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self.get_clock().now().to_msg()
        for wx, wy in self.global_path:
            ps = PoseStamped()
            ps.header              = msg.header
            ps.pose.position.x     = wx
            ps.pose.position.y     = wy
            ps.pose.orientation.w  = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _publish_costmap(self):
        if self.occupancy_grid is None or self.inflated_mask is None:
            return
        info = self.occupancy_grid.info
        raw  = np.array(self.occupancy_grid.data, dtype=np.int8).reshape(
                   (info.height, info.width))
        out  = raw.copy()
        # Inflated but not originally occupied → lethal cost 90 (visible in RViz)
        out[(self.inflated_mask) & (raw <= 50)] = 90
        out[raw > 50] = 100

        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.info            = info
        msg.data            = out.flatten().tolist()
        self.costmap_pub.publish(msg)

    # ── TF helper ─────────────────────────────────────────────────────────────

    def _get_robot_pos(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            return t.transform.translation.x, t.transform.translation.y, True
        except Exception:
            return 0.0, 0.0, False


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
