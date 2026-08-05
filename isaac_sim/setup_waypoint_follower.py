#!/usr/bin/env python3
"""Repeatable Maze3 waypoint route for DR.Nav data collection.

Run setup_sensors.py first, then run this file in Isaac Sim's Script Editor.
The route intentionally visits both dead ends before entering the goal corridor.
Starting this script stops teleop so waypoint control is exclusive.

This is deliberately a simple waypoint controller, not a planner. It has no
obstacle avoidance and is intended only for the current Maze1/Maze3 geometry.
"""

import asyncio
import builtins
import math
import time

import omni.kit.app
import omni.timeline

RUNTIME_CONFIG = getattr(builtins, "_DRNAV_RUNTIME_CONFIG", {})
TOPIC_CONFIG = RUNTIME_CONFIG.get("topics", {})
WAYPOINT_CONFIG = RUNTIME_CONFIG.get("waypoint_follower", {})

ODOM_TOPIC = TOPIC_CONFIG.get("odom", "/odom_lidar")
CMD_TOPIC = TOPIC_CONFIG.get("waypoint_cmd", "/cmd_vel_waypoint")
CONTROL_RATE_HZ = float(WAYPOINT_CONFIG.get("control_rate_hz", 20.0))

# World-frame centerline points recovered from the successful Maze3 recording
# and the persistent semantic-zone geometry. The spawn itself is not a target;
# the controller begins wherever odometry reports the robot currently is.
DEFAULT_WAYPOINTS = [
    ("west_dead_end", -8.80, 0.15),
    ("central_junction_after_west", -1.70, 0.10),
    ("north_dead_end", -1.70, 24.50),
    ("central_junction_after_north", -1.70, 0.10),
    ("south_goal", -1.70, -29.50),
]

WAYPOINTS = [
    (
        str(item.get("name", f"waypoint_{index}")),
        float(item["x"]),
        float(item["y"]),
    )
    for index, item in enumerate(WAYPOINT_CONFIG.get("waypoints", []))
] or DEFAULT_WAYPOINTS

WAYPOINT_TOLERANCE = float(WAYPOINT_CONFIG.get("waypoint_tolerance", 0.35))
FINAL_TOLERANCE = float(WAYPOINT_CONFIG.get("final_tolerance", 0.45))
MAX_LINEAR_SPEED = float(WAYPOINT_CONFIG.get("max_linear_speed", 0.35))
MIN_LINEAR_SPEED = float(WAYPOINT_CONFIG.get("min_linear_speed", 0.08))
MAX_ANGULAR_SPEED = float(WAYPOINT_CONFIG.get("max_angular_speed", 0.60))
LINEAR_GAIN = float(WAYPOINT_CONFIG.get("linear_gain", 0.65))
ANGULAR_GAIN = float(WAYPOINT_CONFIG.get("angular_gain", 1.50))
ROTATE_IN_PLACE_ANGLE = math.radians(
    float(WAYPOINT_CONFIG.get("rotate_in_place_degrees", 25.0))
)
SLOWDOWN_ANGLE = math.radians(
    float(WAYPOINT_CONFIG.get("slowdown_degrees", 10.0))
)

BUILTINS_KEY = "_DRNAV_WAYPOINT_FOLLOWER"
TASK_KEY = "_DRNAV_WAYPOINT_FOLLOWER_TASK"
TELEOP_KEY = "_DRNAV_JACKAL_TELEOP"
SENSOR_BRIDGE_KEY = "_DRNAV_ISAACSIM6_V6_BRIDGE"
REQUIRED_COMMAND_INTERFACE_VERSION = 2


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class WaypointFollower:
    def __init__(self):
        self.rclpy = None
        self.twist_type = None
        self.node = None
        self.publisher = None
        self.odom_sub = None
        self.physics_callback = None
        self.timeline_sub = None
        self.simulation_manager = None
        self.sensor_bridge = None
        self.timeline = omni.timeline.get_timeline_interface()

        self.pose = None
        self.waypoint_index = 0
        self.complete = False
        self.last_update = 0.0
        self.control_elapsed = 0.0
        self.last_status = 0.0
        self.has_reported_odom = False
        self.was_playing = False
        self.control_period = 1.0 / CONTROL_RATE_HZ

    async def start(self):
        self.validate_sensor_bridge()
        self.stop_teleop()
        await self.create_ros()
        self.physics_callback = self.simulation_manager.register_callback(
            self.on_physics_step,
            self.physics_event,
        )
        self.timeline_sub = (
            self.timeline
            .get_timeline_event_stream()
            .create_subscription_to_pop(
                self.on_timeline_event,
                name="DR.Nav waypoint timeline state",
            )
        )

        print("\n[DR.Nav Waypoints] Waiting for odometry on", ODOM_TOPIC)
        print("[DR.Nav Waypoints] Route:")
        for index, (name, x, y) in enumerate(WAYPOINTS, start=1):
            print(f"  {index}. {name}: ({x:.2f}, {y:.2f})")
        print("[DR.Nav Waypoints] Exclusive waypoint-control mode active.")

    @staticmethod
    def stop_teleop():
        teleop = getattr(builtins, TELEOP_KEY, None)
        if teleop is None:
            return
        try:
            teleop.shutdown()
            print(
                "[DR.Nav Waypoints] Stopped teleop; waypoint follower now "
                "owns robot control."
            )
        finally:
            setattr(builtins, TELEOP_KEY, None)

    def validate_sensor_bridge(self):
        bridge = getattr(builtins, SENSOR_BRIDGE_KEY, None)
        version = getattr(bridge, "command_interface_version", None)
        pose_reader = getattr(bridge, "get_world_pose", None)
        command_sink = getattr(bridge, "submit_command", None)
        if (
            version != REQUIRED_COMMAND_INTERFACE_VERSION
            or not callable(pose_reader)
            or not callable(command_sink)
        ):
            raise RuntimeError(
                "Run the current setup_sensors.py before "
                "setup_waypoint_follower.py. The active sensor bridge does "
                "not support direct in-process pose synchronization."
            )
        self.sensor_bridge = bridge

    async def create_ros(self):
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

        import rclpy
        from geometry_msgs.msg import Twist
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
        from nav_msgs.msg import Odometry

        self.rclpy = rclpy
        self.twist_type = Twist
        self.simulation_manager = SimulationManager
        self.physics_event = IsaacEvents.POST_PHYSICS_STEP
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("drnav_maze3_waypoint_follower")
        self.publisher = self.node.create_publisher(Twist, CMD_TOPIC, 10)
        self.odom_sub = self.node.create_subscription(
            Odometry,
            ODOM_TOPIC,
            self.on_odom,
            10,
        )

    def on_odom(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0
            - 2.0
            * (
                orientation.y * orientation.y
                + orientation.z * orientation.z
            ),
        )
        self.pose = (float(position.x), float(position.y), yaw)
        if not self.has_reported_odom:
            print(
                "[DR.Nav Waypoints] Odometry received at "
                f"({position.x:.2f}, {position.y:.2f}); starting route."
            )
            self.has_reported_odom = True

    def synchronize_pose_from_bridge(self):
        """Keep local control independent of ROS executor scheduling latency."""
        if self.sensor_bridge is None:
            return

        x, y, yaw = self.sensor_bridge.get_world_pose()
        self.pose = (float(x), float(y), float(yaw))

        if not self.has_reported_odom:
            print(
                "[DR.Nav Waypoints] Pose synchronized at "
                f"({x:.2f}, {y:.2f}); starting route."
            )
            self.has_reported_odom = True

    def reset_route(self):
        self.publish(0.0, 0.0)
        self.pose = None
        self.waypoint_index = 0
        self.complete = False
        self.last_update = 0.0
        self.control_elapsed = 0.0
        self.last_status = 0.0
        self.has_reported_odom = False
        self.was_playing = False
        print("[DR.Nav Waypoints] Route reset for a new episode.")

    def on_timeline_event(self, event):
        if getattr(builtins, BUILTINS_KEY, None) is not self:
            return

        if event.type == int(omni.timeline.TimelineEventType.STOP):
            if self.was_playing:
                self.publish(0.0, 0.0)
                print(
                    "[DR.Nav Waypoints] Timeline stopped; waypoint control "
                    "suspended."
                )
            self.reset_route()
            return

        if event.type == int(omni.timeline.TimelineEventType.PAUSE):
            if self.was_playing:
                self.publish(0.0, 0.0)
                print(
                    "[DR.Nav Waypoints] Timeline paused; waypoint control "
                    "suspended without resetting the route."
                )
            self.was_playing = False
            return

        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            # Prime the command before the first physics callback. The bridge
            # callback was registered first and would otherwise see the old
            # command timeout for one frame before this controller runs.
            self.synchronize_pose_from_bridge()
            if not self.complete and self.pose is not None:
                self.control_step()
                self.control_elapsed = 0.0

    def on_physics_step(self, dt, _context):
        if (
            getattr(builtins, BUILTINS_KEY, None) is not self
            or self.node is None
        ):
            return

        self.rclpy.spin_once(self.node, timeout_sec=0.0)
        if not self.timeline.is_playing():
            return

        # The follower and sensor bridge run in the same Isaac process. Read
        # the pose directly for control so a delayed ROS odometry callback
        # cannot let /cmd_vel_waypoint expire. ROS odometry remains published
        # and subscribed for observability and eventual external controllers.
        self.synchronize_pose_from_bridge()

        if not self.was_playing:
            self.last_status = 0.0
            self.control_elapsed = self.control_period
            print("[DR.Nav Waypoints] Timeline playing; control resumed.")
        self.was_playing = True

        self.control_elapsed += max(float(dt), 0.0)
        if self.control_elapsed < self.control_period:
            return
        self.control_elapsed %= self.control_period

        if self.complete or self.pose is None:
            return
        self.control_step()

    def control_step(self):
        name, target_x, target_y = WAYPOINTS[self.waypoint_index]
        x, y, yaw = self.pose
        dx, dy = target_x - x, target_y - y
        distance = math.hypot(dx, dy)
        tolerance = (
            FINAL_TOLERANCE
            if self.waypoint_index == len(WAYPOINTS) - 1
            else WAYPOINT_TOLERANCE
        )

        if distance <= tolerance:
            print(
                f"[DR.Nav Waypoints] Reached {name} "
                f"at ({x:.2f}, {y:.2f})."
            )
            self.publish(0.0, 0.0)
            self.waypoint_index += 1

            if self.waypoint_index >= len(WAYPOINTS):
                self.complete = True
                print("[DR.Nav Waypoints] Route complete; robot stopped.")
            return

        heading = math.atan2(dy, dx)
        error = wrap(heading - yaw)
        angular = clamp(
            ANGULAR_GAIN * error,
            -MAX_ANGULAR_SPEED,
            MAX_ANGULAR_SPEED,
        )

        if abs(error) >= ROTATE_IN_PLACE_ANGLE:
            linear = 0.0
        else:
            linear = clamp(
                LINEAR_GAIN * distance,
                MIN_LINEAR_SPEED,
                MAX_LINEAR_SPEED,
            )
            if abs(error) > SLOWDOWN_ANGLE:
                linear *= 0.45

        self.publish(linear, angular)
        now = time.monotonic()
        if now - self.last_status >= 2.0:
            print(
                f"[DR.Nav Waypoints] target={name} "
                f"distance={distance:.2f} heading_error="
                f"{math.degrees(error):.1f}deg "
                f"cmd=({linear:.2f}, {angular:.2f})"
            )
            self.last_status = now

    def publish(self, linear, angular):
        if self.publisher is None:
            return
        message = self.twist_type()
        message.linear.x = float(linear)
        message.angular.z = float(angular)
        self.publisher.publish(message)

        # This follower shares the Isaac process with the bridge. Refresh the
        # local command immediately as well as publishing the ROS topic so
        # DDS/executor latency cannot trip the bridge's safety timeout.
        if self.sensor_bridge is not None:
            self.sensor_bridge.submit_command(
                "waypoint",
                linear,
                angular,
            )

    def shutdown(self):
        self.publish(0.0, 0.0)
        self.timeline_sub = None

        if self.physics_callback is not None and self.simulation_manager is not None:
            try:
                self.simulation_manager.deregister_callback(self.physics_callback)
            except Exception:
                pass
            self.physics_callback = None

        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            self.node = None

        self.publisher = None
        self.sensor_bridge = None
        print("[DR.Nav Waypoints] Stopped and published a zero command.")


async def main():
    previous = getattr(builtins, BUILTINS_KEY, None)
    if previous is not None:
        previous.shutdown()

    follower = WaypointFollower()
    setattr(builtins, BUILTINS_KEY, follower)
    try:
        await follower.start()
    except BaseException:
        follower.shutdown()
        if getattr(builtins, BUILTINS_KEY, None) is follower:
            delattr(builtins, BUILTINS_KEY)
        raise

previous_task = getattr(builtins, TASK_KEY, None)
if previous_task is not None and not previous_task.done():
    previous_task.cancel()

task = asyncio.ensure_future(main())
setattr(builtins, TASK_KEY, task)
