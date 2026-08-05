#!/usr/bin/env python3
"""WebRTC keyboard teleoperation for the DR.Nav Jackal.

Run setup_sensors.py first, then run this file in Isaac Sim's Script Editor.
Click the streamed viewport before using W/S/A/D. Space stops the robot.

This script only publishes geometry_msgs/Twist messages. Wheel actuation,
acceleration limiting, and the command timeout remain in setup_sensors.py.
"""

import asyncio
import builtins
import time

import carb.input
import omni.appwindow
import omni.kit.app

RUNTIME_CONFIG = getattr(builtins, "_DRNAV_RUNTIME_CONFIG", {})
TOPIC_CONFIG = RUNTIME_CONFIG.get("topics", {})
TELEOP_CONFIG = RUNTIME_CONFIG.get("teleop", {})

CMD_TOPIC = TOPIC_CONFIG.get("teleop_cmd", "/cmd_vel_teleop")
LINEAR_SPEED = float(TELEOP_CONFIG.get("linear_speed", 0.15))
ANGULAR_SPEED = float(TELEOP_CONFIG.get("angular_speed", 0.40))
PUBLISH_RATE_HZ = float(TELEOP_CONFIG.get("publish_rate_hz", 30.0))

BUILTINS_KEY = "_DRNAV_JACKAL_TELEOP"
TASK_KEY = "_DRNAV_JACKAL_TELEOP_TASK"
WAYPOINT_FOLLOWER_KEY = "_DRNAV_WAYPOINT_FOLLOWER"
SENSOR_BRIDGE_KEY = "_DRNAV_ISAACSIM6_V6_BRIDGE"
REQUIRED_COMMAND_INTERFACE_VERSION = 2


class JackalTeleop:
    def __init__(self):
        self.rclpy = None
        self.twist_type = None
        self.node = None
        self.publisher = None
        self.input = None
        self.keyboard = None
        self.keyboard_sub = None
        self.update_sub = None
        self.keys = set()
        self.publish_period = 1.0 / PUBLISH_RATE_HZ
        self.last_publish = 0.0
        self.was_active = False

    async def start(self):
        self.validate_sensor_bridge()
        self.stop_waypoint_follower()
        await self.create_ros_publisher()
        self.create_keyboard()
        self.update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self.on_update,
                name="DR.Nav Jackal teleop publisher",
            )
        )
        print("[DR.Nav Teleop] Ready. Click the viewport, then use W/S/A/D.")
        print(
            "[DR.Nav Teleop] Space stops. Publishing active commands to "
            "/cmd_vel_teleop at 30 Hz."
        )

    @staticmethod
    def stop_waypoint_follower():
        follower = getattr(builtins, WAYPOINT_FOLLOWER_KEY, None)
        if follower is None:
            return
        try:
            follower.shutdown()
            print(
                "[DR.Nav Teleop] Stopped waypoint follower; "
                "teleop now owns robot control."
            )
        finally:
            setattr(builtins, WAYPOINT_FOLLOWER_KEY, None)

    @staticmethod
    def validate_sensor_bridge():
        bridge = getattr(builtins, SENSOR_BRIDGE_KEY, None)
        version = getattr(bridge, "command_interface_version", None)
        if version != REQUIRED_COMMAND_INTERFACE_VERSION:
            raise RuntimeError(
                "Run the current setup_sensors.py before setup_teleop.py. "
                "The active sensor bridge does not support separated command "
                "topics."
            )

    async def create_ros_publisher(self):
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

        import rclpy
        from geometry_msgs.msg import Twist

        self.rclpy = rclpy
        self.twist_type = Twist
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("drnav_jackal_teleop")
        self.publisher = self.node.create_publisher(Twist, CMD_TOPIC, 10)

    def create_keyboard(self):
        window = omni.appwindow.get_default_app_window()
        if window is None:
            raise RuntimeError("Isaac Sim app window unavailable.")
        self.keyboard = window.get_keyboard()
        self.input = carb.input.acquire_input_interface()
        self.keyboard_sub = self.input.subscribe_to_keyboard_events(
            self.keyboard,
            self.on_key_event,
        )

    def on_key_event(self, event):
        if getattr(builtins, BUILTINS_KEY, None) is not self:
            return False

        allowed = {
            carb.input.KeyboardInput.W,
            carb.input.KeyboardInput.A,
            carb.input.KeyboardInput.S,
            carb.input.KeyboardInput.D,
            carb.input.KeyboardInput.SPACE,
        }
        if event.input not in allowed:
            return False

        if event.input == carb.input.KeyboardInput.SPACE:
            if event.type in (
                carb.input.KeyboardEventType.KEY_PRESS,
                carb.input.KeyboardEventType.KEY_REPEAT,
            ):
                self.keys.clear()
                self.publish(0.0, 0.0)
            return True

        if event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            self.keys.add(event.input)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.keys.discard(event.input)
        return True

    def command(self):
        linear = LINEAR_SPEED * (
            int(carb.input.KeyboardInput.W in self.keys)
            - int(carb.input.KeyboardInput.S in self.keys)
        )
        angular = ANGULAR_SPEED * (
            int(carb.input.KeyboardInput.A in self.keys)
            - int(carb.input.KeyboardInput.D in self.keys)
        )
        return linear, angular

    def on_update(self, _event):
        if getattr(builtins, BUILTINS_KEY, None) is not self:
            return

        active = bool(self.keys)
        if not active and not self.was_active:
            return

        now = time.monotonic()
        if now - self.last_publish < self.publish_period:
            return
        self.last_publish = now
        self.publish(*self.command())
        self.was_active = active

    def publish(self, linear, angular):
        if self.publisher is None:
            return
        message = self.twist_type()
        message.linear.x = float(linear)
        message.angular.z = float(angular)
        self.publisher.publish(message)

    def shutdown(self):
        self.keys.clear()
        self.publish(0.0, 0.0)
        self.update_sub = None

        if self.input is not None and self.keyboard_sub is not None:
            try:
                self.input.unsubscribe_to_keyboard_events(
                    self.keyboard,
                    self.keyboard_sub,
                )
            except Exception:
                pass
            self.keyboard_sub = None

        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            self.node = None
        self.publisher = None
        print("[DR.Nav Teleop] Stopped and published a zero command.")


async def main():
    previous = getattr(builtins, BUILTINS_KEY, None)
    if previous is not None:
        previous.shutdown()

    teleop = JackalTeleop()
    setattr(builtins, BUILTINS_KEY, teleop)
    try:
        await teleop.start()
    except BaseException:
        teleop.shutdown()
        if getattr(builtins, BUILTINS_KEY, None) is teleop:
            delattr(builtins, BUILTINS_KEY)
        raise

previous_task = getattr(builtins, TASK_KEY, None)
if previous_task is not None and not previous_task.done():
    previous_task.cancel()

task = asyncio.ensure_future(main())
setattr(builtins, TASK_KEY, task)
