#!/usr/bin/env python3
"""
Isaac Sim 6.0 — DR.Nav Jackal runtime setup (compact)
=====================================================

Scene-specific version for the current project:

- Existing Jackal:
    /World/ground/flat_plane/jackal
- Existing Bumblebee left camera
- Existing PhysX LiDAR:
    .../sick_lms1xx_lidar_frame/Lidar
- W/A/S/D WebRTC teleoperation
- ROS 2 camera, LaserScan, PointCloud2, odometry, clock and TF
- ROS 2 /cmd_vel support when keyboard control is disabled
- Runtime camera graph authored only in the USD session layer

Run from Isaac Sim's Script Editor with the maze open.
"""

from __future__ import annotations

import asyncio
import builtins
import math
import time
from typing import Tuple

import carb.input
import numpy as np
import omni.appwindow
import omni.graph.core as og
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom


# -----------------------------------------------------------------------------
# Scene paths
# -----------------------------------------------------------------------------

ROBOT = "/World/ground/flat_plane/jackal"
BASE_LINK = f"{ROBOT}/base_link"

CAMERA_FRAME = (
    f"{BASE_LINK}/bumblebee_stereo_camera_frame/"
    "bumblebee_stereo_left_frame"
)
CAMERA = f"{CAMERA_FRAME}/bumblebee_stereo_left_camera"

LIDAR_FRAME = f"{BASE_LINK}/sick_lms1xx_lidar_frame"
LIDAR = f"{LIDAR_FRAME}/Lidar"

RUNTIME_GRAPH = "/World/DRNav_RuntimeGraph"


# -----------------------------------------------------------------------------
# ROS names
# -----------------------------------------------------------------------------

TOPIC_IMAGE = "/argus/ar0234_front_left/image_raw"
TOPIC_SCAN = "/scan"
TOPIC_POINTS = "/os_cloud_node/points"
TOPIC_ODOM = "/odom_lidar"
TOPIC_CLOCK = "/clock"
TOPIC_CMD_VEL = "/cmd_vel"

FRAME_ODOM = "odom"
FRAME_BASE = "base_link"
FRAME_CAMERA = "bumblebee_stereo_left_frame"
FRAME_LIDAR = "sim_lidar"


# -----------------------------------------------------------------------------
# Robot and input settings
# -----------------------------------------------------------------------------

CAMERA_RESOLUTION = (640, 480)

USE_WEBRTC_KEYBOARD = True
LINEAR_SPEED = 0.15
ANGULAR_SPEED = 0.40

MAX_LINEAR_ACCEL = 0.35
MAX_ANGULAR_ACCEL = 0.90

WHEEL_RADIUS = 0.098
TRACK_WIDTH = 0.37559
MAX_WHEEL_SPEED = 25.0

LEFT_WHEEL_SIGN = 1.0
RIGHT_WHEEL_SIGN = 1.0

CMD_TIMEOUT = 0.5
ODOM_RATE = 30.0
LIDAR_RATE = 10.0

WHEEL_JOINTS = [
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_left_wheel_joint",
    "rear_right_wheel_joint",
]

_BRIDGE_KEY = "_DRNAV_COMPACT_BRIDGE"


def move_toward(value: float, target: float, delta: float) -> float:
    if value < target:
        return min(value + delta, target)
    return max(value - delta, target)


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


class DRNavBridge:
    def __init__(self) -> None:
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()

        self.ros_node = None
        self.cmd_sub = None
        self.scan_pub = None
        self.points_pub = None
        self.odom_pub = None
        self.clock_pub = None
        self.tf_pub = None
        self.static_tf_pub = None

        self.lidar_interface = None
        self.jackal = None
        self.wheel_indices = None
        self.physics_callback = None

        self.input_interface = None
        self.keyboard = None
        self.keyboard_subscription = None
        self.pressed_keys = set()

        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        self.last_cmd_time = 0.0

        self.current_linear = 0.0
        self.current_angular = 0.0

        self.last_odom_time = -1.0
        self.last_lidar_time = -1.0
        self.previous_pose_time = None
        self.previous_position = None
        self.previous_yaw = None

        self.range_min = 0.05
        self.range_max = 100.0
        self.last_error_print = 0.0

    # ------------------------------------------------------------------
    # Startup and shutdown
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self.validate_scene()
        await self.enable_extensions()

        global Articulation, SimulationManager, IsaacEvents
        global set_target_prims, _range_sensor
        global rclpy, Twist, TransformStamped, Odometry, Clock
        global LaserScan, PointCloud2, PointField
        global TransformBroadcaster, StaticTransformBroadcaster

        from isaacsim.core.experimental.prims import Articulation
        from isaacsim.core.nodes.scripts.utils import set_target_prims
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.simulation_manager.impl.isaac_events import (
            IsaacEvents,
        )
        from isaacsim.sensors.physx import _range_sensor

        import rclpy
        from geometry_msgs.msg import TransformStamped, Twist
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import LaserScan, PointCloud2, PointField
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()

        await self.remove_runtime_graph()
        self.disable_lidar_debug_view()
        self.create_camera_graph()
        self.create_ros_node()

        if USE_WEBRTC_KEYBOARD:
            self.create_keyboard_handler()

        self.timeline.play()

        for _ in range(6):
            await omni.kit.app.get_app().next_update_async()

        self.create_articulation()
        self.publish_static_transforms()

        self.physics_callback = SimulationManager.register_callback(
            self.physics_step,
            IsaacEvents.POST_PHYSICS_STEP,
        )

        print("\n[DR.Nav] Compact setup complete.")
        print("[DR.Nav] W/S = forward/reverse, A/D = turn, Space = stop.")
        print("[DR.Nav] Runtime graph is session-layer-only.")

    async def shutdown(self) -> None:
        try:
            self.timeline.stop()
        except Exception:
            pass

        try:
            if self.physics_callback is not None:
                SimulationManager.deregister_callback(self.physics_callback)
        except Exception:
            pass

        self.physics_callback = None

        try:
            self.set_wheel_targets(0.0, 0.0)
        except Exception:
            pass

        if (
            self.input_interface is not None
            and self.keyboard is not None
            and self.keyboard_subscription is not None
        ):
            try:
                self.input_interface.unsubscribe_to_keyboard_events(
                    self.keyboard,
                    self.keyboard_subscription,
                )
            except Exception:
                pass

        self.keyboard_subscription = None
        self.pressed_keys.clear()

        if self.ros_node is not None:
            try:
                self.ros_node.destroy_node()
            except Exception:
                pass

        self.ros_node = None

        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()

        await self.remove_runtime_graph()
        print("[DR.Nav] Previous compact bridge cleaned up.")

    # ------------------------------------------------------------------
    # Scene and extensions
    # ------------------------------------------------------------------

    def validate_scene(self) -> None:
        required = [ROBOT, BASE_LINK, CAMERA_FRAME, CAMERA, LIDAR_FRAME, LIDAR]
        missing = [
            path
            for path in required
            if not self.stage.GetPrimAtPath(path).IsValid()
        ]

        if missing:
            raise RuntimeError(
                "Missing required scene prims:\n  " + "\n  ".join(missing)
            )

        lidar_prim = self.stage.GetPrimAtPath(LIDAR)

        if not lidar_prim.IsActive():
            raise RuntimeError(f"Existing LiDAR is inactive: {LIDAR}")

        if lidar_prim.GetTypeName() != "Lidar":
            raise RuntimeError(
                f"Expected PhysX Lidar at {LIDAR}, "
                f"found type {lidar_prim.GetTypeName()!r}."
            )

    async def enable_extensions(self) -> None:
        manager = omni.kit.app.get_app().get_extension_manager()

        for extension_id in [
            "isaacsim.core.nodes",
            "isaacsim.ros2.bridge",
            "isaacsim.ros2.nodes",
            "isaacsim.sensors.physx",
        ]:
            if not manager.is_extension_enabled(extension_id):
                manager.set_extension_enabled_immediate(extension_id, True)

        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

    def session_target(self):
        return self.stage.GetEditTargetForLocalLayer(
            self.stage.GetSessionLayer()
        )

    async def remove_runtime_graph(self) -> None:
        with Usd.EditContext(self.stage, self.session_target()):
            if self.stage.GetPrimAtPath(RUNTIME_GRAPH).IsValid():
                self.stage.RemovePrim(RUNTIME_GRAPH)

        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()

    def disable_lidar_debug_view(self) -> None:
        lidar_prim = self.stage.GetPrimAtPath(LIDAR)

        with Usd.EditContext(self.stage, self.session_target()):
            for attribute in lidar_prim.GetAttributes():
                name = attribute.GetName().lower()

                if (
                    name.endswith("drawlines")
                    or name.endswith("drawpoints")
                    or name.endswith("showdebugview")
                ):
                    attribute.Set(False)

        self.range_min = self.find_numeric_attribute("minrange", 0.05)
        self.range_max = self.find_numeric_attribute("maxrange", 100.0)

    def find_numeric_attribute(self, suffix: str, fallback: float) -> float:
        prim = self.stage.GetPrimAtPath(LIDAR)

        for attribute in prim.GetAttributes():
            if attribute.GetName().lower().endswith(suffix):
                value = attribute.Get()
                if value is not None:
                    return float(value)

        return fallback

    # ------------------------------------------------------------------
    # Camera graph
    # ------------------------------------------------------------------

    def create_camera_graph(self) -> None:
        with Usd.EditContext(self.stage, self.session_target()):
            og.Controller.edit(
                {
                    "graph_path": RUNTIME_GRAPH,
                    "evaluator_name": "execution",
                    "pipeline_stage": (
                        og.GraphPipelineStage
                        .GRAPH_PIPELINE_STAGE_SIMULATION
                    ),
                },
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("tick", "omni.graph.action.OnPlaybackTick"),
                        (
                            "context",
                            "isaacsim.ros2.bridge.ROS2Context",
                        ),
                        (
                            "once",
                            "isaacsim.core.nodes."
                            "OgnIsaacRunOneSimulationFrame",
                        ),
                        (
                            "render_product",
                            "isaacsim.core.nodes."
                            "IsaacCreateRenderProduct",
                        ),
                        (
                            "camera",
                            "isaacsim.ros2.bridge.ROS2CameraHelper",
                        ),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("tick.outputs:tick", "once.inputs:execIn"),
                        (
                            "once.outputs:step",
                            "render_product.inputs:execIn",
                        ),
                        (
                            "render_product.outputs:execOut",
                            "camera.inputs:execIn",
                        ),
                        (
                            "render_product.outputs:renderProductPath",
                            "camera.inputs:renderProductPath",
                        ),
                        (
                            "context.outputs:context",
                            "camera.inputs:context",
                        ),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("context.inputs:useDomainIDEnvVar", True),
                        (
                            "render_product.inputs:width",
                            CAMERA_RESOLUTION[0],
                        ),
                        (
                            "render_product.inputs:height",
                            CAMERA_RESOLUTION[1],
                        ),
                        ("camera.inputs:topicName", TOPIC_IMAGE),
                        ("camera.inputs:frameId", FRAME_CAMERA),
                        ("camera.inputs:type", "rgb"),
                        ("camera.inputs:enabled", True),
                    ],
                },
            )

            set_target_prims(
                primPath=f"{RUNTIME_GRAPH}/render_product",
                inputName="inputs:cameraPrim",
                targetPrimPaths=[CAMERA],
            )

        graph = self.stage.GetPrimAtPath(RUNTIME_GRAPH)
        session_id = self.stage.GetSessionLayer().identifier
        authored_layers = {
            spec.layer.identifier for spec in graph.GetPrimStack()
        }

        if session_id not in authored_layers:
            raise RuntimeError(
                "Runtime graph was not created in the USD session layer."
            )

    # ------------------------------------------------------------------
    # ROS
    # ------------------------------------------------------------------

    def create_ros_node(self) -> None:
        if not rclpy.ok():
            rclpy.init()

        self.ros_node = rclpy.create_node("drnav_jackal_compact")

        self.cmd_sub = self.ros_node.create_subscription(
            Twist,
            TOPIC_CMD_VEL,
            self.cmd_vel_callback,
            10,
        )

        self.scan_pub = self.ros_node.create_publisher(
            LaserScan,
            TOPIC_SCAN,
            10,
        )
        self.points_pub = self.ros_node.create_publisher(
            PointCloud2,
            TOPIC_POINTS,
            10,
        )
        self.odom_pub = self.ros_node.create_publisher(
            Odometry,
            TOPIC_ODOM,
            10,
        )
        self.clock_pub = self.ros_node.create_publisher(
            Clock,
            TOPIC_CLOCK,
            10,
        )

        self.tf_pub = TransformBroadcaster(self.ros_node)
        self.static_tf_pub = StaticTransformBroadcaster(self.ros_node)

        self.last_cmd_time = time.monotonic()

    def cmd_vel_callback(self, message) -> None:
        self.cmd_linear = float(message.linear.x)
        self.cmd_angular = float(message.angular.z)
        self.last_cmd_time = time.monotonic()

    def sim_time(self) -> float:
        return float(self.timeline.get_current_time())

    @staticmethod
    def set_stamp(stamp, value: float) -> None:
        seconds = int(math.floor(value))
        stamp.sec = seconds
        stamp.nanosec = int((value - seconds) * 1_000_000_000)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def create_keyboard_handler(self) -> None:
        app_window = omni.appwindow.get_default_app_window()

        if app_window is None:
            raise RuntimeError("Default Isaac Sim app window is unavailable.")

        self.keyboard = app_window.get_keyboard()
        self.input_interface = carb.input.acquire_input_interface()
        self.keyboard_subscription = (
            self.input_interface.subscribe_to_keyboard_events(
                self.keyboard,
                self.keyboard_event,
            )
        )

    def keyboard_event(self, event) -> bool:
        handled = {
            carb.input.KeyboardInput.W,
            carb.input.KeyboardInput.A,
            carb.input.KeyboardInput.S,
            carb.input.KeyboardInput.D,
            carb.input.KeyboardInput.SPACE,
        }

        if event.input not in handled:
            return False

        if event.input == carb.input.KeyboardInput.SPACE:
            if event.type in (
                carb.input.KeyboardEventType.KEY_PRESS,
                carb.input.KeyboardEventType.KEY_REPEAT,
            ):
                self.pressed_keys.clear()
                self.current_linear = 0.0
                self.current_angular = 0.0
                self.set_wheel_targets(0.0, 0.0)
            return True

        if event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            self.pressed_keys.add(event.input)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.pressed_keys.discard(event.input)

        return True

    def keyboard_command(self) -> Tuple[float, float]:
        linear = 0.0
        angular = 0.0

        if carb.input.KeyboardInput.W in self.pressed_keys:
            linear += LINEAR_SPEED
        if carb.input.KeyboardInput.S in self.pressed_keys:
            linear -= LINEAR_SPEED
        if carb.input.KeyboardInput.A in self.pressed_keys:
            angular += ANGULAR_SPEED
        if carb.input.KeyboardInput.D in self.pressed_keys:
            angular -= ANGULAR_SPEED

        return linear, angular

    # ------------------------------------------------------------------
    # Articulation
    # ------------------------------------------------------------------

    def create_articulation(self) -> None:
        self.jackal = Articulation(ROBOT)
        self.wheel_indices = self.jackal.get_dof_indices(
            WHEEL_JOINTS
        ).numpy()

        print("[DR.Nav] Wheel indices:", self.wheel_indices)

    def set_wheel_targets(self, linear: float, angular: float) -> None:
        if self.jackal is None or self.wheel_indices is None:
            return

        half_track = TRACK_WIDTH / 2.0

        left = (
            linear - angular * half_track
        ) / WHEEL_RADIUS
        right = (
            linear + angular * half_track
        ) / WHEEL_RADIUS

        left *= LEFT_WHEEL_SIGN
        right *= RIGHT_WHEEL_SIGN

        left = float(np.clip(left, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
        right = float(np.clip(right, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))

        velocities = np.array(
            [[left, right, left, right]],
            dtype=np.float32,
        )

        self.jackal.set_dof_velocity_targets(
            velocities,
            dof_indices=self.wheel_indices,
        )

    # ------------------------------------------------------------------
    # TF, clock and odometry
    # ------------------------------------------------------------------

    def publish_static_transforms(self) -> None:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        base_prim = self.stage.GetPrimAtPath(BASE_LINK)

        transforms = []

        for path, frame in [
            (CAMERA_FRAME, FRAME_CAMERA),
            (LIDAR, FRAME_LIDAR),
        ]:
            prim = self.stage.GetPrimAtPath(path)
            matrix, _ = cache.ComputeRelativeTransform(prim, base_prim)
            transforms.append(
                self.matrix_to_transform(
                    matrix,
                    FRAME_BASE,
                    frame,
                    self.sim_time(),
                )
            )

        self.static_tf_pub.sendTransform(transforms)

    def matrix_to_transform(
        self,
        matrix,
        parent: str,
        child: str,
        sim_time: float,
    ):
        message = TransformStamped()
        self.set_stamp(message.header.stamp, sim_time)
        message.header.frame_id = parent
        message.child_frame_id = child

        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()

        message.transform.translation.x = float(translation[0])
        message.transform.translation.y = float(translation[1])
        message.transform.translation.z = float(translation[2])

        message.transform.rotation.x = float(imaginary[0])
        message.transform.rotation.y = float(imaginary[1])
        message.transform.rotation.z = float(imaginary[2])
        message.transform.rotation.w = float(rotation.GetReal())

        return message

    def publish_clock(self, sim_time: float) -> None:
        message = Clock()
        self.set_stamp(message.clock, sim_time)
        self.clock_pub.publish(message)

    def publish_odom(self, sim_time: float) -> None:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        matrix = cache.GetLocalToWorldTransform(
            self.stage.GetPrimAtPath(BASE_LINK)
        )

        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()

        x = float(translation[0])
        y = float(translation[1])
        z = float(translation[2])

        qx = float(imaginary[0])
        qy = float(imaginary[1])
        qz = float(imaginary[2])
        qw = float(rotation.GetReal())

        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        vx_body = 0.0
        vy_body = 0.0
        wz = 0.0

        if self.previous_pose_time is not None:
            dt = sim_time - self.previous_pose_time

            if dt > 1e-6:
                vx_world = (x - self.previous_position[0]) / dt
                vy_world = (y - self.previous_position[1]) / dt

                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)

                vx_body = cos_yaw * vx_world + sin_yaw * vy_world
                vy_body = -sin_yaw * vx_world + cos_yaw * vy_world
                wz = wrap_angle(yaw - self.previous_yaw) / dt

        self.previous_pose_time = sim_time
        self.previous_position = (x, y, z)
        self.previous_yaw = yaw

        transform = TransformStamped()
        self.set_stamp(transform.header.stamp, sim_time)
        transform.header.frame_id = FRAME_ODOM
        transform.child_frame_id = FRAME_BASE
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_pub.sendTransform(transform)

        odom = Odometry()
        self.set_stamp(odom.header.stamp, sim_time)
        odom.header.frame_id = FRAME_ODOM
        odom.child_frame_id = FRAME_BASE

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = vx_body
        odom.twist.twist.linear.y = vy_body
        odom.twist.twist.angular.z = wz

        self.odom_pub.publish(odom)

    # ------------------------------------------------------------------
    # PhysX LiDAR
    # ------------------------------------------------------------------

    @staticmethod
    def radians(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        finite = values[np.isfinite(values)]

        if finite.size and np.max(np.abs(finite)) > 2.0 * math.pi + 0.1:
            return np.deg2rad(values)

        return values

    def planar_scan(self, depth, azimuth, zenith):
        depth = np.asarray(depth, dtype=np.float32)
        azimuth = self.radians(np.asarray(azimuth, dtype=np.float32))
        zenith = self.radians(np.asarray(zenith, dtype=np.float32))

        if depth.size == 0 or azimuth.size == 0:
            return np.empty(0), np.empty(0)

        if depth.ndim >= 2:
            depth = depth.reshape(depth.shape[0], -1)

            if zenith.ndim == 1 and zenith.size == depth.shape[0]:
                row = int(np.argmin(np.abs(zenith)))
            else:
                row = depth.shape[0] // 2

            ranges = depth[row]

            if azimuth.ndim == 1 and azimuth.size == ranges.size:
                angles = azimuth
            elif azimuth.size == depth.size:
                angles = azimuth.reshape(depth.shape)[row]
            else:
                angles = np.linspace(
                    -math.pi,
                    math.pi,
                    ranges.size,
                    endpoint=False,
                    dtype=np.float32,
                )
        else:
            ranges = depth.reshape(-1)
            angles = (
                azimuth.reshape(-1)
                if azimuth.size == ranges.size
                else np.linspace(
                    -math.pi,
                    math.pi,
                    ranges.size,
                    endpoint=False,
                    dtype=np.float32,
                )
            )

        valid = np.isfinite(angles)
        ranges = ranges[valid]
        angles = angles[valid]
        order = np.argsort(angles)

        return ranges[order], angles[order]

    def publish_lidar(self, sim_time: float) -> None:
        depth = self.lidar_interface.get_linear_depth_data(LIDAR)
        azimuth = self.lidar_interface.get_azimuth_data(LIDAR)
        zenith = self.lidar_interface.get_zenith_data(LIDAR)

        ranges, angles = self.planar_scan(depth, azimuth, zenith)

        if ranges.size >= 2:
            message = LaserScan()
            self.set_stamp(message.header.stamp, sim_time)
            message.header.frame_id = FRAME_LIDAR
            message.angle_min = float(angles[0])
            message.angle_max = float(angles[-1])
            message.angle_increment = float(np.median(np.diff(angles)))
            message.scan_time = 1.0 / LIDAR_RATE
            message.range_min = self.range_min
            message.range_max = self.range_max

            valid = (
                np.isfinite(ranges)
                & (ranges >= self.range_min)
                & (ranges <= self.range_max)
            )
            message.ranges = np.where(
                valid,
                ranges,
                np.inf,
            ).astype(np.float32).tolist()

            self.scan_pub.publish(message)

        points = np.asarray(
            self.lidar_interface.get_point_cloud_data(LIDAR),
            dtype=np.float32,
        )

        if not points.size:
            return

        points = points.reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)]

        message = PointCloud2()
        self.set_stamp(message.header.stamp, sim_time)
        message.header.frame_id = FRAME_LIDAR
        message.height = 1
        message.width = int(points.shape[0])
        message.fields = [
            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]
        message.is_bigendian = False
        message.point_step = 12
        message.row_step = 12 * message.width
        message.is_dense = True
        message.data = np.asarray(points, dtype="<f4").tobytes()

        self.points_pub.publish(message)

    # ------------------------------------------------------------------
    # Physics callback
    # ------------------------------------------------------------------

    def physics_step(self, dt, context) -> None:
        del context

        try:
            rclpy.spin_once(self.ros_node, timeout_sec=0.0)

            if USE_WEBRTC_KEYBOARD:
                target_linear, target_angular = self.keyboard_command()
            elif time.monotonic() - self.last_cmd_time > CMD_TIMEOUT:
                target_linear, target_angular = 0.0, 0.0
            else:
                target_linear = self.cmd_linear
                target_angular = self.cmd_angular

            dt = max(float(dt), 0.0)

            self.current_linear = move_toward(
                self.current_linear,
                target_linear,
                MAX_LINEAR_ACCEL * dt,
            )
            self.current_angular = move_toward(
                self.current_angular,
                target_angular,
                MAX_ANGULAR_ACCEL * dt,
            )

            self.set_wheel_targets(
                self.current_linear,
                self.current_angular,
            )

            now = self.sim_time()
            self.publish_clock(now)

            if (
                self.last_odom_time < 0.0
                or now - self.last_odom_time >= 1.0 / ODOM_RATE
            ):
                self.publish_odom(now)
                self.last_odom_time = now

            if (
                self.last_lidar_time < 0.0
                or now - self.last_lidar_time >= 1.0 / LIDAR_RATE
            ):
                self.publish_lidar(now)
                self.last_lidar_time = now

        except Exception as exc:
            current = time.monotonic()

            if current - self.last_error_print > 2.0:
                print("[DR.Nav] Callback warning:", repr(exc))
                self.last_error_print = current


async def main() -> None:
    previous = getattr(builtins, _BRIDGE_KEY, None)

    if previous is not None:
        try:
            await previous.shutdown()
        except Exception as exc:
            print("[DR.Nav] Cleanup warning:", repr(exc))

    bridge = DRNavBridge()
    setattr(builtins, _BRIDGE_KEY, bridge)

    try:
        await bridge.start()
    except Exception:
        try:
            await bridge.shutdown()
        finally:
            setattr(builtins, _BRIDGE_KEY, None)
        raise


asyncio.ensure_future(main())

