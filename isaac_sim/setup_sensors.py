#!/usr/bin/env python3
"""
Isaac Sim 6.0 — DR.Nav Jackal Setup (v6c, WebRTC hold-key fix)
================================================================

Run from:
    Isaac Sim -> Window -> Script Editor

This version:
    - Reuses the existing Bumblebee left camera.
    - Reuses the existing sensor at:
        .../sick_lms1xx_lidar_frame/lidar
    - Never creates a second LiDAR.
    - Detects whether the existing LiDAR is RTX or legacy PhysX.
    - Publishes an RTX LiDAR through the official ROS2 RTX helper graph.
    - Publishes a PhysX LiDAR directly from its existing range-sensor interface.
    - Disables LiDAR ray/point debug drawing with temporary session-layer
      overrides, so the maze USD is not modified.
    - Creates the camera/RTX graph in the USD session layer, so saving the maze
      does not save DRNav_RuntimeGraph.
    - Supports W/A/S/D WebRTC keyboard teleoperation.
    - Keeps /cmd_vel support for later autonomous control.
    - Publishes /clock, /odom_lidar, TF, camera RGB, LaserScan and PointCloud2.

This script does not create, delete, activate, or relocate any sensor prim.
"""

from __future__ import annotations

import asyncio
import builtins
import math
import time
from typing import Optional, Tuple

import carb.input
import numpy as np
import omni.appwindow
import omni.graph.core as og
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom


# =============================================================================
# SCENE PATHS
# =============================================================================

ROBOT_PRIM = "/World/ground/flat_plane/jackal"
BASE_LINK_PATH = f"{ROBOT_PRIM}/base_link"

BUMBLEBEE_ROOT = f"{BASE_LINK_PATH}/bumblebee_stereo_camera_frame"

LEFT_CAMERA_FRAME_PATH = (
    f"{BUMBLEBEE_ROOT}/bumblebee_stereo_left_frame"
)
LEFT_CAMERA_PATH = (
    f"{LEFT_CAMERA_FRAME_PATH}/bumblebee_stereo_left_camera"
)

RIGHT_CAMERA_FRAME_PATH = (
    f"{BUMBLEBEE_ROOT}/bumblebee_stereo_right_frame"
)
RIGHT_CAMERA_PATH = (
    f"{RIGHT_CAMERA_FRAME_PATH}/bumblebee_stereo_right_camera"
)

LIDAR_FRAME_PATH = f"{BASE_LINK_PATH}/sick_lms1xx_lidar_frame"
LIDAR_SENSOR_PATH = f"{LIDAR_FRAME_PATH}/Lidar"

# This graph is authored only into the anonymous USD session layer.
GRAPH_PATH = "/World/DRNav_RuntimeGraph"


# =============================================================================
# ROS TOPICS AND FRAMES
# =============================================================================

TOPIC_FRONT_RGB = "/argus/ar0234_front_left/image_raw"
TOPIC_RIGHT_RGB = "/argus/ar0234_front_right/image_raw"
TOPIC_POINT_CLOUD = "/os_cloud_node/points"
TOPIC_SCAN = "/scan"
TOPIC_ODOM = "/odom_lidar"
TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_CLOCK = "/clock"

FRAME_BASE = "base_link"
FRAME_LEFT_CAMERA = "bumblebee_stereo_left_frame"
FRAME_RIGHT_CAMERA = "bumblebee_stereo_right_frame"
FRAME_LIDAR = "sim_lidar"
FRAME_ODOM = "odom"


# =============================================================================
# USER SETTINGS
# =============================================================================

CAMERA_RESOLUTION = (640, 480)
PUBLISH_RIGHT_STEREO_CAMERA = False

# The LiDAR debug drawing is forcibly disabled in the session layer.
SHOW_LIDAR_DEBUG_VIEW = False

# Direct keyboard control is useful when Isaac Sim is accessed through WebRTC.
# Change to False when an external ROS node should control /cmd_vel.
ENABLE_WEBRTC_KEYBOARD_TELEOP = True

KEYBOARD_LINEAR_SPEED_M_S = 0.15
KEYBOARD_ANGULAR_SPEED_RAD_S = 0.40

# WebRTC clients do not always emit repeated keyboard events while a key is
# held. The old 1.25-second watchdog therefore cleared W/A/S/D even though the
# user was still holding the key, producing a fixed travel distance and only a
# few degrees of turning.
ENABLE_KEYBOARD_WATCHDOG = False
KEYBOARD_EVENT_TIMEOUT_S = 10.0

# Smooth acceleration reduces wheelspin/slipping from instantaneous commands.
MAX_LINEAR_ACCEL_M_S2 = 0.35
MAX_ANGULAR_ACCEL_RAD_S2 = 0.90

# Jackal dimensions used by the repository controller.
WHEEL_RADIUS_M = 0.098
TRACK_WIDTH_M = 0.37559

WHEEL_JOINT_NAMES = [
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_left_wheel_joint",
    "rear_right_wheel_joint",
]

LEFT_WHEEL_SIGN = 1.0
RIGHT_WHEEL_SIGN = 1.0
MAX_WHEEL_SPEED_RAD_S = 25.0
CMD_TIMEOUT_S = 0.5

ODOM_PUBLISH_RATE_HZ = 30.0
PHYSX_LIDAR_PUBLISH_RATE_HZ = 10.0


# =============================================================================
# UTILITIES
# =============================================================================

def _move_toward(current: float, target: float, maximum_delta: float) -> float:
    if current < target:
        return min(current + maximum_delta, target)
    return max(current - maximum_delta, target)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


# =============================================================================
# MAIN BRIDGE
# =============================================================================

class DRNavIsaacSim6Bridge:
    def __init__(self) -> None:
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()

        self._lidar_backend = "unknown"
        self._physx_lidar_interface = None
        self._physx_min_range = 0.05
        self._physx_max_range = 100.0

        self._ros_node = None
        self._cmd_subscription = None
        self._odom_publisher = None
        self._clock_publisher = None
        self._scan_publisher = None
        self._pointcloud_publisher = None
        self._tf_broadcaster = None
        self._static_tf_broadcaster = None

        self._latest_linear = 0.0
        self._latest_angular = 0.0
        self._last_cmd_time = 0.0

        self._smoothed_linear = 0.0
        self._smoothed_angular = 0.0

        self._input_interface = None
        self._keyboard = None
        self._keyboard_sub_id = None
        self._pressed_keys = set()
        self._last_keyboard_event_time = 0.0

        self._jackal = None
        self._wheel_indices = None
        self._physics_callback_id = None

        self._last_odom_publish_time = -1.0
        self._last_lidar_publish_time = -1.0
        self._previous_pose_time = None
        self._previous_position = None
        self._previous_yaw = None

        self._last_callback_error_time = 0.0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        self._validate_required_prims()
        self._lidar_backend = self._detect_lidar_backend()
        await self._enable_extensions()

        global Articulation, SimulationManager, IsaacEvents
        global set_target_prims
        global rclpy, Twist, TransformStamped, Odometry, Clock
        global LaserScan, PointCloud2, PointField
        global TransformBroadcaster, StaticTransformBroadcaster

        from isaacsim.core.experimental.prims import Articulation
        from isaacsim.core.nodes.scripts.utils import set_target_prims
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents

        import rclpy
        from geometry_msgs.msg import TransformStamped, Twist
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import LaserScan, PointCloud2, PointField
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        if self._lidar_backend == "physx":
            from isaacsim.sensors.physx import _range_sensor

            self._physx_lidar_interface = (
                _range_sensor.acquire_lidar_sensor_interface()
            )

        await self._cleanup_previous_runtime_graph()
        self._disable_lidar_visualization_in_session_layer()
        self._setup_sensor_graph_in_session_layer()
        self._setup_ros_node()

        if ENABLE_WEBRTC_KEYBOARD_TELEOP:
            self._setup_keyboard_teleop()

        self.timeline.play()

        # Give physics, sensor buffers and OmniGraph time to initialize.
        for _ in range(6):
            await omni.kit.app.get_app().next_update_async()

        self._setup_jackal_articulation()
        self._publish_static_sensor_transforms()
        self._register_physics_callback()

        print("\n[DR.Nav] Isaac Sim 6.0 v6 setup complete.")
        print(f"[DR.Nav] Existing LiDAR backend: {self._lidar_backend}")
        print("[DR.Nav] Simulation is playing.")
        self._print_instructions()

    async def shutdown(self) -> None:
        """Stop callbacks and remove only session-layer runtime content."""
        try:
            self.timeline.stop()
        except Exception:
            pass

        try:
            if self._physics_callback_id is not None:
                SimulationManager.deregister_callback(self._physics_callback_id)
                self._physics_callback_id = None
        except Exception as exc:
            print("[DR.Nav] Callback cleanup warning:", repr(exc))

        try:
            self._set_wheel_targets(0.0, 0.0)
        except Exception:
            pass

        if (
            self._input_interface is not None
            and self._keyboard is not None
            and self._keyboard_sub_id is not None
        ):
            try:
                self._input_interface.unsubscribe_to_keyboard_events(
                    self._keyboard,
                    self._keyboard_sub_id,
                )
            except Exception as exc:
                print("[DR.Nav] Keyboard cleanup warning:", repr(exc))

        self._keyboard_sub_id = None
        self._pressed_keys.clear()

        if self._ros_node is not None:
            try:
                self._ros_node.destroy_node()
            except Exception:
                pass
            self._ros_node = None

        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

        await self._remove_graph_from_session_layer()
        print("[DR.Nav] Previous v6 runtime bridge cleaned up.")

    # -------------------------------------------------------------------------
    # Validation and backend detection
    # -------------------------------------------------------------------------

    def _validate_required_prims(self) -> None:
        required = {
            "Jackal": ROBOT_PRIM,
            "base_link": BASE_LINK_PATH,
            "left Bumblebee frame": LEFT_CAMERA_FRAME_PATH,
            "left Bumblebee camera": LEFT_CAMERA_PATH,
            "existing LiDAR frame": LIDAR_FRAME_PATH,
            "existing LiDAR sensor": LIDAR_SENSOR_PATH,
        }

        if PUBLISH_RIGHT_STEREO_CAMERA:
            required["right Bumblebee frame"] = RIGHT_CAMERA_FRAME_PATH
            required["right Bumblebee camera"] = RIGHT_CAMERA_PATH

        missing = [
            f"{name}: {path}"
            for name, path in required.items()
            if not self.stage.GetPrimAtPath(path).IsValid()
        ]

        if missing:
            raise RuntimeError(
                "Required scene prims were not found:\n  "
                + "\n  ".join(missing)
            )

        lidar = self.stage.GetPrimAtPath(LIDAR_SENSOR_PATH)
        if not lidar.IsActive():
            raise RuntimeError(
                "The existing LiDAR is inactive. Reactivate the original "
                f"sensor before running this script:\n{LIDAR_SENSOR_PATH}"
            )

    def _detect_lidar_backend(self) -> str:
        prim = self.stage.GetPrimAtPath(LIDAR_SENSOR_PATH)
        type_name = prim.GetTypeName()
        attribute_names = {attr.GetName() for attr in prim.GetAttributes()}
        lower_attrs = {name.lower() for name in attribute_names}
        schemas = [str(item) for item in prim.GetAppliedSchemas()]

        is_rtx = (
            type_name == "OmniLidar"
            or any(name.startswith("omni:sensor:") for name in lower_attrs)
            or any("omnilidar" in schema.lower() for schema in schemas)
        )

        is_physx = (
            type_name == "Lidar"
            or any("physxlidar" in name for name in lower_attrs)
            or any("rangesensor" in name for name in lower_attrs)
            or any("physxlidar" in schema.lower() for schema in schemas)
        )

        print("\n[DR.Nav] Existing LiDAR inspection:")
        print("  path:", LIDAR_SENSOR_PATH)
        print("  type:", type_name)
        print("  active:", prim.IsActive())
        print("  schemas:", schemas)

        interesting_tokens = (
            "draw",
            "range",
            "rotation",
            "resolution",
            "fov",
            "tickrate",
            "scanrate",
        )
        print("  relevant attributes:")

        for attr in prim.GetAttributes():
            name = attr.GetName()
            if any(token in name.lower() for token in interesting_tokens):
                try:
                    print(f"    {name} = {attr.Get()}")
                except Exception:
                    print(f"    {name} = <unreadable>")

        if is_rtx:
            print("[DR.Nav] Detected RTX/OmniLidar sensor.")
            return "rtx"

        if is_physx:
            print("[DR.Nav] Detected legacy PhysX Lidar sensor.")
            return "physx"

        raise RuntimeError(
            "The existing LiDAR type was not recognized as RTX or PhysX. "
            "No sensor was created or modified. Copy the inspection output "
            "above before changing the script."
        )

    # -------------------------------------------------------------------------
    # Extensions
    # -------------------------------------------------------------------------

    async def _enable_extensions(self) -> None:
        manager = omni.kit.app.get_app().get_extension_manager()

        extension_ids = [
            "isaacsim.core.nodes",
            "isaacsim.ros2.bridge",
            "isaacsim.ros2.nodes",
        ]

        if self._lidar_backend == "rtx":
            extension_ids.append("isaacsim.sensors.rtx.nodes")
        else:
            extension_ids.append("isaacsim.sensors.physx")

        for extension_id in extension_ids:
            if not manager.is_extension_enabled(extension_id):
                manager.set_extension_enabled_immediate(extension_id, True)
                print(f"[DR.Nav] Enabled extension: {extension_id}")
            else:
                print(
                    f"[DR.Nav] Extension already enabled: {extension_id}"
                )

        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

    # -------------------------------------------------------------------------
    # USD session-layer runtime content
    # -------------------------------------------------------------------------

    def _session_edit_target(self):
        return self.stage.GetEditTargetForLocalLayer(
            self.stage.GetSessionLayer()
        )

    async def _cleanup_previous_runtime_graph(self) -> None:
        await self._remove_graph_from_session_layer()

    async def _remove_graph_from_session_layer(self) -> None:
        session_layer = self.stage.GetSessionLayer()

        with Usd.EditContext(self.stage, self._session_edit_target()):
            if self.stage.GetPrimAtPath(GRAPH_PATH).IsValid():
                self.stage.RemovePrim(GRAPH_PATH)

        # Also remove a remaining Sdf prim specification directly when present.
        try:
            session_spec = session_layer.GetPrimAtPath(GRAPH_PATH)
            if session_spec:
                session_layer.RemovePrim(GRAPH_PATH)
        except Exception:
            pass

        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()

    def _disable_lidar_visualization_in_session_layer(self) -> None:
        prim = self.stage.GetPrimAtPath(LIDAR_SENSOR_PATH)
        changed = []

        with Usd.EditContext(self.stage, self._session_edit_target()):
            for attr in prim.GetAttributes():
                name = attr.GetName()
                lower = name.lower()

                is_debug_draw = (
                    lower.endswith("drawlines")
                    or lower.endswith("drawpoints")
                    or lower.endswith("showdebugview")
                    or lower.endswith("debugdraw")
                )

                if not is_debug_draw:
                    continue

                try:
                    old_value = attr.Get()
                    attr.Set(False)
                    changed.append((name, old_value))
                except Exception as exc:
                    print(
                        f"[DR.Nav] Could not disable {name}: {exc!r}"
                    )

        if changed:
            print(
                "[DR.Nav] LiDAR visualization disabled temporarily "
                "in the USD session layer:"
            )
            for name, old_value in changed:
                print(f"  {name}: {old_value} -> False")
        else:
            print(
                "[DR.Nav] No authored LiDAR draw-lines/draw-points "
                "attributes were found."
            )

        if self._lidar_backend == "physx":
            self._physx_min_range = self._read_numeric_lidar_attribute(
                suffixes=("minrange",),
                fallback=0.05,
            )
            self._physx_max_range = self._read_numeric_lidar_attribute(
                suffixes=("maxrange",),
                fallback=100.0,
            )

    def _read_numeric_lidar_attribute(
        self,
        suffixes: Tuple[str, ...],
        fallback: float,
    ) -> float:
        prim = self.stage.GetPrimAtPath(LIDAR_SENSOR_PATH)

        for attr in prim.GetAttributes():
            lower = attr.GetName().lower()
            if any(lower.endswith(suffix) for suffix in suffixes):
                try:
                    value = attr.Get()
                    if value is not None:
                        return float(value)
                except Exception:
                    pass

        return fallback

    def _setup_sensor_graph_in_session_layer(self) -> None:
        """
        Camera is always published through an Isaac Create Render Product node.
        RTX LiDAR uses the same official render-product/helper pipeline.
        PhysX LiDAR is published directly in Python and needs no graph node.
        """
        create_nodes = [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("context", "isaacsim.ros2.bridge.ROS2Context"),
            (
                "sensor_once",
                "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame",
            ),
            (
                "camera_render_product",
                "isaacsim.core.nodes.IsaacCreateRenderProduct",
            ),
            (
                "camera_left",
                "isaacsim.ros2.bridge.ROS2CameraHelper",
            ),
        ]

        connections = [
            ("tick.outputs:tick", "sensor_once.inputs:execIn"),
            (
                "sensor_once.outputs:step",
                "camera_render_product.inputs:execIn",
            ),
            (
                "camera_render_product.outputs:execOut",
                "camera_left.inputs:execIn",
            ),
            (
                "camera_render_product.outputs:renderProductPath",
                "camera_left.inputs:renderProductPath",
            ),
            (
                "context.outputs:context",
                "camera_left.inputs:context",
            ),
        ]

        set_values = [
            ("context.inputs:useDomainIDEnvVar", True),
            (
                "camera_render_product.inputs:width",
                int(CAMERA_RESOLUTION[0]),
            ),
            (
                "camera_render_product.inputs:height",
                int(CAMERA_RESOLUTION[1]),
            ),
            ("camera_left.inputs:topicName", TOPIC_FRONT_RGB),
            ("camera_left.inputs:frameId", FRAME_LEFT_CAMERA),
            ("camera_left.inputs:type", "rgb"),
            ("camera_left.inputs:enabled", True),
        ]

        if PUBLISH_RIGHT_STEREO_CAMERA:
            create_nodes.extend(
                [
                    (
                        "camera_right_render_product",
                        "isaacsim.core.nodes.IsaacCreateRenderProduct",
                    ),
                    (
                        "camera_right",
                        "isaacsim.ros2.bridge.ROS2CameraHelper",
                    ),
                ]
            )
            connections.extend(
                [
                    (
                        "sensor_once.outputs:step",
                        "camera_right_render_product.inputs:execIn",
                    ),
                    (
                        "camera_right_render_product.outputs:execOut",
                        "camera_right.inputs:execIn",
                    ),
                    (
                        "camera_right_render_product.outputs:"
                        "renderProductPath",
                        "camera_right.inputs:renderProductPath",
                    ),
                    (
                        "context.outputs:context",
                        "camera_right.inputs:context",
                    ),
                ]
            )
            set_values.extend(
                [
                    (
                        "camera_right_render_product.inputs:width",
                        int(CAMERA_RESOLUTION[0]),
                    ),
                    (
                        "camera_right_render_product.inputs:height",
                        int(CAMERA_RESOLUTION[1]),
                    ),
                    (
                        "camera_right.inputs:topicName",
                        TOPIC_RIGHT_RGB,
                    ),
                    (
                        "camera_right.inputs:frameId",
                        FRAME_RIGHT_CAMERA,
                    ),
                    ("camera_right.inputs:type", "rgb"),
                    ("camera_right.inputs:enabled", True),
                ]
            )

        if self._lidar_backend == "rtx":
            create_nodes.extend(
                [
                    (
                        "lidar_render_product",
                        "isaacsim.core.nodes.IsaacCreateRenderProduct",
                    ),
                    (
                        "lidar_pointcloud",
                        "isaacsim.ros2.bridge.ROS2RtxLidarHelper",
                    ),
                ]
            )
            connections.extend(
                [
                    (
                        "sensor_once.outputs:step",
                        "lidar_render_product.inputs:execIn",
                    ),
                    (
                        "lidar_render_product.outputs:execOut",
                        "lidar_pointcloud.inputs:execIn",
                    ),
                    (
                        "lidar_render_product.outputs:renderProductPath",
                        "lidar_pointcloud.inputs:renderProductPath",
                    ),
                    (
                        "context.outputs:context",
                        "lidar_pointcloud.inputs:context",
                    ),
                ]
            )
            set_values.extend(
                [
                    ("lidar_render_product.inputs:width", 1),
                    ("lidar_render_product.inputs:height", 1),
                    (
                        "lidar_pointcloud.inputs:topicName",
                        TOPIC_POINT_CLOUD,
                    ),
                    (
                        "lidar_pointcloud.inputs:frameId",
                        FRAME_LIDAR,
                    ),
                    (
                        "lidar_pointcloud.inputs:type",
                        "point_cloud",
                    ),
                    (
                        "lidar_pointcloud.inputs:showDebugView",
                        SHOW_LIDAR_DEBUG_VIEW,
                    ),
                    ("lidar_pointcloud.inputs:enabled", True),
                ]
            )

        with Usd.EditContext(self.stage, self._session_edit_target()):
            og.Controller.edit(
                {
                    "graph_path": GRAPH_PATH,
                    "evaluator_name": "execution",
                    "pipeline_stage": (
                        og.GraphPipelineStage
                        .GRAPH_PIPELINE_STAGE_SIMULATION
                    ),
                },
                {
                    og.Controller.Keys.CREATE_NODES: create_nodes,
                    og.Controller.Keys.CONNECT: connections,
                    og.Controller.Keys.SET_VALUES: set_values,
                },
            )

            set_target_prims(
                primPath=f"{GRAPH_PATH}/camera_render_product",
                inputName="inputs:cameraPrim",
                targetPrimPaths=[LEFT_CAMERA_PATH],
            )

            if PUBLISH_RIGHT_STEREO_CAMERA:
                set_target_prims(
                    primPath=(
                        f"{GRAPH_PATH}/camera_right_render_product"
                    ),
                    inputName="inputs:cameraPrim",
                    targetPrimPaths=[RIGHT_CAMERA_PATH],
                )

            if self._lidar_backend == "rtx":
                set_target_prims(
                    primPath=f"{GRAPH_PATH}/lidar_render_product",
                    inputName="inputs:cameraPrim",
                    targetPrimPaths=[LIDAR_SENSOR_PATH],
                )

        graph_prim = self.stage.GetPrimAtPath(GRAPH_PATH)
        session_id = self.stage.GetSessionLayer().identifier
        authored_layers = [
            spec.layer.identifier for spec in graph_prim.GetPrimStack()
        ]

        print(f"[DR.Nav] Runtime sensor graph created: {GRAPH_PATH}")
        print("[DR.Nav] Graph authored layers:", authored_layers)

        if session_id not in authored_layers:
            raise RuntimeError(
                "Safety check failed: the runtime graph was not authored "
                "into the USD session layer. Do not save the maze."
            )

        print(
            "[DR.Nav] Session-layer safety check passed; "
            "the graph will not be saved with the maze."
        )

    # -------------------------------------------------------------------------
    # ROS Python publishers/subscriber
    # -------------------------------------------------------------------------

    def _setup_ros_node(self) -> None:
        if not rclpy.ok():
            rclpy.init()

        self._ros_node = rclpy.create_node(
            "drnav_jackal_isaacsim6_v6"
        )

        self._cmd_subscription = self._ros_node.create_subscription(
            Twist,
            TOPIC_CMD_VEL,
            self._cmd_vel_callback,
            10,
        )

        self._odom_publisher = self._ros_node.create_publisher(
            Odometry,
            TOPIC_ODOM,
            10,
        )
        self._clock_publisher = self._ros_node.create_publisher(
            Clock,
            TOPIC_CLOCK,
            10,
        )

        if self._lidar_backend == "physx":
            self._scan_publisher = self._ros_node.create_publisher(
                LaserScan,
                TOPIC_SCAN,
                10,
            )
            self._pointcloud_publisher = (
                self._ros_node.create_publisher(
                    PointCloud2,
                    TOPIC_POINT_CLOUD,
                    10,
                )
            )

        self._tf_broadcaster = TransformBroadcaster(self._ros_node)
        self._static_tf_broadcaster = StaticTransformBroadcaster(
            self._ros_node
        )

        self._last_cmd_time = time.monotonic()
        print(f"[DR.Nav] Subscribed to {TOPIC_CMD_VEL}")

        if self._lidar_backend == "physx":
            print(
                "[DR.Nav] Existing PhysX LiDAR will publish directly to:"
            )
            print(f"  {TOPIC_SCAN}")
            print(f"  {TOPIC_POINT_CLOUD}")

    def _cmd_vel_callback(self, message) -> None:
        self._latest_linear = float(message.linear.x)
        self._latest_angular = float(message.angular.z)
        self._last_cmd_time = time.monotonic()

    def _sim_time(self) -> float:
        return float(self.timeline.get_current_time())

    def _set_stamp(self, stamp, sim_time: Optional[float] = None) -> None:
        value = self._sim_time() if sim_time is None else sim_time
        sec = int(math.floor(value))
        nanosec = int((value - sec) * 1_000_000_000)

        stamp.sec = sec
        stamp.nanosec = nanosec

    # -------------------------------------------------------------------------
    # TF and odometry
    # -------------------------------------------------------------------------

    def _matrix_to_transform(
        self,
        matrix,
        parent_frame: str,
        child_frame: str,
    ):
        transform = TransformStamped()
        self._set_stamp(transform.header.stamp)
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame

        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()

        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])

        transform.transform.rotation.x = float(imaginary[0])
        transform.transform.rotation.y = float(imaginary[1])
        transform.transform.rotation.z = float(imaginary[2])
        transform.transform.rotation.w = float(rotation.GetReal())

        return transform

    def _publish_static_sensor_transforms(self) -> None:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        base_prim = self.stage.GetPrimAtPath(BASE_LINK_PATH)

        sensor_frames = [
            (LEFT_CAMERA_FRAME_PATH, FRAME_LEFT_CAMERA),
            (LIDAR_SENSOR_PATH, FRAME_LIDAR),
        ]

        if PUBLISH_RIGHT_STEREO_CAMERA:
            sensor_frames.append(
                (RIGHT_CAMERA_FRAME_PATH, FRAME_RIGHT_CAMERA)
            )

        transforms = []

        for prim_path, frame_id in sensor_frames:
            sensor_prim = self.stage.GetPrimAtPath(prim_path)
            try:
                relative_matrix, _ = cache.ComputeRelativeTransform(
                    sensor_prim,
                    base_prim,
                )
                transforms.append(
                    self._matrix_to_transform(
                        relative_matrix,
                        FRAME_BASE,
                        frame_id,
                    )
                )
            except Exception as exc:
                print(
                    f"[DR.Nav] Static TF warning for {prim_path}: "
                    f"{exc!r}"
                )

        if transforms:
            self._static_tf_broadcaster.sendTransform(transforms)
            print(
                "[DR.Nav] Published static sensor TFs:",
                ", ".join(t.child_frame_id for t in transforms),
            )

    def _publish_clock(self, sim_time: float) -> None:
        message = Clock()
        self._set_stamp(message.clock, sim_time)
        self._clock_publisher.publish(message)

    def _publish_odom_and_tf(self, sim_time: float) -> None:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        base_prim = self.stage.GetPrimAtPath(BASE_LINK_PATH)
        world_matrix = cache.GetLocalToWorldTransform(base_prim)

        translation = world_matrix.ExtractTranslation()
        rotation = world_matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()

        x = float(translation[0])
        y = float(translation[1])
        z = float(translation[2])

        qx = float(imaginary[0])
        qy = float(imaginary[1])
        qz = float(imaginary[2])
        qw = float(rotation.GetReal())
        yaw = _quat_to_yaw(qw, qx, qy, qz)

        vx_body = 0.0
        vy_body = 0.0
        wz = 0.0

        if (
            self._previous_pose_time is not None
            and self._previous_position is not None
            and self._previous_yaw is not None
        ):
            dt = sim_time - self._previous_pose_time

            if dt > 1e-6:
                vx_world = (
                    x - self._previous_position[0]
                ) / dt
                vy_world = (
                    y - self._previous_position[1]
                ) / dt

                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)

                vx_body = cos_yaw * vx_world + sin_yaw * vy_world
                vy_body = -sin_yaw * vx_world + cos_yaw * vy_world
                wz = _wrap_angle(
                    yaw - self._previous_yaw
                ) / dt

        self._previous_pose_time = sim_time
        self._previous_position = (x, y, z)
        self._previous_yaw = yaw

        transform = TransformStamped()
        self._set_stamp(transform.header.stamp, sim_time)
        transform.header.frame_id = FRAME_ODOM
        transform.child_frame_id = FRAME_BASE
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(transform)

        odom = Odometry()
        self._set_stamp(odom.header.stamp, sim_time)
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

        self._odom_publisher.publish(odom)

    # -------------------------------------------------------------------------
    # Existing PhysX LiDAR publishing
    # -------------------------------------------------------------------------

    @staticmethod
    def _angles_to_radians(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)

        finite = values[np.isfinite(values)]
        if finite.size and np.max(np.abs(finite)) > 2.0 * math.pi + 0.1:
            return np.deg2rad(values)

        return values

    def _extract_planar_scan(
        self,
        depth_data,
        azimuth_data,
        zenith_data,
    ) -> Tuple[np.ndarray, np.ndarray]:
        depths = np.asarray(depth_data, dtype=np.float32)
        azimuths = self._angles_to_radians(
            np.asarray(azimuth_data, dtype=np.float32)
        )
        zeniths = self._angles_to_radians(
            np.asarray(zenith_data, dtype=np.float32)
        )

        if depths.size == 0 or azimuths.size == 0:
            return np.empty(0), np.empty(0)

        # Common PhysX layout: vertical rows x horizontal columns.
        if depths.ndim >= 2:
            depth_2d = depths.reshape(depths.shape[0], -1)

            if zeniths.ndim == 1 and zeniths.size == depth_2d.shape[0]:
                row_index = int(np.argmin(np.abs(zeniths)))
            elif (
                zeniths.size == depth_2d.size
            ):
                zenith_2d = zeniths.reshape(depth_2d.shape)
                row_scores = np.nanmean(
                    np.abs(zenith_2d),
                    axis=1,
                )
                row_index = int(np.nanargmin(row_scores))
            else:
                row_index = depth_2d.shape[0] // 2

            ranges = depth_2d[row_index]

            if azimuths.ndim == 1 and azimuths.size == ranges.size:
                angles = azimuths
            elif azimuths.size == depth_2d.size:
                angles = azimuths.reshape(
                    depth_2d.shape
                )[row_index]
            else:
                angles = np.linspace(
                    -math.pi,
                    math.pi,
                    ranges.size,
                    endpoint=False,
                    dtype=np.float32,
                )
        else:
            ranges = depths.reshape(-1)

            if azimuths.size == ranges.size:
                angles = azimuths.reshape(-1)
            else:
                angles = np.linspace(
                    -math.pi,
                    math.pi,
                    ranges.size,
                    endpoint=False,
                    dtype=np.float32,
                )

        finite_mask = np.isfinite(angles)
        ranges = ranges[finite_mask]
        angles = angles[finite_mask]

        if ranges.size == 0:
            return ranges, angles

        order = np.argsort(angles)
        return ranges[order], angles[order]

    def _publish_physx_lidar(self, sim_time: float) -> None:
        interface = self._physx_lidar_interface

        depth = interface.get_linear_depth_data(LIDAR_SENSOR_PATH)
        azimuth = interface.get_azimuth_data(LIDAR_SENSOR_PATH)
        zenith = interface.get_zenith_data(LIDAR_SENSOR_PATH)

        ranges, angles = self._extract_planar_scan(
            depth,
            azimuth,
            zenith,
        )

        if ranges.size >= 2:
            scan = LaserScan()
            self._set_stamp(scan.header.stamp, sim_time)
            scan.header.frame_id = FRAME_LIDAR

            scan.angle_min = float(angles[0])
            scan.angle_max = float(angles[-1])
            scan.angle_increment = float(
                np.median(np.diff(angles))
            )
            scan.time_increment = 0.0
            scan.scan_time = 1.0 / PHYSX_LIDAR_PUBLISH_RATE_HZ
            scan.range_min = float(self._physx_min_range)
            scan.range_max = float(self._physx_max_range)

            valid = (
                np.isfinite(ranges)
                & (ranges >= scan.range_min)
                & (ranges <= scan.range_max)
            )
            output_ranges = np.where(valid, ranges, np.inf)
            scan.ranges = output_ranges.astype(np.float32).tolist()
            scan.intensities = []

            self._scan_publisher.publish(scan)

        points = np.asarray(
            interface.get_point_cloud_data(LIDAR_SENSOR_PATH),
            dtype=np.float32,
        )

        if points.size:
            points = points.reshape(-1, 3)
            valid_points = np.all(np.isfinite(points), axis=1)
            points = points[valid_points]

            cloud = PointCloud2()
            self._set_stamp(cloud.header.stamp, sim_time)
            cloud.header.frame_id = FRAME_LIDAR
            cloud.height = 1
            cloud.width = int(points.shape[0])
            cloud.fields = [
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
            cloud.is_bigendian = False
            cloud.point_step = 12
            cloud.row_step = cloud.point_step * cloud.width
            cloud.is_dense = bool(np.all(valid_points))
            cloud.data = (
                np.asarray(points, dtype="<f4")
                .reshape(-1, 3)
                .tobytes()
            )

            self._pointcloud_publisher.publish(cloud)

    # -------------------------------------------------------------------------
    # WebRTC keyboard teleoperation
    # -------------------------------------------------------------------------

    def _setup_keyboard_teleop(self) -> None:
        app_window = omni.appwindow.get_default_app_window()

        if app_window is None:
            raise RuntimeError(
                "Isaac Sim's default application window was unavailable."
            )

        self._keyboard = app_window.get_keyboard()
        self._input_interface = carb.input.acquire_input_interface()
        self._keyboard_sub_id = (
            self._input_interface.subscribe_to_keyboard_events(
                self._keyboard,
                self._on_keyboard_event,
            )
        )
        self._last_keyboard_event_time = time.monotonic()

        print("[DR.Nav] WebRTC keyboard teleop enabled.")
        print("[DR.Nav] Click the 3D viewport, then hold:")
        print("         W = forward, S = reverse")
        print("         A = turn left, D = turn right")
        print("         SPACE = immediate stop")
        print(
            "[DR.Nav] WebRTC key watchdog disabled: held keys remain active "
            "until KEY_RELEASE."
        )

    def _on_keyboard_event(self, event) -> bool:
        handled_keys = {
            carb.input.KeyboardInput.W,
            carb.input.KeyboardInput.A,
            carb.input.KeyboardInput.S,
            carb.input.KeyboardInput.D,
            carb.input.KeyboardInput.SPACE,
        }

        if event.input not in handled_keys:
            return False

        self._last_keyboard_event_time = time.monotonic()

        if event.input == carb.input.KeyboardInput.SPACE:
            if event.type in (
                carb.input.KeyboardEventType.KEY_PRESS,
                carb.input.KeyboardEventType.KEY_REPEAT,
            ):
                self._pressed_keys.clear()
                self._smoothed_linear = 0.0
                self._smoothed_angular = 0.0
                self._set_wheel_targets(0.0, 0.0)
            return True

        if event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            self._pressed_keys.add(event.input)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed_keys.discard(event.input)

        return True

    def _get_keyboard_command(self) -> Tuple[float, float]:
        if (
            ENABLE_KEYBOARD_WATCHDOG
            and self._pressed_keys
            and time.monotonic() - self._last_keyboard_event_time
            > KEYBOARD_EVENT_TIMEOUT_S
        ):
            self._pressed_keys.clear()

        linear = 0.0
        angular = 0.0

        if carb.input.KeyboardInput.W in self._pressed_keys:
            linear += KEYBOARD_LINEAR_SPEED_M_S
        if carb.input.KeyboardInput.S in self._pressed_keys:
            linear -= KEYBOARD_LINEAR_SPEED_M_S
        if carb.input.KeyboardInput.A in self._pressed_keys:
            angular += KEYBOARD_ANGULAR_SPEED_RAD_S
        if carb.input.KeyboardInput.D in self._pressed_keys:
            angular -= KEYBOARD_ANGULAR_SPEED_RAD_S

        return linear, angular

    # -------------------------------------------------------------------------
    # Jackal articulation and physics loop
    # -------------------------------------------------------------------------

    def _setup_jackal_articulation(self) -> None:
        self._jackal = Articulation(ROBOT_PRIM)

        print("[DR.Nav] Available Jackal DOFs:")
        print(" ", self._jackal.dof_names)

        try:
            self._wheel_indices = (
                self._jackal.get_dof_indices(
                    WHEEL_JOINT_NAMES
                ).numpy()
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not resolve the configured wheel joints.\n"
                f"Configured: {WHEEL_JOINT_NAMES}\n"
                f"Available: {self._jackal.dof_names}"
            ) from exc

        print("[DR.Nav] Wheel DOF indices:", self._wheel_indices)

    def _register_physics_callback(self) -> None:
        self._physics_callback_id = SimulationManager.register_callback(
            self._physics_step,
            IsaacEvents.POST_PHYSICS_STEP,
        )
        print("[DR.Nav] Registered Isaac Sim 6.0 physics callback.")

    def _physics_step(self, dt, context) -> None:
        del context

        try:
            rclpy.spin_once(self._ros_node, timeout_sec=0.0)

            if ENABLE_WEBRTC_KEYBOARD_TELEOP:
                target_linear, target_angular = (
                    self._get_keyboard_command()
                )
            elif time.monotonic() - self._last_cmd_time > CMD_TIMEOUT_S:
                target_linear = 0.0
                target_angular = 0.0
            else:
                target_linear = self._latest_linear
                target_angular = self._latest_angular

            step_dt = max(float(dt), 0.0)
            self._smoothed_linear = _move_toward(
                self._smoothed_linear,
                target_linear,
                MAX_LINEAR_ACCEL_M_S2 * step_dt,
            )
            self._smoothed_angular = _move_toward(
                self._smoothed_angular,
                target_angular,
                MAX_ANGULAR_ACCEL_RAD_S2 * step_dt,
            )

            self._set_wheel_targets(
                self._smoothed_linear,
                self._smoothed_angular,
            )

            sim_time = self._sim_time()
            self._publish_clock(sim_time)

            odom_period = 1.0 / ODOM_PUBLISH_RATE_HZ
            if (
                self._last_odom_publish_time < 0.0
                or sim_time - self._last_odom_publish_time >= odom_period
            ):
                self._publish_odom_and_tf(sim_time)
                self._last_odom_publish_time = sim_time

            if self._lidar_backend == "physx":
                lidar_period = 1.0 / PHYSX_LIDAR_PUBLISH_RATE_HZ

                if (
                    self._last_lidar_publish_time < 0.0
                    or sim_time - self._last_lidar_publish_time
                    >= lidar_period
                ):
                    self._publish_physx_lidar(sim_time)
                    self._last_lidar_publish_time = sim_time

        except Exception as exc:
            now = time.monotonic()

            if now - self._last_callback_error_time > 2.0:
                print(
                    "[DR.Nav] Physics callback warning:",
                    repr(exc),
                )
                self._last_callback_error_time = now

    def _set_wheel_targets(
        self,
        linear_m_s: float,
        angular_rad_s: float,
    ) -> None:
        if self._jackal is None or self._wheel_indices is None:
            return

        half_track = TRACK_WIDTH_M / 2.0

        left_rad_s = (
            linear_m_s - angular_rad_s * half_track
        ) / WHEEL_RADIUS_M

        right_rad_s = (
            linear_m_s + angular_rad_s * half_track
        ) / WHEEL_RADIUS_M

        left_rad_s *= LEFT_WHEEL_SIGN
        right_rad_s *= RIGHT_WHEEL_SIGN

        left_rad_s = float(
            np.clip(
                left_rad_s,
                -MAX_WHEEL_SPEED_RAD_S,
                MAX_WHEEL_SPEED_RAD_S,
            )
        )
        right_rad_s = float(
            np.clip(
                right_rad_s,
                -MAX_WHEEL_SPEED_RAD_S,
                MAX_WHEEL_SPEED_RAD_S,
            )
        )

        wheel_velocities = np.array(
            [[left_rad_s, right_rad_s, left_rad_s, right_rad_s]],
            dtype=np.float32,
        )

        self._jackal.set_dof_velocity_targets(
            wheel_velocities,
            dof_indices=self._wheel_indices,
        )

    # -------------------------------------------------------------------------
    # Instructions
    # -------------------------------------------------------------------------

    @staticmethod
    def _print_instructions() -> None:
        print(
            f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WebRTC keyboard teleop:

  Click the 3D viewport.
  Hold W/S to move forward/reverse.
  Hold A/D to turn left/right.
  Press SPACE to stop immediately.

Expected ROS topics:

  {TOPIC_FRONT_RGB}
  {TOPIC_POINT_CLOUD}
  {TOPIC_SCAN}        (published directly for PhysX LiDAR)
  {TOPIC_ODOM}
  {TOPIC_CLOCK}

The graph path is:

  {GRAPH_PATH}

It is session-layer-only and is not saved with the maze USD.

Important:
  This script reuses {LIDAR_SENSOR_PATH}
  and never creates drnav_rtx_lidar.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        )


# =============================================================================
# SCRIPT EDITOR ENTRY POINT
# =============================================================================

_BRIDGE_KEY = "_DRNAV_ISAACSIM6_V6_BRIDGE"


async def _main() -> None:
    previous = getattr(builtins, _BRIDGE_KEY, None)

    if previous is not None:
        try:
            await previous.shutdown()
        except Exception as exc:
            print(
                "[DR.Nav] Previous bridge cleanup warning:",
                repr(exc),
            )

    bridge = DRNavIsaacSim6Bridge()
    setattr(builtins, _BRIDGE_KEY, bridge)

    try:
        await bridge.start()
    except Exception:
        try:
            await bridge.shutdown()
        finally:
            setattr(builtins, _BRIDGE_KEY, None)
        raise


asyncio.ensure_future(_main())

