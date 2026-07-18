#!/usr/bin/env python3
"""
Isaac Sim 6.0 — DR.Nav Jackal Sensor + WebRTC Keyboard Teleop (v3)
===================================================

Run from:
    Isaac Sim -> Window -> Script Editor

What this version does:
    - Reuses the existing Jackal Bumblebee stereo camera prims.
    - Publishes the left camera as the primary DR.Nav RGB stream.
    - Optionally publishes the right stereo camera.
    - Creates one RTX rotary LiDAR under the existing SICK LiDAR frame.
    - Publishes PointCloud2 and LaserScan through ROS 2.
    - Publishes /odom_lidar and /clock through an Action Graph.
    - Supports direct W/A/S/D keyboard teleop inside the WebRTC viewport.
    - Still subscribes to /cmd_vel for later ROS-based control.
    - Publishes odom -> base_link TF and static sensor transforms.

Important:
    - This is written for Isaac Sim 6.0 APIs.
    - It intentionally does not create the three legacy front/side cameras from
      the repository's Isaac Sim 4.5 script.
    - The stock DR.Nav inference node expects three camera topics. Our current
      project uses the existing forward Bumblebee camera, so the downstream
      model/input code must later be aligned with that sensor choice.
    - The manual navigation-zone annotator is a separate next step.

Before running:
    1. Open the maze USD.
    2. Enable/playable physics must be present on the Jackal.
    3. Launch Isaac Sim from a terminal with ROS 2 sourced, or configure its
       internal ROS libraries.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Optional

import numpy as np
import carb.input
import omni.appwindow
import omni.graph.core as og
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom


# =============================================================================
# USER CONFIGURATION
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

# Existing mount frame in the Jackal asset.
LIDAR_FRAME_PATH = f"{BASE_LINK_PATH}/sick_lms1xx_lidar_frame"

# A new Isaac Sim 6.0 RTX LiDAR sensor is created beneath that mount.
LIDAR_SENSOR_PATH = f"{LIDAR_FRAME_PATH}/drnav_rtx_lidar"

# Official Isaac Sim 6.0 example configuration.
# It is a 360-degree rotary 3D LiDAR running at 10 Hz.
LIDAR_CONFIG = "Example_Rotary"
LIDAR_TICK_RATE_HZ = 10.0

GRAPH_PATH = "/World/DRNav_Graph"

CAMERA_RESOLUTION = (640, 480)
PUBLISH_RIGHT_STEREO_CAMERA = False
SHOW_LIDAR_DEBUG_VIEW = False

TOPIC_FRONT_RGB = "/argus/ar0234_front_left/image_raw"
TOPIC_RIGHT_RGB = "/argus/ar0234_front_right/image_raw"
TOPIC_POINT_CLOUD = "/os_cloud_node/points"
TOPIC_SCAN = "/scan"
TOPIC_ODOM = "/odom_lidar"
TOPIC_CMD_VEL = "/cmd_vel"

FRAME_BASE = "base_link"
FRAME_LEFT_CAMERA = "bumblebee_stereo_left_frame"
FRAME_RIGHT_CAMERA = "bumblebee_stereo_right_frame"
FRAME_LIDAR = "sim_lidar"
FRAME_ODOM = "odom"

# Jackal dimensions used by the original repository controller.
WHEEL_RADIUS_M = 0.098
TRACK_WIDTH_M = 0.37559

WHEEL_JOINT_NAMES = [
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_left_wheel_joint",
    "rear_right_wheel_joint",
]

# Change these only if a movement test shows reversed wheel directions.
LEFT_WHEEL_SIGN = 1.0
RIGHT_WHEEL_SIGN = 1.0

MAX_WHEEL_SPEED_RAD_S = 25.0
CMD_TIMEOUT_S = 0.5

# Direct keyboard teleop for WebRTC users without shell/SSH access.
# Set this to False later when ROS /cmd_vel should control the robot.
ENABLE_WEBRTC_KEYBOARD_TELEOP = True
KEYBOARD_LINEAR_SPEED_M_S = 0.30
KEYBOARD_ANGULAR_SPEED_RAD_S = 0.80

# Safety watchdog in case the browser loses a key-release event.
KEYBOARD_EVENT_TIMEOUT_S = 1.25


# =============================================================================
# MAIN BRIDGE CLASS
# =============================================================================

class DRNavIsaacSim6Bridge:
    def __init__(self) -> None:
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()

        self._ros_node = None
        self._cmd_subscription = None
        self._tf_broadcaster = None
        self._static_tf_broadcaster = None

        self._latest_linear = 0.0
        self._latest_angular = 0.0
        self._last_cmd_time = 0.0

        self._input_interface = None
        self._keyboard = None
        self._keyboard_sub_id = None
        self._pressed_keys = set()
        self._last_keyboard_event_time = 0.0

        self._jackal = None
        self._wheel_indices = None
        self._physics_callback_id = None

        self._camera_render_products = []
        self._lidar_render_product = None
        self._lidar_sensor = None
        self._lidar_pc_writer = None
        self._lidar_scan_writer = None
        self._lidar_debug_writer = None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        self._validate_required_prims()
        await self._enable_extensions()

        # Imports that depend on enabled Isaac Sim extensions.
        global Articulation, SimulationManager, IsaacEvents
        global Lidar, set_target_prims
        global rclpy, Twist, TransformStamped
        global TransformBroadcaster, StaticTransformBroadcaster

        from isaacsim.core.experimental.prims import Articulation
        from isaacsim.core.nodes.scripts.utils import set_target_prims
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
        from isaacsim.sensors.experimental.rtx import Lidar

        import rclpy
        from geometry_msgs.msg import TransformStamped, Twist
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        self._cleanup_owned_scene_prims()

        self._setup_camera_publishers()
        self._setup_lidar_publishers()
        self._setup_odom_and_clock_graph()
        self._setup_ros_node()

        if ENABLE_WEBRTC_KEYBOARD_TELEOP:
            self._setup_keyboard_teleop()

        # Start physics before wrapping/querying the articulation.
        self.timeline.play()
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        self._setup_jackal_articulation()
        self._publish_static_sensor_transforms()
        self._register_physics_callback()

        print("\n[DR.Nav] Isaac Sim 6.0 setup complete.")
        print("[DR.Nav] Simulation is playing.")
        self._print_verification_commands()

    def shutdown(self) -> None:
        """Best-effort cleanup for Script Editor re-runs."""
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

        for writer in [
            self._lidar_pc_writer,
            self._lidar_scan_writer,
            self._lidar_debug_writer,
        ]:
            if writer is not None:
                try:
                    writer.detach()
                except Exception:
                    pass

        for render_product in self._camera_render_products:
            try:
                render_product.destroy()
            except Exception:
                pass

        if self._lidar_render_product is not None:
            try:
                self._lidar_render_product.destroy()
            except Exception:
                pass

        if self._ros_node is not None:
            try:
                self._ros_node.destroy_node()
            except Exception:
                pass
            self._ros_node = None

        print("[DR.Nav] Previous bridge cleaned up.")

    # -------------------------------------------------------------------------
    # Validation and extensions
    # -------------------------------------------------------------------------

    def _validate_required_prims(self) -> None:
        required = {
            "Jackal": ROBOT_PRIM,
            "base_link": BASE_LINK_PATH,
            "left Bumblebee camera": LEFT_CAMERA_PATH,
            "left Bumblebee frame": LEFT_CAMERA_FRAME_PATH,
            "LiDAR mount frame": LIDAR_FRAME_PATH,
        }

        if PUBLISH_RIGHT_STEREO_CAMERA:
            required["right Bumblebee camera"] = RIGHT_CAMERA_PATH
            required["right Bumblebee frame"] = RIGHT_CAMERA_FRAME_PATH

        missing = [
            f"{name}: {path}"
            for name, path in required.items()
            if not self.stage.GetPrimAtPath(path).IsValid()
        ]

        if missing:
            raise RuntimeError(
                "The following required prim paths were not found:\n  "
                + "\n  ".join(missing)
            )

    async def _enable_extensions(self) -> None:
        manager = omni.kit.app.get_app().get_extension_manager()

        extension_ids = [
            "isaacsim.core.nodes",
            "isaacsim.ros2.bridge",
            "isaacsim.ros2.nodes",
            "isaacsim.sensors.experimental.rtx",
        ]

        for extension_id in extension_ids:
            if not manager.is_extension_enabled(extension_id):
                manager.set_extension_enabled_immediate(extension_id, True)
                print(f"[DR.Nav] Enabled extension: {extension_id}")
            else:
                print(f"[DR.Nav] Extension already enabled: {extension_id}")

        # Give Kit a few updates to finish loading Python modules and OGN nodes.
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

    def _cleanup_owned_scene_prims(self) -> None:
        # Only remove prims owned by this script. Existing Jackal cameras,
        # zones, walls, and robot prims are not touched.
        for path in [GRAPH_PATH, LIDAR_SENSOR_PATH]:
            if self.stage.GetPrimAtPath(path).IsValid():
                self.stage.RemovePrim(path)
                print(f"[DR.Nav] Removed prior generated prim: {path}")

    # -------------------------------------------------------------------------
    # Camera
    # -------------------------------------------------------------------------

    def _setup_camera_publishers(self) -> None:
        left_rp = rep.create.render_product(
            LEFT_CAMERA_PATH,
            CAMERA_RESOLUTION,
            name="DRNavLeftCamera",
        )
        self._camera_render_products.append(left_rp)

        camera_specs = [
            (
                "camera_left",
                left_rp.path,
                TOPIC_FRONT_RGB,
                FRAME_LEFT_CAMERA,
            )
        ]

        if PUBLISH_RIGHT_STEREO_CAMERA:
            right_rp = rep.create.render_product(
                RIGHT_CAMERA_PATH,
                CAMERA_RESOLUTION,
                name="DRNavRightCamera",
            )
            self._camera_render_products.append(right_rp)
            camera_specs.append(
                (
                    "camera_right",
                    right_rp.path,
                    TOPIC_RIGHT_RGB,
                    FRAME_RIGHT_CAMERA,
                )
            )

        self._camera_specs = camera_specs

        print("[DR.Nav] Camera render products:")
        for _, rp_path, topic, frame_id in camera_specs:
            print(f"  {rp_path} -> {topic} [{frame_id}]")

    # -------------------------------------------------------------------------
    # LiDAR
    # -------------------------------------------------------------------------

    def _setup_lidar_publishers(self) -> None:
        # The uploaded Jackal scene authors sick_lms1xx_lidar_frame as
        # active=false. USD does not allow a child prim to be defined beneath
        # an inactive parent, so reactivate the mount before creating the RTX
        # LiDAR beneath it.
        lidar_mount_prim = self.stage.GetPrimAtPath(LIDAR_FRAME_PATH)

        if not lidar_mount_prim.IsValid():
            raise RuntimeError(
                f"LiDAR mount frame was not found: {LIDAR_FRAME_PATH}"
            )

        if not lidar_mount_prim.IsActive():
            lidar_mount_prim.SetActive(True)
            print(
                "[DR.Nav] Reactivated inactive LiDAR mount frame: "
                f"{LIDAR_FRAME_PATH}"
            )

        # Create a 6.0 RTX LiDAR exactly at the existing SICK mount frame.
        self._lidar_sensor = Lidar.create(
            path=LIDAR_SENSOR_PATH,
            config=LIDAR_CONFIG,
            tick_rate=LIDAR_TICK_RATE_HZ,
            translations=[[0.0, 0.0, 0.0]],
            orientations=[[1.0, 0.0, 0.0, 0.0]],
        )

        self._lidar_render_product = rep.create.render_product(
            self._lidar_sensor.paths[0],
            [1, 1],
            name="DRNavLidar",
        )

        self._lidar_pc_writer = rep.writers.get(
            "RtxLidarROS2PublishPointCloud"
        )
        self._lidar_pc_writer.initialize(
            topicName=TOPIC_POINT_CLOUD,
            frameId=FRAME_LIDAR,
        )
        self._lidar_pc_writer.attach([self._lidar_render_product])

        # Isaac Sim 6.0's official example uses a separate 2D sensor for
        # RtxLidarROS2PublishLaserScan. Here we publish a LaserScan from the same
        # rotary sensor through the ROS2RtxLidarHelper Action Graph below only if
        # your build supports it. To keep this first migration robust, the
        # PointCloud2 publisher is always enabled; /scan is added in the graph.
        if SHOW_LIDAR_DEBUG_VIEW:
            self._lidar_debug_writer = rep.writers.get(
                "RtxLidarDebugDrawPointCloudBuffer"
            )
            self._lidar_debug_writer.attach(self._lidar_render_product)

        print(
            f"[DR.Nav] RTX LiDAR created: {LIDAR_SENSOR_PATH} "
            f"({LIDAR_CONFIG}, {LIDAR_TICK_RATE_HZ:g} Hz)"
        )

    # -------------------------------------------------------------------------
    # ROS camera, scan, odometry, and clock graph
    # -------------------------------------------------------------------------

    def _setup_odom_and_clock_graph(self) -> None:
        create_nodes = [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("simtime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("lidar_scan", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
            ("odomcompute", "isaacsim.core.nodes.IsaacComputeOdometry"),
            ("odompub", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ("clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ]

        connections = [
            ("tick.outputs:tick", "lidar_scan.inputs:execIn"),
            ("tick.outputs:tick", "odomcompute.inputs:execIn"),
            ("tick.outputs:tick", "odompub.inputs:execIn"),
            ("tick.outputs:tick", "clock.inputs:execIn"),
            ("simtime.outputs:simulationTime", "odompub.inputs:timeStamp"),
            ("simtime.outputs:simulationTime", "clock.inputs:timeStamp"),
            ("odomcompute.outputs:position", "odompub.inputs:position"),
            (
                "odomcompute.outputs:orientation",
                "odompub.inputs:orientation",
            ),
            (
                "odomcompute.outputs:linearVelocity",
                "odompub.inputs:linearVelocity",
            ),
            (
                "odomcompute.outputs:angularVelocity",
                "odompub.inputs:angularVelocity",
            ),
        ]

        set_values = [
            (
                "lidar_scan.inputs:renderProductPath",
                self._lidar_render_product.path,
            ),
            ("lidar_scan.inputs:topicName", TOPIC_SCAN),
            ("lidar_scan.inputs:frameId", FRAME_LIDAR),
            ("lidar_scan.inputs:type", "laser_scan"),
            ("lidar_scan.inputs:showDebugView", False),
            ("odompub.inputs:topicName", TOPIC_ODOM),
            ("odompub.inputs:chassisFrameId", FRAME_BASE),
            ("odompub.inputs:odomFrameId", FRAME_ODOM),
            ("odompub.inputs:robotFront", [1.0, 0.0, 0.0]),
        ]

        for node_name, rp_path, topic_name, frame_id in self._camera_specs:
            create_nodes.append(
                (node_name, "isaacsim.ros2.bridge.ROS2CameraHelper")
            )
            connections.append(
                ("tick.outputs:tick", f"{node_name}.inputs:execIn")
            )
            set_values.extend(
                [
                    (f"{node_name}.inputs:renderProductPath", rp_path),
                    (f"{node_name}.inputs:topicName", topic_name),
                    (f"{node_name}.inputs:frameId", frame_id),
                    (f"{node_name}.inputs:type", "rgb"),
                ]
            )

        og.Controller.edit(
            {
                "graph_path": GRAPH_PATH,
                "evaluator_name": "execution",
                "pipeline_stage": (
                    og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION
                ),
            },
            {
                og.Controller.Keys.CREATE_NODES: create_nodes,
                og.Controller.Keys.CONNECT: connections,
                og.Controller.Keys.SET_VALUES: set_values,
            },
        )

        set_target_prims(
            primPath=f"{GRAPH_PATH}/odomcompute",
            inputName="inputs:chassisPrim",
            targetPrimPaths=[BASE_LINK_PATH],
        )

        print(f"[DR.Nav] ROS 2 Action Graph created: {GRAPH_PATH}")

    # -------------------------------------------------------------------------
    # ROS node and TF
    # -------------------------------------------------------------------------

    def _setup_ros_node(self) -> None:
        if not rclpy.ok():
            rclpy.init()

        self._ros_node = rclpy.create_node("drnav_jackal_isaacsim6")
        self._cmd_subscription = self._ros_node.create_subscription(
            Twist,
            TOPIC_CMD_VEL,
            self._cmd_vel_callback,
            10,
        )

        self._tf_broadcaster = TransformBroadcaster(self._ros_node)
        self._static_tf_broadcaster = StaticTransformBroadcaster(
            self._ros_node
        )

        self._last_cmd_time = time.monotonic()
        print(f"[DR.Nav] Subscribed to {TOPIC_CMD_VEL}")

    def _cmd_vel_callback(self, message) -> None:
        self._latest_linear = float(message.linear.x)
        self._latest_angular = float(message.angular.z)
        self._last_cmd_time = time.monotonic()

    def _sim_stamp(self):
        current_time = float(self.timeline.get_current_time())
        sec = int(math.floor(current_time))
        nanosec = int((current_time - sec) * 1_000_000_000)
        return sec, nanosec

    def _matrix_to_transform(
        self,
        matrix,
        parent_frame: str,
        child_frame: str,
    ):
        transform = TransformStamped()
        sec, nanosec = self._sim_stamp()

        transform.header.stamp.sec = sec
        transform.header.stamp.nanosec = nanosec
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
            (LIDAR_FRAME_PATH, FRAME_LIDAR),
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

    # -------------------------------------------------------------------------
    # Direct WebRTC keyboard teleoperation
    # -------------------------------------------------------------------------

    def _setup_keyboard_teleop(self) -> None:
        app_window = omni.appwindow.get_default_app_window()

        if app_window is None:
            raise RuntimeError(
                "Isaac Sim's default application window was not available."
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
                self._set_wheel_targets(0.0, 0.0)
            return True

        if event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            self._pressed_keys.add(event.input)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed_keys.discard(event.input)

        # Consume handled keys so they are not typed into text fields.
        return True

    def _get_keyboard_command(self):
        # Stop safely if WebRTC/browser focus changes and a release event is lost.
        if (
            self._pressed_keys
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
    # Jackal articulation and physics callback
    # -------------------------------------------------------------------------

    def _setup_jackal_articulation(self) -> None:
        self._jackal = Articulation(ROBOT_PRIM)

        print("[DR.Nav] Available Jackal DOFs:")
        print(" ", self._jackal.dof_names)

        try:
            indices = self._jackal.get_dof_indices(
                WHEEL_JOINT_NAMES
            ).numpy()
        except Exception as exc:
            raise RuntimeError(
                "Could not resolve the configured Jackal wheel joints.\n"
                f"Configured names: {WHEEL_JOINT_NAMES}\n"
                f"Available DOFs: {self._jackal.dof_names}"
            ) from exc

        self._wheel_indices = indices
        print("[DR.Nav] Wheel joint names:", WHEEL_JOINT_NAMES)
        print("[DR.Nav] Wheel DOF indices:", self._wheel_indices)

    def _register_physics_callback(self) -> None:
        self._physics_callback_id = SimulationManager.register_callback(
            self._physics_step,
            IsaacEvents.POST_PHYSICS_STEP,
        )
        print("[DR.Nav] Registered Isaac Sim 6.0 physics callback.")

    def _physics_step(self, dt, context) -> None:
        del dt, context

        try:
            rclpy.spin_once(self._ros_node, timeout_sec=0.0)

            if ENABLE_WEBRTC_KEYBOARD_TELEOP:
                linear, angular = self._get_keyboard_command()
            elif time.monotonic() - self._last_cmd_time > CMD_TIMEOUT_S:
                linear = 0.0
                angular = 0.0
            else:
                linear = self._latest_linear
                angular = self._latest_angular

            self._set_wheel_targets(linear, angular)
            self._publish_dynamic_base_tf()

        except Exception as exc:
            # Avoid crashing the simulation callback. Print sparingly.
            print("[DR.Nav] Physics callback error:", repr(exc))

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

    def _publish_dynamic_base_tf(self) -> None:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        base_prim = self.stage.GetPrimAtPath(BASE_LINK_PATH)
        world_matrix = cache.GetLocalToWorldTransform(base_prim)

        transform = self._matrix_to_transform(
            world_matrix,
            FRAME_ODOM,
            FRAME_BASE,
        )
        self._tf_broadcaster.sendTransform(transform)

    # -------------------------------------------------------------------------
    # User instructions
    # -------------------------------------------------------------------------

    @staticmethod
    def _print_verification_commands() -> None:
        print(
            f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Verify ROS topics from a ROS 2 terminal:

  ros2 topic list
  ros2 topic hz {TOPIC_FRONT_RGB}
  ros2 topic hz {TOPIC_POINT_CLOUD}
  ros2 topic hz {TOPIC_SCAN}
  ros2 topic hz {TOPIC_ODOM}

Verify transforms:

  ros2 run tf2_ros tf2_echo {FRAME_ODOM} {FRAME_BASE}
  ros2 run tf2_ros tf2_echo {FRAME_BASE} {FRAME_LEFT_CAMERA}
  ros2 run tf2_ros tf2_echo {FRAME_BASE} {FRAME_LIDAR}

Drive the Jackal directly through WebRTC:

  1. Click inside the 3D viewport.
  2. Hold W/S for forward/reverse.
  3. Hold A/D to rotate left/right.
  4. Press SPACE to stop immediately.

ROS /cmd_vel remains available when:
  ENABLE_WEBRTC_KEYBOARD_TELEOP = False

Then you can use:
  ros2 run teleop_twist_keyboard teleop_twist_keyboard


If forward/backward or turning is reversed, change:
  LEFT_WHEEL_SIGN
  RIGHT_WHEEL_SIGN
near the top of this file.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        )


# =============================================================================
# SCRIPT EDITOR ENTRY POINT
# =============================================================================

async def _main() -> None:
    global _DRNAV_ISAACSIM6_BRIDGE

    try:
        previous = _DRNAV_ISAACSIM6_BRIDGE
    except NameError:
        previous = None

    if previous is not None:
        try:
            previous.shutdown()
        except Exception as exc:
            print("[DR.Nav] Previous bridge cleanup warning:", repr(exc))

    bridge = DRNavIsaacSim6Bridge()
    _DRNAV_ISAACSIM6_BRIDGE = bridge

    try:
        await bridge.start()
    except Exception:
        bridge.shutdown()
        raise


asyncio.ensure_future(_main())