#!/usr/bin/env python3
"""Isaac Sim 6.0 DR.Nav setup for the current Jackal scene."""

import asyncio, builtins, gc, math, time
import numpy as np
import omni.graph.core as og, omni.kit.app
import omni.timeline, omni.usd
from pxr import Usd, UsdGeom

RUNTIME_CONFIG = getattr(builtins, "_DRNAV_RUNTIME_CONFIG", {})
TOPIC_CONFIG = RUNTIME_CONFIG.get("topics", {})
BASE_CONTROL_CONFIG = RUNTIME_CONFIG.get("base_control", {})
RECORDER_CONFIG = RUNTIME_CONFIG.get("recorder", {})
RECORDER_ENABLED = bool(RECORDER_CONFIG.get("enabled", True))

ROBOT = "/World/ground/flat_plane/jackal"
BASE = f"{ROBOT}/base_link"
CAM_FRAME = f"{BASE}/bumblebee_stereo_camera_frame/bumblebee_stereo_left_frame"
CAMERA = f"{CAM_FRAME}/bumblebee_stereo_left_camera"
LIDAR_FRAME = f"{BASE}/sick_lms1xx_lidar_frame"
LIDAR = f"{LIDAR_FRAME}/Lidar"
GRAPH = "/World/DRNav_RuntimeGraph"

IMAGE_TOPIC = TOPIC_CONFIG.get("image", "/argus/ar0234_front_left/image_raw")
SCAN_TOPIC = TOPIC_CONFIG.get("scan", "/scan")
POINTS_TOPIC = TOPIC_CONFIG.get("points", "/os_cloud_node/points")
ODOM_TOPIC = TOPIC_CONFIG.get("odom", "/odom_lidar")
CLOCK_TOPIC = TOPIC_CONFIG.get("clock", "/clock")
CMD_TOPIC = TOPIC_CONFIG.get("direct_cmd", "/cmd_vel")
TELEOP_CMD_TOPIC = TOPIC_CONFIG.get("teleop_cmd", "/cmd_vel_teleop")
WAYPOINT_CMD_TOPIC = TOPIC_CONFIG.get("waypoint_cmd", "/cmd_vel_waypoint")

ODOM_FRAME = "odom"
BASE_FRAME = "base_link"
CAM_FRAME_ID = "bumblebee_stereo_left_frame"
LIDAR_FRAME_ID = "sim_lidar"

LINEAR_ACCEL = float(BASE_CONTROL_CONFIG.get("linear_accel", 0.35))
ANGULAR_ACCEL = float(BASE_CONTROL_CONFIG.get("angular_accel", 0.90))
WHEEL_RADIUS = 0.098
TRACK_WIDTH = 0.37559
MAX_WHEEL_SPEED = 25.0
LEFT_SIGN = 1.0
RIGHT_SIGN = 1.0
CMD_TIMEOUT = float(BASE_CONTROL_CONFIG.get("command_timeout", 0.5))
COMMAND_EPSILON = 1e-4
ODOM_RATE = float(BASE_CONTROL_CONFIG.get("odom_rate_hz", 30.0))
LIDAR_RATE = float(BASE_CONTROL_CONFIG.get("lidar_rate_hz", 10.0))

WHEEL_JOINTS = [
    "front_left_wheel_joint", "front_right_wheel_joint",
    "rear_left_wheel_joint", "rear_right_wheel_joint",
]
BRIDGE_KEY = "_DRNAV_ISAACSIM6_V6_BRIDGE"
COMMAND_INTERFACE_VERSION = 2
AUXILIARY_CONTROLLER_KEYS = [
    "_DRNAV_JACKAL_TELEOP",
    "_DRNAV_WAYPOINT_FOLLOWER",
]
AUXILIARY_TASK_KEYS = [
    "_DRNAV_JACKAL_TELEOP_TASK",
    "_DRNAV_WAYPOINT_FOLLOWER_TASK",
]
RECORDER_SERVICE_KEYS = [
    "_DRNAV_AUTO_EPISODE_PIPELINE",
    "_DRNAV_REPLICATOR_RECORDER",
]
RUNTIME_OBJECT_KEYS = (
    RECORDER_SERVICE_KEYS
    + AUXILIARY_CONTROLLER_KEYS
    + [BRIDGE_KEY]
)


def approach(value, target, delta):
    return min(value + delta, target) if value < target else max(value - delta, target)


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Bridge:
    def __init__(self):
        self.command_interface_version = COMMAND_INTERFACE_VERSION
        self.rgb_interface_version = 1
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.node = self.cmd_sub = self.rgb_sub = None
        self.lidar = self.robot = None
        self.wheel_ids = self.callback = None
        self.cmd_v = self.cmd_w = self.v = self.w = 0.0
        self.command_sources = {
            "teleop": [0.0, 0.0, float("-inf"), float("-inf")],
            "direct": [0.0, 0.0, float("-inf"), float("-inf")],
            "waypoint": [0.0, 0.0, float("-inf"), float("-inf")],
        }
        self.active_command_source = "none"
        self.reported_command_source = None
        self.last_odom = self.last_lidar = -1.0
        self.prev_time = self.prev_xyz = self.prev_yaw = None
        self.latest_rgb = None
        self.latest_rgb_sequence = 0
        self.unsupported_rgb_encoding = None
        self.range_min, self.range_max = 0.05, 100.0
        self.last_error = 0.0

    async def start(self):
        # Recover normal GUI timeline control if an interrupted Replicator
        # render step previously left automatic timeline updates disabled.
        self.timeline.set_auto_update(True)
        self.timeline.commit_silently()

        if not self.timeline.is_stopped():
            self.timeline.stop()
            self.timeline.commit()
            for _ in range(3):
                await omni.kit.app.get_app().next_update_async()

        self.validate()
        await self.enable_extensions()

        global Articulation, SimulationManager, IsaacEvents, set_target_prims
        global _range_sensor, rclpy, Twist, TransformStamped, Odometry, Clock
        global Image, LaserScan, PointCloud2, PointField
        global qos_profile_sensor_data
        global TransformBroadcaster, StaticTransformBroadcaster

        from isaacsim.core.experimental.prims import Articulation
        from isaacsim.core.nodes.scripts.utils import set_target_prims
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
        from isaacsim.sensors.physx import _range_sensor
        import rclpy
        from geometry_msgs.msg import TransformStamped, Twist
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, LaserScan, PointCloud2, PointField
        from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

        self.lidar = _range_sensor.acquire_lidar_sensor_interface()
        await self.remove_graph()
        self.disable_lidar_debug()
        if not RECORDER_ENABLED:
            self.create_camera_graph()
        self.create_ros()

        for _ in range(6):
            await omni.kit.app.get_app().next_update_async()

        self.robot = Articulation(ROBOT)
        self.wheel_ids = self.robot.get_dof_indices(WHEEL_JOINTS).numpy()
        self.publish_static_tf()
        self.callback = SimulationManager.register_callback(
            self.step, IsaacEvents.POST_PHYSICS_STEP
        )
        print("\n[DR.Nav] Sensor and base bridge setup complete.")
        print(
            "[DR.Nav] Command priority: /cmd_vel_teleop > "
            "/cmd_vel > /cmd_vel_waypoint"
        )
        print("[DR.Nav] Run setup_teleop.py for WebRTC keyboard control.")
        if RECORDER_ENABLED:
            print(
                "[DR.Nav] Camera reserved for exclusive dataset recording; "
                f"ROS image topic {IMAGE_TOPIC} is disabled."
            )
        print("[DR.Nav] Timeline remains stopped until you press Play.")
        print("[DR.Nav] Wheel indices:", self.wheel_ids)

    async def shutdown(self):
        try:
            self.timeline.stop()
            self.timeline.set_auto_update(True)
            self.timeline.commit()
        except Exception:
            pass
        try:
            if self.callback is not None:
                SimulationManager.deregister_callback(self.callback)
        except Exception:
            pass
        self.callback = None
        try:
            self.set_wheels(0.0, 0.0)
        except Exception:
            pass
        if self.node:
            try:
                self.node.destroy_node()
            except Exception:
                pass
        self.node = None
        await self.remove_graph()
        print("[DR.Nav] Previous small bridge cleaned up.")

    def validate(self):
        required = [ROBOT, BASE, CAM_FRAME, CAMERA, LIDAR_FRAME, LIDAR]
        missing = [p for p in required if not self.stage.GetPrimAtPath(p).IsValid()]
        if missing:
            raise RuntimeError("Missing prims:\n  " + "\n  ".join(missing))
        prim = self.stage.GetPrimAtPath(LIDAR)
        if not prim.IsActive() or prim.GetTypeName() != "Lidar":
            raise RuntimeError(
                f"Expected active PhysX Lidar at {LIDAR}; "
                f"active={prim.IsActive()}, type={prim.GetTypeName()!r}"
            )

    async def enable_extensions(self):
        manager = omni.kit.app.get_app().get_extension_manager()
        for name in [
            "isaacsim.core.nodes", "isaacsim.ros2.bridge",
            "isaacsim.ros2.nodes", "isaacsim.sensors.physx",
        ]:
            if not manager.is_extension_enabled(name):
                manager.set_extension_enabled_immediate(name, True)
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

    def session_target(self):
        return self.stage.GetEditTargetForLocalLayer(self.stage.GetSessionLayer())

    async def remove_graph(self):
        with Usd.EditContext(self.stage, self.session_target()):
            if self.stage.GetPrimAtPath(GRAPH).IsValid():
                self.stage.RemovePrim(GRAPH)
        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()

    def disable_lidar_debug(self):
        prim = self.stage.GetPrimAtPath(LIDAR)
        with Usd.EditContext(self.stage, self.session_target()):
            for attr in prim.GetAttributes():
                name = attr.GetName().lower()
                if name.endswith(("drawlines", "drawpoints", "showdebugview")):
                    attr.Set(False)
        self.range_min = self.number_attr("minrange", 0.05)
        self.range_max = self.number_attr("maxrange", 100.0)

    def number_attr(self, suffix, fallback):
        for attr in self.stage.GetPrimAtPath(LIDAR).GetAttributes():
            if attr.GetName().lower().endswith(suffix) and attr.Get() is not None:
                return float(attr.Get())
        return fallback

    def create_camera_graph(self):
        # Evaluate the helper on every playback tick. ROS2CameraHelper is
        # internally idempotent while active, but resets its writer on timeline
        # Stop; a one-shot gate therefore leaves later episodes unpublished.
        nodes = [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("context", "isaacsim.ros2.bridge.ROS2Context"),
            ("render", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("camera", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ]
        links = [
            ("tick.outputs:tick", "render.inputs:execIn"),
            ("render.outputs:execOut", "camera.inputs:execIn"),
            ("render.outputs:renderProductPath", "camera.inputs:renderProductPath"),
            ("context.outputs:context", "camera.inputs:context"),
        ]
        values = [
            ("context.inputs:useDomainIDEnvVar", True),
            ("render.inputs:width", 640), ("render.inputs:height", 480),
            ("camera.inputs:topicName", IMAGE_TOPIC),
            ("camera.inputs:frameId", CAM_FRAME_ID),
            ("camera.inputs:type", "rgb"), ("camera.inputs:enabled", True),
        ]
        with Usd.EditContext(self.stage, self.session_target()):
            og.Controller.edit(
                {
                    "graph_path": GRAPH,
                    "evaluator_name": "execution",
                    "pipeline_stage":
                        og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
                },
                {
                    og.Controller.Keys.CREATE_NODES: nodes,
                    og.Controller.Keys.CONNECT: links,
                    og.Controller.Keys.SET_VALUES: values,
                },
            )
            set_target_prims(
                primPath=f"{GRAPH}/render",
                inputName="inputs:cameraPrim",
                targetPrimPaths=[CAMERA],
            )
        session_id = self.stage.GetSessionLayer().identifier
        layers = {s.layer.identifier for s in self.stage.GetPrimAtPath(GRAPH).GetPrimStack()}
        if session_id not in layers:
            raise RuntimeError("Runtime graph was not authored in session layer.")

    def create_ros(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("drnav_jackal_small")
        self.cmd_sub = [
            self.node.create_subscription(
                Twist,
                topic,
                lambda msg, source=source: self.cmd_callback(msg, source),
                10,
            )
            for source, topic in [
                ("teleop", TELEOP_CMD_TOPIC),
                ("direct", CMD_TOPIC),
                ("waypoint", WAYPOINT_CMD_TOPIC),
            ]
        ]
        if not RECORDER_ENABLED:
            self.rgb_sub = self.node.create_subscription(
                Image,
                IMAGE_TOPIC,
                self.rgb_callback,
                qos_profile_sensor_data,
            )
        self.scan_pub = self.node.create_publisher(LaserScan, SCAN_TOPIC, 10)
        self.points_pub = self.node.create_publisher(PointCloud2, POINTS_TOPIC, 10)
        self.odom_pub = self.node.create_publisher(Odometry, ODOM_TOPIC, 10)
        self.clock_pub = self.node.create_publisher(Clock, CLOCK_TOPIC, 10)
        self.tf_pub = TransformBroadcaster(self.node)
        self.static_tf_pub = StaticTransformBroadcaster(self.node)

    def cmd_callback(self, msg, source):
        self.submit_command(
            source,
            float(msg.linear.x),
            float(msg.angular.z),
        )

    def rgb_callback(self, msg):
        """Keep the latest ROS camera image as a contiguous RGB array."""
        height = int(msg.height)
        width = int(msg.width)
        step = int(msg.step)
        encoding = str(msg.encoding).lower()
        channels_by_encoding = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
        }
        channels = channels_by_encoding.get(encoding)

        if channels is None:
            if encoding != self.unsupported_rgb_encoding:
                print(
                    "[DR.Nav] Unsupported camera image encoding:",
                    repr(encoding),
                )
                self.unsupported_rgb_encoding = encoding
            return

        required_row_bytes = width * channels
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if (
            height <= 0
            or width <= 0
            or step < required_row_bytes
            or data.size < height * step
        ):
            return

        rows = data[: height * step].reshape(height, step)
        image = rows[:, :required_row_bytes].reshape(
            height,
            width,
            channels,
        )

        if encoding == "bgr8":
            image = image[:, :, ::-1]
        elif encoding == "rgba8":
            image = image[:, :, :3]
        elif encoding == "bgra8":
            image = image[:, :, [2, 1, 0]]
        elif encoding == "mono8":
            image = np.repeat(image, 3, axis=2)

        first_frame = self.latest_rgb_sequence == 0
        self.latest_rgb = np.ascontiguousarray(image)
        self.latest_rgb_sequence += 1

        if first_frame:
            print(
                "[DR.Nav] ROS camera frames available: "
                f"{width}x{height} {encoding}."
            )

    def get_latest_rgb(self, after_sequence=None):
        """Return a stable copy of the newest camera frame and its sequence."""
        if (
            self.latest_rgb is None
            or (
                after_sequence is not None
                and self.latest_rgb_sequence <= int(after_sequence)
            )
        ):
            return self.latest_rgb_sequence, None
        return self.latest_rgb_sequence, self.latest_rgb.copy()

    def submit_command(self, source, linear, angular):
        """Update a command source from ROS or an in-process controller."""
        if source not in self.command_sources:
            raise ValueError(f"Unknown DR.Nav command source: {source!r}")

        now = time.monotonic()
        linear = float(linear)
        angular = float(angular)
        previous_nonzero = self.command_sources[source][3]
        if abs(linear) > COMMAND_EPSILON or abs(angular) > COMMAND_EPSILON:
            previous_nonzero = now
        self.command_sources[source] = [
            linear,
            angular,
            now,
            previous_nonzero,
        ]

    def selected_command(self):
        now = time.monotonic()
        for source in ("teleop", "direct", "waypoint"):
            linear, angular, updated, last_nonzero = self.command_sources[source]
            recent = now - updated <= CMD_TIMEOUT
            nonzero = (
                abs(linear) > COMMAND_EPSILON
                or abs(angular) > COMMAND_EPSILON
            )
            # Waypoint commands own the fallback channel, including their final
            # zero. Higher-priority sources only claim control while moving or
            # briefly after stopping, so an idle zero publisher cannot mask the
            # follower forever.
            claims_control = source == "waypoint" or (
                nonzero or now - last_nonzero <= CMD_TIMEOUT
            )
            if recent and claims_control:
                self.active_command_source = source
                self.cmd_v, self.cmd_w = linear, angular
                if source != self.reported_command_source:
                    print("[DR.Nav] Active command source:", source)
                    self.reported_command_source = source
                return linear, angular

        self.active_command_source = "none"
        if self.reported_command_source != "none":
            print("[DR.Nav] Active command source: none")
            self.reported_command_source = "none"
        self.cmd_v = self.cmd_w = 0.0
        return 0.0, 0.0

    def set_wheels(self, linear, angular):
        if self.robot is None or self.wheel_ids is None:
            return
        half = TRACK_WIDTH / 2.0
        left = (linear - angular * half) / WHEEL_RADIUS * LEFT_SIGN
        right = (linear + angular * half) / WHEEL_RADIUS * RIGHT_SIGN
        left = float(np.clip(left, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
        right = float(np.clip(right, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
        self.robot.set_dof_velocity_targets(
            np.array([[left, right, left, right]], dtype=np.float32),
            dof_indices=self.wheel_ids,
        )

    @staticmethod
    def stamp(stamp, value):
        stamp.sec = int(value)
        stamp.nanosec = int((value - int(value)) * 1_000_000_000)

    def matrix_tf(self, matrix, parent, child, now):
        msg = TransformStamped()
        self.stamp(msg.header.stamp, now)
        msg.header.frame_id, msg.child_frame_id = parent, child
        t = matrix.ExtractTranslation()
        q = matrix.ExtractRotationQuat()
        i = q.GetImaginary()
        msg.transform.translation.x = float(t[0])
        msg.transform.translation.y = float(t[1])
        msg.transform.translation.z = float(t[2])
        msg.transform.rotation.x = float(i[0])
        msg.transform.rotation.y = float(i[1])
        msg.transform.rotation.z = float(i[2])
        msg.transform.rotation.w = float(q.GetReal())
        return msg

    def publish_static_tf(self):
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        base = self.stage.GetPrimAtPath(BASE)
        now = float(self.timeline.get_current_time())
        messages = []
        for path, frame in [(CAM_FRAME, CAM_FRAME_ID), (LIDAR, LIDAR_FRAME_ID)]:
            matrix, _ = cache.ComputeRelativeTransform(
                self.stage.GetPrimAtPath(path), base
            )
            messages.append(self.matrix_tf(matrix, BASE_FRAME, frame, now))
        self.static_tf_pub.sendTransform(messages)

    def publish_clock(self, now):
        msg = Clock()
        self.stamp(msg.clock, now)
        self.clock_pub.publish(msg)

    def get_world_pose(self):
        """Return the current Jackal world pose without a ROS round trip."""
        matrix = UsdGeom.XformCache(
            Usd.TimeCode.Default()
        ).GetLocalToWorldTransform(self.stage.GetPrimAtPath(BASE))
        translation = matrix.ExtractTranslation()
        quaternion = matrix.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        qx, qy, qz = map(float, imaginary)
        qw = float(quaternion.GetReal())
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return (
            float(translation[0]),
            float(translation[1]),
            yaw,
        )

    def publish_odom(self, now):
        matrix = UsdGeom.XformCache(
            Usd.TimeCode.Default()
        ).GetLocalToWorldTransform(self.stage.GetPrimAtPath(BASE))
        t = matrix.ExtractTranslation()
        q = matrix.ExtractRotationQuat()
        i = q.GetImaginary()
        x, y, z = map(float, t)
        qx, qy, qz = map(float, i)
        qw = float(q.GetReal())
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        vx = vy = wz = 0.0
        if self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 1e-6:
                wx = (x - self.prev_xyz[0]) / dt
                wy = (y - self.prev_xyz[1]) / dt
                c, s = math.cos(yaw), math.sin(yaw)
                vx, vy = c * wx + s * wy, -s * wx + c * wy
                wz = wrap(yaw - self.prev_yaw) / dt
        self.prev_time, self.prev_xyz, self.prev_yaw = now, (x, y, z), yaw

        tf = TransformStamped()
        self.stamp(tf.header.stamp, now)
        tf.header.frame_id, tf.child_frame_id = ODOM_FRAME, BASE_FRAME
        tf.transform.translation.x, tf.transform.translation.y = x, y
        tf.transform.translation.z = z
        tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf_pub.sendTransform(tf)

        msg = Odometry()
        self.stamp(msg.header.stamp, now)
        msg.header.frame_id, msg.child_frame_id = ODOM_FRAME, BASE_FRAME
        msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
        msg.pose.pose.position.z = z
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y = qx, qy
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = qz, qw
        msg.twist.twist.linear.x, msg.twist.twist.linear.y = vx, vy
        msg.twist.twist.angular.z = wz
        self.odom_pub.publish(msg)

    @staticmethod
    def radians(values):
        values = np.asarray(values, dtype=np.float32)
        finite = values[np.isfinite(values)]
        if finite.size and np.max(np.abs(finite)) > 2.0 * math.pi + 0.1:
            values = np.deg2rad(values)
        return values

    def scan_row(self, depth, azimuth, zenith):
        depth = np.asarray(depth, dtype=np.float32)
        azimuth, zenith = self.radians(azimuth), self.radians(zenith)
        if not depth.size or not azimuth.size:
            return np.empty(0), np.empty(0)
        if depth.ndim >= 2:
            depth = depth.reshape(depth.shape[0], -1)
            row = (
                int(np.argmin(np.abs(zenith)))
                if zenith.ndim == 1 and zenith.size == depth.shape[0]
                else depth.shape[0] // 2
            )
            ranges = depth[row]
            if azimuth.ndim == 1 and azimuth.size == ranges.size:
                angles = azimuth
            elif azimuth.size == depth.size:
                angles = azimuth.reshape(depth.shape)[row]
            else:
                angles = np.linspace(
                    -math.pi, math.pi, ranges.size, endpoint=False, dtype=np.float32
                )
        else:
            ranges = depth.reshape(-1)
            angles = (
                azimuth.reshape(-1)
                if azimuth.size == ranges.size
                else np.linspace(
                    -math.pi, math.pi, ranges.size, endpoint=False, dtype=np.float32
                )
            )
        valid = np.isfinite(angles)
        ranges, angles = ranges[valid], angles[valid]
        order = np.argsort(angles)
        return ranges[order], angles[order]

    def publish_lidar(self, now):
        ranges, angles = self.scan_row(
            self.lidar.get_linear_depth_data(LIDAR),
            self.lidar.get_azimuth_data(LIDAR),
            self.lidar.get_zenith_data(LIDAR),
        )
        if ranges.size >= 2:
            msg = LaserScan()
            self.stamp(msg.header.stamp, now)
            msg.header.frame_id = LIDAR_FRAME_ID
            msg.angle_min, msg.angle_max = float(angles[0]), float(angles[-1])
            msg.angle_increment = float(np.median(np.diff(angles)))
            msg.scan_time = 1.0 / LIDAR_RATE
            msg.range_min, msg.range_max = self.range_min, self.range_max
            valid = (
                np.isfinite(ranges)
                & (ranges >= self.range_min)
                & (ranges <= self.range_max)
            )
            msg.ranges = np.where(valid, ranges, np.inf).astype(np.float32).tolist()
            self.scan_pub.publish(msg)

        points = np.asarray(
            self.lidar.get_point_cloud_data(LIDAR), dtype=np.float32
        )
        if not points.size:
            return
        points = points.reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)]

        msg = PointCloud2()
        self.stamp(msg.header.stamp, now)
        msg.header.frame_id = LIDAR_FRAME_ID
        msg.height, msg.width = 1, int(points.shape[0])
        msg.fields = [
            PointField(
                name="x", offset=0,
                datatype=PointField.FLOAT32, count=1,
            ),
            PointField(
                name="y", offset=4,
                datatype=PointField.FLOAT32, count=1,
            ),
            PointField(
                name="z", offset=8,
                datatype=PointField.FLOAT32, count=1,
            ),
        ]
        msg.is_bigendian, msg.point_step = False, 12
        msg.row_step, msg.is_dense = 12 * msg.width, True
        msg.data = np.asarray(points, dtype="<f4").tobytes()
        self.points_pub.publish(msg)

    def step(self, dt, _context):
        if getattr(builtins, BRIDGE_KEY, None) is not self:
            return
        try:
            # Drain a small bounded batch: camera, command, and other ROS
            # callbacks can all arrive during one physics update.
            for _ in range(4):
                rclpy.spin_once(self.node, timeout_sec=0.0)
            target_v, target_w = self.selected_command()

            dt = max(float(dt), 0.0)
            self.v = approach(self.v, target_v, LINEAR_ACCEL * dt)
            self.w = approach(self.w, target_w, ANGULAR_ACCEL * dt)
            self.set_wheels(self.v, self.w)

            now = float(self.timeline.get_current_time())
            self.publish_clock(now)
            if self.last_odom < 0.0 or now - self.last_odom >= 1.0 / ODOM_RATE:
                self.publish_odom(now)
                self.last_odom = now
            if self.last_lidar < 0.0 or now - self.last_lidar >= 1.0 / LIDAR_RATE:
                self.publish_lidar(now)
                self.last_lidar = now
        except Exception as exc:
            current = time.monotonic()
            if current - self.last_error > 2.0:
                print("[DR.Nav] Callback warning:", repr(exc))
                self.last_error = current


async def cleanup_orphan_runtime_objects():
    """Stop DR.Nav objects that lost their builtins handle on an old run."""
    reachable_ids = {
        id(instance)
        for key in RUNTIME_OBJECT_KEYS
        if (instance := getattr(builtins, key, None)) is not None
    }
    candidates = []

    def cleanup_method(instance, class_name):
        """Return the cleanup method only when the old object is still live."""
        if (
            class_name == "WaypointFollower"
            and hasattr(instance, "waypoint_index")
            and hasattr(instance, "sensor_bridge")
            and any(
                getattr(instance, name, None) is not None
                for name in ("physics_callback", "timeline_sub", "node")
            )
        ):
            return "shutdown"
        if (
            class_name == "JackalTeleop"
            and hasattr(instance, "keys")
            and hasattr(instance, "was_active")
            and any(
                getattr(instance, name, None) is not None
                for name in ("keyboard_sub", "update_sub", "node")
            )
        ):
            return "shutdown"
        if (
            class_name == "AutoEpisodePipeline"
            and hasattr(instance, "episode_number")
            and hasattr(instance, "stop_requested")
            and (
                getattr(instance, "running", False)
                or getattr(instance, "timeline_sub", None) is not None
                or getattr(instance, "monitor_task", None) is not None
                or getattr(instance, "stop_cleanup_task", None) is not None
            )
        ):
            return "stop"
        if (
            class_name == "DRNavRecorder"
            and hasattr(instance, "frame_id")
            and hasattr(instance, "lidar_slices")
            and (
                getattr(instance, "running", False)
                or getattr(instance, "task", None) is not None
                or getattr(instance, "rgb_annotator", None) is not None
                or getattr(instance, "render_product", None) is not None
            )
        ):
            return "stop"
        if (
            class_name == "Bridge"
            and hasattr(instance, "command_sources")
            and hasattr(instance, "command_interface_version")
            and (
                getattr(instance, "callback", None) is not None
                or getattr(instance, "node", None) is not None
            )
        ):
            return "shutdown"
        return None

    for instance in gc.get_objects():
        if id(instance) in reachable_ids:
            continue

        try:
            class_name = type(instance).__name__
            method_name = cleanup_method(instance, class_name)
            if method_name is None:
                continue
            candidates.append((instance, method_name))
        except (AttributeError, ReferenceError, RuntimeError):
            continue

    # Pipelines and recorders can own render/timeline work, so drain them
    # before removing command and sensor callbacks.
    priority = {
        "AutoEpisodePipeline": 0,
        "DRNavRecorder": 1,
        "WaypointFollower": 2,
        "JackalTeleop": 2,
        "Bridge": 3,
    }
    candidates.sort(key=lambda item: priority.get(type(item[0]).__name__, 99))

    cleaned = 0
    cleaned_types = {}
    for instance, method_name in candidates:
        try:
            result = getattr(instance, method_name)()
            if asyncio.iscoroutine(result):
                await asyncio.wait_for(result, timeout=10.0)
            cleaned += 1
            class_name = type(instance).__name__
            cleaned_types[class_name] = cleaned_types.get(class_name, 0) + 1
        except asyncio.TimeoutError:
            print(
                "[DR.Nav] Orphan cleanup timed out "
                f"({type(instance).__name__}); continuing."
            )
        except (ReferenceError, RuntimeError):
            pass
        except Exception as exc:
            print(
                "[DR.Nav] Orphan cleanup warning "
                f"({type(instance).__name__}):",
                repr(exc),
            )

    if cleaned:
        summary = ", ".join(
            f"{name}={count}"
            for name, count in sorted(cleaned_types.items())
        )
        print(
            f"[DR.Nav] Cleaned {cleaned} orphan runtime object(s): "
            f"{summary}."
        )


async def main():
    current_task = asyncio.current_task()
    for key in AUXILIARY_TASK_KEYS:
        task = getattr(builtins, key, None)
        if task is None or task is current_task or task.done():
            continue
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"[DR.Nav] Controller task cleanup warning ({key}):", repr(exc))
        finally:
            setattr(builtins, key, None)

    await cleanup_orphan_runtime_objects()

    for key in RECORDER_SERVICE_KEYS:
        service = getattr(builtins, key, None)
        if service is None:
            continue
        try:
            result = service.stop()
            if asyncio.iscoroutine(result):
                await result
            print(f"[DR.Nav] Stopped residual recorder service: {key}")
        except Exception as exc:
            print(f"[DR.Nav] Recorder cleanup warning ({key}):", repr(exc))
        finally:
            setattr(builtins, key, None)

    for key in AUXILIARY_CONTROLLER_KEYS:
        controller = getattr(builtins, key, None)
        if controller is None:
            continue
        try:
            result = controller.shutdown()
            if asyncio.iscoroutine(result):
                await result
            print(f"[DR.Nav] Cleared residual controller: {key}")
        except Exception as exc:
            print(f"[DR.Nav] Controller cleanup warning ({key}):", repr(exc))
        finally:
            setattr(builtins, key, None)

    previous = getattr(builtins, BRIDGE_KEY, None)
    if previous is not None:
        try:
            await previous.shutdown()
        except Exception as exc:
            print("[DR.Nav] Cleanup warning:", repr(exc))
    bridge = Bridge()
    setattr(builtins, BRIDGE_KEY, bridge)
    try:
        await bridge.start()
    except Exception:
        try:
            await bridge.shutdown()
        finally:
            setattr(builtins, BRIDGE_KEY, None)
        raise


asyncio.ensure_future(main())
