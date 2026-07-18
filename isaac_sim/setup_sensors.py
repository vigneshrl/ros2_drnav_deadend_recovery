#!/usr/bin/env python3
"""Isaac Sim 6.0 DR.Nav setup for the current Jackal scene."""

import asyncio, builtins, math, time
import carb.input, numpy as np
import omni.appwindow, omni.graph.core as og, omni.kit.app
import omni.timeline, omni.usd
from pxr import Usd, UsdGeom

ROBOT = "/World/ground/flat_plane/jackal"
BASE = f"{ROBOT}/base_link"
CAM_FRAME = f"{BASE}/bumblebee_stereo_camera_frame/bumblebee_stereo_left_frame"
CAMERA = f"{CAM_FRAME}/bumblebee_stereo_left_camera"
LIDAR_FRAME = f"{BASE}/sick_lms1xx_lidar_frame"
LIDAR = f"{LIDAR_FRAME}/Lidar"
GRAPH = "/World/DRNav_RuntimeGraph"

IMAGE_TOPIC = "/argus/ar0234_front_left/image_raw"
SCAN_TOPIC = "/scan"
POINTS_TOPIC = "/os_cloud_node/points"
ODOM_TOPIC = "/odom_lidar"
CLOCK_TOPIC = "/clock"
CMD_TOPIC = "/cmd_vel"

ODOM_FRAME = "odom"
BASE_FRAME = "base_link"
CAM_FRAME_ID = "bumblebee_stereo_left_frame"
LIDAR_FRAME_ID = "sim_lidar"

USE_KEYBOARD = True
LINEAR_SPEED = 0.15
ANGULAR_SPEED = 0.40
LINEAR_ACCEL = 0.35
ANGULAR_ACCEL = 0.90
WHEEL_RADIUS = 0.098
TRACK_WIDTH = 0.37559
MAX_WHEEL_SPEED = 25.0
LEFT_SIGN = 1.0
RIGHT_SIGN = 1.0
CMD_TIMEOUT = 0.5
ODOM_RATE = 30.0
LIDAR_RATE = 10.0

WHEEL_JOINTS = [
    "front_left_wheel_joint", "front_right_wheel_joint",
    "rear_left_wheel_joint", "rear_right_wheel_joint",
]
BRIDGE_KEY = "_DRNAV_ISAACSIM6_V6_BRIDGE"


def approach(value, target, delta):
    return min(value + delta, target) if value < target else max(value - delta, target)


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Bridge:
    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.node = self.cmd_sub = self.lidar = self.robot = None
        self.wheel_ids = self.callback = None
        self.input = self.keyboard = self.keyboard_sub = None
        self.keys = set()
        self.cmd_v = self.cmd_w = self.v = self.w = 0.0
        self.last_cmd = time.monotonic()
        self.last_odom = self.last_lidar = -1.0
        self.prev_time = self.prev_xyz = self.prev_yaw = None
        self.range_min, self.range_max = 0.05, 100.0
        self.last_error = 0.0

    async def start(self):
        self.validate()
        await self.enable_extensions()

        global Articulation, SimulationManager, IsaacEvents, set_target_prims
        global _range_sensor, rclpy, Twist, TransformStamped, Odometry, Clock
        global LaserScan, PointCloud2, PointField
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
        from sensor_msgs.msg import LaserScan, PointCloud2, PointField
        from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

        self.lidar = _range_sensor.acquire_lidar_sensor_interface()
        await self.remove_graph()
        self.disable_lidar_debug()
        self.create_camera_graph()
        self.create_ros()
        self.create_keyboard()
        self.timeline.play()

        for _ in range(6):
            await omni.kit.app.get_app().next_update_async()

        self.robot = Articulation(ROBOT)
        self.wheel_ids = self.robot.get_dof_indices(WHEEL_JOINTS).numpy()
        self.publish_static_tf()
        self.callback = SimulationManager.register_callback(
            self.step, IsaacEvents.POST_PHYSICS_STEP
        )
        print("\n[DR.Nav] Small setup complete.")
        print("[DR.Nav] W/S move, A/D turn, Space stops.")
        print("[DR.Nav] Wheel indices:", self.wheel_ids)

    async def shutdown(self):
        try:
            self.timeline.stop()
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
        if self.input and self.keyboard and self.keyboard_sub is not None:
            try:
                self.input.unsubscribe_to_keyboard_events(
                    self.keyboard, self.keyboard_sub
                )
            except Exception:
                pass
        if self.node:
            try:
                self.node.destroy_node()
            except Exception:
                pass
        self.node = self.keyboard_sub = None
        self.keys.clear()
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
        nodes = [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("context", "isaacsim.ros2.bridge.ROS2Context"),
            ("once", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
            ("render", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("camera", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ]
        links = [
            ("tick.outputs:tick", "once.inputs:execIn"),
            ("once.outputs:step", "render.inputs:execIn"),
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
        self.cmd_sub = self.node.create_subscription(
            Twist, CMD_TOPIC, self.cmd_callback, 10
        )
        self.scan_pub = self.node.create_publisher(LaserScan, SCAN_TOPIC, 10)
        self.points_pub = self.node.create_publisher(PointCloud2, POINTS_TOPIC, 10)
        self.odom_pub = self.node.create_publisher(Odometry, ODOM_TOPIC, 10)
        self.clock_pub = self.node.create_publisher(Clock, CLOCK_TOPIC, 10)
        self.tf_pub = TransformBroadcaster(self.node)
        self.static_tf_pub = StaticTransformBroadcaster(self.node)

    def create_keyboard(self):
        if not USE_KEYBOARD:
            return
        window = omni.appwindow.get_default_app_window()
        if window is None:
            raise RuntimeError("Isaac Sim app window unavailable.")
        self.keyboard = window.get_keyboard()
        self.input = carb.input.acquire_input_interface()
        self.keyboard_sub = self.input.subscribe_to_keyboard_events(
            self.keyboard, self.key_event
        )

    def cmd_callback(self, msg):
        self.cmd_v, self.cmd_w = float(msg.linear.x), float(msg.angular.z)
        self.last_cmd = time.monotonic()

    def key_event(self, event):
        allowed = {
            carb.input.KeyboardInput.W, carb.input.KeyboardInput.A,
            carb.input.KeyboardInput.S, carb.input.KeyboardInput.D,
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
                self.v = self.w = 0.0
                self.set_wheels(0.0, 0.0)
            return True
        if event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            self.keys.add(event.input)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.keys.discard(event.input)
        return True

    def keyboard_cmd(self):
        v = LINEAR_SPEED * (
            int(carb.input.KeyboardInput.W in self.keys)
            - int(carb.input.KeyboardInput.S in self.keys)
        )
        w = ANGULAR_SPEED * (
            int(carb.input.KeyboardInput.A in self.keys)
            - int(carb.input.KeyboardInput.D in self.keys)
        )
        return v, w

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
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
            if USE_KEYBOARD:
                target_v, target_w = self.keyboard_cmd()
            elif time.monotonic() - self.last_cmd > CMD_TIMEOUT:
                target_v, target_w = 0.0, 0.0
            else:
                target_v, target_w = self.cmd_v, self.cmd_w

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


async def main():
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

