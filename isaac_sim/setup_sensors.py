#!/usr/bin/env python3
"""
Isaac Sim 4.5 — DR.Nav Sensor Setup Script (Jackal)
=====================================================
Run from: Isaac Sim → Window → Script Editor → Run

Prerequisites:
  - USD file open with /World/ground/flat_plane/jackal
  - isaacsim.ros2.bridge extension enabled (Window → Extensions → search ROS2)

Topics published:
  /argus/ar0234_front_left/image_raw   10 Hz  RGB image      (infer_vis.py)
  /argus/ar0234_side_left/image_raw    10 Hz  RGB image      (infer_vis.py)
  /argus/ar0234_side_right/image_raw   10 Hz  RGB image      (infer_vis.py)
  /os_cloud_node/points                20 Hz  PointCloud2    (pointcloud_segmenter.py)
  /odom_lidar                          60 Hz  Odometry       (goal_generator.py)

TF tree published:
  odom → base_link          (ROS2PublishOdometry — dynamic, 60 Hz)
  base_link → lidar         (StaticTransformBroadcaster — fixed mount)
  base_link → cam_front     (StaticTransformBroadcaster — fixed mount)
  base_link → cam_side_left (StaticTransformBroadcaster — fixed mount)
  base_link → cam_side_right(StaticTransformBroadcaster — fixed mount)

NOTE — find your Jackal body prim name first (run once in Script Editor):
  from omni.isaac.core.utils.stage import get_current_stage
  from pxr import UsdPhysics
  stage = get_current_stage()
  def pt(p, d=0):
      rb = " [RigidBody]" if p.HasAPI(UsdPhysics.RigidBodyAPI) else ""
      print("  "*d + p.GetName() + f" ({p.GetTypeName()}){rb}")
      [pt(c, d+1) for c in p.GetChildren() if d < 3]
  pt(stage.GetPrimAtPath("/World/ground/flat_plane/jackal"))
  # Then set BODY_LINK below to match the printed body prim name.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import math
import omni
import omni.graph.core as og
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TransformStamped
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera
from pxr import Gf, UsdGeom
from tf2_ros import StaticTransformBroadcaster

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0.5 — JACKAL DIFFERENTIAL DRIVE CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════
class JackalController(BaseController):
    """Open-loop unicycle → 4-wheel differential drive."""
    def __init__(self):
        super().__init__(name="jackal_controller")
        self._wheel_radius = 0.098    # metres
        self._wheel_base   = 0.37559  # metres (track width)

    def forward(self, command):
        """command = [linear_x m/s, angular_z rad/s] → ArticulationAction."""
        v, w = command[0], command[1]
        r, b = self._wheel_radius, self._wheel_base
        left  = ((2 * v) - (w * b)) / (2 * r)
        right = ((2 * v) + (w * b)) / (2 * r)
        # Jackal joint order: FL, FR, RL, RR
        return ArticulationAction(joint_velocities=[left, right, left, right])

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ENABLE EXTENSIONS  (must come before everything else)
# ═══════════════════════════════════════════════════════════════════════════════
manager = omni.kit.app.get_app().get_extension_manager()
for ext in ["isaacsim.core.nodes", "isaacsim.ros2.bridge"]:
    if not manager.is_extension_enabled(ext):
        manager.set_extension_enabled_immediate(ext, True)
        print(f"[DR.Nav] Enabled: {ext}")
    else:
        print(f"[DR.Nav] Already enabled: {ext}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
ROBOT_PRIM = "/World/ground/flat_plane/jackal"

# ── Set BODY_LINK to the actual physics body prim name found by the snippet ──
# Common values: "body", "base_link", "chassis"
# chassisFrameId (ROS TF label) stays "base_link" regardless of BODY_LINK.
BODY_LINK  = "body"
ODOM_PRIM  = f"{ROBOT_PRIM}/{BODY_LINK}"

CAM_FL_PATH = f"{ROBOT_PRIM}/base_link/cam_front"
CAM_SL_PATH = f"{ROBOT_PRIM}/base_link/cam_side_left"
CAM_SR_PATH = f"{ROBOT_PRIM}/base_link/cam_side_right"
LIDAR_PATH  = f"{ROBOT_PRIM}/base_link/lidar"
GRAPH       = "/World/DRNav_Graph"

TOPIC_CAM_FL = "/argus/ar0234_front_left/image_raw"
TOPIC_CAM_SL = "/argus/ar0234_side_left/image_raw"
TOPIC_CAM_SR = "/argus/ar0234_side_right/image_raw"
TOPIC_LIDAR  = "/os_cloud_node/points"
TOPIC_ODOM   = "/odom_lidar"

CAM_W, CAM_H = 640, 480

stage = get_current_stage()
print("[DR.Nav] Stage:", stage.GetRootLayer().identifier)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2.5 — CLEANUP  (safe to re-run: removes old prims before recreating)
# ═══════════════════════════════════════════════════════════════════════════════
for _path in [CAM_FL_PATH, CAM_SL_PATH, CAM_SR_PATH, LIDAR_PATH, GRAPH]:
    if stage.GetPrimAtPath(_path).IsValid():
        stage.RemovePrim(_path)
        print(f"[DR.Nav] Removed existing prim: {_path}")

# Remove stale Replicator render products to prevent "[omni.hydra] Invalid USD
# RenderProduct" errors on re-run.
_rp_parent = stage.GetPrimAtPath("/Render/OmniverseKit/HydraTextures")
if _rp_parent.IsValid():
    for _child in _rp_parent.GetChildren():
        stage.RemovePrim(_child.GetPath())
        print(f"[DR.Nav] Removed stale render product: {_child.GetPath()}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ADD CAMERAS
# ═══════════════════════════════════════════════════════════════════════════════
def add_camera(path, translate, rotate_xyz_deg):
    cam = Camera(
        prim_path=path,
        name=path.split("/")[-1],
        translation=translate,
        orientation=None,
        frequency=10,
        resolution=(CAM_W, CAM_H),
    )
    cam.initialize()
    xform = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(path))
    xform.SetTranslate(translate)
    xform.SetRotate(rotate_xyz_deg, UsdGeom.XformCommonAPI.RotationOrderXYZ)
    print(f"[DR.Nav] Camera: {path}")
    return cam

cam_fl = add_camera(CAM_FL_PATH, Gf.Vec3d( 0.40,  0.15, 0.30), Gf.Vec3f(0, 0,   0))
cam_sl = add_camera(CAM_SL_PATH, Gf.Vec3d( 0.00,  0.20, 0.30), Gf.Vec3f(0, 0,  90))
cam_sr = add_camera(CAM_SR_PATH, Gf.Vec3d( 0.00, -0.20, 0.30), Gf.Vec3f(0, 0, -90))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ADD LiDAR + RENDER PRODUCT
# ═══════════════════════════════════════════════════════════════════════════════
# KEY DISCOVERY: in Isaac Sim 4.5, ROS2RtxLidarHelper uses inputs:renderProductPath
# (NOT inputs:lidarPrimPath which no longer exists).
# Always use rep.create.render_product() directly — do NOT use
# get_render_product_path() which returns the viewport camera's render product.

lidar = LidarRtx(
    prim_path=LIDAR_PATH,
    name="lidar",
    translation=Gf.Vec3d(0.0, 0.0, 0.45),
    config_file_name="OS1_128ch20hz1024res",
)

# Get the render product path the sensor created internally.
# If get_render_product_path() isn't available, fall back to creating one manually.
try:
    lidar_rp_path = lidar.get_render_product_path()
    print(f"[DR.Nav] LiDAR render product (from sensor): {lidar_rp_path}")
except AttributeError:
    lidar_rp = rep.create.render_product(LIDAR_PATH, [1, 1])
    lidar_rp_path = lidar_rp.path
    print(f"[DR.Nav] LiDAR render product (created manually): {lidar_rp_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CAMERA RENDER PRODUCTS
# ═══════════════════════════════════════════════════════════════════════════════
rp_fl = rep.create.render_product(CAM_FL_PATH, (CAM_W, CAM_H))
rp_sl = rep.create.render_product(CAM_SL_PATH, (CAM_W, CAM_H))
rp_sr = rep.create.render_product(CAM_SR_PATH, (CAM_W, CAM_H))
print("[DR.Nav] Camera render products created")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — OMNIGRAPH  (sensors + odometry)
# ═══════════════════════════════════════════════════════════════════════════════
# Confirmed attribute names (from discover_attrs.py):
#
#   Node                    Attribute                  Note
#   ──────────────────────────────────────────────────────────────────────────
#   ROS2RtxLidarHelper      inputs:renderProductPath   NOT lidarPrimPath
#   ROS2RtxLidarHelper      inputs:type                "point_cloud"
#   ROS2RtxLidarHelper      inputs:topicName
#   ROS2RtxLidarHelper      inputs:fullScan
#   ROS2RtxLidarHelper      inputs:execIn
#   ROS2PublishOdometry     inputs:topicName           NOT odomTopicName
#   ROS2PublishOdometry     inputs:chassisFrameId
#   ROS2PublishOdometry     inputs:odomFrameId
#   ROS2PublishOdometry     inputs:execIn
#   ROS2CameraHelper        inputs:renderProductPath
#   ROS2CameraHelper        inputs:topicName
#   ROS2CameraHelper        inputs:type
#   ROS2CameraHelper        inputs:execIn
#   IsaacComputeOdometry    inputs:execIn only         NO chassisPrimPath in 4.5
#   IsaacReadSimulationTime  NO execIn                 pure data node
#
# NOTE: ROS2PublishTransformTree is intentionally NOT used here.
#   It publishes the raw USD hierarchy (e.g. "world → body") which is wrong
#   for SLAM.  Sensor static TFs are published by StaticTransformBroadcaster
#   in Section 7, and odom→base_link TF comes from ROS2PublishOdometry.

og.Controller.edit(
    {"graph_path": GRAPH, "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("tick",        "omni.graph.action.OnPlaybackTick"),
            ("simtime",     "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("camfl",       "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("camsl",       "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("camsr",       "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("lidar",       "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
            ("odomcompute", "isaacsim.core.nodes.IsaacComputeOdometry"),
            ("odompub",     "isaacsim.ros2.bridge.ROS2PublishOdometry"),
        ],

        og.Controller.Keys.CONNECT: [
            # tick drives all execution nodes
            # simtime has NO execIn — it is a pure data node
            ("tick.outputs:tick",  "camfl.inputs:execIn"),
            ("tick.outputs:tick",  "camsl.inputs:execIn"),
            ("tick.outputs:tick",  "camsr.inputs:execIn"),
            ("tick.outputs:tick",  "lidar.inputs:execIn"),
            ("tick.outputs:tick",  "odomcompute.inputs:execIn"),
            ("tick.outputs:tick",  "odompub.inputs:execIn"),

            ("simtime.outputs:simulationTime",      "odompub.inputs:timeStamp"),

            ("odomcompute.outputs:position",        "odompub.inputs:position"),
            ("odomcompute.outputs:orientation",     "odompub.inputs:orientation"),
            ("odomcompute.outputs:linearVelocity",  "odompub.inputs:linearVelocity"),
            ("odomcompute.outputs:angularVelocity", "odompub.inputs:angularVelocity"),
        ],

        og.Controller.Keys.SET_VALUES: [
            # Cameras
            ("camfl.inputs:renderProductPath", rp_fl.path),
            ("camfl.inputs:topicName",         TOPIC_CAM_FL),
            ("camfl.inputs:type",              "rgb"),

            ("camsl.inputs:renderProductPath", rp_sl.path),
            ("camsl.inputs:topicName",         TOPIC_CAM_SL),
            ("camsl.inputs:type",              "rgb"),

            ("camsr.inputs:renderProductPath", rp_sr.path),
            ("camsr.inputs:topicName",         TOPIC_CAM_SR),
            ("camsr.inputs:type",              "rgb"),

            # LiDAR
            ("lidar.inputs:renderProductPath", render_product.path),
            ("lidar.inputs:topicName",         TOPIC_LIDAR),
            ("lidar.inputs:type",              "point_cloud"),
            ("lidar.inputs:fullScan",          True),

            # Odometry
            # NOTE: IsaacComputeOdometry has NO chassisPrimPath in Isaac Sim 4.5
            # — the node auto-detects the articulation root in the scene.
            # chassisFrameId = "body" to match mapless.launch.py odom_tf_broadcaster
            # which subscribes to /odom_lidar and publishes odom → body TF.
            ("odompub.inputs:topicName",           TOPIC_ODOM),
            ("odompub.inputs:chassisFrameId",      "body"),
            ("odompub.inputs:odomFrameId",         "odom"),
        ],
    }
)
print(f"[DR.Nav] OmniGraph built: {GRAPH}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ROS 2 NODE, STATIC TF, CMD_VEL → JACKAL CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

# ── 7a. ROS 2 node ────────────────────────────────────────────────────────────
if not rclpy.ok():
    rclpy.init()

try:
    _ros_node.destroy_node()   # noqa: F821 — clean up on re-run
except (NameError, Exception):
    pass

_ros_node     = rclpy.create_node("drnav_jackal_bridge")
_latest_twist = Twist()

def _cmd_vel_cb(msg: Twist):
    global _latest_twist
    _latest_twist = msg

_ros_node.create_subscription(Twist, "/cmd_vel", _cmd_vel_cb, 10)
print("[DR.Nav] Subscribed to /cmd_vel")

# ── 7b. Static TF: base_link → sensors ───────────────────────────────────────
# These are fixed sensor mounts — published once to /tf_static.
# TF buffer stores them permanently (no timeout), so SLAM always has them.
# Dynamic TF (odom → base_link) comes from ROS2PublishOdometry above.
#
# Full SLAM TF chain:
#   map → odom → base_link → lidar
#                           → cam_front / cam_side_left / cam_side_right
#   map→odom    : published by SLAM
#   odom→base_link : ROS2PublishOdometry (dynamic, 60 Hz)
#   base_link→*    : StaticTransformBroadcaster below (fixed mount, once)

def _make_static_tf(parent, child, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    t = TransformStamped()
    t.header.stamp       = _ros_node.get_clock().now().to_msg()
    t.header.frame_id    = parent
    t.child_frame_id     = child
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.translation.z = z
    t.transform.rotation.x = qx
    t.transform.rotation.y = qy
    t.transform.rotation.z = qz
    t.transform.rotation.w = qw
    return t

_static_br = StaticTransformBroadcaster(_ros_node)
_static_br.sendTransform([
    # lidar: 0.3 m forward, 0.1 m up, no rotation
    _make_static_tf("base_link", "lidar",           0.30,  0.00, 0.10),
    # front cam: slight left offset, facing forward (no rotation)
    _make_static_tf("base_link", "cam_front",       0.40,  0.15, 0.30),
    # side-left cam: 90° yaw → qz=sin(45°)=0.7071, qw=cos(45°)=0.7071
    _make_static_tf("base_link", "cam_side_left",   0.00,  0.20, 0.30,
                    0.0, 0.0,  0.7071, 0.7071),
    # side-right cam: -90° yaw → qz=-0.7071, qw=0.7071
    _make_static_tf("base_link", "cam_side_right",  0.00, -0.20, 0.30,
                    0.0, 0.0, -0.7071, 0.7071),
])
print("[DR.Nav] Static TFs published: base_link → lidar / cam_*")

# ── 7c. Jackal articulation controller ───────────────────────────────────────
# Uses omni.kit.app update stream (NOT World.add_physics_callback) because
# _physx_interface is None until physics actually starts after play().
# The articulation is initialized lazily on the first frame.

_jackal        = Articulation(prim_path=ROBOT_PRIM)
_controller    = JackalController()
_jackal_ready  = [False]

# Dynamic TF broadcaster: publishes odom → base_link every frame.
# ROS2PublishOdometry only publishes nav_msgs/Odometry — it does NOT
# auto-publish TF in Isaac Sim 4.5, so we do it here instead.
_tf_broadcaster = TransformBroadcaster(_ros_node)

try:
    _jackal_sub.unsubscribe()   # noqa: F821 — clean up on re-run
except (NameError, Exception):
    pass

def _jackal_step(dt):
    # Process one pending /cmd_vel message (non-blocking)
    rclpy.spin_once(_ros_node, timeout_sec=0)

    if not _jackal_ready[0]:
        try:
            _jackal.initialize()
            _jackal_ready[0] = True
            print("[DR.Nav] Jackal articulation initialized")
        except Exception:
            return   # physics not ready yet — retry next frame
        return

    # ── Publish odom → base_link TF ──────────────────────────────────────
    # get_world_pose() returns (position [x,y,z], orientation [w,x,y,z])
    pos, ori = _jackal.get_world_pose()
    t = TransformStamped()
    t.header.stamp       = _ros_node.get_clock().now().to_msg()
    t.header.frame_id    = "odom"
    t.child_frame_id     = "base_link"
    t.transform.translation.x = float(pos[0])
    t.transform.translation.y = float(pos[1])
    t.transform.translation.z = float(pos[2])
    # Isaac Sim quaternion convention: [w, x, y, z]
    t.transform.rotation.w = float(ori[0])
    t.transform.rotation.x = float(ori[1])
    t.transform.rotation.y = float(ori[2])
    t.transform.rotation.z = float(ori[3])
    _tf_broadcaster.sendTransform(t)

    # ── Drive robot from /cmd_vel ─────────────────────────────────────────
    lin = _latest_twist.linear.x
    ang = _latest_twist.angular.z
    _jackal.apply_action(_controller.forward(command=[lin, ang]))

_jackal_sub = omni.kit.app.get_app() \
    .get_update_event_stream() \
    .create_subscription_to_pop(
        lambda e: _jackal_step(e.payload.get("dt", 1/60)),
        name="drnav_jackal_ctrl"
    )
print("[DR.Nav] Jackal controller registered")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — START SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
omni.timeline.get_timeline_interface().play()
print("[DR.Nav] Simulation playing")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Verify in a terminal:
  source /opt/ros/humble/setup.bash
  ros2 topic list
  ros2 topic hz /argus/ar0234_front_left/image_raw   # expect ~10 Hz
  ros2 topic hz /os_cloud_node/points                # expect ~20 Hz
  ros2 topic hz /odom_lidar                          # expect ~60 Hz

Verify TF chain:
  ros2 run tf2_ros tf2_echo odom base_link           # dynamic, ~60 Hz
  ros2 run tf2_ros tf2_echo base_link lidar          # static, always valid
  ros2 run tf2_tools view_frames                     # full TF tree PDF

Control robot:
  ros2 run teleop_twist_keyboard teleop_twist_keyboard

Then launch DR.Nav:
  ros2 launch map_contruct mapless.launch.py method:=dram
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
