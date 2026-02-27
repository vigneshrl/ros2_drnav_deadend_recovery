#!/usr/bin/env python3
"""
Isaac Sim 4.5 — DR.Nav Teleop (Kinematic / Transform Control)
==============================================================
Run from: Isaac Sim → Window → Script Editor → Run
Run AFTER setup_sensors.py (simulation must already be playing).

WHY KINEMATIC (not physics velocity)?
  Every Isaac Sim 4.5 high-level API that sets rigid body velocity
  (RigidPrim, get_physx_interface, dynamic_control) calls the same
  internal physx interface that returns None unless you go through
  World.reset() — which restarts the whole simulation.

  Kinematic control directly writes the USD transform each physics step.
  No physx interface needed. Zero chance of the NoneType error.

  For DR.Nav experiments this is FINE because:
    • LiDAR ray-casting works against geometry, not rigid bodies
    • Camera rendering ignores physics state
    • DR.Nav detects walls via sensors and steers around them
    • PAD / NPR metrics only care about robot position over time
"""

import math
import rclpy
from geometry_msgs.msg import Twist
from pxr import Gf, UsdGeom, UsdPhysics
from omni.isaac.core.utils.stage import get_current_stage

ROBOT_ROOT = "/World/Spot"

# ── 1. ROS 2 subscriber ───────────────────────────────────────────────────────
if not rclpy.ok():
    rclpy.init()

# Destroy old node if this script is re-run
try:
    _ros_node.destroy_node()   # noqa: F821
except (NameError, Exception):
    pass

_ros_node     = rclpy.create_node("drnav_teleop_bridge")
_latest_twist = Twist()

def _cmd_vel_cb(msg: Twist):
    global _latest_twist
    _latest_twist = msg

_ros_node.create_subscription(Twist, "/cmd_vel", _cmd_vel_cb, 10)
print("[Teleop] Subscribed to /cmd_vel")

# ── 2. Read Spot's initial pose from USD ──────────────────────────────────────
# We integrate position ourselves, so we must start from wherever Spot
# is placed in the scene, not from the origin.

stage      = get_current_stage()
robot_prim = stage.GetPrimAtPath(ROBOT_ROOT)
xformable  = UsdGeom.Xformable(robot_prim)

# Compute the current world transform (read-only, no physx needed)
init_xf  = xformable.ComputeLocalToWorldTransform(0)
init_pos = init_xf.ExtractTranslation()
init_rot = init_xf.ExtractRotationMatrix()

# Isaac Sim is Z-up, X-forward:
#   rotation matrix column 0 = X-axis = forward direction in world space
# atan2(rot[1][0], rot[0][0]) gives the yaw around Z
_pos = [init_pos[0], init_pos[1], init_pos[2]]   # [x, y, z]
_yaw = math.atan2(float(init_rot[1][0]), float(init_rot[0][0]))

print(f"[Teleop] Initial position : ({_pos[0]:.2f}, {_pos[1]:.2f}, {_pos[2]:.2f})")
print(f"[Teleop] Initial yaw      : {math.degrees(_yaw):.1f}°")

# ── 3. Mark robot as kinematic so physics doesn't override our transform ──────
# A kinematic rigid body still participates in collision detection for OTHER
# objects but its own position is driven by the transform, not by forces.
# If Spot has no RigidBodyAPI this step is silently skipped.

rb_api = UsdPhysics.RigidBodyAPI(robot_prim)
if rb_api:
    rb_api.CreateKinematicEnabledAttr().Set(True)
    print("[Teleop] Spot set to kinematic (transform-driven)")
else:
    print("[Teleop] No RigidBodyAPI on /World/Spot — skipping kinematic flag")

# ── 4. Get the translate and rotate XformOps ──────────────────────────────────
# Every USD prim that can be moved has XformOps (translate, rotate, scale).
# We find the existing ones; if none exist we add them.

def _find_or_create_ops(xformable, init_pos, init_yaw):
    """Return (translate_op, orient_op) from the prim's XformOp stack."""
    t_op = None
    r_op = None

    for op in xformable.GetOrderedXformOps():
        ot = op.GetOpType()
        if ot == UsdGeom.XformOp.TypeTranslate:
            t_op = op
        elif ot in (UsdGeom.XformOp.TypeOrient,
                    UsdGeom.XformOp.TypeRotateXYZ,
                    UsdGeom.XformOp.TypeRotateZ):
            r_op = op

    # Create missing ops
    if t_op is None:
        t_op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        t_op.Set(Gf.Vec3d(init_pos[0], init_pos[1], init_pos[2]))
        print("[Teleop] Created TranslateOp")

    if r_op is None:
        r_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        h = init_yaw / 2.0
        r_op.Set(Gf.Quatd(math.cos(h), 0.0, 0.0, math.sin(h)))
        print("[Teleop] Created OrientOp")

    return t_op, r_op

_translate_op, _orient_op = _find_or_create_ops(xformable, _pos, _yaw)
_orient_type = _orient_op.GetOpType()

# ── 5. Transform-update callback ──────────────────────────────────────────────
def _teleop_step(dt: float):
    global _pos, _yaw

    # Non-blocking ROS spin — processes one queued message if available
    rclpy.spin_once(_ros_node, timeout_sec=0)

    lin = _latest_twist.linear.x    # m/s forward
    ang = _latest_twist.angular.z   # rad/s yaw

    # Nothing to do if no command
    if abs(lin) < 1e-4 and abs(ang) < 1e-4:
        return

    # Integrate pose (simple Euler, dt ≈ 1/60 s is fine for this)
    _yaw    += ang * dt
    _pos[0] += lin * math.cos(_yaw) * dt
    _pos[1] += lin * math.sin(_yaw) * dt

    # Write translation — keep original Z so Spot stays on the floor
    _translate_op.Set(Gf.Vec3d(_pos[0], _pos[1], _pos[2]))

    # Write rotation — handle OrientOp (quaternion) or RotateXYZ (euler degrees)
    if _orient_type == UsdGeom.XformOp.TypeOrient:
        h = _yaw / 2.0
        _orient_op.Set(Gf.Quatd(math.cos(h), 0.0, 0.0, math.sin(h)))
    elif _orient_type in (UsdGeom.XformOp.TypeRotateXYZ,
                          UsdGeom.XformOp.TypeRotateZ):
        _orient_op.Set(Gf.Vec3f(0.0, 0.0, math.degrees(_yaw)))

# ── 6. Register callback ──────────────────────────────────────────────────────
# We use omni.kit.app update stream — fires every frame, no World needed.
# This completely avoids all physx initialisation issues.

import omni.kit.app

# Remove old subscription if re-running
try:
    _teleop_sub.unsubscribe()   # noqa: F821
except (NameError, Exception):
    pass

_teleop_sub = omni.kit.app.get_app() \
    .get_update_event_stream() \
    .create_subscription_to_pop(
        lambda e: _teleop_step(e.payload.get("dt", 1/60)),
        name="drnav_teleop"
    )

print("[Teleop] Update callback registered — /cmd_vel now controls Spot")
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test teleop (new terminal):
  source /opt/ros/humble/setup.bash
  ros2 run teleop_twist_keyboard teleop_twist_keyboard

  i / , = forward / backward
  j / l = turn left / right
  k     = stop

Then launch DR.Nav:
  ros2 launch map_contruct mapless.launch.py method:=dram
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
