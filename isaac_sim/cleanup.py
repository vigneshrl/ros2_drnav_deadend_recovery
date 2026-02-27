#!/usr/bin/env python3
"""
Isaac Sim 4.5 — DR.Nav Cleanup Script
=======================================
Run from: Isaac Sim → Window → Script Editor → Run

Removes everything created by setup_sensors.py and setup_teleop.py:
  - OmniGraph  /World/DRNav_Graph
  - OmniGraph  /World/DRNav_DiscoveryGraph  (if left over)
  - Camera prims  /World/Spot/sensor/cam_fl, cam_sl, cam_sr
  - LiDAR prim    /World/Spot/sensor/lidar
  - Sensor folder /World/Spot/sensor
  - All Replicator render products

Run this before re-running setup_sensors.py, or to restore the USD
to its original state before saving.
"""

from omni.isaac.core.utils.stage import get_current_stage
import omni.replicator.core as rep

stage = get_current_stage()

# ── 1. Remove OmniGraphs ──────────────────────────────────────────────────────
GRAPHS = [
    "/World/DRNav_Graph",
    "/World/DRNav_DiscoveryGraph",
    "/World/DRNav_TeleopDiscovery",
]

for path in GRAPHS:
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)
        print(f"[Cleanup] Removed graph: {path}")
    else:
        print(f"[Cleanup] Not found (skip): {path}")

# ── 2. Remove sensor prims ────────────────────────────────────────────────────
SENSOR_PRIMS = [
    "/World/Spot/sensor/cam_fl",
    "/World/Spot/sensor/cam_sl",
    "/World/Spot/sensor/cam_sr",
    "/World/Spot/sensor/lidar",
    "/World/Spot/sensor",          # parent folder — remove last
]

for path in SENSOR_PRIMS:
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)
        print(f"[Cleanup] Removed prim: {path}")
    else:
        print(f"[Cleanup] Not found (skip): {path}")

# ── 3. Clear all Replicator render products ───────────────────────────────────
# rep.orchestrator.stop() flushes the replicator pipeline and
# clears the render product registry (the /Render/... paths).
try:
    rep.orchestrator.stop()
    print("[Cleanup] Replicator render products cleared")
except Exception as e:
    print(f"[Cleanup] Replicator stop skipped: {e}")

# ── 4. Remove teleop update subscription (if setup_teleop.py was run) ─────────
try:
    _teleop_sub.unsubscribe()   # noqa: F821
    print("[Cleanup] Teleop update subscription removed")
except (NameError, Exception):
    print("[Cleanup] No teleop subscription to remove (skip)")

# ── 5. Destroy teleop ROS 2 node (if setup_teleop.py was run) ─────────────────
try:
    _ros_node.destroy_node()    # noqa: F821
    print("[Cleanup] Teleop ROS 2 node destroyed")
except (NameError, Exception):
    print("[Cleanup] No teleop ROS node to destroy (skip)")

print("""
[Cleanup] Done.
Stage is back to scene-only (robot + walls + floor).
You can now save the USD safely, or re-run setup_sensors.py.
""")
