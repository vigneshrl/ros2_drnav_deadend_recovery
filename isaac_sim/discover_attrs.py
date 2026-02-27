"""
Isaac Sim 4.5 — Node Attribute Discovery
==========================================
Run this from the Script Editor BEFORE running setup_sensors.py.
It creates a temporary test graph, prints ALL input/output attributes
for every DR.Nav node, then deletes the graph.

Copy-paste the output here so we know the exact attribute names for your
Isaac Sim 4.5 build.
"""

import omni
import omni.graph.core as og
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Usd

stage = get_current_stage()
manager = omni.kit.app.get_app().get_extension_manager()

# Ensure extensions loaded
for ext in ["isaacsim.core.nodes", "isaacsim.ros2.bridge"]:
    if not manager.is_extension_enabled(ext):
        manager.set_extension_enabled_immediate(ext, True)

TEST_GRAPH = "/World/DRNav_DiscoveryGraph"

# Remove leftover graph from a previous run if it exists
if stage.GetPrimAtPath(TEST_GRAPH):
    stage.RemovePrim(TEST_GRAPH)

# Create nodes — no connections, no SET_VALUES — just instantiate them
og.Controller.edit(
    {"graph_path": TEST_GRAPH, "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("lidar",       "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
            ("odomcompute", "isaacsim.core.nodes.IsaacComputeOdometry"),
            ("odompub",     "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ("camhelper",   "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("simtime",     "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ],
    }
)

# Print every attribute on each node
NODES = {
    "ROS2RtxLidarHelper":        f"{TEST_GRAPH}/lidar",
    "IsaacComputeOdometry":      f"{TEST_GRAPH}/odomcompute",
    "ROS2PublishOdometry":       f"{TEST_GRAPH}/odompub",
    "ROS2CameraHelper":          f"{TEST_GRAPH}/camhelper",
    "IsaacReadSimulationTime":   f"{TEST_GRAPH}/simtime",
}

print("\n" + "═"*60)
print("DR.Nav — Isaac Sim 4.5 Node Attribute Discovery")
print("═"*60)

for label, prim_path in NODES.items():
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"\n[MISSING] {label} — prim not found at {prim_path}")
        continue

    print(f"\n{'─'*60}")
    print(f"NODE: {label}")
    print(f"PATH: {prim_path}")

    inputs  = []
    outputs = []
    for attr in prim.GetAttributes():
        name = attr.GetName()
        typ  = str(attr.GetTypeName())
        val  = attr.Get()
        if name.startswith("inputs:"):
            inputs.append((name, typ, val))
        elif name.startswith("outputs:"):
            outputs.append((name, typ, val))

    if inputs:
        print("  INPUTS:")
        for n, t, v in inputs:
            print(f"    {n:<45} [{t}]  default={v}")
    else:
        print("  INPUTS:  (none)")

    if outputs:
        print("  OUTPUTS:")
        for n, t, v in outputs:
            print(f"    {n:<45} [{t}]")
    else:
        print("  OUTPUTS: (none)")

print("\n" + "═"*60)
print("Paste everything above ^ into the chat so we can fix setup_sensors.py")
print("═"*60 + "\n")
