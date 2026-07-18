#!/usr/bin/env python3
"""
DR.Nav Replicator recorder — Phase 1
====================================

Run after setup_sensors_isaacsim6_small_v2.py.

Captures synchronized:
- RGB from the existing Bumblebee left camera
- Existing PhysX LiDAR depth/angles/point cloud
- Jackal world pose and estimated velocity
- Maze ID
- Semantic zone label

Output:
~/drnav_dataset/<maze>/<timestamp>/
    manifest.json
    rgb/000000.png
    lidar/000000.npz
    metadata/000000.json

Run this script again to stop the previous recorder and start a new run.
"""

import asyncio
import builtins
import io
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from omni.replicator.core import BackendDispatch
from pxr import Usd, UsdGeom


# -----------------------------------------------------------------------------
# Current scene
# -----------------------------------------------------------------------------

ROBOT = "/World/ground/flat_plane/jackal"
BASE = f"{ROBOT}/base_link"

CAMERA = (
    f"{BASE}/bumblebee_stereo_camera_frame/"
    "bumblebee_stereo_left_frame/"
    "bumblebee_stereo_left_camera"
)

LIDAR = f"{BASE}/sick_lms1xx_lidar_frame/Lidar"

ZONE_GROUPS = [
    # Highest priority first.
    ("dead_end_terminal", 3, "/World/Zones/dead_end_terminal_zone"),
    ("dead_end_branch", 2, "/World/Zones/dead_end_branch_zone"),
    ("goal_path", 1, "/World/Zones/goal_path_zone"),
]

ZONE_OUTSIDE = ("outside", 0)

RESOLUTION = (640, 480)
CAPTURE_HZ = 10.0
USE_Z_FOR_ZONE_TEST = True
OUTPUT_ROOT = Path.home() / "drnav_dataset"

RECORDER_KEY = "_DRNAV_REPLICATOR_RECORDER"


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class DRNavRecorder:
    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()

        self.render_product = None
        self.rgb_annotator = None
        self.lidar_interface = None
        self.backend = None
        self.task = None
        self.running = False

        self.frame_id = 0
        self.next_capture_time = None
        self.zone_volumes = []

        self.previous_time = None
        self.previous_position = None
        self.previous_yaw = None

        self.maze_id = self.get_maze_id()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = OUTPUT_ROOT / self.maze_id / timestamp

    async def start(self):
        self.validate_scene()
        await self.enable_extensions()

        from isaacsim.sensors.physx import _range_sensor

        self.lidar_interface = (
            _range_sensor.acquire_lidar_sensor_interface()
        )

        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.backend = BackendDispatch(
            {"paths": {"out_dir": str(self.output_dir)}}
        )

        self.build_zone_index()
        self.write_manifest()

        # We trigger capture manually at CAPTURE_HZ.
        rep.orchestrator.set_capture_on_play(False)

        self.render_product = rep.create.render_product(
            CAMERA,
            RESOLUTION,
            name="drnav_dataset_render_product",
        )
        self.render_product.hydra_texture.set_updates_enabled(False)

        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach(self.render_product)

        self.running = True
        self.task = asyncio.ensure_future(self.capture_loop())

        print("\n[DR.Nav Recorder] Started")
        print("[DR.Nav Recorder] Maze:", self.maze_id)
        print("[DR.Nav Recorder] Output:", self.output_dir)
        print("[DR.Nav Recorder] Capture rate:", CAPTURE_HZ, "Hz")
        print(
            "[DR.Nav Recorder] Zone volumes:",
            len(self.zone_volumes),
        )

    async def stop(self):
        self.running = False

        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print("[DR.Nav Recorder] Loop cleanup warning:", repr(exc))

        self.task = None

        try:
            await rep.orchestrator.wait_until_complete_async()
        except Exception:
            pass

        if self.rgb_annotator is not None:
            try:
                self.rgb_annotator.detach()
            except Exception:
                pass

        if self.render_product is not None:
            try:
                self.render_product.destroy()
            except Exception:
                pass

        self.rgb_annotator = None
        self.render_product = None

        print(
            "[DR.Nav Recorder] Stopped after",
            self.frame_id,
            "frames.",
        )

    def validate_scene(self):
        missing = [
            path
            for path in [ROBOT, BASE, CAMERA, LIDAR]
            if not self.stage.GetPrimAtPath(path).IsValid()
        ]

        if missing:
            raise RuntimeError(
                "Missing required scene prims:\n  "
                + "\n  ".join(missing)
            )

        lidar_prim = self.stage.GetPrimAtPath(LIDAR)

        if lidar_prim.GetTypeName() != "Lidar":
            raise RuntimeError(
                f"Expected existing PhysX Lidar at {LIDAR}, "
                f"found {lidar_prim.GetTypeName()!r}"
            )

    async def enable_extensions(self):
        manager = omni.kit.app.get_app().get_extension_manager()

        for extension in [
            "omni.replicator.core",
            "isaacsim.sensors.physx",
        ]:
            if not manager.is_extension_enabled(extension):
                manager.set_extension_enabled_immediate(
                    extension,
                    True,
                )

        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

    def get_maze_id(self):
        root = self.stage.GetRootLayer()
        source = root.realPath or root.identifier
        name = Path(source).stem

        if not name or name.startswith("anon"):
            return "unknown_maze"

        return name

    # ------------------------------------------------------------------
    # Zones
    # ------------------------------------------------------------------

    def build_zone_index(self):
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_],
            useExtentsHint=True,
        )

        for label, zone_id, group_path in ZONE_GROUPS:
            group = self.stage.GetPrimAtPath(group_path)

            if not group.IsValid():
                print(
                    "[DR.Nav Recorder] Missing zone group:",
                    group_path,
                )
                continue

            found = 0

            for prim in Usd.PrimRange(group):
                if prim == group or prim.GetTypeName() != "Cube":
                    continue

                world_bound = cache.ComputeWorldBound(prim)
                aligned = world_bound.ComputeAlignedRange()
                minimum = np.array(
                    aligned.GetMin(),
                    dtype=np.float64,
                )
                maximum = np.array(
                    aligned.GetMax(),
                    dtype=np.float64,
                )

                self.zone_volumes.append(
                    {
                        "label": label,
                        "id": zone_id,
                        "path": str(prim.GetPath()),
                        "min": minimum,
                        "max": maximum,
                    }
                )
                found += 1

            print(
                f"[DR.Nav Recorder] {label}: "
                f"{found} cube volume(s)"
            )

        if not self.zone_volumes:
            raise RuntimeError(
                "No zone cube volumes found under /World/Zones."
            )

    def zone_at(self, position):
        point = np.asarray(position, dtype=np.float64)

        for volume in self.zone_volumes:
            minimum = volume["min"]
            maximum = volume["max"]

            inside_xy = (
                minimum[0] <= point[0] <= maximum[0]
                and minimum[1] <= point[1] <= maximum[1]
            )

            inside_z = (
                minimum[2] <= point[2] <= maximum[2]
                if USE_Z_FOR_ZONE_TEST
                else True
            )

            if inside_xy and inside_z:
                return (
                    volume["label"],
                    volume["id"],
                    volume["path"],
                )

        return ZONE_OUTSIDE[0], ZONE_OUTSIDE[1], None

    # ------------------------------------------------------------------
    # Robot state
    # ------------------------------------------------------------------

    def read_robot_state(self, now):
        matrix = UsdGeom.XformCache(
            Usd.TimeCode.Default()
        ).GetLocalToWorldTransform(
            self.stage.GetPrimAtPath(BASE)
        )

        translation = matrix.ExtractTranslation()
        quaternion = matrix.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()

        x, y, z = map(float, translation)
        qx, qy, qz = map(float, imaginary)
        qw = float(quaternion.GetReal())

        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        vx = vy = vz = wz = 0.0

        if self.previous_time is not None:
            dt = now - self.previous_time

            if dt > 1e-6:
                vx = (x - self.previous_position[0]) / dt
                vy = (y - self.previous_position[1]) / dt
                vz = (z - self.previous_position[2]) / dt
                wz = wrap_angle(yaw - self.previous_yaw) / dt

        self.previous_time = now
        self.previous_position = (x, y, z)
        self.previous_yaw = yaw

        zone_label, zone_id, zone_path = self.zone_at((x, y, z))

        return {
            "position": {"x": x, "y": y, "z": z},
            "orientation": {
                "x": qx,
                "y": qy,
                "z": qz,
                "w": qw,
            },
            "yaw": yaw,
            "velocity_world": {
                "x": vx,
                "y": vy,
                "z": vz,
                "yaw_rate": wz,
            },
            "zone": {
                "label": zone_label,
                "id": zone_id,
                "volume_path": zone_path,
            },
        }

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write_manifest(self):
        manifest = {
            "format": "drnav_replicator_phase1",
            "version": 1,
            "maze_id": self.maze_id,
            "created": datetime.now().isoformat(),
            "capture_hz": CAPTURE_HZ,
            "resolution": list(RESOLUTION),
            "camera_prim": CAMERA,
            "lidar_prim": LIDAR,
            "robot_base_prim": BASE,
            "zone_priority": [
                item[0] for item in ZONE_GROUPS
            ],
            "use_z_for_zone_test": USE_Z_FOR_ZONE_TEST,
            "zone_ids": {
                "outside": 0,
                "goal_path": 1,
                "dead_end_branch": 2,
                "dead_end_terminal": 3,
            },
        }

        self.write_json("manifest.json", manifest)

    def write_json(self, relative_path, data):
        blob = json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        self.backend.write_blob(relative_path, blob)

    def capture_lidar(self):
        depth = np.asarray(
            self.lidar_interface.get_linear_depth_data(LIDAR),
            dtype=np.float32,
        )
        azimuth = np.asarray(
            self.lidar_interface.get_azimuth_data(LIDAR),
            dtype=np.float32,
        )
        zenith = np.asarray(
            self.lidar_interface.get_zenith_data(LIDAR),
            dtype=np.float32,
        )
        points = np.asarray(
            self.lidar_interface.get_point_cloud_data(LIDAR),
            dtype=np.float32,
        )

        if points.size:
            points = points.reshape(-1, 3)

        return depth, azimuth, zenith, points

    def write_frame(self, rgb, now):
        frame_name = f"{self.frame_id:06d}"
        state = self.read_robot_state(now)

        depth, azimuth, zenith, points = self.capture_lidar()

        self.backend.write_image(
            f"rgb/{frame_name}.png",
            np.asarray(rgb),
        )

        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            depth=depth,
            azimuth=azimuth,
            zenith=zenith,
            points=points,
        )
        self.backend.write_blob(
            f"lidar/{frame_name}.npz",
            buffer.getvalue(),
        )

        metadata = {
            "frame_id": self.frame_id,
            "simulation_time": now,
            "maze_id": self.maze_id,
            "rgb_path": f"rgb/{frame_name}.png",
            "lidar_path": f"lidar/{frame_name}.npz",
            "robot": state,
            "lidar_shapes": {
                "depth": list(depth.shape),
                "azimuth": list(azimuth.shape),
                "zenith": list(zenith.shape),
                "points": list(points.shape),
            },
        }

        self.write_json(
            f"metadata/{frame_name}.json",
            metadata,
        )

        if self.frame_id % 20 == 0:
            print(
                f"[DR.Nav Recorder] frame={self.frame_id} "
                f"time={now:.2f} "
                f"zone={state['zone']['label']}"
            )

        self.frame_id += 1

    # ------------------------------------------------------------------
    # Capture loop
    # ------------------------------------------------------------------

    async def capture_loop(self):
        interval = 1.0 / CAPTURE_HZ

        while self.running:
            await omni.kit.app.get_app().next_update_async()

            if not self.timeline.is_playing():
                self.next_capture_time = None
                continue

            now = float(self.timeline.get_current_time())

            if self.next_capture_time is None:
                self.next_capture_time = now

            if now + 1e-9 < self.next_capture_time:
                continue

            self.render_product.hydra_texture.set_updates_enabled(True)

            try:
                await rep.orchestrator.step_async(
                    delta_time=0.0,
                    pause_timeline=False,
                    rt_subframes=1,
                )
                rgb = self.rgb_annotator.get_data()

                if rgb is not None and np.asarray(rgb).size:
                    capture_time = float(
                        self.timeline.get_current_time()
                    )
                    self.write_frame(rgb, capture_time)
            finally:
                self.render_product.hydra_texture.set_updates_enabled(
                    False
                )

            while self.next_capture_time <= now:
                self.next_capture_time += interval


async def main():
    previous = getattr(builtins, RECORDER_KEY, None)

    if previous is not None:
        try:
            await previous.stop()
        except Exception as exc:
            print(
                "[DR.Nav Recorder] Previous recorder cleanup warning:",
                repr(exc),
            )

    recorder = DRNavRecorder()
    setattr(builtins, RECORDER_KEY, recorder)

    try:
        await recorder.start()
    except Exception:
        try:
            await recorder.stop()
        finally:
            setattr(builtins, RECORDER_KEY, None)
        raise


asyncio.ensure_future(main())
