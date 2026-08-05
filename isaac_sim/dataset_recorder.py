#!/usr/bin/env python3
"""Synchronized RGB, full-scan LiDAR, pose, and semantic-zone recorder."""

import asyncio
import builtins
import io
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from omni.replicator.core import BackendDispatch
from pxr import Usd, UsdGeom


ROBOT = "/World/ground/flat_plane/jackal"
BASE = f"{ROBOT}/base_link"
CAMERA = (
    f"{BASE}/bumblebee_stereo_camera_frame/"
    "bumblebee_stereo_left_frame/bumblebee_stereo_left_camera"
)
LIDAR = f"{BASE}/sick_lms1xx_lidar_frame/Lidar"
ZONE_RULES = [
    (
        "dead_end_terminal",
        3,
        ("dead_end_terminal", "deadendterminal", "terminal_zone"),
    ),
    (
        "dead_end_branch",
        2,
        ("dead_end_branch", "deadendbranch", "branch_zone"),
    ),
    ("goal_path", 1, ("goal_path", "goalpath", "path_zone")),
]
ZONE_OUTSIDE = ("outside", 0)
RECORDER_KEY = "_DRNAV_REPLICATOR_RECORDER"

CONFIG = getattr(builtins, "_DRNAV_RUNTIME_CONFIG", {}).get("recorder", {})
RESOLUTION = tuple(int(value) for value in CONFIG.get("resolution", [640, 480]))
CAPTURE_HZ = float(CONFIG.get("capture_hz", 10.0))
USE_Z_FOR_ZONE_TEST = bool(CONFIG.get("use_z_for_zone_test", True))
REQUIRE_LIDAR_DATA = bool(CONFIG.get("require_lidar_data", True))
MIN_LIDAR_COVERAGE = float(CONFIG.get("minimum_lidar_coverage", 0.95))
MAX_LIDAR_SLICES = int(CONFIG.get("max_lidar_slices", 12))
OUTPUT_ROOT = Path(CONFIG.get("output_root", "~/drnav_dataset")).expanduser()


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class DRNavRecorder:
    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.render_product = self.rgb_annotator = None
        self.lidar_interface = self.backend = self.task = None
        self.running = self.capture_in_progress = False
        self.stopped = False
        self.stop_lock = asyncio.Lock()
        self.timeline_auto_update = self.timeline.is_auto_updating()

        self.frame_id = 0
        self.last_capture_time = self.last_written_time = None
        self.skipped_frames = self.empty_rgb_frames = 0
        self.empty_lidar_frames = self.incomplete_lidar_scans = 0
        self.lidar_slices = []
        self.expected_azimuth_bins = None
        self.has_seen_playback = False
        self.zone_volumes = []
        self.previous_time = self.previous_position = self.previous_yaw = None

        self.maze_id = self.get_maze_id()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.output_dir = OUTPUT_ROOT / self.maze_id / timestamp

    async def start(self):
        self.validate_scene()
        await self.enable_extensions()
        from isaacsim.sensors.physx import _range_sensor

        self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.backend = BackendDispatch(
            {"paths": {"out_dir": str(self.output_dir)}}
        )
        self.build_zone_index()
        self.configure_lidar()
        self.write_manifest()

        rep.orchestrator.set_capture_on_play(False)
        self.render_product = rep.create.render_product(
            CAMERA,
            RESOLUTION,
            name="drnav_dataset_render_product",
        )
        self.render_product.hydra_texture.set_updates_enabled(False)
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach(self.render_product)
        print("[DR.Nav Recorder] RGB source: exclusive Replicator render product")

        self.running = True
        self.task = asyncio.ensure_future(self.capture_loop())
        print("\n[DR.Nav Recorder] Started")
        print("[DR.Nav Recorder] Maze:", self.maze_id)
        print("[DR.Nav Recorder] Output:", self.output_dir)
        print("[DR.Nav Recorder] Capture rate:", CAPTURE_HZ, "Hz")
        print("[DR.Nav Recorder] Zone volumes:", len(self.zone_volumes))

    async def stop(self):
        current = asyncio.current_task()
        async with self.stop_lock:
            if self.stopped:
                return
            self.running = False
            task_to_wait = (
                self.task
                if self.task is not None
                and self.task is not current
                and not self.task.done()
                else None
            )

        if task_to_wait is not None:
            try:
                await task_to_wait
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print("[DR.Nav Recorder] Loop cleanup warning:", repr(exc))

        async with self.stop_lock:
            if self.stopped:
                return
            self.task = None
            try:
                if not rep.orchestrator.get_is_stopped():
                    await rep.orchestrator.stop_async()
            except Exception as exc:
                print("[DR.Nav Recorder] Replicator cleanup warning:", repr(exc))

            self.timeline.set_auto_update(self.timeline_auto_update)
            self.timeline.commit_silently()
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
            self.rgb_annotator = self.render_product = None
            self.stopped = True

            print("[DR.Nav Recorder] Stopped after", self.frame_id, "frames.")
            print("[DR.Nav Recorder] Skipped frames:", self.skipped_frames)
            print("[DR.Nav Recorder] Empty RGB frames:", self.empty_rgb_frames)
            print(
                "[DR.Nav Recorder] Empty LiDAR slices:",
                self.empty_lidar_frames,
            )
            print(
                "[DR.Nav Recorder] Incomplete LiDAR scans:",
                self.incomplete_lidar_scans,
            )

    def validate_scene(self):
        if self.stage is None:
            raise RuntimeError("No USD stage is currently open.")
        missing = [
            path
            for path in (ROBOT, BASE, CAMERA, LIDAR)
            if not self.stage.GetPrimAtPath(path).IsValid()
        ]
        if missing:
            raise RuntimeError("Missing required scene prims:\n  " + "\n  ".join(missing))
        lidar = self.stage.GetPrimAtPath(LIDAR)
        if lidar.GetTypeName() != "Lidar":
            raise RuntimeError(
                f"Expected PhysX Lidar at {LIDAR}, "
                f"found {lidar.GetTypeName()!r}"
            )

    async def enable_extensions(self):
        manager = omni.kit.app.get_app().get_extension_manager()
        for extension in ("omni.replicator.core", "isaacsim.sensors.physx"):
            if not manager.is_extension_enabled(extension):
                manager.set_extension_enabled_immediate(extension, True)
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

    def get_maze_id(self):
        if self.stage is None:
            return "unknown_maze"
        root = self.stage.GetRootLayer()
        name = Path(root.realPath or root.identifier).stem
        return "unknown_maze" if not name or name.startswith("anon") else name

    def build_zone_index(self):
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_],
            useExtentsHint=True,
        )
        counts = {label: 0 for label, _, _ in ZONE_RULES}
        zone_like_paths = []

        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            normalized = path.lower().replace("-", "_").replace(" ", "_")
            if any(
                token in normalized
                for token in (
                    "zone",
                    "dead_end",
                    "deadend",
                    "goal_path",
                    "goalpath",
                )
            ):
                zone_like_paths.append(path)
            if prim.GetTypeName() != "Cube":
                continue

            match = next(
                (
                    (label, zone_id)
                    for label, zone_id, aliases in ZONE_RULES
                    if any(alias in normalized for alias in aliases)
                ),
                None,
            )
            if match is None:
                continue

            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            self.zone_volumes.append(
                {
                    "label": match[0],
                    "id": match[1],
                    "path": path,
                    "min": np.asarray(bounds.GetMin(), dtype=np.float64),
                    "max": np.asarray(bounds.GetMax(), dtype=np.float64),
                }
            )
            counts[match[0]] += 1

        self.zone_volumes.sort(key=lambda zone: zone["id"], reverse=True)
        for label, _, _ in ZONE_RULES:
            print(f"[DR.Nav Recorder] {label}: {counts[label]} cube volume(s)")
        if not self.zone_volumes:
            print("[DR.Nav Recorder] Zone-like prims discovered:")
            for path in zone_like_paths[:100]:
                print("   ", path)
            raise RuntimeError(
                "No semantic zone Cube prims were matched. Rename the zone "
                "cubes or add their path keywords to ZONE_RULES."
            )

    def zone_at(self, position):
        point = np.asarray(position, dtype=np.float64)
        for zone in self.zone_volumes:
            minimum, maximum = zone["min"], zone["max"]
            inside = (
                minimum[0] <= point[0] <= maximum[0]
                and minimum[1] <= point[1] <= maximum[1]
                and (
                    not USE_Z_FOR_ZONE_TEST
                    or minimum[2] <= point[2] <= maximum[2]
                )
            )
            if inside:
                return zone["label"], zone["id"], zone["path"]
        return ZONE_OUTSIDE[0], ZONE_OUTSIDE[1], None

    def read_robot_state(self, now):
        matrix = UsdGeom.XformCache(
            Usd.TimeCode.Default()
        ).GetLocalToWorldTransform(self.stage.GetPrimAtPath(BASE))
        translation = matrix.ExtractTranslation()
        quaternion = matrix.ExtractRotationQuat()
        x, y, z = map(float, translation)
        qx, qy, qz = map(float, quaternion.GetImaginary())
        qw = float(quaternion.GetReal())
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        vx = vy = vz = wz = 0.0
        if self.previous_time is not None:
            elapsed = now - self.previous_time
            if elapsed > 1e-6:
                vx = (x - self.previous_position[0]) / elapsed
                vy = (y - self.previous_position[1]) / elapsed
                vz = (z - self.previous_position[2]) / elapsed
                wz = wrap_angle(yaw - self.previous_yaw) / elapsed

        self.previous_time = now
        self.previous_position = (x, y, z)
        self.previous_yaw = yaw
        zone_label, zone_id, zone_path = self.zone_at((x, y, z))
        return {
            "position": {"x": x, "y": y, "z": z},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
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

    def configure_lidar(self):
        prim = self.stage.GetPrimAtPath(LIDAR)
        fov = prim.GetAttribute("horizontalFov").Get()
        resolution = prim.GetAttribute("horizontalResolution").Get()
        if fov is None or resolution is None or float(resolution) <= 0.0:
            raise RuntimeError("PhysX LiDAR angular configuration is unavailable.")
        self.expected_azimuth_bins = int(round(float(fov) / float(resolution)))
        if self.expected_azimuth_bins <= 0:
            raise RuntimeError("Computed invalid PhysX LiDAR azimuth bin count.")
        print(
            "[DR.Nav Recorder] Expected full LiDAR scan:",
            self.expected_azimuth_bins,
            "azimuth bins.",
        )

    def capture_lidar(self):
        interface = self.lidar_interface
        depth = np.asarray(
            interface.get_linear_depth_data(LIDAR), dtype=np.float32
        ).copy()
        azimuth = np.asarray(
            interface.get_azimuth_data(LIDAR), dtype=np.float32
        ).copy()
        zenith = np.asarray(
            interface.get_zenith_data(LIDAR), dtype=np.float32
        ).copy()
        points = np.asarray(
            interface.get_point_cloud_data(LIDAR), dtype=np.float32
        ).copy()
        return depth, azimuth, zenith, points.reshape(-1, 3) if points.size else points

    def accumulate_lidar(self, data):
        depth, azimuth, zenith, points = data
        vertical_count = max(1, int(zenith.size))
        if not depth.size or not azimuth.size or not points.size:
            self.empty_lidar_frames += 1
            return
        if (
            depth.shape[0] != azimuth.size
            or points.size != azimuth.size * vertical_count * 3
        ):
            self.empty_lidar_frames += 1
            print(
                "[DR.Nav Recorder] LiDAR buffer shape mismatch:",
                depth.shape,
                azimuth.shape,
                zenith.shape,
                points.shape,
            )
            return
        self.lidar_slices.append(
            (
                depth,
                azimuth,
                zenith,
                points.reshape(azimuth.size, vertical_count, 3),
            )
        )
        if len(self.lidar_slices) > MAX_LIDAR_SLICES:
            self.lidar_slices.pop(0)

    def completed_lidar_scan(self):
        if not self.lidar_slices:
            return None
        depth = np.concatenate([item[0] for item in self.lidar_slices], axis=0)
        azimuth = np.concatenate([item[1] for item in self.lidar_slices])
        zenith = self.lidar_slices[-1][2]
        point_rows = np.concatenate(
            [item[3] for item in self.lidar_slices], axis=0
        )
        _, unique = np.unique(
            np.round(azimuth.astype(np.float64), 6),
            return_index=True,
        )
        unique.sort()
        depth, azimuth, point_rows = (
            depth[unique],
            azimuth[unique],
            point_rows[unique],
        )
        order = np.argsort(azimuth)
        depth, azimuth, point_rows = (
            depth[order],
            azimuth[order],
            point_rows[order],
        )
        if azimuth.size / self.expected_azimuth_bins < MIN_LIDAR_COVERAGE:
            return None
        limit = self.expected_azimuth_bins
        self.lidar_slices.clear()
        return (
            depth[:limit],
            azimuth[:limit],
            zenith,
            point_rows[:limit].reshape(-1, 3),
        )

    def write_manifest(self):
        self.write_json(
            "manifest.json",
            {
                "format": "drnav_replicator_phase1",
                "version": 2,
                "require_lidar_data": REQUIRE_LIDAR_DATA,
                "minimum_lidar_azimuth_coverage": MIN_LIDAR_COVERAGE,
                "expected_lidar_azimuth_bins": self.expected_azimuth_bins,
                "maze_id": self.maze_id,
                "created": datetime.now().isoformat(),
                "capture_hz": CAPTURE_HZ,
                "resolution": list(RESOLUTION),
                "camera_prim": CAMERA,
                "lidar_prim": LIDAR,
                "robot_base_prim": BASE,
                "zone_priority": [item[0] for item in ZONE_RULES],
                "zone_overlap_policy": "highest_zone_id_wins",
                "use_z_for_zone_test": USE_Z_FOR_ZONE_TEST,
                "appearance_randomization": getattr(
                    builtins, "_DRNAV_APPEARANCE_STATE", None
                ),
                "zone_ids": {
                    "outside": 0,
                    "goal_path": 1,
                    "dead_end_branch": 2,
                    "dead_end_terminal": 3,
                },
            },
        )

    def write_json(self, path, data):
        self.backend.write_blob(
            path,
            json.dumps(data, indent=2, sort_keys=True).encode("utf-8"),
        )

    def write_frame(self, rgb, now, lidar):
        name = f"{self.frame_id:06d}"
        state = self.read_robot_state(now)
        depth, azimuth, zenith, points = lidar
        self.backend.write_image(f"rgb/{name}.png", np.asarray(rgb))
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            depth=depth,
            azimuth=azimuth,
            zenith=zenith,
            points=points,
        )
        self.backend.write_blob(f"lidar/{name}.npz", buffer.getvalue())
        self.write_json(
            f"metadata/{name}.json",
            {
                "frame_id": self.frame_id,
                "simulation_time": now,
                "maze_id": self.maze_id,
                "rgb_path": f"rgb/{name}.png",
                "lidar_path": f"lidar/{name}.npz",
                "robot": state,
                "lidar_shapes": {
                    "depth": list(depth.shape),
                    "azimuth": list(azimuth.shape),
                    "zenith": list(zenith.shape),
                    "points": list(points.shape),
                },
            },
        )
        if self.frame_id % 20 == 0:
            print(
                f"[DR.Nav Recorder] frame={self.frame_id} time={now:.2f} "
                f"zone={state['zone']['label']}"
            )
        self.frame_id += 1

    async def capture_loop(self):
        interval = 1.0 / CAPTURE_HZ
        while self.running:
            await omni.kit.app.get_app().next_update_async()
            if (
                not self.running
                or getattr(builtins, RECORDER_KEY, None) is not self
            ):
                self.running = False
                return
            if not self.timeline.is_playing():
                self.last_capture_time = None
                self.lidar_slices.clear()
                if self.has_seen_playback:
                    print("[DR.Nav Recorder] Timeline stopped; closing recorder.")
                    await self.stop()
                    if getattr(builtins, RECORDER_KEY, None) is self:
                        setattr(builtins, RECORDER_KEY, None)
                    return
                continue

            self.has_seen_playback = True
            now = float(self.timeline.get_current_time())
            self.accumulate_lidar(self.capture_lidar())
            if (
                self.last_capture_time is not None
                and now - self.last_capture_time < interval - 1e-9
            ):
                continue

            lidar = self.completed_lidar_scan()
            if REQUIRE_LIDAR_DATA and lidar is None:
                self.incomplete_lidar_scans += 1
                count = self.incomplete_lidar_scans
                if count <= 3 or count % 20 == 0:
                    print(
                        "[DR.Nav Recorder] Waiting for complete 360-degree "
                        "LiDAR scan."
                    )
                continue

            auto_update = self.timeline.is_auto_updating()
            self.render_product.hydra_texture.set_updates_enabled(True)
            self.capture_in_progress = True
            try:
                await rep.orchestrator.step_async(
                    delta_time=None,
                    pause_timeline=False,
                    rt_subframes=1,
                )
                if not self.running or not self.timeline.is_playing():
                    self.skipped_frames += 1
                    continue
                capture_time = float(self.timeline.get_current_time())
                if (
                    self.last_written_time is not None
                    and capture_time <= self.last_written_time
                ):
                    self.skipped_frames += 1
                    print(
                        "[DR.Nav Recorder] Non-increasing simulation time; "
                        f"discarding frame at {capture_time:.6f}."
                    )
                    continue
                rgb = self.rgb_annotator.get_data(do_array_copy=True)
                if rgb is None or not np.asarray(rgb).size:
                    self.skipped_frames += 1
                    self.empty_rgb_frames += 1
                    count = self.empty_rgb_frames
                    if count <= 3 or count % 20 == 0:
                        print(
                            "[DR.Nav Recorder] Empty RGB frame; skipping "
                            f"capture (count={count})."
                        )
                    continue
                self.write_frame(rgb, capture_time, lidar)
                self.last_capture_time = capture_time
                self.last_written_time = capture_time
            finally:
                self.timeline.set_auto_update(auto_update)
                self.timeline.commit_silently()
                self.render_product.hydra_texture.set_updates_enabled(False)
                self.capture_in_progress = False


async def main():
    previous = getattr(builtins, RECORDER_KEY, None)
    if previous is not None:
        try:
            await previous.stop()
        except Exception as exc:
            print("[DR.Nav Recorder] Previous cleanup warning:", repr(exc))

    recorder = DRNavRecorder()
    setattr(builtins, RECORDER_KEY, recorder)
    try:
        await recorder.start()
    except BaseException:
        try:
            await recorder.stop()
        finally:
            if getattr(builtins, RECORDER_KEY, None) is recorder:
                setattr(builtins, RECORDER_KEY, None)
        raise


if globals().get("_DRNAV_AUTOSTART", True):
    asyncio.ensure_future(main())
