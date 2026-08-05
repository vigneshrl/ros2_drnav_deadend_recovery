#!/usr/bin/env python3
"""Prepare, record, and safely finalize repeated DR.Nav Isaac Sim episodes."""

import asyncio
import builtins
import random
import time
import traceback
from pathlib import Path

import carb
import carb.eventdispatcher
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


CONTROLLER_KEY = "_DRNAV_AUTO_EPISODE_PIPELINE"
RECORDER_KEY = "_DRNAV_REPLICATOR_RECORDER"
FOLLOWER_KEY = "_DRNAV_WAYPOINT_FOLLOWER"
APPEARANCE_LAYER_KEY = "_DRNAV_APPEARANCE_LAYER"
APPEARANCE_STATE_KEY = "_DRNAV_APPEARANCE_STATE"
CONFIG = getattr(builtins, "_DRNAV_RUNTIME_CONFIG", {})
RECORDER_CONFIG = CONFIG.get("recorder", {})
WARMUP_UPDATES = int(RECORDER_CONFIG.get("warmup_updates", 5))
STOP_DEBUG = bool(RECORDER_CONFIG.get("debug_stop_state", False))

CAMERA = (
    "/World/ground/flat_plane/jackal/base_link/"
    "bumblebee_stereo_camera_frame/bumblebee_stereo_left_frame/"
    "bumblebee_stereo_left_camera"
)
LIGHT_PATHS = [
    "/World/ground/DistantLight",
    "/World/ground/flat_plane/DistantLight",
    "/World/ground/flat_plane/Environment/defaultLight",
    "/World/ground/CylinderLight",
]
REGION_ROOTS = [
    "/World/ground/flat_plane/dead_end",
    "/World/ground/flat_plane/dead_end_01",
    "/World/ground/flat_plane/cross",
]
MATERIAL_NAMES = [
    "Large_Granite_Paving",
    "Facade_Brick_Red_Clinker",
    "Carpet_Woven_Cherry_Red",
]


def clamp_color(value):
    return max(0.0, min(1.0, value))


def remove_appearance_layer():
    layer = getattr(builtins, APPEARANCE_LAYER_KEY, None)
    if layer is None:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is not None:
        session = stage.GetSessionLayer()
        session.subLayerPaths = [
            path
            for path in session.subLayerPaths
            if path != layer.identifier
        ]
    setattr(builtins, APPEARANCE_LAYER_KEY, None)
    setattr(builtins, APPEARANCE_STATE_KEY, None)


def randomize_lights(stage, rng):
    multiplier = rng.uniform(0.70, 1.30)
    amount = rng.uniform(-0.12, 0.12)
    tint = (
        (
            1.0,
            clamp_color(1.0 - 0.35 * amount),
            clamp_color(1.0 - amount),
        )
        if amount >= 0.0
        else (
            clamp_color(1.0 + amount),
            clamp_color(1.0 + 0.30 * amount),
            1.0,
        )
    )
    changes = []
    for path in LIGHT_PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsActive():
            changes.append({"path": path, "status": "missing_or_inactive"})
            continue
        intensity = prim.GetAttribute("inputs:intensity")
        if not intensity.IsValid() or intensity.Get() is None:
            changes.append({"path": path, "status": "no_intensity"})
            continue

        original_intensity = float(intensity.Get())
        new_intensity = original_intensity * multiplier
        intensity.Set(new_intensity)
        color = prim.GetAttribute("inputs:color")
        original_color = new_color = None
        if color.IsValid() and color.Get() is not None:
            original_color = [float(component) for component in color.Get()]
            new_color = [
                clamp_color(original_color[index] * tint[index])
                for index in range(3)
            ]
            color.Set(Gf.Vec3f(*new_color))
        changes.append(
            {
                "path": path,
                "status": "changed",
                "original_intensity": original_intensity,
                "intensity": new_intensity,
                "original_color": original_color,
                "color": new_color,
            }
        )
    return {
        "global_multiplier": multiplier,
        "global_tint_factors": list(tint),
        "lights": changes,
    }


def material_leaf(material):
    if not material or material.GetPath() == Sdf.Path.emptyPath:
        return None
    return material.GetPath().name


def randomize_materials(stage, rng):
    shuffled = list(MATERIAL_NAMES)
    rng.shuffle(shuffled)
    mapping = dict(zip(MATERIAL_NAMES, shuffled))
    changed, skipped = [], []
    for root_path in REGION_ROOTS:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            skipped.append({"path": root_path, "reason": "missing_region"})
            continue
        materials = {
            name: UsdShade.Material.Get(stage, f"{root_path}/Looks/{name}")
            for name in MATERIAL_NAMES
        }
        for prim in Usd.PrimRange(root):
            if prim.GetTypeName() != "Mesh":
                continue
            imageable = UsdGeom.Imageable(prim)
            if (
                imageable
                and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
            ):
                continue
            bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            original = bound[0] if bound else None
            original_name = material_leaf(original)
            if original_name not in mapping:
                skipped.append(
                    {
                        "path": str(prim.GetPath()),
                        "reason": "unrecognized_or_unbound_material",
                        "bound_material": (
                            str(original.GetPath()) if original else None
                        ),
                    }
                )
                continue
            target_name = mapping[original_name]
            target = materials.get(target_name)
            if not target:
                skipped.append(
                    {
                        "path": str(prim.GetPath()),
                        "reason": "target_material_missing",
                        "target_name": target_name,
                    }
                )
                continue
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(target)
            changed.append(
                {
                    "mesh": str(prim.GetPath()),
                    "original_material": str(original.GetPath()),
                    "target_material": str(target.GetPath()),
                    "original_class": original_name,
                    "target_class": target_name,
                }
            )
    return {
        "palette_mapping": mapping,
        "changed_meshes": changed,
        "skipped_meshes": skipped,
    }


def randomize_camera(stage, rng):
    prim = stage.GetPrimAtPath(CAMERA)
    if not prim.IsValid():
        return {"status": "camera_missing", "path": CAMERA}
    exposure = prim.GetAttribute("exposure")
    if not exposure.IsValid():
        return {"status": "exposure_not_supported", "path": CAMERA}
    value = rng.uniform(-0.5, 0.5)
    exposure.Set(value)
    return {"status": "changed", "path": CAMERA, "exposure": value}


def apply_randomization():
    """Apply one curated appearance and retain its manifest metadata."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is currently open.")
    remove_appearance_layer()
    seed = time.time_ns() & 0xFFFFFFFF
    rng = random.Random(seed)
    layer = Sdf.Layer.CreateAnonymous("DRNav_CuratedAppearance.usda")
    session = stage.GetSessionLayer()
    session.subLayerPaths = [*session.subLayerPaths, layer.identifier]
    setattr(builtins, APPEARANCE_LAYER_KEY, layer)

    with Usd.EditContext(stage, stage.GetEditTargetForLocalLayer(layer)):
        lighting = randomize_lights(stage, rng)
        materials = randomize_materials(stage, rng)
        camera = randomize_camera(stage, rng)

    root = stage.GetRootLayer()
    state = {
        "mode": "curated_appearance_only",
        "seed": seed,
        "stage": root.realPath or root.identifier,
        "layer": layer.identifier,
        "lighting": lighting,
        "materials": materials,
        "camera": camera,
        "physics_changed": False,
        "geometry_changed": False,
        "sensor_pose_changed": False,
        "robot_materials_changed": False,
        "zone_materials_changed": False,
    }
    setattr(builtins, APPEARANCE_STATE_KEY, state)
    print("\n[DR.Nav Appearance] Curated randomization applied")
    print("[DR.Nav Appearance] Seed:", seed)
    print(
        "[DR.Nav Appearance] Brightness multiplier:",
        round(lighting["global_multiplier"], 4),
    )
    print("[DR.Nav Appearance] Material palette:", materials["palette_mapping"])
    print(
        "[DR.Nav Appearance] Mesh bindings changed:",
        len(materials["changed_meshes"]),
    )
    print("[DR.Nav Appearance] Camera:", camera["status"])
    print("[DR.Nav Appearance] Full metadata will be saved in manifest.json.")


def find_runtime_dir():
    """Locate sibling runtime modules even when Isaac executes this as a string."""
    config_path = CONFIG.get("_config_path")
    if config_path:
        candidate = Path(config_path).expanduser().resolve().parent.parent
        candidate /= "isaac_sim"
        if candidate.is_dir():
            return candidate

    source_path = Path(str(globals().get("__file__", ""))).expanduser()
    if source_path.is_file():
        return source_path.resolve().parent

    for candidate in (
        Path.home() / "DRNav" / "ros2_drnav_deadend_recovery" / "isaac_sim",
        Path.home() / "ros2_drnav_deadend_recovery" / "isaac_sim",
    ):
        if candidate.is_dir():
            return candidate.resolve()
    raise RuntimeError(
        "Could not locate DR.Nav Isaac modules. Run start_drnav.py so the "
        "runtime configuration supplies the repository path."
    )


RUNTIME_DIR = find_runtime_dir()


def execute_module(filename, name, **injected):
    path = RUNTIME_DIR / filename
    if not path.is_file():
        raise RuntimeError(f"Missing DR.Nav runtime module: {path}")
    namespace = {
        "__name__": name,
        "__file__": str(path),
        "__builtins__": __builtins__,
        **injected,
    }
    source = path.read_text(encoding="utf-8")
    exec(compile(source, str(path), "exec"), namespace, namespace)
    return namespace


class AutoEpisodePipeline:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.running = self.preparing = self.stop_requested = False
        self.stop_event_seen = False
        self.monitor_task = self.stop_cleanup_task = self.timeline_sub = None
        self.timeline_auto_update = self.timeline.is_auto_updating()
        self.episode_number = 0

    @staticmethod
    def task_state(task):
        if task is None:
            return "none"
        if task.cancelled():
            return "cancelled"
        return "done" if task.done() else "pending"

    def timeline_state(self):
        if self.timeline.is_playing():
            return "playing"
        return "stopped" if self.timeline.is_stopped() else "paused"

    def debug_snapshot(self, label):
        if not STOP_DEBUG:
            return
        recorder = getattr(builtins, RECORDER_KEY, None)
        follower = getattr(builtins, FOLLOWER_KEY, None)
        settings = carb.settings.get_settings()
        try:
            replicator = rep.orchestrator.get_status()
        except Exception as exc:
            replicator = f"unavailable:{exc!r}"
        print(
            f"[DR.Nav Stop Debug] {label}: timeline={self.timeline_state()} "
            f"auto_update={self.timeline.is_auto_updating()} "
            f"time={self.timeline.get_current_time():.6f} "
            f"replicator={replicator} "
            "capture_on_play="
            f"{settings.get('/omni/replicator/captureOnPlay')}"
        )
        print(
            "[DR.Nav Stop Debug] "
            f"running={self.running} preparing={self.preparing} "
            f"stop_requested={self.stop_requested} "
            f"monitor={self.task_state(self.monitor_task)} "
            f"cleanup={self.task_state(self.stop_cleanup_task)} "
            f"recorder={id(recorder) if recorder else 'none'} "
            f"recording={getattr(recorder, 'running', None)} "
            f"capture={getattr(recorder, 'capture_in_progress', None)} "
            f"follower={id(follower) if follower else 'none'}"
        )

    async def start(self):
        self.timeline.set_auto_update(True)
        self.timeline.commit_silently()
        self.timeline_auto_update = True
        if not self.timeline.is_stopped():
            self.timeline.stop()
            for _ in range(3):
                await omni.kit.app.get_app().next_update_async()

        await self.stop_recorder()
        self.timeline_sub = (
            self.timeline.get_timeline_event_stream().create_subscription_to_pop(
                self.on_timeline_event,
                name="DR.Nav recorder timeline state",
            )
        )
        self.running = True
        self.monitor_task = asyncio.ensure_future(self.monitor())
        print("\n[DR.Nav Auto] Curated episode pipeline started.")
        print("[DR.Nav Auto] Press Play to randomize and record.")
        print("[DR.Nav Auto] Press Stop to finalize the episode.")
        print("[DR.Nav Auto] Reset the robot while stopped.")

    async def stop(self):
        self.running = False
        current = asyncio.current_task()
        cleanup = self.stop_cleanup_task
        if cleanup is not None and cleanup is not current and not cleanup.done():
            await cleanup
        self.stop_cleanup_task = None

        monitor = self.monitor_task
        if monitor is not None and monitor is not current and not monitor.done():
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass
        self.monitor_task = self.timeline_sub = None
        await self.stop_recorder()
        self.remove_appearance()
        print("[DR.Nav Auto] Curated episode pipeline stopped.")

    def on_timeline_event(self, event):
        if getattr(builtins, CONTROLLER_KEY, None) is not self:
            return
        if STOP_DEBUG and event.type in (0, 1, 2):
            name = {0: "PLAY", 1: "PAUSE", 2: "STOP"}[event.type]
            print(
                f"[DR.Nav Stop Debug] event={name} "
                f"timeline={self.timeline_state()} "
                f"auto_update={self.timeline.is_auto_updating()}"
            )
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self.stop_event_seen = False
        elif event.type == int(omni.timeline.TimelineEventType.STOP):
            self.stop_event_seen = True
            self.request_stop_cleanup("timeline event")

    def request_stop_cleanup(self, source):
        if getattr(builtins, CONTROLLER_KEY, None) is not self:
            return
        self.stop_requested = True
        cleanup_running = (
            self.stop_cleanup_task is not None
            and not self.stop_cleanup_task.done()
        )
        if self.preparing or cleanup_running:
            if STOP_DEBUG:
                print(
                    "[DR.Nav Stop Debug] cleanup request coalesced: "
                    f"source={source} preparing={self.preparing}"
                )
            return
        self.debug_snapshot(f"detected:{source}")
        self.stop_cleanup_task = asyncio.ensure_future(self.finish_stop_cleanup())
        print(
            f"[DR.Nav Auto] Stop detected by {source}; finalizing recorder "
            "and camera capture."
        )

    def notify_missed_stop(self):
        if self.stop_event_seen:
            return
        payload = dict(self.timeline.get_event_key())
        payload.setdefault("name", "")
        observers = carb.eventdispatcher.get_eventdispatcher().dispatch_event(
            omni.timeline.GLOBAL_EVENT_STOP,
            payload=payload,
        )
        self.stop_event_seen = True
        follower = getattr(builtins, FOLLOWER_KEY, None)
        reset = getattr(follower, "reset_route", None)
        if callable(reset):
            reset()
        print(
            "[DR.Nav Auto] Re-emitted missed STOP notification "
            f"to {observers} observer(s) without starting physics."
        )

    async def finish_stop_cleanup(self):
        if getattr(builtins, CONTROLLER_KEY, None) is not self:
            self.stop_requested = False
            return
        final_state = replicator_status = "unknown"
        auto_update = self.timeline.is_auto_updating()
        try:
            self.debug_snapshot("cleanup:begin")
            await self.stop_recorder()
            self.timeline.set_auto_update(self.timeline_auto_update)
            self.timeline.commit_silently()

            # Never synthesize Play here: doing so recreates physics while old
            # tensor views are still being released and can crash Isaac Sim.
            self.timeline.stop()
            self.timeline.commit()
            self.notify_missed_stop()

            stable_updates = 0
            for _ in range(120):
                await omni.kit.app.get_app().next_update_async()
                if getattr(builtins, CONTROLLER_KEY, None) is not self:
                    return
                if not self.timeline.is_stopped():
                    stable_updates = 0
                    self.timeline.stop()
                    self.timeline.commit()
                    continue
                stable_updates += 1
                if stable_updates >= 12:
                    break

            final_state = self.timeline_state()
            auto_update = self.timeline.is_auto_updating()
            replicator_status = str(rep.orchestrator.get_status())
            self.debug_snapshot("cleanup:final")
        except Exception as exc:
            print("[DR.Nav Auto] Stop cleanup warning:", repr(exc))
            print(traceback.format_exc())
            self.timeline.set_auto_update(True)
            self.timeline.stop()
            self.timeline.commit()
            final_state = self.timeline_state()
            auto_update = self.timeline.is_auto_updating()
        finally:
            self.stop_requested = False
            if getattr(builtins, CONTROLLER_KEY, None) is self:
                print(
                    "[DR.Nav Auto] Stop cleanup complete "
                    f"(timeline={final_state}, auto_update={auto_update}, "
                    f"replicator={replicator_status}); press Play for the "
                    "next episode."
                )

    async def stop_recorder(self):
        recorder = getattr(builtins, RECORDER_KEY, None)
        if recorder is None:
            return
        try:
            await recorder.stop()
        except Exception as exc:
            print("[DR.Nav Auto] Recorder cleanup warning:", repr(exc))
        if getattr(builtins, RECORDER_KEY, None) is recorder:
            setattr(builtins, RECORDER_KEY, None)

    @staticmethod
    def remove_appearance():
        remove_appearance_layer()

    async def wait_for_recorder(self):
        for _ in range(300):
            recorder = getattr(builtins, RECORDER_KEY, None)
            if recorder is not None and getattr(recorder, "running", False):
                return recorder
            await omni.kit.app.get_app().next_update_async()
        raise RuntimeError("Recorder did not become ready.")

    async def cancel_if_stopped(self):
        if not self.stop_requested:
            return False
        if self.timeline.is_playing():
            self.timeline.stop()
            for _ in range(2):
                await omni.kit.app.get_app().next_update_async()
        await self.stop_recorder()
        self.remove_appearance()
        self.stop_requested = False
        print("[DR.Nav Auto] Episode preparation cancelled by Stop.")
        return True

    @staticmethod
    def reset_follower():
        follower = getattr(builtins, FOLLOWER_KEY, None)
        if follower is None:
            print(
                "[DR.Nav Auto] No waypoint follower active; episode will "
                "record without autonomous motion."
            )
            return
        reset = getattr(follower, "reset_route", None)
        if not callable(reset):
            raise RuntimeError(
                "The waypoint follower is outdated. Rerun "
                "setup_waypoint_follower.py before pressing Play."
            )
        reset()
        print("[DR.Nav Auto] Waypoint follower reset for this episode.")

    async def prepare_episode(self):
        if (
            self.preparing
            or not self.running
            or getattr(builtins, CONTROLLER_KEY, None) is not self
        ):
            return
        self.preparing = True
        self.stop_requested = False
        try:
            self.timeline.pause()
            for _ in range(2):
                await omni.kit.app.get_app().next_update_async()
            await self.stop_recorder()
            apply_randomization()
            dataset = execute_module(
                "dataset_recorder.py",
                "__drnav_dataset_module__",
                _DRNAV_AUTOSTART=False,
            )
            for _ in range(WARMUP_UPDATES):
                await omni.kit.app.get_app().next_update_async()
            if await self.cancel_if_stopped():
                return

            asyncio.ensure_future(dataset["main"]())
            recorder = await self.wait_for_recorder()
            if await self.cancel_if_stopped():
                return
            self.episode_number += 1
            appearance = getattr(builtins, "_DRNAV_APPEARANCE_STATE", {})
            print(f"\n[DR.Nav Auto] Episode {self.episode_number} ready")
            print("[DR.Nav Auto] Seed:", appearance.get("seed", "unknown"))
            print("[DR.Nav Auto] Output:", recorder.output_dir)
            if await self.cancel_if_stopped():
                return

            self.reset_follower()
            self.timeline.play()
            for _ in range(2):
                await omni.kit.app.get_app().next_update_async()
        except Exception as exc:
            print("[DR.Nav Auto] Episode preparation failed:", repr(exc))
            print(traceback.format_exc())
            await self.stop_recorder()
            self.timeline.stop()
            self.timeline.commit()
            self.stop_requested = False
        finally:
            self.preparing = False

    async def monitor(self):
        previous_playing = self.timeline.is_playing()
        while self.running:
            await omni.kit.app.get_app().next_update_async()
            if getattr(builtins, CONTROLLER_KEY, None) is not self:
                self.running = False
                return
            playing = self.timeline.is_playing()
            if not playing and previous_playing and not self.preparing:
                self.request_stop_cleanup("timeline state")
            if playing and not previous_playing and not self.preparing:
                if self.stop_requested:
                    previous_playing = playing
                    continue
                await self.prepare_episode()
                playing = self.timeline.is_playing()
            previous_playing = playing


async def main():
    previous = getattr(builtins, CONTROLLER_KEY, None)
    if previous is not None:
        try:
            await previous.stop()
        except Exception as exc:
            print("[DR.Nav Auto] Previous cleanup warning:", repr(exc))

    controller = AutoEpisodePipeline()
    setattr(builtins, CONTROLLER_KEY, controller)
    try:
        await controller.start()
    except BaseException:
        try:
            await controller.stop()
        finally:
            if getattr(builtins, CONTROLLER_KEY, None) is controller:
                setattr(builtins, CONTROLLER_KEY, None)
        raise


asyncio.ensure_future(main())
