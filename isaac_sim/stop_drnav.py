#!/usr/bin/env python3
"""Finalize and remove the complete DR.Nav Isaac Sim runtime."""

import asyncio
import builtins
import gc

import omni.kit.app
import omni.timeline
import omni.usd


STARTUP_TASK_KEY = "_DRNAV_STARTUP_TASK"
RUNTIME_CONFIG_KEY = "_DRNAV_RUNTIME_CONFIG"
STOP_TASK_KEY = "_DRNAV_STOP_TASK"
CONTROLLER_TASK_KEYS = [
    "_DRNAV_WAYPOINT_FOLLOWER_TASK",
    "_DRNAV_JACKAL_TELEOP_TASK",
]
RUNTIME_OBJECT_KEYS = [
    "_DRNAV_AUTO_EPISODE_PIPELINE",
    "_DRNAV_REPLICATOR_RECORDER",
    "_DRNAV_WAYPOINT_FOLLOWER",
    "_DRNAV_JACKAL_TELEOP",
    "_DRNAV_ISAACSIM6_V6_BRIDGE",
]


def orphan_cleanup_method(instance, class_name):
    """Identify live DR.Nav objects left behind by an interrupted run."""
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


async def cleanup_orphan_runtime_objects():
    """Stop live DR.Nav objects no longer reachable through builtins."""
    reachable_ids = {
        id(instance)
        for key in RUNTIME_OBJECT_KEYS
        if (instance := getattr(builtins, key, None)) is not None
    }
    candidates = []

    for instance in gc.get_objects():
        if id(instance) in reachable_ids:
            continue
        try:
            class_name = type(instance).__name__
            method_name = orphan_cleanup_method(instance, class_name)
            if method_name is not None:
                candidates.append((instance, method_name))
        except (AttributeError, ReferenceError, RuntimeError):
            continue

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
                "[DR.Nav Stop] Orphan cleanup timed out "
                f"({type(instance).__name__}); continuing."
            )
        except (ReferenceError, RuntimeError):
            pass
        except Exception as exc:
            print(
                "[DR.Nav Stop] Orphan cleanup warning "
                f"({type(instance).__name__}):",
                repr(exc),
            )

    if cleaned:
        summary = ", ".join(
            f"{name}={count}"
            for name, count in sorted(cleaned_types.items())
        )
        print(
            f"[DR.Nav Stop] Cleaned {cleaned} orphan runtime object(s): "
            f"{summary}."
        )


async def call_method(key, method_name):
    instance = getattr(builtins, key, None)
    if instance is None:
        return

    try:
        result = getattr(instance, method_name)()
        if asyncio.iscoroutine(result):
            await result
        print(f"[DR.Nav Stop] Cleared {key}")
    except Exception as exc:
        print(f"[DR.Nav Stop] Cleanup warning ({key}):", repr(exc))
    finally:
        setattr(builtins, key, None)


async def stop_replicator_runtime():
    """Drain global Replicator state even when its recorder handle was lost."""
    try:
        import omni.replicator.core as rep

        rep.orchestrator.set_capture_on_play(False)
        if not rep.orchestrator.get_is_stopped():
            await asyncio.wait_for(
                rep.orchestrator.stop_async(),
                timeout=10.0,
            )
        status = str(rep.orchestrator.get_status())
        print(f"[DR.Nav Stop] Replicator status: {status}")
        return status
    except asyncio.TimeoutError:
        print("[DR.Nav Stop] Replicator cleanup timed out; continuing.")
        return "timeout"
    except Exception as exc:
        print("[DR.Nav Stop] Replicator cleanup warning:", repr(exc))
        return "unknown"


def remove_appearance_layer():
    key = "_DRNAV_APPEARANCE_LAYER"
    layer = getattr(builtins, key, None)
    if layer is None:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is not None:
        session = stage.GetSessionLayer()
        session.subLayerPaths = [
            path for path in session.subLayerPaths if path != layer.identifier
        ]

    setattr(builtins, key, None)
    setattr(builtins, "_DRNAV_APPEARANCE_STATE", None)
    print("[DR.Nav Stop] Removed temporary appearance layer.")


async def stop_runtime():
    startup = getattr(builtins, STARTUP_TASK_KEY, None)
    current = asyncio.current_task()
    if startup is not None and startup is not current and not startup.done():
        startup.cancel()
        try:
            await startup
        except asyncio.CancelledError:
            pass
    setattr(builtins, STARTUP_TASK_KEY, None)

    for key in CONTROLLER_TASK_KEYS:
        task = getattr(builtins, key, None)
        if task is not None and task is not current and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print(f"[DR.Nav Stop] Task cleanup warning ({key}):", repr(exc))
        setattr(builtins, key, None)

    await cleanup_orphan_runtime_objects()

    # Stop recording before control and sensor teardown so pending output can
    # be finalized while the required Isaac resources still exist.
    await call_method("_DRNAV_AUTO_EPISODE_PIPELINE", "stop")
    await call_method("_DRNAV_REPLICATOR_RECORDER", "stop")
    remove_appearance_layer()

    await call_method("_DRNAV_WAYPOINT_FOLLOWER", "shutdown")
    await call_method("_DRNAV_JACKAL_TELEOP", "shutdown")
    await call_method("_DRNAV_ISAACSIM6_V6_BRIDGE", "shutdown")
    replicator_status = await stop_replicator_runtime()

    timeline = omni.timeline.get_timeline_interface()
    before_state = (
        "playing"
        if timeline.is_playing()
        else "stopped"
        if timeline.is_stopped()
        else "paused"
    )
    print(
        "[DR.Nav Stop] Timeline before UI resync: "
        f"state={before_state}, "
        f"auto_update={timeline.is_auto_updating()}, "
        f"time={timeline.get_current_time():.6f}"
    )
    timeline.set_auto_update(True)
    timeline.stop()
    timeline.commit()

    # Delayed render work from an interrupted recorder can briefly put the
    # timeline back into Play. Require a stable stopped window before handing
    # control back to the GUI.
    stable_updates = 0
    for _ in range(120):
        await omni.kit.app.get_app().next_update_async()

        if not timeline.is_stopped():
            stable_updates = 0
            timeline.stop()
            timeline.commit()
        else:
            stable_updates += 1
            if stable_updates >= 12:
                break

    if timeline.is_playing():
        final_state = "playing"
    elif timeline.is_stopped():
        final_state = "stopped"
    else:
        final_state = "paused"
    auto_update = timeline.is_auto_updating()

    setattr(builtins, RUNTIME_CONFIG_KEY, None)
    print("\n[DR.Nav Stop] COMPLETE")
    print(
        "[DR.Nav Stop] Recorder finalized and all runtime services removed "
        f"(timeline={final_state}, auto_update={auto_update}, "
        f"replicator={replicator_status})."
    )


previous_task = getattr(builtins, STOP_TASK_KEY, None)
if previous_task is not None and not previous_task.done():
    previous_task.cancel()

stop_task = asyncio.ensure_future(stop_runtime())
setattr(builtins, STOP_TASK_KEY, stop_task)
