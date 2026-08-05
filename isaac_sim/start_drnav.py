#!/usr/bin/env python3
"""Start the configured DR.Nav Isaac Sim runtime.

Run this file once from Isaac Sim's Script Editor. It loads the YAML, replaces
any residual runtime, prepares sensors/recording/control, and leaves the
timeline stopped. Press Play once after the READY message.
"""

import asyncio
import builtins
import os
import traceback
from pathlib import Path

import omni.kit.app
import omni.timeline
import omni.usd


CONFIG_RELATIVE_PATH = Path("config") / "isaac_sim_maze3.yaml"
RUNTIME_CONFIG_KEY = "_DRNAV_RUNTIME_CONFIG"
STARTUP_TASK_KEY = "_DRNAV_STARTUP_TASK"


def find_project_root():
    """Find this checkout without depending on a particular username."""
    candidates = []
    override = getattr(builtins, "_DRNAV_PROJECT_ROOT", None)
    if override is None:
        override = os.environ.get("DRNAV_PROJECT_ROOT")
    if override:
        candidates.append(Path(override).expanduser())

    working_directory = Path.cwd()
    candidates.extend([working_directory, *working_directory.parents])
    candidates.extend(
        [
            Path.home() / "DRNav" / "ros2_drnav_deadend_recovery",
            Path.home() / "ros2_drnav_deadend_recovery",
        ]
    )

    checked = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.add(candidate)
        if (candidate / CONFIG_RELATIVE_PATH).is_file():
            return candidate

    raise RuntimeError(
        "Could not find the DR.Nav repository. Run from the checkout, place "
        "it at ~/DRNav/ros2_drnav_deadend_recovery, set DRNAV_PROJECT_ROOT, "
        "or set builtins._DRNAV_PROJECT_ROOT before running this script."
    )


def load_config(discovered_root):
    import yaml

    config_path = discovered_root / CONFIG_RELATIVE_PATH

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    mode = str(config.get("control_mode", "none")).lower()
    if mode not in {"waypoint", "teleop", "none"}:
        raise RuntimeError(
            "control_mode must be one of: waypoint, teleop, none; "
            f"found {mode!r}"
    )
    config["control_mode"] = mode
    config["_config_path"] = str(config_path)
    return config, config_path


def resolve_configured_root(config, config_path):
    configured = Path(str(config.get("project_root", ".."))).expanduser()
    if not configured.is_absolute():
        configured = config_path.parent / configured
    return configured.resolve()


def validate_stage(config):
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")

    root = stage.GetRootLayer()
    source = root.realPath or root.identifier
    current = Path(source).stem
    expected = str(config.get("expected_maze", "")).strip()

    if (
        expected
        and current != expected
        and not bool(config.get("allow_other_mazes", False))
    ):
        raise RuntimeError(
            f"Configuration expects {expected}, but the open stage is "
            f"{current}. Set allow_other_mazes only when intentional."
        )
    return current


def execute_script(path, name):
    source = path.read_text(encoding="utf-8")
    namespace = {
        "__name__": name,
        "__file__": str(path),
        "__builtins__": __builtins__,
    }
    exec(compile(source, str(path), "exec"), namespace, namespace)


async def wait_for(name, predicate, maximum_updates=900):
    for _ in range(maximum_updates):
        value = predicate()
        if value:
            return value
        await omni.kit.app.get_app().next_update_async()
    raise RuntimeError(f"Timed out waiting for {name}.")


async def cleanup_failed_startup():
    """Remove every runtime object still reachable after startup failure."""
    current = asyncio.current_task()

    for key in [
        "_DRNAV_WAYPOINT_FOLLOWER_TASK",
        "_DRNAV_JACKAL_TELEOP_TASK",
    ]:
        task = getattr(builtins, key, None)
        if task is not None and task is not current and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        setattr(builtins, key, None)

    for key, method_name in [
        ("_DRNAV_AUTO_EPISODE_PIPELINE", "stop"),
        ("_DRNAV_REPLICATOR_RECORDER", "stop"),
        ("_DRNAV_WAYPOINT_FOLLOWER", "shutdown"),
        ("_DRNAV_JACKAL_TELEOP", "shutdown"),
        ("_DRNAV_ISAACSIM6_V6_BRIDGE", "shutdown"),
    ]:
        instance = getattr(builtins, key, None)
        if instance is None:
            continue
        try:
            result = getattr(instance, method_name)()
            if asyncio.iscoroutine(result):
                await result
        except Exception as cleanup_error:
            print(
                f"[DR.Nav Start] Cleanup warning ({key}):",
                repr(cleanup_error),
            )
        finally:
            setattr(builtins, key, None)

    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    timeline.set_auto_update(True)
    timeline.commit()


async def start_runtime():
    try:
        discovered_root = find_project_root()
        config, config_path = load_config(discovered_root)
        maze = validate_stage(config)
        project_root = resolve_configured_root(config, config_path)
        isaac_dir = project_root / "isaac_sim"

        required = [
            isaac_dir / "setup_sensors.py",
            isaac_dir / "setup_teleop.py",
            isaac_dir / "setup_waypoint_follower.py",
            isaac_dir / "episode_recorder.py",
            isaac_dir / "dataset_recorder.py",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("Missing runtime scripts:\n  " + "\n  ".join(missing))

        setattr(builtins, RUNTIME_CONFIG_KEY, config)
        print(f"\n[DR.Nav Start] Configuration: {config_path}")
        print(f"[DR.Nav Start] Stage: {maze}")

        execute_script(
            isaac_dir / "setup_sensors.py",
            "__drnav_start_sensors__",
        )
        bridge = await wait_for(
            "sensor bridge",
            lambda: getattr(builtins, "_DRNAV_ISAACSIM6_V6_BRIDGE", None),
        )
        await wait_for(
            "sensor bridge readiness",
            lambda: bridge.node is not None and bridge.robot is not None,
        )

        recorder_enabled = bool(config.get("recorder", {}).get("enabled", True))
        if recorder_enabled:
            execute_script(
                isaac_dir / "episode_recorder.py",
                "__drnav_start_recorder__",
            )
            await wait_for(
                "episode recorder pipeline",
                lambda: getattr(
                    builtins,
                    "_DRNAV_AUTO_EPISODE_PIPELINE",
                    None,
                )
                and getattr(
                    builtins._DRNAV_AUTO_EPISODE_PIPELINE,
                    "running",
                    False,
                ),
            )
        else:
            omni.timeline.get_timeline_interface().stop()
            for _ in range(3):
                await omni.kit.app.get_app().next_update_async()

        mode = config["control_mode"]
        controller_key = None
        if mode == "waypoint":
            execute_script(
                isaac_dir / "setup_waypoint_follower.py",
                "__drnav_start_waypoints__",
            )
            controller_key = "_DRNAV_WAYPOINT_FOLLOWER"
        elif mode == "teleop":
            execute_script(
                isaac_dir / "setup_teleop.py",
                "__drnav_start_teleop__",
            )
            controller_key = "_DRNAV_JACKAL_TELEOP"

        if controller_key is not None:
            controller = await wait_for(
                mode + " controller",
                lambda: getattr(builtins, controller_key, None),
            )
            await wait_for(
                mode + " controller readiness",
                lambda: controller.node is not None,
            )

        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_stopped():
            timeline.stop()
            timeline.commit()
            for _ in range(3):
                await omni.kit.app.get_app().next_update_async()

        timeline.set_auto_update(True)
        timeline.commit_silently()

        print("\n[DR.Nav Start] READY")
        print("[DR.Nav Start] Control mode:", mode)
        print("[DR.Nav Start] Recorder enabled:", recorder_enabled)
        print("[DR.Nav Start] Press Play once to begin the episode.")
        print("[DR.Nav Start] Run stop_drnav.py when completely finished.")
    except Exception as exc:
        print("[DR.Nav Start] Startup failed:", repr(exc))
        print(traceback.format_exc())
        await cleanup_failed_startup()
        print("[DR.Nav Start] Failed startup cleaned up; timeline stopped.")


previous_task = getattr(builtins, STARTUP_TASK_KEY, None)
if previous_task is not None and not previous_task.done():
    previous_task.cancel()

startup_task = asyncio.ensure_future(start_runtime())
setattr(builtins, STARTUP_TASK_KEY, startup_task)
