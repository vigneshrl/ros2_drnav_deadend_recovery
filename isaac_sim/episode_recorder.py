#!/usr/bin/env python3
"""
Automatic DR.Nav episode pipeline using curated Maze2 appearance paths, including CylinderLight and the Isaac Sim 6.0 Sdf.Path fix.

Run the ROS bridge once, then run this controller once while stopped.

Each Play:
- pauses briefly,
- applies a new curated appearance randomization,
- waits for render/material updates,
- starts a new recorder episode,
- resumes playback.

Each Stop:
- finalizes that recorder episode,
- leaves this controller waiting for the next Play.
"""

import asyncio
import builtins
import traceback

import carb
import carb.eventdispatcher
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd


CONTROLLER_KEY = "_DRNAV_AUTO_EPISODE_PIPELINE"
RECORDER_KEY = "_DRNAV_REPLICATOR_RECORDER"
WAYPOINT_FOLLOWER_KEY = "_DRNAV_WAYPOINT_FOLLOWER"
APPEARANCE_LAYER_KEY = "_DRNAV_APPEARANCE_LAYER"
RUNTIME_CONFIG = getattr(builtins, "_DRNAV_RUNTIME_CONFIG", {})
RECORDER_CONFIG = RUNTIME_CONFIG.get("recorder", {})
WARMUP_UPDATES = int(RECORDER_CONFIG.get("warmup_updates", 5))
STOP_DEBUG = bool(RECORDER_CONFIG.get("debug_stop_state", False))

APPEARANCE_SOURCE = '#!/usr/bin/env python3\n"""\nDR.Nav curated appearance randomizer for Maze2.\n\nUses only exact paths confirmed by drnav_scene_paths.json.\n\nPer episode:\n- one global brightness multiplier for the three distant lights and the cylinder light,\n- one coherent warm/cool light tint,\n- one camera exposure value when supported,\n- one discrete permutation of the three existing environment materials.\n\nIt does not touch robot materials, zones, physics, colliders, joints, sensor\nposes, or the ROS runtime graph. All changes are stored in an anonymous\nsession sublayer and are removed before the next episode.\n"""\n\nimport builtins\nimport json\nimport random\nimport time\n\nimport omni.usd\nfrom pxr import Gf, Sdf, Usd, UsdGeom, UsdShade\n\n\nSEED = None\n\nLIGHT_PATHS = [\n    "/World/ground/DistantLight",\n    "/World/ground/flat_plane/DistantLight",\n    "/World/ground/flat_plane/Environment/defaultLight",\n    "/World/ground/CylinderLight",\n]\n\nCAMERA_PATH = (\n    "/World/ground/flat_plane/jackal/base_link/"\n    "bumblebee_stereo_camera_frame/"\n    "bumblebee_stereo_left_frame/"\n    "bumblebee_stereo_left_camera"\n)\n\nREGION_ROOTS = [\n    "/World/ground/flat_plane/dead_end",\n    "/World/ground/flat_plane/dead_end_01",\n    "/World/ground/flat_plane/cross",\n]\n\nMATERIAL_NAMES = [\n    "Large_Granite_Paving",\n    "Facade_Brick_Red_Clinker",\n    "Carpet_Woven_Cherry_Red",\n]\n\nBRIGHTNESS_MULTIPLIER_RANGE = (0.70, 1.30)\nLIGHT_TINT_STRENGTH = 0.12\nCAMERA_EXPOSURE_RANGE = (-0.5, 0.5)\n\nSTATE_KEY = "_DRNAV_APPEARANCE_STATE"\nLAYER_KEY = "_DRNAV_APPEARANCE_LAYER"\n\n\ndef clamp(value, low=0.0, high=1.0):\n    return max(low, min(high, value))\n\n\ndef remove_previous_layer(stage):\n    previous = getattr(builtins, LAYER_KEY, None)\n\n    if previous is None:\n        return\n\n    session = stage.GetSessionLayer()\n    session.subLayerPaths = [\n        path\n        for path in session.subLayerPaths\n        if path != previous.identifier\n    ]\n\n    setattr(builtins, LAYER_KEY, None)\n    setattr(builtins, STATE_KEY, None)\n\n\ndef create_layer(stage):\n    layer = Sdf.Layer.CreateAnonymous("DRNav_CuratedAppearance.usda")\n    session = stage.GetSessionLayer()\n    paths = list(session.subLayerPaths)\n    paths.append(layer.identifier)\n    session.subLayerPaths = paths\n    return layer\n\n\ndef tint_factors(rng):\n    amount = rng.uniform(-LIGHT_TINT_STRENGTH, LIGHT_TINT_STRENGTH)\n\n    if amount >= 0.0:\n        return (\n            1.0,\n            clamp(1.0 - 0.35 * amount),\n            clamp(1.0 - amount),\n        )\n\n    amount = abs(amount)\n\n    return (\n        clamp(1.0 - amount),\n        clamp(1.0 - 0.30 * amount),\n        1.0,\n    )\n\n\ndef randomize_lights(stage, rng):\n    multiplier = rng.uniform(*BRIGHTNESS_MULTIPLIER_RANGE)\n    tint = tint_factors(rng)\n    changes = []\n\n    for path in LIGHT_PATHS:\n        prim = stage.GetPrimAtPath(path)\n\n        if not prim.IsValid() or not prim.IsActive():\n            changes.append({"path": path, "status": "missing_or_inactive"})\n            continue\n\n        intensity_attr = prim.GetAttribute("inputs:intensity")\n\n        if not intensity_attr.IsValid() or intensity_attr.Get() is None:\n            changes.append({"path": path, "status": "no_intensity"})\n            continue\n\n        original_intensity = float(intensity_attr.Get())\n        new_intensity = original_intensity * multiplier\n        intensity_attr.Set(new_intensity)\n\n        color_attr = prim.GetAttribute("inputs:color")\n        original_color = None\n        new_color = None\n\n        if color_attr.IsValid() and color_attr.Get() is not None:\n            original = color_attr.Get()\n            original_color = [\n                float(original[0]),\n                float(original[1]),\n                float(original[2]),\n            ]\n            new_color = [\n                clamp(original_color[0] * tint[0]),\n                clamp(original_color[1] * tint[1]),\n                clamp(original_color[2] * tint[2]),\n            ]\n            color_attr.Set(Gf.Vec3f(*new_color))\n\n        changes.append(\n            {\n                "path": path,\n                "status": "changed",\n                "original_intensity": original_intensity,\n                "intensity": new_intensity,\n                "original_color": original_color,\n                "color": new_color,\n            }\n        )\n\n    return {\n        "global_multiplier": multiplier,\n        "global_tint_factors": list(tint),\n        "lights": changes,\n    }\n\n\ndef get_bound_material(prim):\n    result = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()\n\n    if not result:\n        return None\n\n    return result[0]\n\n\ndef material_leaf(material):\n    if not material:\n        return None\n\n    path = material.GetPath()\n\n    # Sdf.Path in Isaac Sim 6.0 does not expose an IsEmpty attribute in\n    # Python. Compare against the canonical empty path instead.\n    if path == Sdf.Path.emptyPath:\n        return None\n\n    return path.name\n\n\ndef randomize_material_bindings(stage, rng):\n    # Six discrete palette choices: all permutations of the three materials.\n    shuffled = list(MATERIAL_NAMES)\n    rng.shuffle(shuffled)\n    mapping = dict(zip(MATERIAL_NAMES, shuffled))\n\n    changes = []\n    skipped = []\n\n    for root_path in REGION_ROOTS:\n        root = stage.GetPrimAtPath(root_path)\n\n        if not root.IsValid():\n            skipped.append({"path": root_path, "reason": "missing_region"})\n            continue\n\n        local_materials = {}\n\n        for name in MATERIAL_NAMES:\n            material = UsdShade.Material.Get(\n                stage,\n                f"{root_path}/Looks/{name}",\n            )\n\n            if material:\n                local_materials[name] = material\n\n        for prim in Usd.PrimRange(root):\n            if prim.GetTypeName() != "Mesh":\n                continue\n\n            imageable = UsdGeom.Imageable(prim)\n\n            if (\n                imageable\n                and imageable.ComputeVisibility()\n                == UsdGeom.Tokens.invisible\n            ):\n                continue\n\n            original_material = get_bound_material(prim)\n            original_name = material_leaf(original_material)\n\n            if original_name not in mapping:\n                skipped.append(\n                    {\n                        "path": str(prim.GetPath()),\n                        "reason": "unrecognized_or_unbound_material",\n                        "bound_material": (\n                            str(original_material.GetPath())\n                            if original_material\n                            else None\n                        ),\n                    }\n                )\n                continue\n\n            target_name = mapping[original_name]\n            target_material = local_materials.get(target_name)\n\n            if not target_material:\n                skipped.append(\n                    {\n                        "path": str(prim.GetPath()),\n                        "reason": "target_material_missing",\n                        "target_name": target_name,\n                    }\n                )\n                continue\n\n            UsdShade.MaterialBindingAPI.Apply(prim).Bind(target_material)\n\n            changes.append(\n                {\n                    "mesh": str(prim.GetPath()),\n                    "original_material": str(\n                        original_material.GetPath()\n                    ),\n                    "target_material": str(\n                        target_material.GetPath()\n                    ),\n                    "original_class": original_name,\n                    "target_class": target_name,\n                }\n            )\n\n    return {\n        "palette_mapping": mapping,\n        "changed_meshes": changes,\n        "skipped_meshes": skipped,\n    }\n\n\ndef randomize_camera_exposure(stage, rng):\n    prim = stage.GetPrimAtPath(CAMERA_PATH)\n\n    if not prim.IsValid():\n        return {"status": "camera_missing", "path": CAMERA_PATH}\n\n    attr = prim.GetAttribute("exposure")\n\n    if not attr.IsValid():\n        return {\n            "status": "exposure_not_supported",\n            "path": CAMERA_PATH,\n        }\n\n    exposure = rng.uniform(*CAMERA_EXPOSURE_RANGE)\n    attr.Set(exposure)\n\n    return {\n        "status": "changed",\n        "path": CAMERA_PATH,\n        "exposure": exposure,\n    }\n\n\ndef apply_randomization():\n    stage = omni.usd.get_context().get_stage()\n\n    if stage is None:\n        raise RuntimeError("No USD stage is currently open.")\n\n    remove_previous_layer(stage)\n\n    seed = int(SEED) if SEED is not None else time.time_ns() & 0xFFFFFFFF\n    rng = random.Random(seed)\n    layer = create_layer(stage)\n\n    setattr(builtins, LAYER_KEY, layer)\n\n    with Usd.EditContext(\n        stage,\n        stage.GetEditTargetForLocalLayer(layer),\n    ):\n        lighting = randomize_lights(stage, rng)\n        materials = randomize_material_bindings(stage, rng)\n        camera = randomize_camera_exposure(stage, rng)\n\n    root = stage.GetRootLayer()\n    source = root.realPath or root.identifier\n\n    state = {\n        "mode": "curated_appearance_only",\n        "seed": seed,\n        "stage": source,\n        "layer": layer.identifier,\n        "lighting": lighting,\n        "materials": materials,\n        "camera": camera,\n        "physics_changed": False,\n        "geometry_changed": False,\n        "sensor_pose_changed": False,\n        "robot_materials_changed": False,\n        "zone_materials_changed": False,\n    }\n\n    setattr(builtins, STATE_KEY, state)\n\n    print("\\n[DR.Nav Appearance] Curated randomization applied")\n    print("[DR.Nav Appearance] Seed:", seed)\n    print(\n        "[DR.Nav Appearance] Brightness multiplier:",\n        round(lighting["global_multiplier"], 4),\n    )\n    print(\n        "[DR.Nav Appearance] Material palette:",\n        materials["palette_mapping"],\n    )\n    print(\n        "[DR.Nav Appearance] Mesh bindings changed:",\n        len(materials["changed_meshes"]),\n    )\n    print("[DR.Nav Appearance] Camera:", camera["status"])\n    print("[DR.Nav Appearance] Run recorder next.")\n\n    print("\\n[DR.Nav Appearance] Metadata:")\n    print(json.dumps(state, indent=2))\n\n\napply_randomization()\n'
RECORDER_SOURCE = '#!/usr/bin/env python3\n"""\nDR.Nav Replicator recorder — Phase 1\n====================================\n\nRun after setup_sensors_isaacsim6_small_v2.py.\n\nCaptures synchronized:\n- RGB from the existing Bumblebee left camera\n- Existing PhysX LiDAR depth/angles/point cloud\n- Jackal world pose and estimated velocity\n- Maze ID\n- Semantic zone label\n\nOutput:\n~/drnav_dataset/<maze>/<timestamp>/\n    manifest.json\n    rgb/000000.png\n    lidar/000000.npz\n    metadata/000000.json\n\nRun this script again to stop the previous recorder and start a new run.\n"""\n\nimport asyncio\nimport builtins\nimport io\nimport json\nimport math\nimport os\nimport time\nfrom datetime import datetime\nfrom pathlib import Path\n\nimport numpy as np\nimport omni.kit.app\nimport omni.replicator.core as rep\nimport omni.timeline\nimport omni.usd\nfrom omni.replicator.core import BackendDispatch\nfrom pxr import Usd, UsdGeom\n\n\n# -----------------------------------------------------------------------------\n# Current scene\n# -----------------------------------------------------------------------------\n\nROBOT = "/World/ground/flat_plane/jackal"\nBASE = f"{ROBOT}/base_link"\n\nCAMERA = (\n    f"{BASE}/bumblebee_stereo_camera_frame/"\n    "bumblebee_stereo_left_frame/"\n    "bumblebee_stereo_left_camera"\n)\n\nLIDAR = f"{BASE}/sick_lms1xx_lidar_frame/Lidar"\n\n# Zone cubes are discovered from their prim paths instead of assuming that\n# /World/Zones uses one exact hierarchy. Highest priority is listed first.\nZONE_RULES = [\n    (\n        "dead_end_terminal",\n        3,\n        ("dead_end_terminal", "deadendterminal", "terminal_zone"),\n    ),\n    (\n        "dead_end_branch",\n        2,\n        ("dead_end_branch", "deadendbranch", "branch_zone"),\n    ),\n    (\n        "goal_path",\n        1,\n        ("goal_path", "goalpath", "path_zone"),\n    ),\n]\n\nZONE_OUTSIDE = ("outside", 0)\n\nRESOLUTION = (640, 480)\nCAPTURE_HZ = 10.0\nUSE_Z_FOR_ZONE_TEST = True\nREQUIRE_LIDAR_DATA = True\nMIN_LIDAR_AZIMUTH_COVERAGE = 0.95\nMAX_LIDAR_SLICES = 12\n\n# Starting while paused is allowed. After playback has begun once, stopping the\n# timeline automatically closes the recorder and finalizes its resources.\nSTOP_WHEN_TIMELINE_STOPS = True\n\nOUTPUT_ROOT = Path.home() / "drnav_dataset"\n\nRECORDER_KEY = "_DRNAV_REPLICATOR_RECORDER"\n\n\ndef wrap_angle(angle):\n    return (angle + math.pi) % (2.0 * math.pi) - math.pi\n\n\nclass DRNavRecorder:\n    def __init__(self):\n        self.stage = omni.usd.get_context().get_stage()\n        self.timeline = omni.timeline.get_timeline_interface()\n\n        self.render_product = None\n        self.rgb_annotator = None\n        self.lidar_interface = None\n        self.backend = None\n        self.task = None\n        self.running = False\n\n        self.frame_id = 0\n        self.last_capture_time = None\n        self.last_written_time = None\n        self.skipped_frames = 0\n        self.empty_lidar_frames = 0\n        self.incomplete_lidar_scans = 0\n        self.lidar_slices = []\n        self.expected_azimuth_bins = None\n        self.has_seen_playback = False\n        self.zone_volumes = []\n\n        self.previous_time = None\n        self.previous_position = None\n        self.previous_yaw = None\n\n        self.maze_id = self.get_maze_id()\n        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")\n        self.output_dir = OUTPUT_ROOT / self.maze_id / timestamp\n\n    async def start(self):\n        self.validate_scene()\n        await self.enable_extensions()\n\n        from isaacsim.sensors.physx import _range_sensor\n\n        self.lidar_interface = (\n            _range_sensor.acquire_lidar_sensor_interface()\n        )\n\n        self.output_dir.mkdir(parents=True, exist_ok=False)\n        self.backend = BackendDispatch(\n            {"paths": {"out_dir": str(self.output_dir)}}\n        )\n\n        self.build_zone_index()\n        self.configure_lidar_accumulator()\n        self.write_manifest()\n\n        # We trigger capture manually at CAPTURE_HZ.\n        rep.orchestrator.set_capture_on_play(False)\n\n        self.render_product = rep.create.render_product(\n            CAMERA,\n            RESOLUTION,\n            name="drnav_dataset_render_product",\n        )\n        self.render_product.hydra_texture.set_updates_enabled(False)\n\n        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")\n        self.rgb_annotator.attach(self.render_product)\n\n        self.running = True\n        self.task = asyncio.ensure_future(self.capture_loop())\n\n        print("\\n[DR.Nav Recorder] Started")\n        print("[DR.Nav Recorder] Maze:", self.maze_id)\n        print("[DR.Nav Recorder] Output:", self.output_dir)\n        print("[DR.Nav Recorder] Capture rate:", CAPTURE_HZ, "Hz")\n        print(\n            "[DR.Nav Recorder] Zone volumes:",\n            len(self.zone_volumes),\n        )\n\n    async def stop(self):\n        self.running = False\n\n        current_task = asyncio.current_task()\n\n        if (\n            self.task is not None\n            and self.task is not current_task\n            and not self.task.done()\n        ):\n            self.task.cancel()\n            try:\n                await self.task\n            except asyncio.CancelledError:\n                pass\n            except Exception as exc:\n                print("[DR.Nav Recorder] Loop cleanup warning:", repr(exc))\n\n        self.task = None\n\n        try:\n            await rep.orchestrator.wait_until_complete_async()\n        except Exception:\n            pass\n\n        if self.rgb_annotator is not None:\n            try:\n                self.rgb_annotator.detach()\n            except Exception:\n                pass\n\n        if self.render_product is not None:\n            try:\n                self.render_product.destroy()\n            except Exception:\n                pass\n\n        self.rgb_annotator = None\n        self.render_product = None\n\n        print(\n            "[DR.Nav Recorder] Stopped after",\n            self.frame_id,\n            "frames.",\n        )\n        print("[DR.Nav Recorder] Skipped frames:", self.skipped_frames)\n        print(\n            "[DR.Nav Recorder] Empty LiDAR slices:",\n            self.empty_lidar_frames,\n        )\n        print(\n            "[DR.Nav Recorder] Incomplete LiDAR scans:",\n            self.incomplete_lidar_scans,\n        )\n\n    def validate_scene(self):\n        missing = [\n            path\n            for path in [ROBOT, BASE, CAMERA, LIDAR]\n            if not self.stage.GetPrimAtPath(path).IsValid()\n        ]\n\n        if missing:\n            raise RuntimeError(\n                "Missing required scene prims:\\n  "\n                + "\\n  ".join(missing)\n            )\n\n        lidar_prim = self.stage.GetPrimAtPath(LIDAR)\n\n        if lidar_prim.GetTypeName() != "Lidar":\n            raise RuntimeError(\n                f"Expected existing PhysX Lidar at {LIDAR}, "\n                f"found {lidar_prim.GetTypeName()!r}"\n            )\n\n    async def enable_extensions(self):\n        manager = omni.kit.app.get_app().get_extension_manager()\n\n        for extension in [\n            "omni.replicator.core",\n            "isaacsim.sensors.physx",\n        ]:\n            if not manager.is_extension_enabled(extension):\n                manager.set_extension_enabled_immediate(\n                    extension,\n                    True,\n                )\n\n        for _ in range(3):\n            await omni.kit.app.get_app().next_update_async()\n\n    def get_maze_id(self):\n        root = self.stage.GetRootLayer()\n        source = root.realPath or root.identifier\n        name = Path(source).stem\n\n        if not name or name.startswith("anon"):\n            return "unknown_maze"\n\n        return name\n\n    # ------------------------------------------------------------------\n    # Zones\n    # ------------------------------------------------------------------\n\n    def build_zone_index(self):\n        """Discover semantic Cube volumes anywhere in the stage."""\n        cache = UsdGeom.BBoxCache(\n            Usd.TimeCode.Default(),\n            [UsdGeom.Tokens.default_],\n            useExtentsHint=True,\n        )\n\n        found_counts = {label: 0 for label, _, _ in ZONE_RULES}\n        zone_like_paths = []\n\n        for prim in self.stage.Traverse():\n            path = str(prim.GetPath())\n            normalized = path.lower().replace("-", "_").replace(" ", "_")\n\n            if any(\n                token in normalized\n                for token in ("zone", "dead_end", "deadend", "goal_path", "goalpath")\n            ):\n                zone_like_paths.append(path)\n\n            if prim.GetTypeName() != "Cube":\n                continue\n\n            matched = None\n\n            for label, zone_id, aliases in ZONE_RULES:\n                if any(alias in normalized for alias in aliases):\n                    matched = (label, zone_id)\n                    break\n\n            if matched is None:\n                continue\n\n            world_bound = cache.ComputeWorldBound(prim)\n            aligned = world_bound.ComputeAlignedRange()\n\n            self.zone_volumes.append(\n                {\n                    "label": matched[0],\n                    "id": matched[1],\n                    "path": path,\n                    "min": np.array(\n                        aligned.GetMin(),\n                        dtype=np.float64,\n                    ),\n                    "max": np.array(\n                        aligned.GetMax(),\n                        dtype=np.float64,\n                    ),\n                }\n            )\n            found_counts[matched[0]] += 1\n\n        # Stage traversal order is not a semantic priority. Sort explicitly so\n        # overlaps always resolve terminal > branch > goal path.\n        self.zone_volumes.sort(\n            key=lambda volume: volume["id"],\n            reverse=True,\n        )\n\n        for label, _, _ in ZONE_RULES:\n            print(\n                f"[DR.Nav Recorder] {label}: "\n                f"{found_counts[label]} cube volume(s)"\n            )\n\n        if not self.zone_volumes:\n            print("[DR.Nav Recorder] Zone-like prims discovered:")\n            for path in zone_like_paths[:100]:\n                print("   ", path)\n\n            raise RuntimeError(\n                "No semantic zone Cube prims were matched. "\n                "Rename the zone cubes or add their path keywords to ZONE_RULES."\n            )\n\n    def zone_at(self, position):\n        point = np.asarray(position, dtype=np.float64)\n\n        for volume in self.zone_volumes:\n            minimum = volume["min"]\n            maximum = volume["max"]\n\n            inside_xy = (\n                minimum[0] <= point[0] <= maximum[0]\n                and minimum[1] <= point[1] <= maximum[1]\n            )\n\n            inside_z = (\n                minimum[2] <= point[2] <= maximum[2]\n                if USE_Z_FOR_ZONE_TEST\n                else True\n            )\n\n            if inside_xy and inside_z:\n                return (\n                    volume["label"],\n                    volume["id"],\n                    volume["path"],\n                )\n\n        return ZONE_OUTSIDE[0], ZONE_OUTSIDE[1], None\n\n    # ------------------------------------------------------------------\n    # Robot state\n    # ------------------------------------------------------------------\n\n    def read_robot_state(self, now):\n        matrix = UsdGeom.XformCache(\n            Usd.TimeCode.Default()\n        ).GetLocalToWorldTransform(\n            self.stage.GetPrimAtPath(BASE)\n        )\n\n        translation = matrix.ExtractTranslation()\n        quaternion = matrix.ExtractRotationQuat()\n        imaginary = quaternion.GetImaginary()\n\n        x, y, z = map(float, translation)\n        qx, qy, qz = map(float, imaginary)\n        qw = float(quaternion.GetReal())\n\n        yaw = math.atan2(\n            2.0 * (qw * qz + qx * qy),\n            1.0 - 2.0 * (qy * qy + qz * qz),\n        )\n\n        vx = vy = vz = wz = 0.0\n\n        if self.previous_time is not None:\n            dt = now - self.previous_time\n\n            if dt > 1e-6:\n                vx = (x - self.previous_position[0]) / dt\n                vy = (y - self.previous_position[1]) / dt\n                vz = (z - self.previous_position[2]) / dt\n                wz = wrap_angle(yaw - self.previous_yaw) / dt\n\n        self.previous_time = now\n        self.previous_position = (x, y, z)\n        self.previous_yaw = yaw\n\n        zone_label, zone_id, zone_path = self.zone_at((x, y, z))\n\n        return {\n            "position": {"x": x, "y": y, "z": z},\n            "orientation": {\n                "x": qx,\n                "y": qy,\n                "z": qz,\n                "w": qw,\n            },\n            "yaw": yaw,\n            "velocity_world": {\n                "x": vx,\n                "y": vy,\n                "z": vz,\n                "yaw_rate": wz,\n            },\n            "zone": {\n                "label": zone_label,\n                "id": zone_id,\n                "volume_path": zone_path,\n            },\n        }\n\n    # ------------------------------------------------------------------\n    # Writing\n    # ------------------------------------------------------------------\n\n    def write_manifest(self):\n        manifest = {\n            "format": "drnav_replicator_phase1",\n            "version": 2,\n            "require_lidar_data": REQUIRE_LIDAR_DATA,\n            "minimum_lidar_azimuth_coverage": MIN_LIDAR_AZIMUTH_COVERAGE,\n            "expected_lidar_azimuth_bins": self.expected_azimuth_bins,\n            "maze_id": self.maze_id,\n            "created": datetime.now().isoformat(),\n            "capture_hz": CAPTURE_HZ,\n            "resolution": list(RESOLUTION),\n            "camera_prim": CAMERA,\n            "lidar_prim": LIDAR,\n            "robot_base_prim": BASE,\n            "zone_priority": [\n                item[0] for item in ZONE_RULES\n            ],\n            "zone_overlap_policy": "highest_zone_id_wins",\n            "use_z_for_zone_test": USE_Z_FOR_ZONE_TEST,\n            "appearance_randomization": getattr(\n                builtins,\n                "_DRNAV_APPEARANCE_STATE",\n                None,\n            ),\n            "zone_ids": {\n                "outside": 0,\n                "goal_path": 1,\n                "dead_end_branch": 2,\n                "dead_end_terminal": 3,\n            },\n        }\n\n        self.write_json("manifest.json", manifest)\n\n    def write_json(self, relative_path, data):\n        blob = json.dumps(\n            data,\n            indent=2,\n            sort_keys=True,\n        ).encode("utf-8")\n        self.backend.write_blob(relative_path, blob)\n\n    def configure_lidar_accumulator(self):\n        prim = self.stage.GetPrimAtPath(LIDAR)\n        fov = prim.GetAttribute("horizontalFov").Get()\n        resolution = prim.GetAttribute("horizontalResolution").Get()\n\n        if fov is None or resolution is None or float(resolution) <= 0.0:\n            raise RuntimeError(\n                "PhysX LiDAR horizontalFov/horizontalResolution is unavailable."\n            )\n\n        self.expected_azimuth_bins = int(\n            round(float(fov) / float(resolution))\n        )\n\n        if self.expected_azimuth_bins <= 0:\n            raise RuntimeError("Computed invalid PhysX LiDAR azimuth bin count.")\n\n        print(\n            "[DR.Nav Recorder] Expected full LiDAR scan:",\n            self.expected_azimuth_bins,\n            "azimuth bins.",\n        )\n\n    def capture_lidar(self):\n        # Copy the PhysX buffers before Replicator performs its zero-time\n        # render step. The buffers may be invalidated or cleared by that step.\n        depth = np.asarray(\n            self.lidar_interface.get_linear_depth_data(LIDAR),\n            dtype=np.float32,\n        ).copy()\n        azimuth = np.asarray(\n            self.lidar_interface.get_azimuth_data(LIDAR),\n            dtype=np.float32,\n        ).copy()\n        zenith = np.asarray(\n            self.lidar_interface.get_zenith_data(LIDAR),\n            dtype=np.float32,\n        ).copy()\n        points = np.asarray(\n            self.lidar_interface.get_point_cloud_data(LIDAR),\n            dtype=np.float32,\n        ).copy()\n\n        if points.size:\n            points = points.reshape(-1, 3)\n\n        return depth, azimuth, zenith, points\n\n    def accumulate_lidar_slice(self, lidar_data):\n        depth, azimuth, zenith, points = lidar_data\n\n        if not depth.size or not azimuth.size or not points.size:\n            self.empty_lidar_frames += 1\n            return False\n\n        if depth.shape[0] != azimuth.size:\n            print(\n                "[DR.Nav Recorder] LiDAR depth/azimuth mismatch:",\n                depth.shape,\n                azimuth.shape,\n            )\n            self.empty_lidar_frames += 1\n            return False\n\n        vertical_count = max(1, int(zenith.size))\n\n        if points.size != azimuth.size * vertical_count * 3:\n            print(\n                "[DR.Nav Recorder] LiDAR point/angle mismatch:",\n                points.shape,\n                azimuth.shape,\n                zenith.shape,\n            )\n            self.empty_lidar_frames += 1\n            return False\n\n        point_rows = points.reshape(azimuth.size, vertical_count, 3)\n        self.lidar_slices.append(\n            (depth, azimuth, zenith, point_rows)\n        )\n\n        if len(self.lidar_slices) > MAX_LIDAR_SLICES:\n            self.lidar_slices.pop(0)\n\n        return True\n\n    def completed_lidar_scan(self):\n        if not self.lidar_slices:\n            return None\n\n        depth = np.concatenate(\n            [item[0] for item in self.lidar_slices],\n            axis=0,\n        )\n        azimuth = np.concatenate(\n            [item[1] for item in self.lidar_slices],\n            axis=0,\n        )\n        zenith = self.lidar_slices[-1][2]\n        point_rows = np.concatenate(\n            [item[3] for item in self.lidar_slices],\n            axis=0,\n        )\n\n        # Repeated rotations contain the same angular bins. Quantize at a\n        # tighter precision than the configured angular resolution and keep one\n        # row per azimuth before sorting into scan order.\n        quantized = np.round(azimuth.astype(np.float64), decimals=6)\n        _, unique_indices = np.unique(quantized, return_index=True)\n        unique_indices.sort()\n        depth = depth[unique_indices]\n        azimuth = azimuth[unique_indices]\n        point_rows = point_rows[unique_indices]\n\n        order = np.argsort(azimuth)\n        depth = depth[order]\n        azimuth = azimuth[order]\n        point_rows = point_rows[order]\n\n        coverage = azimuth.size / float(self.expected_azimuth_bins)\n\n        if coverage < MIN_LIDAR_AZIMUTH_COVERAGE:\n            return None\n\n        # Do not let duplicate rotations make a completed scan larger than the\n        # sensor\'s configured azimuth resolution.\n        if azimuth.size > self.expected_azimuth_bins:\n            azimuth = azimuth[:self.expected_azimuth_bins]\n            depth = depth[:self.expected_azimuth_bins]\n            point_rows = point_rows[:self.expected_azimuth_bins]\n\n        points = point_rows.reshape(-1, 3)\n        self.lidar_slices.clear()\n        return depth, azimuth, zenith, points\n\n    def write_frame(self, rgb, now, lidar_data):\n        frame_name = f"{self.frame_id:06d}"\n        state = self.read_robot_state(now)\n\n        depth, azimuth, zenith, points = lidar_data\n\n        self.backend.write_image(\n            f"rgb/{frame_name}.png",\n            np.asarray(rgb),\n        )\n\n        buffer = io.BytesIO()\n        np.savez_compressed(\n            buffer,\n            depth=depth,\n            azimuth=azimuth,\n            zenith=zenith,\n            points=points,\n        )\n        self.backend.write_blob(\n            f"lidar/{frame_name}.npz",\n            buffer.getvalue(),\n        )\n\n        metadata = {\n            "frame_id": self.frame_id,\n            "simulation_time": now,\n            "maze_id": self.maze_id,\n            "rgb_path": f"rgb/{frame_name}.png",\n            "lidar_path": f"lidar/{frame_name}.npz",\n            "robot": state,\n            "lidar_shapes": {\n                "depth": list(depth.shape),\n                "azimuth": list(azimuth.shape),\n                "zenith": list(zenith.shape),\n                "points": list(points.shape),\n            },\n        }\n\n        self.write_json(\n            f"metadata/{frame_name}.json",\n            metadata,\n        )\n\n        if self.frame_id % 20 == 0:\n            print(\n                f"[DR.Nav Recorder] frame={self.frame_id} "\n                f"time={now:.2f} "\n                f"zone={state[\'zone\'][\'label\']}"\n            )\n\n        self.frame_id += 1\n\n    # ------------------------------------------------------------------\n    # Capture loop\n    # ------------------------------------------------------------------\n\n    async def capture_loop(self):\n        interval = 1.0 / CAPTURE_HZ\n\n        while self.running:\n            await omni.kit.app.get_app().next_update_async()\n\n            if not self.timeline.is_playing():\n                self.last_capture_time = None\n                self.lidar_slices.clear()\n\n                if (\n                    STOP_WHEN_TIMELINE_STOPS\n                    and self.has_seen_playback\n                ):\n                    print(\n                        "[DR.Nav Recorder] Timeline stopped; "\n                        "closing recorder."\n                    )\n                    await self.stop()\n\n                    if getattr(builtins, RECORDER_KEY, None) is self:\n                        setattr(builtins, RECORDER_KEY, None)\n\n                    return\n\n                # If launched while paused, wait for the first Play event.\n                continue\n\n            self.has_seen_playback = True\n            now = float(self.timeline.get_current_time())\n\n            # Read every physics slice, not only the 10 Hz output instants.\n            # The 20 Hz rotary LiDAR yields a 120-degree slice at 60 Hz physics.\n            self.accumulate_lidar_slice(self.capture_lidar())\n\n            if (\n                self.last_capture_time is not None\n                and now - self.last_capture_time < interval - 1e-9\n            ):\n                continue\n\n            lidar_data = self.completed_lidar_scan()\n\n            if REQUIRE_LIDAR_DATA and lidar_data is None:\n                self.incomplete_lidar_scans += 1\n\n                if (\n                    self.incomplete_lidar_scans <= 3\n                    or self.incomplete_lidar_scans % 20 == 0\n                ):\n                    print(\n                        "[DR.Nav Recorder] Waiting for complete 360-degree "\n                        "LiDAR scan."\n                    )\n                continue\n\n            self.render_product.hydra_texture.set_updates_enabled(True)\n\n            try:\n                await rep.orchestrator.step_async(\n                    delta_time=0.0,\n                    pause_timeline=False,\n                    rt_subframes=1,\n                )\n\n                # Stop can reset simulation time while the render await is in\n                # progress. Never write that reset frame into the episode.\n                if not self.running or not self.timeline.is_playing():\n                    self.skipped_frames += 1\n                    continue\n\n                capture_time = float(self.timeline.get_current_time())\n\n                if (\n                    self.last_written_time is not None\n                    and capture_time <= self.last_written_time\n                ):\n                    self.skipped_frames += 1\n                    print(\n                        "[DR.Nav Recorder] Non-increasing simulation time; "\n                        f"discarding frame at {capture_time:.6f}."\n                    )\n                    continue\n\n                rgb = self.rgb_annotator.get_data()\n\n                if rgb is None or not np.asarray(rgb).size:\n                    self.skipped_frames += 1\n                    print("[DR.Nav Recorder] Empty RGB frame; skipping capture.")\n                    continue\n\n                self.write_frame(rgb, capture_time, lidar_data)\n                self.last_capture_time = capture_time\n                self.last_written_time = capture_time\n            finally:\n                self.render_product.hydra_texture.set_updates_enabled(False)\n\n\nasync def main():\n    previous = getattr(builtins, RECORDER_KEY, None)\n\n    if previous is not None:\n        try:\n            await previous.stop()\n        except Exception as exc:\n            print(\n                "[DR.Nav Recorder] Previous recorder cleanup warning:",\n                repr(exc),\n            )\n\n    recorder = DRNavRecorder()\n    setattr(builtins, RECORDER_KEY, recorder)\n\n    try:\n        await recorder.start()\n    except Exception:\n        try:\n            await recorder.stop()\n        finally:\n            setattr(builtins, RECORDER_KEY, None)\n        raise\n\n\nasyncio.ensure_future(main())\n'


def configured_appearance_source():
    """Keep full appearance metadata in the manifest, not the terminal."""
    verbose_output = '''    print("[DR.Nav Appearance] Run recorder next.")

    print("\\n[DR.Nav Appearance] Metadata:")
    print(json.dumps(state, indent=2))
'''
    concise_output = '''    print(
        "[DR.Nav Appearance] Full metadata will be saved in manifest.json."
    )
'''

    if verbose_output not in APPEARANCE_SOURCE:
        raise RuntimeError("Appearance console-output patch target is missing.")

    return APPEARANCE_SOURCE.replace(
        verbose_output,
        concise_output,
        1,
    )


def configured_recorder_source():
    """Apply YAML recorder values to the embedded standalone recorder."""
    source = RECORDER_SOURCE
    resolution = RECORDER_CONFIG.get("resolution", [640, 480])
    output_root = RECORDER_CONFIG.get("output_root", "~/drnav_dataset")
    replacements = {
        "RESOLUTION = (640, 480)": (
            f"RESOLUTION = ({int(resolution[0])}, {int(resolution[1])})"
        ),
        "CAPTURE_HZ = 10.0": (
            f"CAPTURE_HZ = {float(RECORDER_CONFIG.get('capture_hz', 10.0))!r}"
        ),
        "USE_Z_FOR_ZONE_TEST = True": (
            "USE_Z_FOR_ZONE_TEST = "
            f"{bool(RECORDER_CONFIG.get('use_z_for_zone_test', True))!r}"
        ),
        "REQUIRE_LIDAR_DATA = True": (
            "REQUIRE_LIDAR_DATA = "
            f"{bool(RECORDER_CONFIG.get('require_lidar_data', True))!r}"
        ),
        "MIN_LIDAR_AZIMUTH_COVERAGE = 0.95": (
            "MIN_LIDAR_AZIMUTH_COVERAGE = "
            f"{float(RECORDER_CONFIG.get('minimum_lidar_coverage', 0.95))!r}"
        ),
        "MAX_LIDAR_SLICES = 12": (
            "MAX_LIDAR_SLICES = "
            f"{int(RECORDER_CONFIG.get('max_lidar_slices', 12))}"
        ),
        '''        # Copy the PhysX buffers before Replicator performs its zero-time
        # render step. The buffers may be invalidated or cleared by that step.
''': '''        # Copy the PhysX buffers immediately because the sensor reuses its
        # live buffers on later simulation updates.
''',
        'OUTPUT_ROOT = Path.home() / "drnav_dataset"': (
            f"OUTPUT_ROOT = Path({str(output_root)!r}).expanduser()"
        ),
        "        self.rgb_annotator.attach(self.render_product)": (
            "        self.rgb_annotator.attach(self.render_product)\n"
            "        print(\n"
            '            "[DR.Nav Recorder] RGB source: exclusive "\n'
            '            "Replicator render product"\n'
            "        )"
        ),
        '''        # We trigger capture manually at CAPTURE_HZ.
        rep.orchestrator.set_capture_on_play(False)
''': '''        # Capture is manually stepped only for RGB frames. Disabling
        # capture-on-play prevents Replicator from owning GUI playback.
        rep.orchestrator.set_capture_on_play(False)
''',
        "        self.running = False\n\n        self.frame_id = 0": (
            "        self.running = False\n"
            "        self.capture_in_progress = False\n"
            "        self.stopped = False\n"
            "        self.stop_lock = asyncio.Lock()\n"
            "        self.timeline_auto_update = (\n"
            "            self.timeline.is_auto_updating()\n"
            "        )\n\n"
            "        self.frame_id = 0\n"
            "        self.empty_rgb_frames = 0"
        ),
        '''    async def stop(self):
        self.running = False

        current_task = asyncio.current_task()

        if (
            self.task is not None
            and self.task is not current_task
            and not self.task.done()
        ):
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
        print("[DR.Nav Recorder] Skipped frames:", self.skipped_frames)
        print(
            "[DR.Nav Recorder] Empty LiDAR slices:",
            self.empty_lidar_frames,
        )
        print(
            "[DR.Nav Recorder] Incomplete LiDAR scans:",
            self.incomplete_lidar_scans,
        )
''': '''    async def stop(self):
        current_task = asyncio.current_task()

        async with self.stop_lock:
            if self.stopped:
                return

            self.running = False
            task_to_wait = (
                self.task
                if (
                    self.task is not None
                    and self.task is not current_task
                    and not self.task.done()
                )
                else None
            )

        # Let the capture loop leave at its next app update. Wait outside the
        # lock because the capture task may also be the caller that completes
        # cleanup.
        if task_to_wait is not None:
            try:
                await task_to_wait
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print(
                    "[DR.Nav Recorder] Loop cleanup warning:",
                    repr(exc),
                )

        async with self.stop_lock:
            if self.stopped:
                return

            self.task = None

            try:
                if not rep.orchestrator.get_is_stopped():
                    await rep.orchestrator.stop_async()
            except Exception as exc:
                print(
                    "[DR.Nav Recorder] Replicator cleanup warning:",
                    repr(exc),
                )

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

            self.rgb_annotator = None
            self.render_product = None
            self.stopped = True

            print(
                "[DR.Nav Recorder] Stopped after",
                self.frame_id,
                "frames.",
            )
            print(
                "[DR.Nav Recorder] Skipped frames:",
                self.skipped_frames,
            )
            print(
                "[DR.Nav Recorder] Empty RGB frames:",
                self.empty_rgb_frames,
            )
            print(
                "[DR.Nav Recorder] Empty LiDAR slices:",
                self.empty_lidar_frames,
            )
            print(
                "[DR.Nav Recorder] Incomplete LiDAR scans:",
                self.incomplete_lidar_scans,
            )
''',
        '''        while self.running:
            await omni.kit.app.get_app().next_update_async()

            if not self.timeline.is_playing():
''': '''        while self.running:
            await omni.kit.app.get_app().next_update_async()

            if (
                not self.running
                or getattr(builtins, RECORDER_KEY, None) is not self
            ):
                self.running = False
                return

            if not self.timeline.is_playing():
''',
        '''            self.render_product.hydra_texture.set_updates_enabled(True)

            try:
''': '''            auto_update_before_capture = (
                self.timeline.is_auto_updating()
            )
            self.render_product.hydra_texture.set_updates_enabled(True)

            try:
                self.capture_in_progress = True
''',
        "                    delta_time=0.0,": (
            "                    # Keep normal timeline auto-update active.\n"
            "                    delta_time=None,"
        ),
        '''            finally:
                self.render_product.hydra_texture.set_updates_enabled(False)
''': '''            finally:
                # Replicator temporarily disables automatic timeline updates
                # while rendering. Restore the pre-capture value even when
                # capture is interrupted by Stop or an exception.
                self.timeline.set_auto_update(auto_update_before_capture)
                self.timeline.commit_silently()
                self.render_product.hydra_texture.set_updates_enabled(False)
                self.capture_in_progress = False
''',
        "                rgb = self.rgb_annotator.get_data()": (
            "                rgb = self.rgb_annotator.get_data("
            "do_array_copy=True)"
        ),
        '''                if rgb is None or not np.asarray(rgb).size:
                    self.skipped_frames += 1
                    print("[DR.Nav Recorder] Empty RGB frame; skipping capture.")
                    continue
''': '''                if rgb is None or not np.asarray(rgb).size:
                    self.skipped_frames += 1
                    self.empty_rgb_frames += 1

                    if (
                        self.empty_rgb_frames <= 3
                        or self.empty_rgb_frames % 20 == 0
                    ):
                        print(
                            "[DR.Nav Recorder] Empty RGB frame; skipping "
                            f"capture (count={self.empty_rgb_frames})."
                        )
                    continue
''',
    }

    for original, configured in replacements.items():
        if original not in source:
            raise RuntimeError(
                "Recorder configuration target is missing: " + original
            )
        source = source.replace(original, configured, 1)

    return source


class AutoEpisodePipeline:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.running = False
        self.preparing = False
        self.monitor_task = None
        self.timeline_sub = None
        self.stop_requested = False
        self.stop_event_seen = False
        self.stop_cleanup_task = None
        self.timeline_auto_update = self.timeline.is_auto_updating()
        self.episode_number = 0

    @staticmethod
    def task_state(task):
        if task is None:
            return "none"
        if task.cancelled():
            return "cancelled"
        if task.done():
            return "done"
        return "pending"

    def timeline_state(self):
        if self.timeline.is_playing():
            return "playing"
        if self.timeline.is_stopped():
            return "stopped"
        return "paused"

    def debug_stop_snapshot(self, label):
        if not STOP_DEBUG:
            return

        recorder = getattr(builtins, RECORDER_KEY, None)
        follower = getattr(builtins, WAYPOINT_FOLLOWER_KEY, None)
        recorder_task = getattr(recorder, "task", None)
        render_product = getattr(recorder, "render_product", None)
        render_path = getattr(render_product, "path", None)
        settings = carb.settings.get_settings()
        try:
            replicator_status = rep.orchestrator.get_status()
        except Exception as exc:
            replicator_status = f"unavailable:{exc!r}"

        print(f"[DR.Nav Stop Debug] snapshot={label}")
        print(
            "[DR.Nav Stop Debug] timeline="
            f"{self.timeline_state()} "
            f"playing={self.timeline.is_playing()} "
            f"stopped={self.timeline.is_stopped()} "
            f"auto_update={self.timeline.is_auto_updating()} "
            f"time={self.timeline.get_current_time():.6f}"
        )
        print(
            "[DR.Nav Stop Debug] replicator="
            f"{replicator_status} "
            "capture_on_play="
            f"{settings.get('/omni/replicator/captureOnPlay')} "
            "async_rendering="
            f"{settings.get('/app/asyncRendering')}"
        )
        print(
            "[DR.Nav Stop Debug] pipeline="
            f"running={self.running} preparing={self.preparing} "
            f"stop_requested={self.stop_requested} "
            f"stop_event_seen={self.stop_event_seen} "
            f"monitor={self.task_state(self.monitor_task)} "
            f"cleanup={self.task_state(self.stop_cleanup_task)}"
        )
        print(
            "[DR.Nav Stop Debug] recorder="
            f"present={recorder is not None} "
            f"id={id(recorder) if recorder is not None else 'none'} "
            f"running={getattr(recorder, 'running', None)} "
            f"stopped={getattr(recorder, 'stopped', None)} "
            "capture_in_progress="
            f"{getattr(recorder, 'capture_in_progress', None)} "
            f"task={self.task_state(recorder_task)} "
            f"frames={getattr(recorder, 'frame_id', None)} "
            f"render_product={render_path}"
        )
        print(
            "[DR.Nav Stop Debug] follower="
            f"present={follower is not None} "
            f"id={id(follower) if follower is not None else 'none'} "
            "physics_callback="
            f"{getattr(follower, 'physics_callback', None) is not None}"
        )

    async def start(self):
        # This pipeline is controlled from the GUI timeline and therefore
        # requires normal automatic timeline updates. Also repairs the setting
        # if an older recorder was interrupted inside a Replicator step.
        self.timeline.set_auto_update(True)
        self.timeline.commit_silently()
        self.timeline_auto_update = True

        if not self.timeline.is_stopped():
            self.timeline.stop()

            for _ in range(3):
                await omni.kit.app.get_app().next_update_async()

        await self.stop_active_recorder()

        self.timeline_sub = (
            self.timeline
            .get_timeline_event_stream()
            .create_subscription_to_pop(
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

        if (
            self.stop_cleanup_task is not None
            and self.stop_cleanup_task is not current
            and not self.stop_cleanup_task.done()
        ):
            await self.stop_cleanup_task

        self.stop_cleanup_task = None

        if (
            self.monitor_task is not None
            and self.monitor_task is not current
            and not self.monitor_task.done()
        ):
            self.monitor_task.cancel()

            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        self.monitor_task = None
        self.timeline_sub = None
        await self.stop_active_recorder()
        self.remove_appearance_layer()

        print("[DR.Nav Auto] Curated episode pipeline stopped.")

    def request_stop_cleanup(self, source):
        if getattr(builtins, CONTROLLER_KEY, None) is not self:
            return

        self.stop_requested = True

        if (
            self.preparing
            or (
                self.stop_cleanup_task is not None
                and not self.stop_cleanup_task.done()
            )
        ):
            if STOP_DEBUG:
                print(
                    "[DR.Nav Stop Debug] cleanup request coalesced: "
                    f"source={source} preparing={self.preparing}"
                )
            return

        self.debug_stop_snapshot(f"detected:{source}")
        self.stop_cleanup_task = asyncio.ensure_future(
            self.finish_stop_cleanup()
        )
        print(
            f"[DR.Nav Auto] Stop detected by {source}; finalizing recorder "
            "and camera capture."
        )

    def on_timeline_event(self, event):
        if getattr(builtins, CONTROLLER_KEY, None) is not self:
            return

        if STOP_DEBUG and event.type in (0, 1, 2):
            event_name = {0: "PLAY", 1: "PAUSE", 2: "STOP"}[event.type]
            print(
                "[DR.Nav Stop Debug] event="
                f"{event_name} timeline={self.timeline_state()} "
                f"auto_update={self.timeline.is_auto_updating()} "
                f"time={self.timeline.get_current_time():.6f}"
            )

        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self.stop_event_seen = False

        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self.stop_event_seen = True
            self.request_stop_cleanup("timeline event")

    def notify_missed_stop_event(self):
        """Refresh STOP listeners without transitioning through Play."""
        if self.stop_event_seen:
            return

        payload = dict(self.timeline.get_event_key())
        payload.setdefault("name", "")
        observers = (
            carb.eventdispatcher.get_eventdispatcher().dispatch_event(
                omni.timeline.GLOBAL_EVENT_STOP,
                payload=payload,
            )
        )
        self.stop_event_seen = True

        follower = getattr(builtins, WAYPOINT_FOLLOWER_KEY, None)
        reset_route = getattr(follower, "reset_route", None)
        if callable(reset_route):
            reset_route()

        print(
            "[DR.Nav Auto] Re-emitted missed STOP notification "
            f"to {observers} observer(s) without starting physics."
        )

    async def finish_stop_cleanup(self):
        if getattr(builtins, CONTROLLER_KEY, None) is not self:
            self.stop_requested = False
            return

        try:
            self.debug_stop_snapshot("cleanup:begin")
            await self.stop_active_recorder()
            self.debug_stop_snapshot("cleanup:recorder-stopped")

            self.timeline.set_auto_update(self.timeline_auto_update)
            self.timeline.commit_silently()

            # A Stop that lands while Replicator is committing a render can
            # change the internal state without notifying the toolbar or
            # reset listeners. Dispatch only the missing STOP notification;
            # never synthesize Play because that initializes physics tensors.
            self.timeline.stop()
            self.timeline.commit()
            self.notify_missed_stop_event()

            # Require several consecutive app updates in the stopped state.
            # This finite settling window catches delayed render/graph work
            # that can otherwise put the timeline back into Play immediately
            # after cleanup without trapping future user Play requests.
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

            if self.timeline.is_playing():
                final_state = "playing"
            elif self.timeline.is_stopped():
                final_state = "stopped"
            else:
                final_state = "paused"
            auto_update = self.timeline.is_auto_updating()
            replicator_status = str(rep.orchestrator.get_status())
            self.debug_stop_snapshot("cleanup:final")
        except Exception as exc:
            print("[DR.Nav Auto] Stop cleanup warning:", repr(exc))
            print(traceback.format_exc())
            self.timeline.set_auto_update(True)
            self.timeline.stop()
            self.timeline.commit()
            final_state = (
                "stopped" if self.timeline.is_stopped() else "unknown"
            )
            auto_update = self.timeline.is_auto_updating()
            replicator_status = "unknown"
        finally:
            self.stop_requested = False
            if getattr(builtins, CONTROLLER_KEY, None) is self:
                print(
                    "[DR.Nav Auto] Stop cleanup complete "
                    f"(timeline={locals().get('final_state', 'unknown')}, "
                    f"auto_update={locals().get('auto_update', 'unknown')}, "
                    "replicator="
                    f"{locals().get('replicator_status', 'unknown')}); "
                    "press Play for the next episode."
                )

    async def stop_active_recorder(self):
        recorder = getattr(builtins, RECORDER_KEY, None)

        if recorder is None:
            return

        try:
            await recorder.stop()
        except Exception as exc:
            print("[DR.Nav Auto] Recorder cleanup warning:", repr(exc))

        if getattr(builtins, RECORDER_KEY, None) is recorder:
            setattr(builtins, RECORDER_KEY, None)

    def remove_appearance_layer(self):
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
        setattr(builtins, "_DRNAV_APPEARANCE_STATE", None)

    async def execute_source(self, source, name):
        namespace = {
            "__name__": name,
            "__builtins__": __builtins__,
        }
        exec(compile(source, name, "exec"), namespace, namespace)

    async def wait_for_recorder(self):
        for _ in range(300):
            recorder = getattr(builtins, RECORDER_KEY, None)

            if recorder is not None and getattr(recorder, "running", False):
                return recorder

            await omni.kit.app.get_app().next_update_async()

        raise RuntimeError("Recorder did not become ready.")

    async def cancel_preparation_if_stopped(self):
        if not self.stop_requested:
            return False

        if self.timeline.is_playing():
            self.timeline.stop()
            for _ in range(2):
                await omni.kit.app.get_app().next_update_async()

        await self.stop_active_recorder()
        self.remove_appearance_layer()
        self.stop_requested = False
        print("[DR.Nav Auto] Episode preparation cancelled by Stop.")
        return True

    @staticmethod
    def reset_waypoint_follower():
        follower = getattr(builtins, WAYPOINT_FOLLOWER_KEY, None)
        if follower is None:
            print(
                "[DR.Nav Auto] No waypoint follower active; episode will "
                "record without autonomous motion."
            )
            return

        reset = getattr(follower, "reset_route", None)
        if reset is None:
            raise RuntimeError(
                "The active waypoint follower is outdated. Rerun the current "
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

            await self.stop_active_recorder()

            await self.execute_source(
                configured_appearance_source(),
                "__drnav_curated_appearance__",
            )

            for _ in range(WARMUP_UPDATES):
                await omni.kit.app.get_app().next_update_async()

            if await self.cancel_preparation_if_stopped():
                return

            await self.execute_source(
                configured_recorder_source(),
                "__drnav_episode_recorder__",
            )

            recorder = await self.wait_for_recorder()
            if await self.cancel_preparation_if_stopped():
                return
            self.episode_number += 1

            appearance = getattr(
                builtins,
                "_DRNAV_APPEARANCE_STATE",
                {},
            )

            print(
                f"\n[DR.Nav Auto] Episode {self.episode_number} ready"
            )
            print(
                "[DR.Nav Auto] Seed:",
                appearance.get("seed", "unknown"),
            )
            print(
                "[DR.Nav Auto] Output:",
                getattr(recorder, "output_dir", "unknown"),
            )

            if await self.cancel_preparation_if_stopped():
                return

            self.reset_waypoint_follower()
            self.timeline.play()

            for _ in range(2):
                await omni.kit.app.get_app().next_update_async()

        except Exception as exc:
            print("[DR.Nav Auto] Episode preparation failed:", repr(exc))
            print(traceback.format_exc())
            await self.stop_active_recorder()
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

            if (
                not playing
                and previous_playing
                and not self.preparing
            ):
                # GUI timeline events can be missed if Stop lands while
                # Replicator is temporarily controlling the timeline during a
                # render step. Polling the actual state makes cleanup immune to
                # viewport/terminal focus and event-delivery timing.
                self.request_stop_cleanup("timeline state")

            if playing and not previous_playing and not self.preparing:
                if self.stop_requested:
                    # A pending Replicator step may briefly report Play while
                    # the one-shot cleanup task is draining it. Do not create
                    # another Stop event or renew cleanup from this monitor.
                    previous_playing = playing
                    continue
                else:
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
    except Exception:
        try:
            await controller.stop()
        finally:
            setattr(builtins, CONTROLLER_KEY, None)
        raise


asyncio.ensure_future(main())
