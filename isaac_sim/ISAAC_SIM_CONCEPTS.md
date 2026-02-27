# Isaac Sim 4.5 — Concepts for DR.Nav

A concise reference explaining the Isaac Sim ideas you'll encounter while
running `setup_sensors.py`.

---

## Core Concept: Everything is a Prim

Isaac Sim uses **USD (Universal Scene Description)** — a scene format from Pixar
also used in film VFX. Every object in your scene is a **prim** (primitive) with a
path, like a filesystem:

```
/World
├── flat_plane       ← ground
├── Spot             ← your robot
│   └── sensor
│       ├── camera_front_left
│       ├── camera_side_left
│       ├── camera_side_right
│       └── lidar
├── wall0
├── wall1
└── wall2
```

In code you refer to prims by their path string: `"/World/Spot"`.

---

## Coordinate System

Isaac Sim uses **Z-up, X-forward** (same as ROS REP-103):

| Axis | Direction |
|------|-----------|
| X    | Forward   |
| Y    | Left      |
| Z    | Up        |

Translations are in **metres**, rotations are in **degrees** (in the Python API).

---

## Sensors

### Camera (`omni.isaac.sensor.Camera`)
- Creates a `UsdGeom.Camera` prim
- Does NOT publish images by itself — you need a **render product**
- A render product tells the GPU renderer: "render this camera at this resolution"
- OmniGraph then reads the render buffer and pushes it to ROS 2

### LiDAR RTX (`omni.isaac.sensor.LidarRtx`)
- Ray-traced LiDAR — each beam is a real ray cast against scene geometry
- `config_file_name` picks a pre-defined sensor profile (beam pattern, range, Hz)
- `"OS1_128ch20hz1024res"` = Ouster OS1, 128 beams, 20 Hz, 1024 horizontal steps

---

## OmniGraph

OmniGraph is Isaac Sim's **dataflow compute system** — like a visual programming
environment where data flows through nodes.

```
OnPlaybackTick ──► ROS2CameraHelper  ──► /argus/.../image_raw  (ROS 2)
               ──► ROS2RtxLidarHelper ──► /os_cloud_node/points (ROS 2)
               ──► IsaacComputeOdometry
                        │
                        ▼
                   ROS2PublishOdometry ──► /odom_lidar  (ROS 2)
```

Key node types:

| Node | What it does |
|------|-------------|
| `OnPlaybackTick` | Fires every sim step (the clock) |
| `IsaacReadSimulationTime` | Reads sim clock → for message headers |
| `ROS2CameraHelper` | Render product → `sensor_msgs/Image` |
| `ROS2RtxLidarHelper` | RTX hits → `sensor_msgs/PointCloud2` |
| `IsaacComputeOdometry` | Robot prim pose/vel → odometry data |
| `ROS2PublishOdometry` | Odometry data → `nav_msgs/Odometry` |

---

## Extensions

Extensions are like ROS 2 packages — optional features you load.
The ROS 2 bridge is an extension: `isaacsim.ros2.bridge`.

Enable it via: **Window → Extensions** → search "ROS2" → enable.
Or in Python:
```python
manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
```

---

## Common Issues

### Topics don't appear in `ros2 topic list`
- Check the ROS 2 bridge extension is enabled
- Check simulation is playing (press ▶ or call `timeline.play()`)
- Check `ROS_DOMAIN_ID` matches between Isaac Sim and your terminal:
  ```bash
  export ROS_DOMAIN_ID=0   # default
  ```
- Source ROS 2 before launching Isaac Sim:
  ```bash
  source /opt/ros/humble/setup.bash
  ```

### Camera images look wrong / black
- Render products need a frame to warm up — wait 2–3 seconds after pressing play
- Check the camera prim path in `cam_fl.inputs:renderProductPath`

### LiDAR pointcloud is empty
- Make sure `fullScan=True` — it waits for a complete 360° sweep
- The Ouster OS1 profile rotates at 20 Hz so the first cloud arrives after ~50 ms

### Odometry is always zero
- `odom_compute.inputs:chassisPrimPath` must point to the **root** articulation prim
  (i.e. `/World/Spot`, not a link inside Spot)
- The robot must be moving — drive it with the keyboard (`W/A/S/D`) or publish
  `geometry_msgs/Twist` to `/cmd_vel`

---

## Running DR.Nav After Setup

```bash
# Terminal 1 — run SLAM so /map is available
ros2 launch slam_toolbox online_async_launch.py

# Terminal 2 — launch DR.Nav pipeline
ros2 launch map_contruct mapless.launch.py method:=dram

# Terminal 3 — record bag for metrics
ros2 bag record /ede_score /odom_lidar /dead_end_detection/path_status
```
