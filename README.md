# DR.Nav — Dead-End Risk-Aware Navigation

**DR.Nav** is a ROS 2 navigation system that uses RGB-LiDAR sensor fusion and a deep learning model to predict dead ends proactively and recover from them autonomously. The system operates in both mapless (exploration) and map-based (structured) environments.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture: DR.Nav Method](#2-architecture-drnav-method)
   - [Full Pipeline Flowchart](#full-pipeline-flowchart)
   - [Model Inference Flowchart](#model-inference-flowchart)
   - [Risk Mapping and Recovery Flowchart](#risk-mapping-and-recovery-flowchart)
   - [Goal Generation Flowchart](#goal-generation-flowchart)
3. [Node Reference](#3-node-reference)
4. [ROS 2 Topic Map](#4-ros-2-topic-map)
5. [Hardware Requirements](#5-hardware-requirements)
6. [Software Requirements](#6-software-requirements)
7. [Installation](#7-installation)
8. [Path Configuration — Required Before First Run](#8-path-configuration--required-before-first-run)
9. [Running on a Real Robot](#9-running-on-a-real-robot)
   - [Mapless Mode](#91-mapless-mode)
   - [Map-Based Mode](#92-map-based-mode)
10. [Launch File Arguments](#10-launch-file-arguments)
11. [RViz Visualization Topics](#11-rviz-visualization-topics)
12. [Key Parameters](#12-key-parameters)
13. [Package Structure](#13-package-structure)

---

## 1. System Overview

DR.Nav consists of four sequential processing stages that run in parallel as independent ROS 2 nodes:

```
[Sensors]  →  [Inference]  →  [Risk Mapping]  →  [Goal Generation]  →  [Planner]  →  [Robot]
```

| Stage | Node | Role |
|---|---|---|
| Sensor preprocessing | `pointcloud_segmenter` | Splits omnidirectional LiDAR into directional sectors |
| Dead-end inference | `infer_vis` | Runs the DRaM model; outputs path probabilities per direction |
| Risk mapping | `dram_risk_map` | Builds a Bayesian safety grid; tracks recovery waypoints |
| Goal generation | `goal_generator` | Scores candidate headings; sends waypoints to the planner |
| Local planning | `dwa_planner` | Executes velocity commands toward the current waypoint |

The system uses a unified scoring formula across all methods:

```
Score(θ) = J_geom(θ) + λ · EDE(θ)
```

- **J_geom**: Geometric cost — collision avoidance, feasibility, smoothness, range bias
- **EDE**: Exposure to Dead-End — accumulated dead-end risk integral along heading θ
- **λ**: Weighting coefficient — `λ = 1.0` for DR.Nav, `λ = 0.0` for all baselines

---

## 2. Architecture: DR.Nav Method

### Full Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HARDWARE SENSORS                                   │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │  Camera     │  │  Camera     │  │  Camera     │  │  Ouster LiDAR    │  │
│  │  Front-Left │  │  Side-Left  │  │  Side-Right │  │  (Omnidirectional│  │
│  │  (AR0234)   │  │  (AR0234)   │  │  (AR0234)   │  │   Full Scan)     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘  │
│         │                │                │                   │            │
└─────────┼────────────────┼────────────────┼───────────────────┼────────────┘
          │                │                │                   │
          │ /argus/ar0234_front_left/image_raw                  │ /os_cloud_node/points
          │ /argus/ar0234_side_left/image_raw                   │
          │ /argus/ar0234_side_right/image_raw                  │
          │                │                │                   ▼
          │                │                │        ┌──────────────────────┐
          │                │                │        │  pointcloud_segmenter│
          │                │                │        │  (Sector Splitter)   │
          │                │                │        └──────────┬───────────┘
          │                │                │                   │
          │                │                │    ┌──────────────┼──────────────┐
          │                │                │    │              │              │
          │                │                │  /lidar/    /lidar/        /lidar/
          │                │                │  front/     left/          right/
          │                │                │  points     points         points
          │                │                │    │              │              │
          ▼                ▼                ▼    ▼              ▼              ▼
          └────────────────┴────────────────┴────┴──────────────┴──────────────┘
                                            │
                              ┌─────────────▼──────────────┐
                              │        infer_vis            │
                              │  (DRaM Model Inference)     │
                              │                             │
                              │  Input:                     │
                              │   3× RGB image (224×224)    │
                              │   3× PointCloud (1024 pts)  │
                              │                             │
                              │  Output:                    │
                              │   path_status  [3×float]    │
                              │   is_dead_end  [bool]       │
                              └─────────┬───────────────────┘
                                        │
               /dead_end_detection/path_status  [front, left, right] ∈ [0,1]
               /dead_end_detection/is_dead_end  [true/false]
                                        │
                              ┌─────────▼───────────────────┐
                              │       dram_risk_map          │
                              │  (Bayesian Risk Grid +       │
                              │   Recovery Point Manager)    │
                              │                             │
                              │  Inputs:                    │
                              │   path_status               │
                              │   /map (OccupancyGrid)      │
                              │   TF: map → body            │
                              │                             │
                              │  Outputs:                   │
                              │   dram_exploration_map      │
                              │   recovery_points           │
                              │   cost_layer (RViz only)    │
                              └─────────┬───────────────────┘
                                        │
               /dram_exploration_map  (MarkerArray — EDE heatmap)
               /dead_end_detection/recovery_points  (Float32MultiArray)
                                        │
                              ┌─────────▼───────────────────┐
                              │      goal_generator          │
                              │  (Unified Scoring + Waypoint │
                              │   Selection)                 │
                              │                             │
                              │  Score(θ) = J_geom + λ·EDE  │
                              │  λ = 1.0  (DR.Nav)          │
                              │  36 rays × [-π, π]          │
                              │                             │
                              │  Recovery mode:             │
                              │   triggered if stuck > 5s   │
                              │   navigates to best stored  │
                              │   recovery point            │
                              └─────────┬───────────────────┘
                                        │
                              /move_base_simple/goal  (PoseStamped)
                                        │
                              ┌─────────▼───────────────────┐
                              │        dwa_planner           │
                              │  (Dynamic Window Approach)   │
                              │                             │
                              │  Inputs:  goal, /map, TF    │
                              │  Output: /cmd_vel            │
                              └─────────────────────────────┘
```

---

### Model Inference Flowchart

```
                    ┌────────────────────────────────────────┐
                    │          infer_vis Node                │
                    │     (dead_end_detection_visual_node)   │
                    └────────────────────────────────────────┘

  ┌──────────────┐      ┌────────────────────────────────────────────────────┐
  │  Camera      │      │  PREPROCESSING                                     │
  │  Callbacks   │      │                                                    │
  │  (3× Image)  │─────▶│  Resize to 224×224                                 │
  └──────────────┘      │  Normalize: mean=[0.485,0.456,0.406]               │
                        │             std=[0.229,0.224,0.225]                │
  ┌──────────────┐      │  Stack → image_tensors [3, 3, 224, 224]            │
  │  LiDAR       │      │                                                    │
  │  Callbacks   │      │  Point cloud: sample 1024 pts (robot_mode=True)    │
  │  (3× PC2)    │─────▶│  Fields: x, y, z, intensity                        │
  └──────────────┘      │  Stack → lidar_tensors [3, 1024, 4]                │
                        └───────────────────┬────────────────────────────────┘
                                            │
                                            │  (batched every 1/5 s in robot_mode)
                                            ▼
                        ┌───────────────────────────────────────────────────────┐
                        │           DeadEndDetectionModel (model_CA.py)         │
                        │                                                       │
                        │   Input branches:                                     │
                        │   ┌─────────────┐    ┌────────────────────────────┐  │
                        │   │ Image branch│    │ LiDAR branch               │  │
                        │   │ (CNN embed) │    │ (PointNet-style embed)     │  │
                        │   └──────┬──────┘    └─────────────┬──────────────┘  │
                        │          └──────────┬───────────────┘                │
                        │                     ▼                                │
                        │           Cross-Attention Fusion                     │
                        │                     ▼                                │
                        │           Shared Trunk (MLP)                        │
                        │                     ▼                                │
                        │   ┌─────────────────┴─────────────────┐             │
                        │   │  path_status head                 │             │
                        │   │  (3 logits → sigmoid → probs)     │             │
                        │   │  [front_prob, left_prob,          │             │
                        │   │   right_prob]  ∈ [0, 1]           │             │
                        │   └───────────────────────────────────┘             │
                        └───────────────────────────────────────────────────┘
                                            │
                                            ▼
                        ┌───────────────────────────────────────────────────────┐
                        │            DECISION LOGIC                             │
                        │                                                       │
                        │   threshold = 0.56                                   │
                        │                                                       │
                        │   for each direction d ∈ {front, left, right}:       │
                        │       open[d] = (prob[d] > 0.56)                     │
                        │                                                       │
                        │   is_dead_end = (open[front] == open[left]            │
                        │                  == open[right] == False)             │
                        └───────────────────┬───────────────────────────────────┘
                                            │
                    ┌───────────────────────┴────────────────────────┐
                    │                                                │
                    ▼                                                ▼
     /dead_end_detection/path_status               /dead_end_detection/is_dead_end
     Float32MultiArray                             std_msgs/Bool
     [front_prob, left_prob, right_prob]           true  if all 3 blocked
     published at ~5 Hz                            false otherwise
```

---

### Risk Mapping and Recovery Flowchart

```
                   ┌────────────────────────────────────────────┐
                   │           dram_risk_map Node               │
                   └────────────────────────────────────────────┘

  /dead_end_detection/path_status ──────────────────────────────┐
  /map  (OccupancyGrid)  ────────────────────────────────────┐  │
  TF: map → body  ────────────────────────────────────────┐  │  │
                                                          │  │  │
                                                          ▼  ▼  ▼
                              ┌───────────────────────────────────────┐
                              │  path_status_callback()               │
                              │                                       │
                              │  1. TF lookup: robot_x, robot_y, yaw  │
                              │  2. Decode probs [front, left, right] │
                              │  3. Classify: open if prob > 0.56     │
                              └──────────────┬────────────────────────┘
                                             │
               ┌─────────────────────────────┼────────────────────────────────┐
               │                             │                                │
               ▼                             ▼                                ▼
  ┌────────────────────────┐   ┌──────────────────────────┐   ┌────────────────────────┐
  │  RecoveryPointManager  │   │  Bayesian Safety Grid    │   │  Sector Visualisation  │
  │                        │   │                          │   │                        │
  │  If any direction open:│   │  For each cell within    │   │  3 triangle sectors    │
  │  store (x, y, rank)    │   │  3.0 m of robot:         │   │  front / left / right  │
  │                        │   │  safety_level:           │   │  green  = open         │
  │  rank = # open dirs    │   │   1.0 if any open        │   │  red    = blocked      │
  │  (1 = tight, 3 = free) │   │   0.0 if dead end        │   │  alpha  = 0.6          │
  │                        │   │                          │   │                        │
  │  Deduplication:        │   │  weight by distance:     │   │  Historical fade:      │
  │  within 1.0 m keep     │   │  w = 1 - dist/radius     │   │  past sectors shown    │
  │  highest rank only     │   │                          │   │  at alpha = 0.25       │
  │                        │   │  Navigability check:     │   │  for 20 s              │
  │  Expiry: 60 s          │   │  skip cells with         │   └────────────┬───────────┘
  │  Capacity: 50 points   │   │  occupancy > 0           │                │
  └────────────┬───────────┘   └──────────────┬───────────┘                │
               │                              │                             │
               ▼                              ▼                             ▼
  /dead_end_detection/recovery_points   /dram_exploration_map         /cost_layer
  Float32MultiArray                     MarkerArray                   MarkerArray
  [type, x, y, type, x, y, ...]        ns="exploration_heatmap"      (RViz debug)
  type 1.0 = rank ≥ 2 (preferred)      green = safe  (safety ≥ 0.5)
  type 2.0 = rank = 1  (fallback)      red   = risky (safety < 0.5)
```

---

### Goal Generation Flowchart

```
                    ┌───────────────────────────────────────┐
                    │        goal_generator Node            │
                    │     method_type = multi_camera_dram   │
                    │     lambda_ede  = 1.0                 │
                    │     rate        = 7.0 Hz              │
                    │     horizon     = 4.0 m               │
                    └───────────────────────────────────────┘

  Inputs:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  /dram_exploration_map      — Bayesian risk grid (from dram_risk_map)   │
  │  /dead_end_detection/is_dead_end   — current blocked status             │
  │  /dead_end_detection/path_status   — per-direction probabilities        │
  │  /dead_end_detection/recovery_points — stored safe waypoints            │
  │  /map                       — OccupancyGrid                             │
  │  TF: map → base_link                                                    │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                     NORMAL NAVIGATION MODE                              │
  │                                                                         │
  │  Sample 36 headings θ uniformly over [-π, π]  (10° spacing)            │
  │                                                                         │
  │  For each heading θ:                                                    │
  │  ┌───────────────────────────────────────────────────────────────────┐  │
  │  │  Cast ray at θ for horizon_distance = 4.0 m                      │  │
  │  │  step = 0.15 m                                                   │  │
  │  │                                                                  │  │
  │  │  J_geom(θ):                                                      │  │
  │  │    collision_cost   = Σ inflated_costmap(step)  × 50.0           │  │
  │  │    feasibility_cost = 1000.0  if ray hits obstacle               │  │
  │  │    smoothness_cost  = Δyaw × 0.5                                 │  │
  │  │    range_bias_cost  = (horizon - actual_length) × 0.1            │  │
  │  │                                                                  │  │
  │  │  EDE(θ):  (DR.Nav only, λ = 1.0)                                │  │
  │  │    EDE = Σᵢ risk_prob[i] × Δs                                   │  │
  │  │    bilinear sample of explored_grid at each step i              │  │
  │  │                                                                  │  │
  │  │  Score(θ) = J_geom(θ) + 1.0 × EDE(θ)                          │  │
  │  └───────────────────────────────────────────────────────────────┘  │
  │                                                                         │
  │  Select θ* = argmin Score(θ)                                           │
  │  Waypoint = robot_pos + horizon × [cos(θ*), sin(θ*)]                  │
  │  Publish  /move_base_simple/goal                                        │
  └─────────────────────────────────────────────────────────────────────────┘

                              │
               ┌──────────────┴──────────────┐
               │                             │
     is_dead_end = True          robot_displacement < 0.1 m
     (all dirs blocked)          for > 5 consecutive seconds
               └──────────────┬──────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        RECOVERY MODE                                    │
  │                                                                         │
  │  Query RecoveryPointManager for best stored recovery point:             │
  │    get_best_recovery_point()  → highest rank (most open directions)     │
  │    get_nearest_recovery_point() → fallback if no best                   │
  │                                                                         │
  │  Score each recovery candidate:                                         │
  │    recovery_score = α × goal_progress                                   │
  │                   + β × clearance                                       │
  │                   - γ × roughness                                       │
  │                   - δ × ede_from_recovery_point                         │
  │                                                                         │
  │    α = 1.0, β = 2.0, γ = 0.5, δ = 0.3                                 │
  │                                                                         │
  │  Publish recovery waypoint → robot backtracks to safe position          │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Node Reference

### `pointcloud_segmenter`

Splits the full Ouster LiDAR scan into three directional sectors using azimuth angle.

| | |
|---|---|
| **Subscribes** | `/os_cloud_node/points` — full LiDAR scan |
| **Publishes** | `/lidar/front/points`, `/lidar/left/points`, `/lidar/right/points` |
| **Sector angles** | Front: −30° to +30° / Left: +60° to +120° / Right: −120° to −60° |

---

### `infer_vis`

Runs the DRaM deep learning model (RGB + LiDAR fusion) and publishes per-direction path probabilities.

| | |
|---|---|
| **Subscribes** | 3× camera image topics, 3× directional LiDAR point clouds |
| **Publishes** | `/dead_end_detection/path_status` (Float32MultiArray), `/dead_end_detection/is_dead_end` (Bool) |
| **Model input** | 3× image tensors (224×224) + 3× point clouds (1024 points each) |
| **Model output** | Sigmoid probabilities per direction — threshold 0.56 |
| **Processing rate** | 5 Hz (robot mode), 10 Hz (rosbag mode) |
| **Executor** | MultiThreadedExecutor, 4 threads |

---

### `dram_risk_map`

Merges Bayesian risk grid construction and recovery point tracking into a single node.

| | |
|---|---|
| **Subscribes** | `/dead_end_detection/path_status`, `/map` |
| **Publishes** | `/dram_exploration_map` (risk heatmap), `/dead_end_detection/recovery_points`, `/cost_layer` (RViz) |
| **Grid resolution** | 0.3 m per cell |
| **Exploration radius** | 3.0 m around robot |
| **Threshold** | 0.56 (consistent with infer_vis) |

---

### `goal_generator`

Scores 36 candidate headings using the unified DR.Nav formula and publishes waypoints at 7 Hz.

| | |
|---|---|
| **Subscribes** | `/dram_exploration_map`, `/dead_end_detection/recovery_points`, `/dead_end_detection/is_dead_end`, `/dead_end_detection/path_status`, `/map` |
| **Publishes** | `/move_base_simple/goal` (PoseStamped), `/goal_generator/waypoints`, `/goal_generator/rays` (visualization) |
| **Parameters** | `method_type=multi_camera_dram`, `lambda_ede=1.0`, `goal_generation_rate=7.0`, `horizon_distance=4.0` |

---

### `dwa_planner`

Executes velocity commands toward the current waypoint using the Dynamic Window Approach.

| | |
|---|---|
| **Subscribes** | `/move_base_simple/goal`, `/map` |
| **Publishes** | `/cmd_vel` (Twist) |
| **TF lookup** | `map` → `base_link` |
| **Control rate** | 10 Hz |

---

### `odom_tf_broadcaster`

Publishes the `odom → body` transform from odometry data.

| | |
|---|---|
| **Subscribes** | `/odom_lidar` (Odometry) |
| **Publishes** | TF: `odom → body` |
| **Parameters** | `odom_topic=/odom_lidar`, `parent_frame=odom`, `child_frame=body` |

---

## 4. ROS 2 Topic Map

```
SENSORS
  /os_cloud_node/points         ──▶  pointcloud_segmenter
  /argus/ar0234_front_left/...  ──┐
  /argus/ar0234_side_left/...   ──┤─▶  infer_vis
  /argus/ar0234_side_right/...  ──┘
  /lidar/front/points           ──┐
  /lidar/left/points            ──┤─▶  infer_vis
  /lidar/right/points           ──┘
  /odom_lidar                   ──▶  odom_tf_broadcaster  ──▶  TF(odom→body)
  /map                          ──▶  dram_risk_map, goal_generator, dwa_planner

INFERENCE
  /dead_end_detection/path_status    ──▶  dram_risk_map, goal_generator
  /dead_end_detection/is_dead_end    ──▶  goal_generator

RISK MAPPING
  /dram_exploration_map              ──▶  goal_generator
  /dead_end_detection/recovery_points──▶  goal_generator
  /cost_layer                        ──▶  RViz

GOAL GENERATION
  /move_base_simple/goal             ──▶  dwa_planner
  /goal_generator/waypoints          ──▶  RViz
  /goal_generator/rays               ──▶  RViz

CONTROL
  /cmd_vel                           ──▶  Robot hardware driver
```

---

## 5. Hardware Requirements

| Component | Specification |
|---|---|
| Cameras | 3× Argus AR0234 (front-left, side-left, side-right) |
| LiDAR | Ouster OS-series (omnidirectional, publishes `/os_cloud_node/points`) |
| Odometry | Topic `/odom_lidar` (nav_msgs/Odometry) |
| GPU | NVIDIA GPU with CUDA (required for real-time model inference) |
| CPU | Minimum 4 cores recommended (inference runs MultiThreadedExecutor) |
| RAM | Minimum 8 GB |

---

## 6. Software Requirements

| Dependency | Version |
|---|---|
| ROS 2 | Humble or later |
| Python | 3.8+ |
| PyTorch | Compatible with installed CUDA version |
| slam_toolbox | Any ROS 2 compatible version (mapless mode only) |
| nav2_bringup | Any ROS 2 compatible version (map-based mode only) |

Python packages:

```bash
pip install torch torchvision numpy opencv-python pillow matplotlib
```

ROS 2 packages:

```
sensor_msgs  tf2_ros  tf2_geometry_msgs  tf2_sensor_msgs  nav_msgs
visualization_msgs  geometry_msgs  std_msgs  rclpy
```

---

## 7. Installation

```bash
# Clone into your ROS 2 workspace source directory
cd ~/ros2_ws/src
git clone <repository-url> map_contruct_pkg

# Install Python dependencies
pip install torch torchvision numpy opencv-python pillow matplotlib

# Build the package
cd ~/ros2_ws
colcon build --packages-select map_contruct
source install/setup.bash
```

---

## 8. Path Configuration — Required Before First Run

The following hardcoded paths must be updated to match your machine before the system will start correctly.

---

### Critical — System will fail to start without these

**File:** `map_contruct/scripts/inference/infer_vis.py`

**Line 69 — Model weights path:**

```python
# Change this to the full absolute path of your model_best.pth file
model_path = '/home/mrvik/dram_ws/model_wts/model_best.pth'
```

Change to:

```python
model_path = '/your/path/to/model_best.pth'
```

The model weights file (`model_best.pth`) must exist at this path. If the file is missing or the path is wrong, the node will log an error and no inference will be produced.

---

### Optional — Only required if `save_visualizations=True`

**File:** `map_contruct/scripts/inference/infer_vis.py`

**Line 42 — Inference output directory:**

```python
self.output_dir = '/home/mrvik/dram_ws/inference_results'
```

Change to any writable directory on your system:

```python
self.output_dir = '/your/path/to/output_directory'
```

This directory is used to save per-frame PNG images and a JSON metrics file. It is **only active** when the node is launched with `save_visualizations:=true`. In standard robot operation (`save_visualizations:=false`, which is the default in the launch files), this path is never accessed.

---

### Optional — Only required if using `slam.py`

**File:** `map_contruct/scripts/utilities/slam.py`

**Line 21 — SLAM labels CSV path:**

```python
self.declare_parameter('csv_path',
    '/home/mrvik/dram_ws/src/map_contruct/map_contruct/labels.csv')
```

This is a ROS 2 parameter with a default value. Override it at launch:

```bash
ros2 run map_contruct slam --ros-args -p csv_path:=/your/path/to/labels.csv
```

---

### Optional — Only required if using evaluation scripts

The following files contain hardcoded paths used only during offline evaluation and post-processing. They are not part of the live robot pipeline.

| File | Line | Path to change |
|---|---|---|
| `scripts/viz/evaluation_framework.py` | 79 | Results output: `/home/mrvik/dram_ws/evaluation_results/` |
| `scripts/viz/evaluation_framework.py` | 185 | Metrics scan base dir: `/home/mrvik/dram_ws/evaluation_results` |
| `scripts/viz/enhanced_evaluation_framework.py` | 49 | Global map file: `/path/to/your/global_map.yaml` |
| `scripts/viz/enhanced_evaluation_framework.py` | 124 | Results output: `/home/mrvik/dram_ws/evaluation_results/` |
| `scripts/viz/enhanced_evaluation_framework.py` | 235 | Metrics scan base dir: `/home/mrvik/dram_ws/evaluation_results` |
| `scripts/viz/deadend_prediction_visualizer.py` | 25 | Figures output: `/home/mrvik/dram_ws/deadend_prediction_figures` |
| `scripts/viz/comprehensive_deadend_visualizer.py` | 29 | Figures output: `/home/mrvik/dram_ws/comprehensive_figures` |
| `scripts/viz/method_comparison_analyzer.py` | 35 | Base dir: `/home/mrvik/dram_ws` |

---

## 9. Running on a Real Robot

### TF Frame Requirements

The system requires the following TF tree to be present:

```
map ──▶ odom ──▶ body
```

- `map → odom`: Provided by SLAM (slam_toolbox) or AMCL (nav2_bringup)
- `odom → body`: Provided by `odom_tf_broadcaster` (included in all launch files)

Ensure your hardware drivers publish `/odom_lidar` (nav_msgs/Odometry) and that the LiDAR and camera drivers are running before launching.

---

### 9.1 Mapless Mode

Use this mode for exploration in an unknown environment. SLAM builds the map on the fly.

**Terminal 1 — SLAM:**

```bash
ros2 launch slam_toolbox online_async_launch.py
```

**Terminal 2 — DR.Nav:**

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch map_contruct mapless.launch.py method:=dram
```

To open RViz alongside:

```bash
ros2 launch map_contruct mapless.launch.py method:=dram use_rviz:=true
```

Nodes started by this command:

```
odom_tf_broadcaster
goal_generator       (method_type=multi_camera_dram, lambda_ede=1.0)
pointcloud_segmenter
infer_vis            (robot_mode=true, save_visualizations=false)
dram_risk_map
dwa_planner
```

---

### 9.2 Map-Based Mode

Use this mode when a pre-built map is available. Nav2 handles localization and global planning. DR.Nav adds the dead-end perception stack on top.

**Terminal 1 — Nav2 with your map:**

```bash
ros2 launch nav2_bringup bringup_launch.py \
  map:=/path/to/your_map.yaml \
  params_file:=/path/to/nav2_params.yaml
```

**Terminal 2 — DR.Nav perception stack:**

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch map_contruct map_based.launch.py method:=dram
```

Nodes started by this command:

```
pointcloud_segmenter
infer_vis            (robot_mode=true, save_visualizations=false)
dram_risk_map
```

In map-based mode, `goal_generator` and `dwa_planner` are **not** launched. Navigation goals are sent via the RViz "Nav2 Goal" button, which feeds directly into Nav2's `bt_navigator` → `planner_server` → `controller_server` pipeline.

---

## 10. Launch File Arguments

### `mapless.launch.py`

```bash
ros2 launch map_contruct mapless.launch.py method:=<method> [use_rviz:=true]
```

| Argument | Options | Default | Description |
|---|---|---|---|
| `method` | `dram`, `dwa`, `mppi`, `nav2_dwb` | `dwa` | Navigation method to run |
| `use_rviz` | `true`, `false` | `false` | Launch RViz for visualization |

Method behaviour:

| Method | Nodes launched | `lambda_ede` |
|---|---|---|
| `dram` | odom_tf_broadcaster + infer_vis + pointcloud_segmenter + dram_risk_map + goal_generator + dwa_planner | 1.0 |
| `dwa` | odom_tf_broadcaster + goal_generator + dwa_planner | 0.0 |
| `mppi` | odom_tf_broadcaster + goal_generator + mppi_planner | 0.0 |
| `nav2_dwb` | odom_tf_broadcaster + goal_generator + nav2_dwb_planner | 0.0 |

---

### `map_based.launch.py`

```bash
ros2 launch map_contruct map_based.launch.py method:=<method> [use_rviz:=true]
```

| Argument | Options | Default | Description |
|---|---|---|---|
| `method` | `dram`, `dwa`, `mppi`, `nav2_dwb` | `dram` | Navigation method |
| `use_rviz` | `true`, `false` | `false` | Launch RViz |

Method behaviour:

| Method | Additional nodes launched | Notes |
|---|---|---|
| `dram` | pointcloud_segmenter + infer_vis + dram_risk_map | Nav2 drives; DR.Nav provides risk perception |
| `dwa` | None | Handled entirely by nav2_bringup (use nav2_params_dwa.yaml) |
| `mppi` | None | Handled entirely by nav2_bringup (use nav2_params_mppi.yaml) |
| `nav2_dwb` | None | Handled entirely by nav2_bringup (default nav2_params.yaml) |

---

## 11. RViz Visualization Topics

Add these topics to an RViz configuration to monitor the system:

| Topic | Type | What it shows |
|---|---|---|
| `/dram_exploration_map` | MarkerArray | Risk heatmap — green=safe, red=dead-end risk |
| `/cost_layer` | MarkerArray | Directional sector overlay at robot position |
| `/goal_generator/waypoints` | MarkerArray | Current target waypoint |
| `/goal_generator/rays` | MarkerArray | All 36 scored heading rays |
| `/map` | OccupancyGrid | Occupancy grid (from SLAM or Nav2) |

Recovery points and direction status spheres are included within `/dram_exploration_map` (namespaces: `recovery_pins`, `recovery_labels`, `status_center`).

---

## 12. Key Parameters

| Parameter | Node | Default | Description |
|---|---|---|---|
| `robot_mode` | `infer_vis` | `true` | Use BEST_EFFORT QoS, 1024 pts/cloud, 5 Hz inference |
| `save_visualizations` | `infer_vis` | `false` | Save per-frame PNGs and JSON metrics to disk |
| `method_type` | `goal_generator` | `multi_camera_dram` | Selects EDE scoring and recovery logic |
| `lambda_ede` | `goal_generator` | `1.0` | EDE weight in unified scoring (0 = baseline) |
| `goal_generation_rate` | `goal_generator` | `7.0` | Waypoint sampling frequency (Hz) |
| `horizon_distance` | `goal_generator` | `4.0` | Look-ahead ray length (metres) |
| `odom_topic` | `odom_tf_broadcaster` | `/odom_lidar` | Source odometry topic |
| `parent_frame` | `odom_tf_broadcaster` | `odom` | Parent frame for TF broadcast |
| `child_frame` | `odom_tf_broadcaster` | `body` | Child frame for TF broadcast |

---

## 13. Package Structure

```
ros2_drnav_deadend_recovery/
├── map_contruct/
│   ├── scripts/
│   │   ├── inference/
│   │   │   └── infer_vis.py          # DRaM model inference node
│   │   ├── models/
│   │   │   └── model_CA.py           # DeadEndDetectionModel architecture
│   │   └── utilities/
│   │       ├── pointcloud_segmenter.py   # LiDAR sector splitter
│   │       ├── odom_tf_brodcaster.py     # Odometry → TF broadcaster
│   │       ├── dram_risk_map.py          # Bayesian risk grid + recovery manager
│   │       └── recovery_points.py        # RecoveryPointManager class
│   ├── goal_generator/
│   │   └── goal_generator.py             # Unified scoring and waypoint selection
│   └── baselines/
│       ├── dwa/
│       │   └── dwa_planner.py            # DWA local planner
│       ├── mppi/
│       │   └── mppi_planner.py           # MPPI local planner
│       └── nav2_dwb/
│           └── nav2_dwb_planner.py       # Nav2 DWB local planner
├── launch/
│   ├── mapless.launch.py                 # Mapless mode: all methods
│   └── map_based.launch.py               # Map-based mode: all methods
├── config/
│   └── evaluation_config.yaml            # Evaluation trial parameters
├── setup.py
└── package.xml
```
